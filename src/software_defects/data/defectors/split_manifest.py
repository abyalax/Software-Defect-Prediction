"""Row-level split manifest: which dataset rows are train, val and test in every scenario and fold."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import MAPPING_RELATIVE_PATH, load_holdout_repos
from software_defects.data.defectors.exploratory.run_report import JIT_SPLIT_NAMES, RunReport
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitResult,
    SplitScenario,
)
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

MANIFEST_FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
# Must stay equal to the defaults of ``DefectorsSplitGenerator.generate_split`` (guarded by a unit test),
# because ``run-pipeline`` calls it with those defaults and the manifest has to describe the same split.
SPLIT_SEED = 42
VAL_RATIO = 0.1
MANIFEST_FILENAME = "manifest.json"
ROW_IDS_FILENAME = "row_ids.parquet"
INDEX_FILENAME = "index.json"
# The legacy ``Cross_Domain_Default`` scenario duplicates ML_to_NonML and is not one of the seven thesis scenarios
MANIFEST_SCENARIOS = tuple(scenario for scenario in SplitScenario if scenario is not SplitScenario.CROSS_DOMAIN_DEFAULT)
TRACKED_PACKAGES = ("pandas", "numpy", "scikit-learn", "pyarrow")

ROW_ID_DEFINITION = (
    "0-based position of the row in concat(jit train, jit test, jit val) of the original Defectors "
    "'random' parquet files, i.e. the row index the pipeline holds before filtering"
)
IDS_HASH_DEFINITION = "sha256 over the ascending row ids written as little-endian int64"

ManifestEntry = tuple[SplitScenario, int | None]


def hash_row_ids(row_ids: np.ndarray | pd.Index | list[int]) -> str:
    """Order-independent SHA-256 of a set of row ids.

    Args:
        row_ids: Row ids of one split.

    Returns:
        Hex digest of the ascending ids written as little-endian int64.
    """
    ordered = np.sort(np.asarray(row_ids, dtype="<i8"))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def hash_contract(contract: dict) -> str:
    """SHA-256 of the canonical JSON form of a manifest contract.

    Args:
        contract: The hashed part of a manifest (everything except the provenance).

    Returns:
        Hex digest of the contract serialised with sorted keys and no whitespace.
    """
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    """Run a git command in the project root; None when git or the repository is unavailable."""
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fold_label(scenario: SplitScenario, fold: int | None) -> str:
    """Readable name of a manifest entry, e.g. ``Group_KFold_ML_DL fold 2``."""
    return f"{scenario.value} fold {fold}" if fold is not None else scenario.value


class DefectorsSplitManifest:
    """Persist and verify the row ids of every split scenario (and fold) with their hashes.

    A manifest freezes the split the pipeline produces: for each scenario and fold it stores the ids of the
    train, val and test rows and the SHA-256 of each id set. Every training and evaluation run can then cite
    the manifest hash, and ``verify`` regenerates the splits from the raw data to prove that nothing drifted.

    Layout under ``artifacts/split_manifest/<dataset>/``::

        index.json                                  all manifest hashes and one hash over the whole set
        <scenario>/manifest.json                    parameters, per-split counts and hashes, provenance
        <scenario>/row_ids.parquet                  columns ``row_id`` and ``split``
        Group_KFold_*/fold_<n>/...                  same two files per fold
    """

    def __init__(self, config: DatasetConfig, manifest_dir: Path | None = None) -> None:
        """Initialize the manifest builder and verifier.

        Args:
            config: Dataset configuration.
            manifest_dir: Root directory of the manifests; defaults to ``config.split_manifest_dir``.
        """
        self._config = config
        self._manifest_dir = Path(manifest_dir) if manifest_dir is not None else config.split_manifest_dir
        self._loader = DefectorsDatasetLoader(config)
        self._generator = DefectorsSplitGenerator(config)
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="data_pipeline",
            class_name="DefectorsSplitManifest",
        )

    @property
    def manifest_dir(self) -> Path:
        """Root directory of the manifests."""
        return self._manifest_dir

    @staticmethod
    def manifest_entries(scenarios: list[SplitScenario] | None = None, fold: int | None = None) -> list[ManifestEntry]:
        """List the (scenario, fold) pairs that have a manifest.

        Args:
            scenarios: Scenarios to include; all seven scenarios when omitted.
            fold: Single fold index, only valid together with Group K-Fold scenarios.

        Returns:
            One entry per scenario, or one per fold (fold 0 to 4) for the Group K-Fold scenarios.

        Raises:
            ValueError: If a scenario has no manifest, the fold is out of range, or a fold is given for a
                scenario that is not Group K-Fold.
        """
        selected = list(scenarios) if scenarios else list(MANIFEST_SCENARIOS)
        for scenario in selected:
            if scenario not in MANIFEST_SCENARIOS:
                raise ValueError(f"No manifest is kept for scenario {scenario.value}")
        if fold is not None:
            if not 0 <= fold < GROUP_KFOLD_N_FOLDS:
                raise ValueError(f"fold must be in [0, {GROUP_KFOLD_N_FOLDS - 1}], got {fold}")
            if any(scenario not in GROUP_KFOLD_SCENARIOS for scenario in selected):
                raise ValueError("--fold is only valid together with a Group_KFold scenario (pass --scenario)")

        entries: list[ManifestEntry] = []
        for scenario in selected:
            if scenario in GROUP_KFOLD_SCENARIOS:
                folds = [fold] if fold is not None else list(range(GROUP_KFOLD_N_FOLDS))
                entries.extend((scenario, fold_index) for fold_index in folds)
            else:
                entries.append((scenario, None))
        return entries

    def entry_dir(self, scenario: SplitScenario, fold: int | None = None) -> Path:
        """Directory that holds the manifest of one scenario (and fold).

        Args:
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            Directory with ``manifest.json`` and ``row_ids.parquet``.
        """
        return self._generator.scenario_dir(self._manifest_dir, scenario, fold or 0)

    def write_from_split_result(
        self,
        result: SplitResult,
        scenario: SplitScenario,
        fold: int | None = None,
    ) -> Path:
        """Write a manifest from an existing SplitResult (called by run-pipeline).

        This method is used by the pipeline to write the manifest immediately after
        generating the split, ensuring the manifest describes the exact same split that
        produced the parquet files.

        Args:
            result: The generated split result.
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            Directory where the manifest was written.

        Raises:
            ValueError: If the dataset is row-limited or the split leaks commits/repositories.
        """
        self._ensure_full_dataset()

        project_root = self._config.project_root
        dirty = _git(project_root, "status", "--porcelain")
        if dirty:
            self._reporter.display_warning(
                "  Working tree has uncommitted changes; commit first so the recorded commit describes this manifest"
            )

        data_hashes = self._data_hashes()
        provenance = self._provenance(dirty)

        contract, row_ids = self._build_contract_from_result(result, scenario, fold, data_hashes)
        issues = self._overlap_issues(result)
        if issues:
            raise ValueError(f"{_fold_label(scenario, fold)} leaks between splits: {'; '.join(issues)}")

        manifest = {"manifest_sha256": hash_contract(contract), "contract": contract, "provenance": provenance}
        directory = self._save_entry(scenario, fold, manifest, row_ids)

        # Update index to include this new entry
        self._write_index()

        counts = {name: contract["splits"][name]["samples"] for name in SPLIT_NAMES}
        self._reporter.display_text(
            f"  {_fold_label(scenario, fold):32} train={counts['train']:>8,} val={counts['val']:>7,} "
            f"test={counts['test']:>7,}  sha256={manifest['manifest_sha256'][:16]}"
        )

        return directory

    def load(self, scenario: SplitScenario, fold: int | None = None) -> tuple[dict, dict[str, np.ndarray]]:
        """Load the stored manifest of one scenario (and fold).

        Args:
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            Tuple of (manifest dictionary, ascending row ids per split name).

        Raises:
            FileNotFoundError: If the manifest or its row ids file does not exist.
        """
        directory = self.entry_dir(scenario, fold)
        manifest_path = directory / MANIFEST_FILENAME
        row_ids_path = directory / ROW_IDS_FILENAME
        for path in (manifest_path, row_ids_path):
            if not path.is_file():
                raise FileNotFoundError(f"Split manifest file not found: {path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(row_ids_path)
        row_ids = {
            name: np.sort(frame.loc[frame["split"] == name, "row_id"].to_numpy(dtype="int64")) for name in SPLIT_NAMES
        }
        return manifest, row_ids

    def build(self, scenarios: list[SplitScenario] | None = None, fold: int | None = None) -> list[Path]:
        """Generate and save the manifests, then refresh ``index.json``.

        Args:
            scenarios: Scenarios to build; all seven when omitted.
            fold: Single fold index, only valid together with Group K-Fold scenarios.

        Returns:
            Directories of the manifests that were written.

        Raises:
            ValueError: If the dataset is row-limited or a generated split leaks commits or repositories.
        """
        entries = self.manifest_entries(scenarios, fold)
        self._reporter.display_section("Split manifest: build")
        self._reporter.display_text()
        self._ensure_full_dataset()

        project_root = self._config.project_root
        dirty = _git(project_root, "status", "--porcelain")
        if dirty:
            self._reporter.display_warning(
                "  Working tree has uncommitted changes; commit first so the recorded commit describes this manifest"
            )

        self._reporter.display_text("Loading data...")
        all_data = self._loader.load_all_jit()
        data_hashes = self._data_hashes()
        provenance = self._provenance(dirty)
        self._reporter.display_text(f"  Loaded {len(all_data):,} records")
        self._reporter.display_text()

        written: list[Path] = []
        summaries: list[dict] = []
        for scenario, fold_index in entries:
            contract, row_ids, result = self._build_entry(all_data, scenario, fold_index, data_hashes)
            issues = self._overlap_issues(result)
            if issues:
                raise ValueError(f"{_fold_label(scenario, fold_index)} leaks between splits: {'; '.join(issues)}")

            manifest = {"manifest_sha256": hash_contract(contract), "contract": contract, "provenance": provenance}
            directory = self._save_entry(scenario, fold_index, manifest, row_ids)
            written.append(directory)
            counts = {name: contract["splits"][name]["samples"] for name in SPLIT_NAMES}
            self._reporter.display_text(
                f"  {_fold_label(scenario, fold_index):32} train={counts['train']:>8,} val={counts['val']:>7,} "
                f"test={counts['test']:>7,}  sha256={manifest['manifest_sha256'][:16]}"
            )
            summaries.append(
                {
                    "scenario": scenario.value,
                    "fold": fold_index,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "samples": counts,
                    "path": str(directory),
                }
            )

        index = self._write_index()
        self._reporter.display_text()
        self._reporter.display_text(
            f"Manifests written: {len(written)} (manifest set sha256: {index['manifest_set_sha256']})"
        )
        self._reporter.update_summary(
            {
                "manifest_dir": str(self._manifest_dir),
                "manifest_set_sha256": index["manifest_set_sha256"],
                "git_dirty": bool(dirty),
                "entries": summaries,
            }
        )
        self._reporter.save_reports("split_manifest_build.txt", "split_manifest_build.json")
        return written

    def verify(self, scenarios: list[SplitScenario] | None = None, fold: int | None = None) -> bool:
        """Check the stored manifests against a fresh regeneration from the raw data.

        For each entry it checks that the manifest and its row ids file agree with their hashes, that the raw
        files and repository mapping are the ones the manifest was built on, that regenerating the split gives
        the same ids, and that train/val/test share no commit and train/test share no repository.

        Args:
            scenarios: Scenarios to verify; all seven when omitted.
            fold: Single fold index, only valid together with Group K-Fold scenarios.

        Returns:
            True when every checked entry passes and ``index.json`` is up to date.
        """
        entries = self.manifest_entries(scenarios, fold)
        self._reporter.display_section("Split manifest: verify")
        self._reporter.display_text()
        self._ensure_full_dataset()

        self._reporter.display_text("Loading data...")
        all_data = self._loader.load_all_jit()
        data_hashes = self._data_hashes()
        self._reporter.display_text(f"  Loaded {len(all_data):,} records")
        self._reporter.display_text()

        results: list[dict] = []
        for scenario, fold_index in entries:
            issues = self._verify_entry(all_data, data_hashes, scenario, fold_index)
            status = "OK" if not issues else "FAIL"
            self._reporter.display_text(f"  [{status}] {_fold_label(scenario, fold_index)}")
            for issue in issues:
                self._reporter.display_warning(f"      {issue}")
            results.append({"scenario": scenario.value, "fold": fold_index, "ok": not issues, "issues": issues})

        index_issues = self._index_issues()
        self._reporter.display_text(f"  [{'OK' if not index_issues else 'FAIL'}] {INDEX_FILENAME}")
        for issue in index_issues:
            self._reporter.display_warning(f"      {issue}")

        failed = [result for result in results if not result["ok"]]
        ok = not failed and not index_issues
        self._reporter.display_text()
        self._reporter.display_text(
            f"Verified {len(results)} manifests: {len(results) - len(failed)} passed, {len(failed)} failed"
        )
        self._reporter.update_summary(
            {"manifest_dir": str(self._manifest_dir), "ok": ok, "index_issues": index_issues, "entries": results}
        )
        self._reporter.save_reports("split_manifest_verify.txt", "split_manifest_verify.json")
        return ok

    def _ensure_full_dataset(self) -> None:
        """Refuse a row-limited dataset, because its row ids would not match the full data."""
        if self._config.row_limit is not None:
            raise ValueError("A split manifest needs the full dataset; do not use a row limit")

    def _data_hashes(self) -> dict[str, str | None]:
        """SHA-256 of the three JIT parquet files and of the repository mapping."""
        raw_dir = self._config.raw_dir / "defectors" / "jit_bug_prediction_splits" / "random"
        hashes = {
            f"jit_random_{name}": RunReport.sha256_of(raw_dir / f"{name}.parquet.gzip") for name in JIT_SPLIT_NAMES
        }
        hashes["repository_mapping"] = RunReport.sha256_of(self._config.project_root / MAPPING_RELATIVE_PATH)
        return hashes

    @staticmethod
    def _provenance(dirty: str | None) -> dict:
        """Facts about the machine and code state that built the manifest (not part of the hash)."""
        versions = {}
        for name in TRACKED_PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not installed"
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "library_versions": versions,
            "git_dirty": bool(dirty) if dirty is not None else None,
        }

    def _build_entry(
        self,
        all_data: pd.DataFrame,
        scenario: SplitScenario,
        fold: int | None,
        data_hashes: dict[str, str | None],
    ) -> tuple[dict, dict[str, np.ndarray], SplitResult]:
        """Generate one split and describe it as a manifest contract.

        Args:
            all_data: The concatenated JIT data with its original row index.
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.
            data_hashes: SHA-256 of the raw files and the repository mapping.

        Returns:
            Tuple of (contract, ascending row ids per split, the generated split).

        Raises:
            ValueError: If a row id repeats inside a split or appears in more than one split.
        """
        result = self._generator.generate_split(
            all_data,
            scenario,
            val_ratio=VAL_RATIO,
            equalize_train_size=None,
            seed=SPLIT_SEED,
            fold=fold or 0,
        )
        contract, row_ids = self._build_contract_from_result(result, scenario, fold, data_hashes)
        return contract, row_ids, result

    def _build_contract_from_result(
        self,
        result: SplitResult,
        scenario: SplitScenario,
        fold: int | None,
        data_hashes: dict[str, str | None],
    ) -> tuple[dict, dict[str, np.ndarray]]:
        """Build a manifest contract from an existing SplitResult.

        Args:
            result: The generated split result.
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.
            data_hashes: SHA-256 of the raw files and the repository mapping.

        Returns:
            Tuple of (contract, ascending row ids per split).

        Raises:
            ValueError: If a row id repeats inside a split or appears in more than one split.
        """
        frames = {"train": result.train_df, "val": result.val_df, "test": result.test_df}

        row_ids: dict[str, np.ndarray] = {}
        for name, frame in frames.items():
            if frame.index.has_duplicates:
                raise ValueError(f"{_fold_label(scenario, fold)}: duplicated row ids in the {name} split")
            row_ids[name] = np.sort(frame.index.to_numpy(dtype="int64"))
        every_id = np.concatenate([row_ids[name] for name in SPLIT_NAMES])
        if len(np.unique(every_id)) != len(every_id):
            raise ValueError(f"{_fold_label(scenario, fold)}: a row appears in more than one split")

        splits = {
            name: {
                "samples": int(len(frame)),
                "defective": int(frame["label"].sum()),
                "commits": int(frame["commit"].nunique()),
                "repos": sorted(frame["repo"].unique().tolist()),
                "ids_sha256": hash_row_ids(row_ids[name]),
            }
            for name, frame in frames.items()
        }
        contract = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "dataset": self._config.dataset_name,
            "scenario": scenario.value,
            "fold": fold,
            "n_folds": GROUP_KFOLD_N_FOLDS if scenario in GROUP_KFOLD_SCENARIOS else None,
            "seed": SPLIT_SEED,
            "val_ratio": VAL_RATIO,
            "equalize_train_size": None,
            "row_id_definition": ROW_ID_DEFINITION,
            "ids_hash_definition": IDS_HASH_DEFINITION,
            "holdout_repositories": load_holdout_repos(self._config.project_root),
            "data_sha256": data_hashes,
            "splits": splits,
        }
        return contract, row_ids

    def _overlap_issues(self, result: SplitResult) -> list[str]:
        """Leakage findings of a split: shared commits between any two splits, shared repos with test."""
        stats = self._generator.check_overlap(result)
        issues = [f"{count} commits shared between {pair}" for pair, count in stats["commit_overlap"].items() if count]
        for pair in ("train_test", "val_test"):
            if stats["repo_overlap"][pair]:
                issues.append(f"{stats['repo_overlap'][pair]} repositories shared between {pair}")
        return issues

    def _save_entry(
        self,
        scenario: SplitScenario,
        fold: int | None,
        manifest: dict,
        row_ids: dict[str, np.ndarray],
    ) -> Path:
        """Write ``manifest.json`` and ``row_ids.parquet`` of one entry."""
        directory = self.entry_dir(scenario, fold)
        directory.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(
            {
                "row_id": np.concatenate([row_ids[name] for name in SPLIT_NAMES]),
                "split": np.repeat(np.array(SPLIT_NAMES, dtype=object), [len(row_ids[name]) for name in SPLIT_NAMES]),
            }
        )
        # Ascending ids compress about 7x better with delta encoding than with the default (~70 KB instead of ~480 KB)
        frame.to_parquet(
            directory / ROW_IDS_FILENAME,
            index=False,
            compression="zstd",
            use_dictionary=["split"],
            column_encoding={"row_id": "DELTA_BINARY_PACKED"},
        )
        (directory / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return directory

    def _index_payload(self) -> dict:
        """Index of the manifests present on disk, with one hash over the whole set."""
        entries = []
        for scenario, fold in self.manifest_entries():
            path = self.entry_dir(scenario, fold) / MANIFEST_FILENAME
            if not path.is_file():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "scenario": scenario.value,
                    "fold": fold,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "path": path.relative_to(self._manifest_dir).as_posix(),
                }
            )
        joined = "\n".join(f"{entry['scenario']}:{entry['fold']}:{entry['manifest_sha256']}" for entry in entries)
        return {
            "format_version": MANIFEST_FORMAT_VERSION,
            "dataset": self._config.dataset_name,
            "n_manifests": len(entries),
            "manifest_set_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
            "entries": entries,
        }

    def _write_index(self) -> dict:
        """Rewrite ``index.json`` from the manifests on disk (so partial builds keep earlier entries)."""
        index = self._index_payload()
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        (self._manifest_dir / INDEX_FILENAME).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return index

    def _index_issues(self) -> list[str]:
        """Problems of ``index.json`` compared with the manifests on disk."""
        path = self._manifest_dir / INDEX_FILENAME
        if not path.is_file():
            return [f"{INDEX_FILENAME} is missing; run build-split-manifest"]
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != self._index_payload():
            return [f"{INDEX_FILENAME} is stale or edited; run build-split-manifest"]
        return []

    def _verify_entry(
        self,
        all_data: pd.DataFrame,
        data_hashes: dict[str, str | None],
        scenario: SplitScenario,
        fold: int | None,
    ) -> list[str]:
        """Return every problem found for one stored manifest (empty list when it passes)."""
        try:
            manifest, stored_ids = self.load(scenario, fold)
        except FileNotFoundError as error:
            return [str(error)]

        issues: list[str] = []
        contract = manifest.get("contract", {})
        declared_splits = contract.get("splits", {})
        if hash_contract(contract) != manifest.get("manifest_sha256"):
            issues.append("manifest_sha256 does not match the stored contract (manifest.json was edited)")
        for name in SPLIT_NAMES:
            declared = declared_splits.get(name, {})
            if hash_row_ids(stored_ids[name]) != declared.get("ids_sha256") or len(stored_ids[name]) != declared.get(
                "samples"
            ):
                issues.append(f"{ROW_IDS_FILENAME} does not match the {name} hash or count in manifest.json")

        fresh_contract, _, result = self._build_entry(all_data, scenario, fold, data_hashes)
        if fresh_contract["data_sha256"] != contract.get("data_sha256"):
            issues.append("raw dataset files or repository mapping differ from the ones the manifest was built on")
        for name in SPLIT_NAMES:
            if fresh_contract["splits"][name]["ids_sha256"] != declared_splits.get(name, {}).get("ids_sha256"):
                issues.append(f"{name} row ids differ from a fresh regeneration of the split")
        if not issues and json.loads(json.dumps(fresh_contract)) != contract:
            issues.append("contract differs from a fresh regeneration (split parameters or holdout changed)")

        issues.extend(self._overlap_issues(result))
        return issues
