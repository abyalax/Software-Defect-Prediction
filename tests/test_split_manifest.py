"""Tests for the row-level split manifest (ids, hashes, persistence, verification, CLI wiring)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from software_defects.cli.parser import ParserArgs
from software_defects.config import DatasetConfig
from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.data.defectors.split_manifest import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCENARIOS,
    ROW_IDS_FILENAME,
    SPLIT_NAMES,
    SPLIT_SEED,
    VAL_RATIO,
    DefectorsSplitManifest,
    hash_contract,
    hash_row_ids,
)
from tests.conftest import make_synthetic_jit, write_raw_jit

N_ENTRIES = 5 + 2 * GROUP_KFOLD_N_FOLDS  # five single-split scenarios + two Group K-Fold scenarios with 5 folds


@pytest.fixture
def raw_config(config: DatasetConfig, project_root: Path, synthetic_jit: pd.DataFrame) -> DatasetConfig:
    """Config whose project root already holds the synthetic raw JIT files."""
    write_raw_jit(project_root, synthetic_jit)
    return config


@pytest.fixture
def manifest(raw_config: DatasetConfig) -> DefectorsSplitManifest:
    """Manifest builder over the synthetic raw data."""
    return DefectorsSplitManifest(raw_config)


class TestHashing:
    """Hash helpers."""

    def test_row_id_hash_ignores_order(self) -> None:
        assert hash_row_ids([3, 1, 2]) == hash_row_ids(np.array([1, 2, 3]))

    def test_row_id_hash_changes_with_the_set(self) -> None:
        assert hash_row_ids([1, 2, 3]) != hash_row_ids([1, 2, 4])
        assert hash_row_ids([1, 2, 3]) != hash_row_ids([1, 2])

    def test_contract_hash_ignores_key_order(self) -> None:
        assert hash_contract({"a": 1, "b": [1, 2]}) == hash_contract({"b": [1, 2], "a": 1})
        assert hash_contract({"a": 1}) != hash_contract({"a": 2})


class TestEntries:
    """Which scenarios and folds have a manifest."""

    def test_default_covers_seven_scenarios_with_five_folds_each_for_kfold(self) -> None:
        entries = DefectorsSplitManifest.manifest_entries()
        assert len(entries) == N_ENTRIES
        assert SplitScenario.CROSS_DOMAIN_DEFAULT not in {scenario for scenario, _ in entries}
        assert len(MANIFEST_SCENARIOS) == 7

    def test_single_fold(self) -> None:
        entries = DefectorsSplitManifest.manifest_entries([SplitScenario.GROUP_KFOLD_ML_DL], fold=2)
        assert entries == [(SplitScenario.GROUP_KFOLD_ML_DL, 2)]

    def test_fold_needs_a_group_kfold_scenario(self) -> None:
        with pytest.raises(ValueError, match="Group_KFold"):
            DefectorsSplitManifest.manifest_entries(fold=1)
        with pytest.raises(ValueError, match="Group_KFold"):
            DefectorsSplitManifest.manifest_entries([SplitScenario.ML_TO_NON_ML], fold=1)

    def test_fold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="fold must be"):
            DefectorsSplitManifest.manifest_entries([SplitScenario.GROUP_KFOLD_NON_ML], fold=5)

    def test_legacy_scenario_has_no_manifest(self) -> None:
        with pytest.raises(ValueError, match="No manifest"):
            DefectorsSplitManifest.manifest_entries([SplitScenario.CROSS_DOMAIN_DEFAULT])

    def test_seed_and_val_ratio_match_the_pipeline_defaults(self) -> None:
        parameters = inspect.signature(DefectorsSplitGenerator.generate_split).parameters
        assert parameters["seed"].default == SPLIT_SEED
        assert parameters["val_ratio"].default == VAL_RATIO
        assert parameters["equalize_train_size"].default is None


class TestRowIdentity:
    """The row id is the position in concat(train, test, val) and survives filtering."""

    def test_filter_keeps_the_original_row_index(
        self, raw_config: DatasetConfig, generator_for: DefectorsSplitGenerator
    ) -> None:
        all_data = DefectorsDatasetLoader(raw_config).load_all_jit()
        filtered = generator_for.filter_data(all_data)
        assert filtered.index.is_unique
        assert set(filtered.index) <= set(all_data.index)
        assert len(filtered) < len(all_data)  # freqtrade and empty diffs were dropped
        # an id still points at the same raw row
        sample_id = int(filtered.index[10])
        assert filtered.loc[sample_id, "commit"] == all_data.loc[sample_id, "commit"]


@pytest.fixture
def generator_for(raw_config: DatasetConfig) -> DefectorsSplitGenerator:
    """Split generator over the synthetic config."""
    return DefectorsSplitGenerator(raw_config)


class TestBuild:
    """Building and persisting the manifests."""

    def test_build_writes_every_entry_and_the_index(self, manifest: DefectorsSplitManifest) -> None:
        written = manifest.build()
        assert len(written) == N_ENTRIES
        for directory in written:
            assert (directory / MANIFEST_FILENAME).is_file()
            assert (directory / ROW_IDS_FILENAME).is_file()
        index = json.loads((manifest.manifest_dir / INDEX_FILENAME).read_text(encoding="utf-8"))
        assert index["n_manifests"] == N_ENTRIES
        assert len(index["manifest_set_sha256"]) == 64

    def test_group_kfold_folds_use_their_own_directories(self, manifest: DefectorsSplitManifest) -> None:
        written = {path.relative_to(manifest.manifest_dir).as_posix() for path in manifest.build()}
        assert "Group_KFold_ML_DL/fold_0" in written
        assert "Group_KFold_Non_ML/fold_4" in written
        assert "ML_to_NonML" in written

    def test_ids_equal_the_generated_split(
        self, manifest: DefectorsSplitManifest, raw_config: DatasetConfig, generator_for: DefectorsSplitGenerator
    ) -> None:
        manifest.build([SplitScenario.REPO_LEVEL_SPLIT])
        stored, row_ids = manifest.load(SplitScenario.REPO_LEVEL_SPLIT)
        all_data = DefectorsDatasetLoader(raw_config).load_all_jit()
        result = generator_for.generate_split(all_data, SplitScenario.REPO_LEVEL_SPLIT)
        assert np.array_equal(row_ids["train"], np.sort(result.train_df.index.to_numpy()))
        assert np.array_equal(row_ids["val"], np.sort(result.val_df.index.to_numpy()))
        assert np.array_equal(row_ids["test"], np.sort(result.test_df.index.to_numpy()))
        assert stored["contract"]["splits"]["test"]["samples"] == result.test_samples
        assert stored["contract"]["splits"]["train"]["repos"] == sorted(result.train_df["repo"].unique())

    def test_splits_are_disjoint_in_every_entry(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build()
        for scenario, fold in DefectorsSplitManifest.manifest_entries():
            _, row_ids = manifest.load(scenario, fold)
            every_id = np.concatenate([row_ids[name] for name in SPLIT_NAMES])
            assert len(np.unique(every_id)) == len(every_id)

    def test_manifest_hash_is_deterministic(self, raw_config: DatasetConfig, tmp_path: Path) -> None:
        first = DefectorsSplitManifest(raw_config, manifest_dir=tmp_path / "a")
        second = DefectorsSplitManifest(raw_config, manifest_dir=tmp_path / "b")
        first.build([SplitScenario.GROUP_KFOLD_ML_DL], fold=1)
        second.build([SplitScenario.GROUP_KFOLD_ML_DL], fold=1)
        hash_a = first.load(SplitScenario.GROUP_KFOLD_ML_DL, 1)[0]["manifest_sha256"]
        hash_b = second.load(SplitScenario.GROUP_KFOLD_ML_DL, 1)[0]["manifest_sha256"]
        assert hash_a == hash_b

    def test_partial_build_keeps_earlier_entries_in_the_index(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build([SplitScenario.ML_TO_NON_ML])
        manifest.build([SplitScenario.NON_ML_TO_ML])
        index = json.loads((manifest.manifest_dir / INDEX_FILENAME).read_text(encoding="utf-8"))
        assert [entry["scenario"] for entry in index["entries"]] == ["ML_to_NonML", "NonML_to_ML"]

    def test_row_limited_dataset_is_refused(self, raw_config: DatasetConfig) -> None:
        limited = DatasetConfig(project_root=raw_config.project_root, dataset_name="defectors", row_limit=100)
        with pytest.raises(ValueError, match="full dataset"):
            DefectorsSplitManifest(limited).build([SplitScenario.ML_TO_NON_ML])

    def test_load_missing_manifest(self, manifest: DefectorsSplitManifest) -> None:
        with pytest.raises(FileNotFoundError):
            manifest.load(SplitScenario.ML_TO_NON_ML)

    def test_write_from_split_result_matches_build(
        self, manifest: DefectorsSplitManifest, raw_config: DatasetConfig, generator_for: DefectorsSplitGenerator
    ) -> None:
        """Test that write_from_split_result produces the same manifest as build."""
        # Generate a split
        all_data = DefectorsDatasetLoader(raw_config).load_all_jit()
        result = generator_for.generate_split(all_data, SplitScenario.REPO_LEVEL_SPLIT)

        # Write manifest from the split result
        manifest.write_from_split_result(result, SplitScenario.REPO_LEVEL_SPLIT)

        # Load the manifest
        stored, row_ids = manifest.load(SplitScenario.REPO_LEVEL_SPLIT)

        # Verify the row ids match the split result
        assert np.array_equal(row_ids["train"], np.sort(result.train_df.index.to_numpy()))
        assert np.array_equal(row_ids["val"], np.sort(result.val_df.index.to_numpy()))
        assert np.array_equal(row_ids["test"], np.sort(result.test_df.index.to_numpy()))
        assert stored["contract"]["splits"]["test"]["samples"] == result.test_samples
        assert stored["contract"]["splits"]["train"]["repos"] == sorted(result.train_df["repo"].unique())

    def test_write_from_split_result_with_row_limit_raises(self, raw_config: DatasetConfig) -> None:
        """Test that write_from_split_result raises ValueError when row_limit is set."""
        limited = DatasetConfig(project_root=raw_config.project_root, dataset_name="defectors", row_limit=100)
        manifest = DefectorsSplitManifest(limited)
        generator = DefectorsSplitGenerator(limited)
        all_data = DefectorsDatasetLoader(limited).load_all_jit()
        result = generator.generate_split(all_data, SplitScenario.ML_TO_NON_ML)
        with pytest.raises(ValueError, match="full dataset"):
            manifest.write_from_split_result(result, SplitScenario.ML_TO_NON_ML)


class TestVerify:
    """Verification against a fresh regeneration."""

    def test_fresh_manifests_pass(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build()
        assert manifest.verify() is True

    def test_missing_manifest_fails(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build([SplitScenario.ML_TO_NON_ML])
        assert manifest.verify([SplitScenario.NON_ML_TO_ML]) is False

    def test_tampered_row_ids_fail(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build([SplitScenario.WITHIN_DOMAIN_ML_DL])
        path = manifest.entry_dir(SplitScenario.WITHIN_DOMAIN_ML_DL) / ROW_IDS_FILENAME
        frame = pd.read_parquet(path)
        frame.loc[frame.index[0], "row_id"] += 1
        frame.to_parquet(path, index=False)
        assert manifest.verify([SplitScenario.WITHIN_DOMAIN_ML_DL]) is False

    def test_edited_manifest_json_fails(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build([SplitScenario.WITHIN_DOMAIN_NON_ML])
        path = manifest.entry_dir(SplitScenario.WITHIN_DOMAIN_NON_ML) / MANIFEST_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data["contract"]["seed"] = 7
        path.write_text(json.dumps(data), encoding="utf-8")
        assert manifest.verify([SplitScenario.WITHIN_DOMAIN_NON_ML]) is False

    def test_changed_raw_data_fails(self, manifest: DefectorsSplitManifest, raw_config: DatasetConfig) -> None:
        manifest.build([SplitScenario.REPO_LEVEL_SPLIT])
        write_raw_jit(raw_config.project_root, make_synthetic_jit(rows_per_repo=61))
        assert manifest.verify([SplitScenario.REPO_LEVEL_SPLIT]) is False

    def test_stale_index_fails(self, manifest: DefectorsSplitManifest) -> None:
        manifest.build([SplitScenario.ML_TO_NON_ML])
        manifest.build([SplitScenario.NON_ML_TO_ML])
        (manifest.manifest_dir / INDEX_FILENAME).unlink()
        assert manifest.verify([SplitScenario.ML_TO_NON_ML]) is False


class TestCli:
    """Parser wiring of the new commands."""

    def test_build_parser_defaults_to_all_scenarios(self) -> None:
        arguments = ParserArgs.build_parser().parse_args(["build-split-manifest", "--dataset", "defectors"])
        assert arguments.command == "build-split-manifest"
        assert arguments.scenario is None
        assert arguments.fold is None
        assert arguments.manifest_dir is None

    def test_verify_parser_accepts_scenario_and_fold(self) -> None:
        arguments = ParserArgs.build_parser().parse_args(
            ["verify-split-manifest", "--dataset", "defectors", "--scenario", "Group_KFold_ML_DL", "--fold", "3"]
        )
        assert arguments.scenario == "Group_KFold_ML_DL"
        assert arguments.fold == 3

    def test_legacy_scenario_is_not_a_choice(self) -> None:
        with pytest.raises(SystemExit):
            ParserArgs.build_parser().parse_args(
                ["build-split-manifest", "--dataset", "defectors", "--scenario", "Cross_Domain_Default"]
            )

    def test_manifest_commands_have_no_row_limit(self) -> None:
        with pytest.raises(SystemExit):
            ParserArgs.build_parser().parse_args(["build-split-manifest", "--dataset", "defectors", "--nrows", "10"])
