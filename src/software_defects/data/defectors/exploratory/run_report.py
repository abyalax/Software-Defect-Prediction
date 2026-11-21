"""Checkpoint 18: Run report with the exact code and data state of the run."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import MAPPING_RELATIVE_PATH
from software_defects.reporting.reporter import ReportWritter

TRACKED_PACKAGES = ["pandas", "numpy", "scikit-learn", "pyarrow", "transformers", "torch", "pyyaml"]
JIT_SPLIT_NAMES = ("train", "test", "val")
HASH_CHUNK_BYTES = 1 << 20


class RunReport:
    """Checkpoint 18: Run report with the exact code and data state of the run.

    The report records the git commit, whether the working tree was clean, the library versions and
    the SHA-256 of the dataset files and the repository mapping, so every number in the reports can
    be traced back to the code and data that produced it. It always runs after the other checkpoints.
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def _git(project_root: Path, *args: str) -> str | None:
        """Run a git command in the project root; None when git or the repository is unavailable."""
        try:
            return subprocess.check_output(
                ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    @staticmethod
    def sha256_of(path: Path) -> str | None:
        """Streaming SHA-256 of a file, or None when the file does not exist."""
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _versions() -> dict[str, str]:
        """Installed versions of the libraries that affect results."""
        versions = {}
        for name in TRACKED_PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not installed"
        return versions

    def _git_state(self, project_root: Path) -> dict:
        """Commit, commit date, branch and dirty flag of the working tree."""
        commit = self._git(project_root, "rev-parse", "HEAD")
        status = self._git(project_root, "status", "--porcelain")
        return {
            "git_commit": commit or "unknown",
            "git_commit_date": self._git(project_root, "log", "-1", "--format=%cI") or "unknown",
            "git_branch": self._git(project_root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "git_dirty": bool(status) if status is not None else None,
        }

    def _data_hashes(self, project_root: Path) -> dict[str, str | None]:
        """SHA-256 of the three JIT parquet files and of the repository mapping."""
        raw_dir = self._loader.config.raw_dir / "defectors" / "jit_bug_prediction_splits" / "random"
        hashes = {f"jit_random_{name}": self.sha256_of(raw_dir / f"{name}.parquet.gzip") for name in JIT_SPLIT_NAMES}
        hashes["repository_mapping"] = self.sha256_of(project_root / MAPPING_RELATIVE_PATH)
        return hashes

    def run(self) -> None:
        """Execute run report generation."""
        self._reporter.display_section("Checkpoint 18: Run Report")
        self._reporter.display_text()

        project_root = self._loader.config.project_root
        warnings = self._reporter.get_warnings()
        git_state = self._git_state(project_root)
        if git_state["git_dirty"]:
            self._reporter.display_warning(
                "  Working tree has uncommitted changes; commit first so the recorded commit describes this run"
            )
            warnings = self._reporter.get_warnings()

        self._reporter.display_text(f"Git commit: {git_state['git_commit']} ({git_state['git_branch']})")
        self._reporter.display_text(f"Working tree dirty: {git_state['git_dirty']}")
        self._reporter.display_text(f"Total warnings collected: {len(warnings)}")
        self._reporter.display_text()

        structured_warnings = [
            {"id": f"W{i:03d}", "severity": "warning", "message": message} for i, message in enumerate(warnings, 1)
        ]
        checkpoints_run = sorted(key for key in self._reporter.summary if key.startswith("checkpoint_"))
        metadata_block = {
            "timestamp": datetime.now().isoformat(),
            **git_state,
            "library_versions": self._versions(),
            "data_sha256": self._data_hashes(project_root),
            "checkpoints_run": checkpoints_run,
            "total_warnings": len(warnings),
        }
        combined_report = {
            "metadata": metadata_block,
            "warnings": structured_warnings,
            "checkpoint_summary": self._reporter.summary,
        }
        json_path = self._reporter.save_json(combined_report, "combined_run_report.json")
        self._reporter.display_text(f"Combined run report saved to: {json_path}")

        lines = [
            "# Defectors Dataset Validation Report",
            "",
            f"Generated: {metadata_block['timestamp']}",
            f"Git commit: {git_state['git_commit']} ({git_state['git_branch']}, {git_state['git_commit_date']})",
            f"Working tree dirty: {git_state['git_dirty']}",
            "",
            "## Summary",
            "",
            f"- Checkpoints run: {', '.join(checkpoints_run) or 'none'}",
            f"- Total warnings: {len(warnings)}",
            "",
            "## Warnings",
            "",
            *[f"- **{item['id']}**: {item['message'].strip()}" for item in structured_warnings],
            "",
            "## Checkpoint Summary",
            "",
            "See the JSON report for detailed checkpoint results.",
            "",
        ]
        md_path = self._reporter.save_text("\n".join(lines), "combined_run_report.md")
        self._reporter.display_text(f"Markdown report saved to: {md_path}")
        self._reporter.display_text()

        self._reporter.update_summary(
            {
                "checkpoint_18": {
                    "git_commit": git_state["git_commit"],
                    "git_dirty": git_state["git_dirty"],
                    "total_warnings": len(warnings),
                    "combined_report_json": str(json_path),
                    "combined_report_md": str(md_path),
                }
            }
        )
