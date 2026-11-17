"""Checkpoint 5: Leakage and unit analysis verification ."""

import hashlib
import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class LeakageUnitAnalysis:
    """Checkpoint 5: Leakage and unit analysis verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute leakage and unit analysis verification."""
        self._reporter.display_section("Checkpoint 5: Leakage and Unit Analysis")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()

        # Key overlap analysis: (repo, commit, filepath)
        if all(col in jit_train.columns for col in ["repo", "commit", "filepath"]):
            self._reporter.display_text("Key Overlap Analysis (repo+commit+filepath):")

            # Check duplicates within each split
            train_duplicates = jit_train.duplicated(subset=["repo", "commit", "filepath"]).sum()
            test_duplicates = jit_test.duplicated(subset=["repo", "commit", "filepath"]).sum()
            val_duplicates = jit_val.duplicated(subset=["repo", "commit", "filepath"]).sum()

            self._reporter.display_text(f"  Train duplicates: {train_duplicates:,}")
            self._reporter.display_text(f"  Test duplicates: {test_duplicates:,}")
            self._reporter.display_text(f"  Val duplicates: {val_duplicates:,}")
            self._reporter.display_text()

            # Check overlap between splits
            train_keys = set(jit_train[["repo", "commit", "filepath"]].itertuples(index=False, name=None))
            test_keys = set(jit_test[["repo", "commit", "filepath"]].itertuples(index=False, name=None))
            val_keys = set(jit_val[["repo", "commit", "filepath"]].itertuples(index=False, name=None))

            train_test_overlap = len(train_keys.intersection(test_keys))
            train_val_overlap = len(train_keys.intersection(val_keys))
            test_val_overlap = len(test_keys.intersection(val_keys))

            self._reporter.display_text("  Overlap between splits:")
            self._reporter.display_text(f"    Train-Test: {train_test_overlap:,}")
            self._reporter.display_text(f"    Train-Val: {train_val_overlap:,}")
            self._reporter.display_text(f"    Test-Val: {test_val_overlap:,}")

            if train_test_overlap > 0:
                self._reporter.display_warning("  WARNING: Records overlap between train and test!")
            self._reporter.display_text()

        # Commit overlap: commits whose files are scattered across different splits
        if "commit" in jit_train.columns:
            self._reporter.display_text("Commit Overlap Analysis:")

            train_commits = set(jit_train["commit"].unique())
            test_commits = set(jit_test["commit"].unique())
            val_commits = set(jit_val["commit"].unique())

            train_test_commit_overlap = len(train_commits.intersection(test_commits))
            train_val_commit_overlap = len(train_commits.intersection(val_commits))
            test_val_commit_overlap = len(test_commits.intersection(val_commits))

            self._reporter.display_text(f"  Train-Test commit overlap: {train_test_commit_overlap:,}")
            self._reporter.display_text(f"  Train-Val commit overlap: {train_val_commit_overlap:,}")
            self._reporter.display_text(f"  Test-Val commit overlap: {test_val_commit_overlap:,}")

            if train_test_commit_overlap > 0:
                self._reporter.display_warning(
                    "  WARNING: Commits overlap between train and test (files from same commit in different splits)!"
                )
            self._reporter.display_text()

        # Content hash overlap (exact duplicate diffs)
        if "content" in jit_train.columns:
            self._reporter.display_text("Content Hash Overlap Analysis:")

            # Compute hashes (sample for efficiency)
            def compute_hash(content: str) -> str:
                if content is None:
                    return ""
                return hashlib.md5(str(content).encode()).hexdigest()

            # Sample for efficiency
            train_sample = jit_train.head(1000).copy()
            test_sample = jit_test.head(1000).copy()

            train_sample["content_hash"] = train_sample["content"].apply(compute_hash)
            test_sample["content_hash"] = test_sample["content"].apply(compute_hash)

            hash_overlap = len(set(train_sample["content_hash"]).intersection(set(test_sample["content_hash"])))
            self._reporter.display_text(f"  Content hash overlap (sample 1000 each): {hash_overlap:,}")

            if hash_overlap > 0:
                self._reporter.display_warning("  WARNING: Exact duplicate diffs found between train and test!")
            self._reporter.display_text()

        # Datetime overlap between train and test
        if "datetime" in jit_train.columns and "datetime" in jit_test.columns:
            self._reporter.display_text("Datetime Overlap Analysis:")

            try:
                train_dts = pd.to_datetime(jit_train["datetime"])
                test_dts = pd.to_datetime(jit_test["datetime"])

                train_min, train_max = train_dts.min(), train_dts.max()
                test_min, test_max = test_dts.min(), test_dts.max()

                self._reporter.display_text(f"  Train datetime range: {train_min} to {train_max}")
                self._reporter.display_text(f"  Test datetime range: {test_min} to {test_max}")

                # Check for overlap
                if train_max >= test_min and test_max >= train_min:
                    self._reporter.display_warning("  WARNING: Datetime ranges overlap between train and test!")
                    self._reporter.display_text(
                        "  NOTE: This is problematic for time-based splits but acceptable for cross-domain"
                    )
                else:
                    self._reporter.display_text("  No datetime overlap (good for time-based split)")
                self._reporter.display_text()
            except Exception:
                self._reporter.display_warning("  Could not parse datetime for overlap analysis")
                self._reporter.display_text()

        # Contextual warnings
        self._reporter.display_text("Contextual Warnings:")
        self._reporter.display_text("  Repository overlap warning applies to:")
        self._reporter.display_text("    - Within-domain evaluation (not acceptable)")
        self._reporter.display_text("    - Cross-domain evaluation (may be acceptable if domains are disjoint)")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_5": {
                    "train_duplicates": int(train_duplicates) if "train_duplicates" in locals() else 0,
                    "test_duplicates": int(test_duplicates) if "test_duplicates" in locals() else 0,
                    "train_test_key_overlap": int(train_test_overlap) if "train_test_overlap" in locals() else 0,
                    "train_test_commit_overlap": int(train_test_commit_overlap)
                    if "train_test_commit_overlap" in locals()
                    else 0,
                    "content_hash_overlap": int(hash_overlap) if "hash_overlap" in locals() else 0,
                }
            }
        )
