"""Checkpoint 1: Dataset integrity verification ."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class DatasetIntegrity:
    """Checkpoint 1: Dataset integrity verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute dataset integrity verification."""
        self._reporter.display_section("Checkpoint 1: Dataset Integrity")
        self._reporter.display_text()

        # Load JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()

        self._reporter.display_text("JIT Splits:")
        self._reporter.display_text(f"  Train rows: {len(jit_train):,}")
        self._reporter.display_text(f"  Test rows: {len(jit_test):,}")
        self._reporter.display_text(f"  Val rows: {len(jit_val):,}")
        self._reporter.display_text(f"  Total JIT rows: {len(jit_train) + len(jit_test) + len(jit_val):,}")
        self._reporter.display_text()

        # Load line-level data
        line_train = self._loader.load_line_train()
        line_test = self._loader.load_line_test()
        line_val = self._loader.load_line_val()

        self._reporter.display_text("Line-level Splits:")
        self._reporter.display_text(f"  Train rows: {len(line_train):,}")
        self._reporter.display_text(f"  Test rows: {len(line_test):,}")
        self._reporter.display_text(f"  Val rows: {len(line_val):,}")
        self._reporter.display_text(f"  Total line-level rows: {len(line_train) + len(line_test) + len(line_val):,}")
        self._reporter.display_text()

        # Expected columns for JIT dataset
        expected_jit_columns = {"datetime", "commit", "repo", "filepath", "content", "methods", "lines"}
        jit_columns = set(jit_train.columns)

        self._reporter.display_text("JIT Column Verification:")
        missing_jit_columns = expected_jit_columns - jit_columns
        if missing_jit_columns:
            self._reporter.display_warning(f"  MISSING columns: {missing_jit_columns}")
        else:
            self._reporter.display_text("  All expected columns present")
        self._reporter.display_text(f"  JIT columns: {list(jit_train.columns)}")
        self._reporter.display_text()

        # Check for commit message column
        if "commit_message" not in jit_columns:
            self._reporter.display_warning("  commit_message column NOT found in JIT dataset")
        else:
            self._reporter.display_text("  commit_message column found")
        self._reporter.display_text()

        # Profile methods column
        if "methods" in jit_columns:
            self._reporter.display_text("Methods Column Profile:")
            self._reporter.display_text(f"  Type: {type(jit_train['methods'].iloc[0])}")
            self._reporter.display_text(f"  Sample: {jit_train['methods'].iloc[0]}")

            # Check for empty/None values
            empty_methods = (
                jit_train["methods"].apply(lambda x: x is None or (isinstance(x, list) and len(x) == 0)).sum()
            )
            self._reporter.display_text(
                f"  Empty/None count: {empty_methods:,} ({empty_methods / len(jit_train) * 100:.2f}%)"
            )
            self._reporter.display_text()
        else:
            self._reporter.display_warning("  methods column NOT found")
            self._reporter.display_text()

        # Compare JIT vs line-level datasets
        self._reporter.display_text("JIT vs Line-level Comparison:")
        line_columns = set(line_train.columns)

        if jit_columns == line_columns:
            self._reporter.display_text("  JIT and line-level have IDENTICAL columns")
            self._reporter.display_warning("  WARNING: This may indicate double-counting if both are used")
        else:
            self._reporter.display_text(f"  JIT-only columns: {jit_columns - line_columns}")
            self._reporter.display_text(f"  Line-level-only columns: {line_columns - jit_columns}")
        self._reporter.display_text()

        # Check for content overlap (samples that appear in both)
        self._reporter.display_text("Content Overlap Check:")
        jit_test_sample = jit_train[["commit", "filepath"]].head(100)
        line_test_sample = line_train[["commit", "filepath"]].head(100)
        overlap = pd.merge(jit_test_sample, line_test_sample, on=["commit", "filepath"], how="inner")
        if len(overlap) > 0:
            self._reporter.display_warning(f"  Found {len(overlap)} overlapping samples in first 100 rows")
        else:
            self._reporter.display_text("  No overlap in first 100 rows (good)")
        self._reporter.display_text()

        # Check for empty or corrupt data
        self._reporter.display_text("Data Quality Checks:")
        self._reporter.display_text(f"  JIT train empty rows: {jit_train.isnull().all(axis=1).sum()}")
        self._reporter.display_text(f"  JIT test empty rows: {jit_test.isnull().all(axis=1).sum()}")
        self._reporter.display_text(f"  Line train empty rows: {line_train.isnull().all(axis=1).sum()}")
        self._reporter.display_text(f"  Line test empty rows: {line_test.isnull().all(axis=1).sum()}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_1": {
                    "jit_train_rows": len(jit_train),
                    "jit_test_rows": len(jit_test),
                    "jit_val_rows": len(jit_val),
                    "jit_total_rows": len(jit_train) + len(jit_test) + len(jit_val),
                    "line_train_rows": len(line_train),
                    "line_test_rows": len(line_test),
                    "line_val_rows": len(line_val),
                    "line_total_rows": len(line_train) + len(line_test) + len(line_val),
                    "jit_columns": list(jit_train.columns),
                    "line_columns": list(line_train.columns),
                    "missing_jit_columns": list(missing_jit_columns),
                    "has_commit_message": "commit_message" in jit_columns,
                    "methods_empty_pct": float(empty_methods / len(jit_train) * 100)
                    if "methods" in jit_columns
                    else None,
                    "jit_line_identical_columns": jit_columns == line_columns,
                }
            }
        )
