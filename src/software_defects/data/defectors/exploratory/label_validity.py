"""Checkpoint 2: Label validity verification ."""

import re

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class LabelValidity:
    """Checkpoint 2: Label validity verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute label validity verification."""
        self._reporter.display_section("Checkpoint 2: Label Validity")
        self._reporter.display_text()

        # Check lines column in JIT dataset
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()

        self._reporter.display_text("JIT Label Distribution (lines):")

        if "lines" in jit_train.columns:
            # Check if lines column contains arrays
            sample_lines = jit_train["lines"].iloc[0]
            self._reporter.display_text(f"  Lines column type: {type(sample_lines)}")
            self._reporter.display_text(f"  Sample lines: {sample_lines}")
            self._reporter.display_text()

            # Count empty vs non-empty arrays with absolute counts
            for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
                empty_count = (
                    df["lines"]
                    .apply(lambda x: len(x) == 0 if isinstance(x, (list, type(sample_lines))) else pd.isna(x))
                    .sum()
                )
                non_empty_count = (
                    df["lines"]
                    .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(sample_lines))) else not pd.isna(x))
                    .sum()
                )
                null_count = df["lines"].isnull().sum()

                self._reporter.display_text(f"  {split_name}:")
                self._reporter.display_text(
                    f"    Empty (defect-free): {empty_count:,} ({empty_count / len(df) * 100:.2f}%)"
                )
                self._reporter.display_text(
                    f"    Non-empty (defective): {non_empty_count:,} ({non_empty_count / len(df) * 100:.2f}%)"
                )
                self._reporter.display_text(f"    Null: {null_count:,}")
                self._reporter.display_text()
        else:
            self._reporter.display_warning("  lines column not found in JIT dataset")
            self._reporter.display_text()

        # Create binary labels
        self._reporter.display_text("Binary Label Interpretation:")
        self._reporter.display_text("  lines = empty array -> label = 0 (defect-free)")
        self._reporter.display_text("  lines = non-empty array -> label = 1 (defective)")
        self._reporter.display_text()

        # Statistics of lines per defective row
        if "lines" in jit_train.columns:
            self._reporter.display_text("Lines Statistics for Defective Rows:")

            for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
                # Filter defective rows
                defective_df = df[
                    df["lines"].apply(lambda x: len(x) > 0 if isinstance(x, (list, type(sample_lines))) else False)
                ]

                if len(defective_df) > 0:
                    line_counts = defective_df["lines"].apply(len)
                    self._reporter.display_text(f"  {split_name} (n={len(defective_df):,}):")
                    self._reporter.display_text(f"    Number of lines per defective row:")
                    self._reporter.display_text(f"      Mean: {line_counts.mean():.2f}")
                    self._reporter.display_text(f"      Median: {line_counts.median():.2f}")
                    self._reporter.display_text(f"      Min: {line_counts.min()}")
                    self._reporter.display_text(f"      Max: {line_counts.max()}")
                    self._reporter.display_text(f"      P90: {line_counts.quantile(0.90):.2f}")
                    self._reporter.display_text(f"      P95: {line_counts.quantile(0.95):.2f}")
                else:
                    self._reporter.display_text(f"  {split_name}: No defective rows")
                self._reporter.display_text()

        # Check consistency of defective lines against diff hunk ranges (old and new side)
        if "content" in jit_train.columns and "lines" in jit_train.columns:
            self._reporter.display_text("Hunk Consistency Check:")
            self._reporter.display_text("  Checking if defective lines fall within diff hunk ranges...")

            # Hunk header format: @@ -old_start[,old_len] +new_start[,new_len] @@
            hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

            def hunk_ranges(content: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
                old_ranges: list[tuple[int, int]] = []
                new_ranges: list[tuple[int, int]] = []
                for old_start, old_len, new_start, new_len in hunk_pattern.findall(content):
                    old_n = max(int(old_len) if old_len else 1, 1)
                    new_n = max(int(new_len) if new_len else 1, 1)
                    old_ranges.append((int(old_start), int(old_start) + old_n - 1))
                    new_ranges.append((int(new_start), int(new_start) + new_n - 1))
                return old_ranges, new_ranges

            def all_in_ranges(line_numbers: list[int], ranges: list[tuple[int, int]]) -> bool:
                return all(any(lo <= n <= hi for lo, hi in ranges) for n in line_numbers)

            defective_mask = jit_train["lines"].apply(len) > 0
            n_defective = int(defective_mask.sum())
            sample_df = jit_train[defective_mask].sample(n=min(2000, n_defective), random_state=42)

            in_old = in_new = in_either = no_hunk = 0
            for content, lines in zip(sample_df["content"], sample_df["lines"], strict=True):
                old_ranges, new_ranges = hunk_ranges(str(content) if content is not None else "")
                if not new_ranges:
                    no_hunk += 1
                    continue
                line_numbers = [int(n) for n in lines]
                is_old = all_in_ranges(line_numbers, old_ranges)
                is_new = all_in_ranges(line_numbers, new_ranges)
                in_old += is_old
                in_new += is_new
                in_either += is_old or is_new

            n_checked = len(sample_df) - no_hunk
            if n_checked > 0:
                self._reporter.display_text(f"  Defective rows checked: {n_checked:,} (no hunk header: {no_hunk:,})")
                self._reporter.display_text(f"  All lines inside NEW-side hunks: {in_new / n_checked * 100:.2f}%")
                self._reporter.display_text(f"  All lines inside OLD-side hunks: {in_old / n_checked * 100:.2f}%")
                self._reporter.display_text(f"  All lines inside either side: {in_either / n_checked * 100:.2f}%")
                if in_either / n_checked < 0.9:
                    self._reporter.display_warning(
                        "  Many defective lines fall outside the diff hunks; verify what `lines` refers to"
                    )
            else:
                self._reporter.display_warning("  No defective rows with parsable hunk headers in sample")
            self._reporter.display_text()

        # Cross-tabulation: empty content vs label
        if "content" in jit_train.columns and "lines" in jit_train.columns:
            self._reporter.display_text("Content vs Label Cross-Tabulation:")

            # Determine labels
            jit_train_cp = jit_train.copy()
            jit_train_cp["label"] = jit_train_cp["lines"].apply(
                lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(sample_lines))) else 0
            )
            jit_train_cp["empty_content"] = jit_train_cp["content"].apply(lambda x: x == "" or x is None)

            # Cross-tab
            cross_tab = pd.crosstab(jit_train_cp["empty_content"], jit_train_cp["label"], margins=True)
            self._reporter.display_table(cross_tab)
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_2": {
                    "label_column": "lines",
                    "label_interpretation": {
                        "empty_array": 0,
                        "non_empty_array": 1,
                    },
                    "train_defective_count": int(
                        jit_train["lines"]
                        .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(sample_lines))) else False)
                        .sum()
                    ),
                    "test_defective_count": int(
                        jit_test["lines"]
                        .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(sample_lines))) else False)
                        .sum()
                    ),
                    "val_defective_count": int(
                        jit_val["lines"]
                        .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(sample_lines))) else False)
                        .sum()
                    ),
                }
            }
        )
