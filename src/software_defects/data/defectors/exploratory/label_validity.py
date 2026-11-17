"""Checkpoint 2: Label validity verification ."""

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
                    line_counts = defective_df["lines"].apply(lambda x: len(x) if isinstance(x, list) else 0)
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

        # Check consistency of lines against hunk header (simplified check)
        if "content" in jit_train.columns and "lines" in jit_train.columns:
            self._reporter.display_text("Hunk Consistency Check:")
            self._reporter.display_text("  Checking if lines fall within diff hunk ranges...")

            # Sample 100 rows for manual inspection
            sample_df = jit_train.head(100)
            out_of_range_count = 0

            for idx, row in sample_df.iterrows():
                content = str(row["content"]) if row["content"] is not None else ""
                lines = row["lines"] if isinstance(row["lines"], list) else []

                if len(lines) > 0:
                    # Parse hunk headers (e.g., @@ -10,5 +10,7 @@)
                    import re

                    hunk_pattern = r"@@ -(\d+),?\d* \+(\d+),?\d* @@"
                    hunks = re.findall(hunk_pattern, content)

                    if hunks:
                        # Get the first hunk's new file line range
                        try:
                            new_start = int(hunks[0][1])
                            # Count added lines to estimate range
                            added_lines = content.count("\n+") - content.count("\n-")
                            new_end = new_start + added_lines

                            # Check if any line is outside range
                            for line_num in lines:
                                if line_num < new_start or line_num > new_end:
                                    out_of_range_count += 1
                                    break
                        except (ValueError, IndexError):
                            pass

            if out_of_range_count > 0:
                self._reporter.display_warning(
                    f"  {out_of_range_count} samples have lines outside hunk range (in first 100)"
                )
            else:
                self._reporter.display_text("  All sampled lines appear within hunk ranges")
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
