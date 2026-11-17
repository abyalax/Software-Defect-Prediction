"""Checkpoint 3: JIT input validity verification ."""

import re

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.reporting.reporter import ReportWritter


class JitInputValidity:
    """Checkpoint 3: JIT input validity verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute JIT input validity verification."""
        self._reporter.display_section("Checkpoint 3: JIT Input Validity")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()

        # Check content column
        if "content" not in jit_train.columns:
            self._reporter.display_warning("  content column not found")
            return

        self._reporter.display_text("Content Length Statistics (All Splits):")

        for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
            # Check for empty content
            empty_content = df["content"].apply(lambda x: x == "" or x is None).sum()
            self._reporter.display_text(f"  {split_name}:")
            self._reporter.display_text(f"    Empty content: {empty_content:,} ({empty_content / len(df) * 100:.2f}%)")

            # Check content length
            content_lengths = df["content"].apply(lambda x: len(str(x)) if x is not None else 0)
            self._reporter.display_text(f"    Content length stats:")
            self._reporter.display_text(f"      Mean: {content_lengths.mean():.2f}")
            self._reporter.display_text(f"      Median: {content_lengths.median():.2f}")
            self._reporter.display_text(f"      P90: {content_lengths.quantile(0.90):.2f}")
            self._reporter.display_text(f"      P95: {content_lengths.quantile(0.95):.2f}")
            self._reporter.display_text(f"      P99: {content_lengths.quantile(0.99):.2f}")
            self._reporter.display_text()

        # Diff structure analysis
        self._reporter.display_text("Diff Structure Analysis (Train):")

        # Count hunks per diff
        def count_hunks(content: str) -> int:
            if not content:
                return 0
            return len(re.findall(r"^@@", content, re.MULTILINE))

        jit_train["num_hunks"] = jit_train["content"].apply(lambda x: count_hunks(str(x)) if x is not None else 0)

        self._reporter.display_text(f"  Hunks per diff:")
        self._reporter.display_text(f"    Mean: {jit_train['num_hunks'].mean():.2f}")
        self._reporter.display_text(f"    Median: {jit_train['num_hunks'].median():.2f}")
        self._reporter.display_text(f"    Max: {jit_train['num_hunks'].max()}")
        self._reporter.display_text()

        # Count added and removed lines
        def count_lines(content: str, line_type: str) -> int:
            if not content:
                return 0
            pattern = f"\n{line_type}"
            return content.count(pattern)

        jit_train["added_lines"] = jit_train["content"].apply(
            lambda x: count_lines(str(x), "+") if x is not None else 0
        )
        jit_train["removed_lines"] = jit_train["content"].apply(
            lambda x: count_lines(str(x), "-") if x is not None else 0
        )

        self._reporter.display_text("  Added lines per diff:")
        self._reporter.display_text(f"    Mean: {jit_train['added_lines'].mean():.2f}")
        self._reporter.display_text(f"    Median: {jit_train['added_lines'].median():.2f}")
        self._reporter.display_text(f"    Max: {jit_train['added_lines'].max()}")
        self._reporter.display_text()

        self._reporter.display_text("  Removed lines per diff:")
        self._reporter.display_text(f"    Mean: {jit_train['removed_lines'].mean():.2f}")
        self._reporter.display_text(f"    Median: {jit_train['removed_lines'].median():.2f}")
        self._reporter.display_text(f"    Max: {jit_train['removed_lines'].max()}")
        self._reporter.display_text()

        # Proportion of diffs that are only additions
        only_additions = (jit_train["removed_lines"] == 0).sum()
        only_additions_pct = only_additions / len(jit_train) * 100
        self._reporter.display_text(f"  Diffs with only additions: {only_additions:,} ({only_additions_pct:.2f}%)")
        self._reporter.display_text()

        # File extension analysis
        if "filepath" in jit_train.columns:
            self._reporter.display_text("File Extension Distribution:")

            def get_extension(filepath: str) -> str:
                if not filepath:
                    return "unknown"
                parts = filepath.rsplit(".", 1)
                return parts[-1].lower() if len(parts) > 1 else "no_ext"

            jit_train["extension"] = jit_train["filepath"].apply(get_extension)
            ext_counts = jit_train["extension"].value_counts().head(10)

            for ext, count in ext_counts.items():
                self._reporter.display_text(f"    .{ext}: {count:,} ({count / len(jit_train) * 100:.2f}%)")
            self._reporter.display_text()

        # Per-label statistics
        if "lines" in jit_train.columns:
            self._reporter.display_text("Content Length Statistics by Label:")

            sample_lines = jit_train["lines"].iloc[0]
            jit_train["label"] = jit_train["lines"].apply(
                lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(sample_lines))) else 0
            )

            for label_val, label_name in [(0, "Defect-free"), (1, "Defective")]:
                label_df = jit_train[jit_train["label"] == label_val]
                if len(label_df) > 0:
                    content_lengths = label_df["content"].apply(lambda x: len(str(x)) if x is not None else 0)
                    self._reporter.display_text(f"  {label_name} (n={len(label_df):,}):")
                    self._reporter.display_text(f"    Mean length: {content_lengths.mean():.2f}")
                    self._reporter.display_text(f"    Median length: {content_lengths.median():.2f}")
                    self._reporter.display_text(f"    P95 length: {content_lengths.quantile(0.95):.2f}")
            self._reporter.display_text()

        # Per-domain statistics (if repo available)
        if "repo" in jit_train.columns:
            self._reporter.display_text("Content Length Statistics by Domain:")

            # Domains come from the shared YAML; ambiguous/unmapped repos are excluded
            domain_sets = load_domain_sets(self._loader.config.project_root)
            ml_dl_df = jit_train[jit_train["repo"].isin(domain_sets["ml_dl"])]
            non_ml_df = jit_train[jit_train["repo"].isin(domain_sets["non_ml"])]

            for domain_name, domain_df in [("ML/DL", ml_dl_df), ("Non-ML", non_ml_df)]:
                if len(domain_df) > 0:
                    content_lengths = domain_df["content"].apply(lambda x: len(str(x)) if x is not None else 0)
                    self._reporter.display_text(f"  {domain_name} (n={len(domain_df):,}):")
                    self._reporter.display_text(f"    Mean length: {content_lengths.mean():.2f}")
                    self._reporter.display_text(f"    Median length: {content_lengths.median():.2f}")
                    self._reporter.display_text(f"    P95 length: {content_lengths.quantile(0.95):.2f}")
            self._reporter.display_text()

        # Sample content
        self._reporter.display_text("Sample content (first 200 chars):")
        sample = jit_train["content"].iloc[0]
        if sample:
            self._reporter.display_text(f"  {str(sample)[:200]}...")
        else:
            self._reporter.display_text("  (empty)")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_3": {
                    "train_empty_content_count": int(jit_train["content"].apply(lambda x: x == "" or x is None).sum()),
                    "train_content_length_mean": float(
                        jit_train["content"].apply(lambda x: len(str(x)) if x is not None else 0).mean()
                    ),
                    "train_content_length_p95": float(
                        jit_train["content"].apply(lambda x: len(str(x)) if x is not None else 0).quantile(0.95)
                    ),
                    "mean_hunks_per_diff": float(jit_train["num_hunks"].mean()),
                    "mean_added_lines": float(jit_train["added_lines"].mean()),
                    "mean_removed_lines": float(jit_train["removed_lines"].mean()),
                    "only_additions_pct": float(only_additions_pct),
                }
            }
        )
