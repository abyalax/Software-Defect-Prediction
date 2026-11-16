"""Dataset explorer for Defectors dataset investigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsDatasetExplorer:
    """Explore and analyze Defectors dataset following TODO.md checkpoints."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset explorer.

        Args:
            config: Dataset configuration.
        """
        self._config = config
        self._loader = DefectorsDatasetLoader(config)
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="dataset_investigation",
            class_name="DefectorsDatasetExplorer",
        )

    def run(self) -> None:
        """Run full dataset investigation."""
        self._reporter.display_section("Defectors Dataset Investigation")
        self._reporter.display_text()

        # Checkpoint 1: Dataset integrity
        self._checkpoint_1_dataset_integrity()

        # Checkpoint 2: Label validity
        self._checkpoint_2_label_validity()

        # Checkpoint 3: JIT input validity
        self._checkpoint_3_jit_input_validity()

        # Checkpoint 4: Repository and domain validity
        self._checkpoint_4_repository_domain_validity()

        # Checkpoint 5: Leakage and unit analysis
        self._checkpoint_5_leakage_unit_analysis()

        # Checkpoint 6: Cross-domain feasibility
        self._checkpoint_6_cross_domain_feasibility()

        # Save reports
        self._reporter.save_reports(
            text_filename="dataset_investigation.txt",
            json_filename="dataset_investigation.json",
        )

    def _checkpoint_1_dataset_integrity(self) -> None:
        """Checkpoint 1: Dataset integrity verification."""
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

        # Check columns
        self._reporter.display_text("JIT Columns:")
        self._reporter.display_text(f"  {list(jit_train.columns)}")
        self._reporter.display_text()

        self._reporter.display_text("Line-level Columns:")
        self._reporter.display_text(f"  {list(line_train.columns)}")
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
                }
            }
        )

    def _checkpoint_2_label_validity(self) -> None:
        """Checkpoint 2: Label validity verification."""
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

            # Count empty vs non-empty arrays
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

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_2": {
                    "label_column": "lines",
                    "label_interpretation": {
                        "empty_array": 0,
                        "non_empty_array": 1,
                    },
                }
            }
        )

    def _checkpoint_3_jit_input_validity(self) -> None:
        """Checkpoint 3: JIT input validity verification."""
        self._reporter.display_section("Checkpoint 3: JIT Input Validity")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()

        # Check content column
        if "content" not in jit_train.columns:
            self._reporter.display_warning("  content column not found")
            return

        self._reporter.display_text("Content Column Analysis:")

        # Check for empty content
        empty_content = jit_train["content"].apply(lambda x: x == "" or x is None).sum()
        self._reporter.display_text(f"  Empty content: {empty_content:,} ({empty_content / len(jit_train) * 100:.2f}%)")

        # Check content length
        content_lengths = jit_train["content"].apply(lambda x: len(str(x)) if x is not None else 0)
        self._reporter.display_text("  Content length stats:")
        self._reporter.display_text(f"    Mean: {content_lengths.mean():.2f}")
        self._reporter.display_text(f"    Median: {content_lengths.median():.2f}")
        self._reporter.display_text(f"    P90: {content_lengths.quantile(0.90):.2f}")
        self._reporter.display_text(f"    P95: {content_lengths.quantile(0.95):.2f}")
        self._reporter.display_text(f"    P99: {content_lengths.quantile(0.99):.2f}")
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
                    "empty_content_count": int(empty_content),
                    "empty_content_percentage": float(empty_content / len(jit_train) * 100),
                    "content_length_mean": float(content_lengths.mean()),
                    "content_length_median": float(content_lengths.median()),
                    "content_length_p90": float(content_lengths.quantile(0.90)),
                    "content_length_p95": float(content_lengths.quantile(0.95)),
                    "content_length_p99": float(content_lengths.quantile(0.99)),
                }
            }
        )

    def _checkpoint_4_repository_domain_validity(self) -> None:
        """Checkpoint 4: Repository and domain validity verification."""
        self._reporter.display_section("Checkpoint 4: Repository and Domain Validity")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()

        # Exclude freqtrade (only 1 record)
        jit_train = jit_train[jit_train["repo"] != "freqtrade"]

        # Check repo column
        if "repo" not in jit_train.columns:
            self._reporter.display_warning("  repo column not found")
            return

        self._reporter.display_text("Repository Distribution:")

        # Count records per repository
        repo_counts = jit_train["repo"].value_counts()
        self._reporter.display_text(f"  Number of repositories: {len(repo_counts)}")
        self._reporter.display_text()

        # Show top repositories
        self._reporter.display_text("Top 10 repositories by record count:")
        for i, (repo, count) in enumerate(repo_counts.head(10).items(), 1):
            self._reporter.display_text(f"    {i}. {repo}: {count:,} records")
        self._reporter.display_text()

        # Count unique commits per repository
        if "commit" in jit_train.columns:
            self._reporter.display_text("Unique commits per repository:")
            commits_per_repo = jit_train.groupby("repo")["commit"].nunique().sort_values(ascending=False)
            for i, (repo, count) in enumerate(commits_per_repo.head(10).items(), 1):
                self._reporter.display_text(f"    {i}. {repo}: {count:,} unique commits")
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_4": {
                    "num_repositories": len(repo_counts),
                    "top_repositories": repo_counts.head(10).to_dict(),
                }
            }
        )

    def _checkpoint_5_leakage_unit_analysis(self) -> None:
        """Checkpoint 5: Leakage and unit analysis verification."""
        self._reporter.display_section("Checkpoint 5: Leakage and Unit Analysis")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()

        # Check for duplicate (repo + commit + filepath)
        if "repo" in jit_train.columns and "commit" in jit_train.columns and "filepath" in jit_train.columns:
            self._reporter.display_text("Duplicate Analysis:")

            # Check duplicates in train
            train_duplicates = jit_train.duplicated(subset=["repo", "commit", "filepath"]).sum()
            self._reporter.display_text(f"  Train duplicates (repo+commit+filepath): {train_duplicates:,}")

            # Check duplicates in test
            test_duplicates = jit_test.duplicated(subset=["repo", "commit", "filepath"]).sum()
            self._reporter.display_text(f"  Test duplicates (repo+commit+filepath): {test_duplicates:,}")
            self._reporter.display_text()

        # Check files per commit
        if "commit" in jit_train.columns:
            self._reporter.display_text("Files per Commit Analysis:")
            files_per_commit = jit_train.groupby("commit").size()
            self._reporter.display_text(f"  Unique commits: {len(files_per_commit):,}")
            self._reporter.display_text(f"  Mean files per commit: {files_per_commit.mean():.2f}")
            self._reporter.display_text(f"  Median files per commit: {files_per_commit.median():.2f}")
            self._reporter.display_text(f"  Max files per commit: {files_per_commit.max()}")
            self._reporter.display_text()

        # Check repository overlap between train and test
        if "repo" in jit_train.columns and "repo" in jit_test.columns:
            self._reporter.display_text("Repository Overlap Analysis:")
            train_repos = set(jit_train["repo"].unique())
            test_repos = set(jit_test["repo"].unique())
            overlap = train_repos.intersection(test_repos)

            self._reporter.display_text(f"  Train repositories: {len(train_repos)}")
            self._reporter.display_text(f"  Test repositories: {len(test_repos)}")
            self._reporter.display_text(f"  Overlapping repositories: {len(overlap)}")

            if overlap:
                self._reporter.display_warning(
                    f"  WARNING: {len(overlap)} repositories overlap between train and test!"
                )
                self._reporter.display_text(f"  Overlapping repos: {overlap}")
            else:
                self._reporter.display_text("  No repository overlap (good for repository-level split)")
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_5": {
                    "train_duplicates": int(train_duplicates) if "train_duplicates" in locals() else 0,
                    "test_duplicates": int(test_duplicates) if "test_duplicates" in locals() else 0,
                    "unique_commits": int(len(files_per_commit)) if "files_per_commit" in locals() else 0,
                    "mean_files_per_commit": float(files_per_commit.mean()) if "files_per_commit" in locals() else 0.0,
                    "repo_overlap_count": int(len(overlap)) if "overlap" in locals() else 0,
                }
            }
        )

    def _checkpoint_6_cross_domain_feasibility(self) -> None:
        """Checkpoint 6: Cross-domain feasibility verification."""
        self._reporter.display_section("Checkpoint 6: Cross-domain Feasibility")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()

        # Exclude freqtrade (only 1 record)
        jit_train = jit_train[jit_train["repo"] != "freqtrade"]
        jit_test = jit_test[jit_test["repo"] != "freqtrade"]

        if "repo" not in jit_train.columns:
            self._reporter.display_warning("  repo column not found")
            return

        # Define ML/DL repositories (from TODO.md)
        ml_dl_repos = {
            "lightning",
            "jax",
            "transformers",
            "ray",
            "scikit-learn",
            "yolov5",
        }

        # Classify repositories
        self._reporter.display_text("Domain Classification:")

        train_repos = set(jit_train["repo"].unique())
        test_repos = set(jit_test["repo"].unique())

        ml_dl_train = train_repos.intersection(ml_dl_repos)
        non_ml_train = train_repos - ml_dl_repos

        ml_dl_test = test_repos.intersection(ml_dl_repos)
        non_ml_test = test_repos - ml_dl_repos

        self._reporter.display_text(f"  ML/DL repositories in train: {len(ml_dl_train)}")
        self._reporter.display_text(f"  Non-ML repositories in train: {len(non_ml_train)}")
        self._reporter.display_text(f"  ML/DL repositories in test: {len(ml_dl_test)}")
        self._reporter.display_text(f"  Non-ML repositories in test: {len(non_ml_test)}")
        self._reporter.display_text()

        # Check label distribution per domain (using JIT data with lines column)
        if "lines" in jit_train.columns:
            self._reporter.display_text("Label Distribution by Domain (Train - JIT):")

            # ML/DL domain
            ml_dl_df = jit_train[jit_train["repo"].isin(ml_dl_repos)]
            if len(ml_dl_df) > 0:
                ml_dl_defective = (
                    ml_dl_df["lines"]
                    .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(ml_dl_df["lines"].iloc[0]))) else False)
                    .sum()
                )
                self._reporter.display_text(
                    f"  ML/DL: {ml_dl_defective:,} defective / {len(ml_dl_df):,} total ({ml_dl_defective / len(ml_dl_df) * 100:.2f}%)"  # noqa: E501
                )
            else:
                self._reporter.display_text("  ML/DL: No data")

            # Non-ML domain
            non_ml_df = jit_train[~jit_train["repo"].isin(ml_dl_repos)]
            if len(non_ml_df) > 0:
                non_ml_defective = (
                    non_ml_df["lines"]
                    .apply(lambda x: len(x) > 0 if isinstance(x, (list, type(non_ml_df["lines"].iloc[0]))) else False)
                    .sum()
                )
                self._reporter.display_text(
                    f"  Non-ML: {non_ml_defective:,} defective / {len(non_ml_df):,} total ({non_ml_defective / len(non_ml_df) * 100:.2f}%)"  # noqa: E501
                )
            else:
                self._reporter.display_text("  Non-ML: No data")
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_6": {
                    "ml_dl_train_repos": len(ml_dl_train),
                    "non_ml_train_repos": len(non_ml_train),
                    "ml_dl_test_repos": len(ml_dl_test),
                    "non_ml_test_repos": len(non_ml_test),
                    "ml_dl_repos_list": list(ml_dl_repos),
                }
            }
        )
