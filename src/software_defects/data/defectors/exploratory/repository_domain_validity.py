"""Checkpoint 4: Repository and domain validity verification ."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class RepositoryDomainValidity:
    """Checkpoint 4: Repository and domain validity verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute repository and domain validity verification."""
        self._reporter.display_section("Checkpoint 4: Repository and Domain Validity")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Exclude freqtrade (only 1 record)
        all_jit = all_jit[all_jit["repo"] != "freqtrade"]

        # Check repo column
        if "repo" not in all_jit.columns:
            self._reporter.display_warning("  repo column not found")
            return

        self._reporter.display_text("Repository Distribution:")

        # Count records per repository
        repo_counts = all_jit["repo"].value_counts()
        self._reporter.display_text(f"  Number of repositories: {len(repo_counts)}")
        self._reporter.display_text()

        # Show top repositories
        self._reporter.display_text("Top 10 repositories by record count:")
        for i, (repo, count) in enumerate(repo_counts.head(10).items(), 1):
            self._reporter.display_text(f"    {i}. {repo}: {count:,} records")
        self._reporter.display_text()

        # Files-per-commit distribution with percentiles
        if "commit" in all_jit.columns:
            self._reporter.display_text("Files-per-Commit Distribution:")
            files_per_commit = all_jit.groupby("commit").size()
            self._reporter.display_text(f"  Mean: {files_per_commit.mean():.2f}")
            self._reporter.display_text(f"  Median: {files_per_commit.median():.2f}")
            self._reporter.display_text(f"  P25: {files_per_commit.quantile(0.25):.2f}")
            self._reporter.display_text(f"  P75: {files_per_commit.quantile(0.75):.2f}")
            self._reporter.display_text(f"  P90: {files_per_commit.quantile(0.90):.2f}")
            self._reporter.display_text(f"  P95: {files_per_commit.quantile(0.95):.2f}")
            self._reporter.display_text(f"  P99: {files_per_commit.quantile(0.99):.2f}")
            self._reporter.display_text(f"  Max: {files_per_commit.max()}")
            self._reporter.display_text()

        # Top-3 repo contribution to total rows and positives
        if "lines" in all_jit.columns:
            self._reporter.display_text("Top-3 Repository Contribution:")

            # Calculate labels
            sample_lines = all_jit["lines"].iloc[0]
            all_jit["label"] = all_jit["lines"].apply(
                lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(sample_lines))) else 0
            )

            top_3_repos = repo_counts.head(3).index.tolist()
            total_rows = len(all_jit)
            total_positives = all_jit["label"].sum()

            for repo in top_3_repos:
                repo_df = all_jit[all_jit["repo"] == repo]
                repo_rows = len(repo_df)
                repo_positives = repo_df["label"].sum()
                self._reporter.display_text(f"  {repo}:")
                self._reporter.display_text(f"    Rows: {repo_rows:,} ({repo_rows / total_rows * 100:.2f}% of total)")
                self._reporter.display_text(
                    f"    Positives: {repo_positives:,} ({repo_positives / total_positives * 100:.2f}% of total positives)"  # noqa: E501
                )
            self._reporter.display_text()

        # Datetime range per repository
        if "datetime" in all_jit.columns:
            self._reporter.display_text("Datetime Range per Repository:")

            # Convert datetime if needed
            try:
                all_jit["datetime"] = pd.to_datetime(all_jit["datetime"])
            except Exception:
                self._reporter.display_warning("  Could not convert datetime column")
            else:
                datetime_ranges = []
                for repo in sorted(all_jit["repo"].unique()):
                    repo_df = all_jit[all_jit["repo"] == repo]
                    min_dt = repo_df["datetime"].min()
                    max_dt = repo_df["datetime"].max()
                    datetime_ranges.append({"repo": repo, "min": str(min_dt), "max": str(max_dt)})

                # Save to CSV
                datetime_df = pd.DataFrame(datetime_ranges)
                csv_path = self._reporter.save_csv(datetime_df, "datetime_ranges.csv")
                self._reporter.display_text(f"  Datetime ranges saved to: {csv_path}")
                self._reporter.display_text()

                # Show summary
                overall_min = all_jit["datetime"].min()
                overall_max = all_jit["datetime"].max()
                self._reporter.display_text(f"  Overall datetime range: {overall_min} to {overall_max}")
                self._reporter.display_text()

        # Datetime range per split
        if "datetime" in all_jit.columns:
            self._reporter.display_text("Datetime Range per Split:")

            for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
                if len(df) > 0:
                    try:
                        df_copy = df.copy()
                        df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])
                        min_dt = df_copy["datetime"].min()
                        max_dt = df_copy["datetime"].max()
                        self._reporter.display_text(f"  {split_name}: {min_dt} to {max_dt}")
                    except Exception:
                        self._reporter.display_text(f"  {split_name}: Could not parse datetime")
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_4": {
                    "num_repositories": len(repo_counts),
                    "top_repositories": repo_counts.head(10).to_dict(),
                    "files_per_commit_median": float(files_per_commit.median())
                    if "files_per_commit" in locals()
                    else 0,
                    "files_per_commit_p90": float(files_per_commit.quantile(0.90))
                    if "files_per_commit" in locals()
                    else 0,
                    "files_per_commit_p99": float(files_per_commit.quantile(0.99))
                    if "files_per_commit" in locals()
                    else 0,
                }
            }
        )
