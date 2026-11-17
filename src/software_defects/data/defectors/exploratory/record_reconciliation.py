"""Checkpoint 7: Record reconciliation and repository distribution ."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_repository_mapping
from software_defects.reporting.reporter import ReportWritter


class RecordReconciliation:
    """Checkpoint 7: Record reconciliation and repository distribution ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute record reconciliation."""
        self._reporter.display_section("Checkpoint 7: Record Reconciliation")
        self._reporter.display_text()

        # Load all JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Load repository mapping from the shared YAML (single source of truth)
        repo_mapping = load_repository_mapping(self._loader.config.project_root)

        ml_dl_repos = set(repo_mapping.get("ml_dl_repositories", {}).keys())
        non_ml_repos = set(repo_mapping.get("non_ml_repositories", {}).keys())
        unmapped_repos = set(repo_mapping.get("unmapped_repositories", {}).keys())

        # Assert total rows = 213,420
        expected_total = 213420
        actual_total = len(all_jit)
        self._reporter.display_text("Total Row Count Assertion:")
        self._reporter.display_text(f"  Expected: {expected_total:,}")
        self._reporter.display_text(f"  Actual: {actual_total:,}")

        if actual_total != expected_total:
            difference = actual_total - expected_total
            self._reporter.display_warning(
                f"  DIFFERENCE: {difference:,} ({'+' if difference > 0 else ''}{difference})"
            )
        else:
            self._reporter.display_text("  MATCH: Total rows match expected count")
        self._reporter.display_text()

        # Repository status CSV with inclusion/exclusion reasons
        self._reporter.display_text("Repository Status:")

        repo_status_data = []
        for repo in sorted(all_jit["repo"].unique()):
            repo_df = all_jit[all_jit["repo"] == repo]
            repo_count = len(repo_df)

            if repo in unmapped_repos:
                status = "excluded"
                reason = repo_mapping.get("unmapped_repositories", {}).get(repo, {}).get("criteria", "Unknown")
            elif repo in ml_dl_repos or repo in non_ml_repos:
                status = "included"
                reason = (
                    repo_mapping.get("ml_dl_repositories", {})
                    .get(repo, {})
                    .get("criteria", repo_mapping.get("non_ml_repositories", {}).get(repo, {}).get("criteria", "Known"))
                )
            else:
                status = "ambiguous"
                reason = "Domain not clearly defined"

            repo_status_data.append(
                {
                    "repository": repo,
                    "status": status,
                    "reason": reason,
                    "record_count": repo_count,
                }
            )

        repo_status_df = pd.DataFrame(repo_status_data)
        csv_path = self._reporter.save_csv(repo_status_df, "repository_status.csv")
        self._reporter.display_text(f"  Repository status saved to: {csv_path}")
        self._reporter.display_text()

        # Calculate totals per repo and domain
        self._reporter.display_text("Record Counts by Repository and Domain:")

        # Exclude unmapped repos for domain totals
        included_repos = ml_dl_repos.union(non_ml_repos)
        all_jit[all_jit["repo"].isin(included_repos)]

        repo_totals = {}
        for repo in sorted(included_repos):
            repo_df = all_jit[all_jit["repo"] == repo]
            repo_totals[repo] = len(repo_df)

        total_included = sum(repo_totals.values())
        self._reporter.display_text(f"  Total rows (included repos): {total_included:,}")

        # Every row must be accounted for as included, ambiguous or excluded
        status_totals = repo_status_df.groupby("status")["record_count"].sum().to_dict()
        for status_name, status_total in sorted(status_totals.items()):
            self._reporter.display_text(f"  Status '{status_name}': {int(status_total):,}")
        reconciled_total = int(repo_status_df["record_count"].sum())
        if reconciled_total != actual_total:
            self._reporter.display_warning(f"  MISMATCH: {reconciled_total:,} != {actual_total:,}")
        else:
            self._reporter.display_text("  MATCH: all rows are accounted for by repository status")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_7": {
                    "expected_total_rows": expected_total,
                    "actual_total_rows": actual_total,
                    "row_count_match": actual_total == expected_total,
                    "row_count_difference": actual_total - expected_total,
                    "included_repos_count": len(included_repos),
                    "excluded_repos_count": len(unmapped_repos),
                    "repository_status_csv": str(csv_path),
                }
            }
        )
