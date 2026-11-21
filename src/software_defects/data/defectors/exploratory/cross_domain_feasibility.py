"""Checkpoint 6: Cross-domain feasibility verification ."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_repository_mapping
from software_defects.reporting.reporter import ReportWritter


class CrossDomainFeasibility:
    """Checkpoint 6: Cross-domain feasibility verification ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute cross-domain feasibility verification."""
        self._reporter.display_section("Checkpoint 6: Cross-Domain Feasibility")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Load repository mapping from the shared YAML (single source of truth)
        repo_mapping = load_repository_mapping(self._loader.config.project_root)

        # Extract domain mappings
        ml_dl_repos = set(repo_mapping.get("ml_dl_repositories", {}).keys())
        non_ml_repos = set(repo_mapping.get("non_ml_repositories", {}).keys())
        ambiguous_repos = set(repo_mapping.get("ambiguous_repositories", {}).keys())
        unmapped_repos = set(repo_mapping.get("unmapped_repositories", {}).keys())

        # Assert all repos have domain mapping
        all_repos = set(all_jit["repo"].unique())
        unmapped_in_data = all_repos - ml_dl_repos - non_ml_repos - ambiguous_repos - unmapped_repos

        if unmapped_in_data:
            self._reporter.display_warning(f"  UNMAPPED repositories in data: {unmapped_in_data}")
        else:
            self._reporter.display_text("  All repositories have domain mapping")
        self._reporter.display_text()

        # Label distribution per domain for all splits
        if "lines" in all_jit.columns:
            # The label must exist on every frame used below, not only on the pooled one
            for frame in (jit_train, jit_test, jit_val, all_jit):
                frame["label"] = frame["lines"].apply(lambda x: int(len(x) > 0))

            for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
                self._reporter.display_text(f"{split_name} Domain Summary:")

                ml_dl_df = df[df["repo"].isin(ml_dl_repos)]
                non_ml_df = df[df["repo"].isin(non_ml_repos)]

                if len(ml_dl_df) > 0:
                    ml_dl_positives = ml_dl_df["label"].sum()
                    self._reporter.display_text(
                        f"  ML/DL: {len(ml_dl_df):,} rows, {ml_dl_positives:,} positives ({ml_dl_positives / len(ml_dl_df) * 100:.2f}%)"  # noqa: E501
                    )
                else:
                    self._reporter.display_text("  ML/DL: No data")

                if len(non_ml_df) > 0:
                    non_ml_positives = non_ml_df["label"].sum()
                    self._reporter.display_text(
                        f"  Non-ML: {len(non_ml_df):,} rows, {non_ml_positives:,} positives ({non_ml_positives / len(non_ml_df) * 100:.2f}%)"  # noqa: E501
                    )
                else:
                    self._reporter.display_text("  Non-ML: No data")
                self._reporter.display_text()

        # Target domain positive counts per direction
        self._reporter.display_text("Cross-Direction Positive Counts:")

        # ML -> Non-ML
        ml_df = all_jit[all_jit["repo"].isin(ml_dl_repos)]
        non_ml_df = all_jit[all_jit["repo"].isin(non_ml_repos)]

        ml_positives = ml_df["label"].sum() if len(ml_df) > 0 else 0
        non_ml_positives = non_ml_df["label"].sum() if len(non_ml_df) > 0 else 0

        self._reporter.display_text(f"  ML -> Non-ML: Target (Non-ML) has {non_ml_positives:,} positives")
        self._reporter.display_text(f"  Non-ML -> ML: Target (ML) has {ml_positives:,} positives")
        self._reporter.display_text()

        # Domain size ratios
        self._reporter.display_text("Domain Size Ratios:")
        ml_size = len(ml_df)
        non_ml_size = len(non_ml_df)

        if non_ml_size > 0:
            self._reporter.display_text(f"  ML/Non-ML ratio: {ml_size / non_ml_size:.2f}")
        if ml_size > 0:
            self._reporter.display_text(f"  Non-ML/ML ratio: {non_ml_size / ml_size:.2f}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_6": {
                    "ml_dl_repos_count": len(ml_dl_repos),
                    "non_ml_repos_count": len(non_ml_repos),
                    "ambiguous_repos_count": len(ambiguous_repos),
                    "unmapped_in_data": list(unmapped_in_data),
                    "ml_domain_rows": len(ml_df),
                    "non_ml_domain_rows": len(non_ml_df),
                    "ml_domain_positives": int(ml_positives),
                    "non_ml_domain_positives": int(non_ml_positives),
                }
            }
        )
