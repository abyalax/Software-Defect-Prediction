"""Checkpoint 9: Label distribution analysis (global, per domain, per repository)."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.reporting.reporter import ReportWritter


class LabelDistribution:
    """Checkpoint 9: Label distribution analysis (global, per domain, per repository)."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute label distribution analysis."""
        self._reporter.display_section("Checkpoint 9: Label Distribution Analysis")
        self._reporter.display_text()

        # Load all JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Exclude freqtrade (only 1 record)
        all_jit = all_jit[all_jit["repo"] != "freqtrade"]

        # Domains come from the shared YAML; ambiguous repos are reported separately
        domain_sets = load_domain_sets(self._loader.config.project_root)
        ml_dl_repos = domain_sets["ml_dl"]
        non_ml_repos = domain_sets["non_ml"]

        # Calculate binary labels
        all_jit["label"] = all_jit["lines"].apply(
            lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(all_jit["lines"].iloc[0]))) else 0
        )

        # Global distribution
        global_stats = all_jit["label"].value_counts(normalize=True).to_dict()
        global_counts = all_jit["label"].value_counts().to_dict()

        # Per-domain distribution
        ml_dl_df = all_jit[all_jit["repo"].isin(ml_dl_repos)]
        non_ml_df = all_jit[all_jit["repo"].isin(non_ml_repos)]

        domain_stats = {
            "ml_dl": ml_dl_df["label"].value_counts(normalize=True).to_dict() if len(ml_dl_df) > 0 else {},
            "non_ml": non_ml_df["label"].value_counts(normalize=True).to_dict() if len(non_ml_df) > 0 else {},
        }

        # Per-repository distribution
        repo_stats = {}
        for repo in all_jit["repo"].unique():
            repo_df = all_jit[all_jit["repo"] == repo]
            repo_stats[repo] = {
                "total": len(repo_df),
                "defective": int(repo_df["label"].sum()),
                "defect_free": int(len(repo_df) - repo_df["label"].sum()),
                "defective_pct": float(repo_df["label"].mean() * 100),
            }

        # Create CSV
        label_dist_data = []
        for repo, stats in repo_stats.items():
            domain = "ml_dl" if repo in ml_dl_repos else "non_ml" if repo in non_ml_repos else "ambiguous"
            label_dist_data.append(
                {
                    "repository": repo,
                    "domain": domain,
                    "total_samples": stats["total"],
                    "defective": stats["defective"],
                    "defect_free": stats["defect_free"],
                    "defective_percentage": stats["defective_pct"],
                }
            )

        # Add global and domain summaries
        label_dist_data.append(
            {
                "repository": "GLOBAL",
                "domain": "global",
                "total_samples": len(all_jit),
                "defective": global_counts.get(1, 0),
                "defect_free": global_counts.get(0, 0),
                "defective_percentage": global_stats.get(1, 0) * 100,
            }
        )

        label_dist_data.append(
            {
                "repository": "ML_DL_DOMAIN",
                "domain": "ml_dl",
                "total_samples": len(ml_dl_df),
                "defective": int(ml_dl_df["label"].sum()) if len(ml_dl_df) > 0 else 0,
                "defect_free": int(len(ml_dl_df) - ml_dl_df["label"].sum()) if len(ml_dl_df) > 0 else 0,
                "defective_percentage": domain_stats["ml_dl"].get(1, 0) * 100 if domain_stats["ml_dl"] else 0,
            }
        )

        label_dist_data.append(
            {
                "repository": "NON_ML_DOMAIN",
                "domain": "non_ml",
                "total_samples": len(non_ml_df),
                "defective": int(non_ml_df["label"].sum()) if len(non_ml_df) > 0 else 0,
                "defect_free": int(len(non_ml_df) - non_ml_df["label"].sum()) if len(non_ml_df) > 0 else 0,
                "defective_percentage": domain_stats["non_ml"].get(1, 0) * 100 if domain_stats["non_ml"] else 0,
            }
        )

        label_dist_df = pd.DataFrame(label_dist_data)
        csv_path = self._reporter.save_csv(label_dist_df, "label_distribution.csv")

        self._reporter.display_text(f"Label distribution saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_9": {
                    "global_defective_pct": float(global_stats.get(1, 0) * 100),
                    "ml_dl_defective_pct": float(domain_stats["ml_dl"].get(1, 0) * 100) if domain_stats["ml_dl"] else 0,
                    "non_ml_defective_pct": float(domain_stats["non_ml"].get(1, 0) * 100)
                    if domain_stats["non_ml"]
                    else 0,
                    "csv_path": str(csv_path),
                }
            }
        )
