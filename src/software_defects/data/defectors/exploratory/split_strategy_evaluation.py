"""Checkpoint 8: Split strategy evaluation and recommendations."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets, load_holdout_repos
from software_defects.data.defectors.split_generator import DefectorsSplitGenerator
from software_defects.reporting.reporter import ReportWritter


class SplitStrategyEvaluation:
    """Checkpoint 8: Split strategy evaluation and recommendations."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute split strategy evaluation."""
        self._reporter.display_section("Checkpoint 8: Split Strategy Evaluation")
        self._reporter.display_text()

        # Load JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()

        # Domains come from the shared YAML; ambiguous repos are excluded
        domain_sets = load_domain_sets(self._loader.config.project_root)
        ml_dl_repos = domain_sets["ml_dl"]
        non_ml_repos = domain_sets["non_ml"]

        # Calculate labels
        def calculate_label(df: pd.DataFrame) -> pd.DataFrame:
            sample_lines = df["lines"].iloc[0] if len(df) > 0 else []
            df["label"] = df["lines"].apply(
                lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(sample_lines))) else 0
            )
            return df

        jit_train_labeled = calculate_label(jit_train.copy())
        jit_test_labeled = calculate_label(jit_test.copy())

        # Exclude freqtrade
        train_no_freq = jit_train_labeled[jit_train_labeled["repo"] != "freqtrade"]
        test_no_freq = jit_test_labeled[jit_test_labeled["repo"] != "freqtrade"]

        # Current split analysis
        train_repos = set(train_no_freq["repo"].unique())
        test_repos = set(test_no_freq["repo"].unique())

        train_ml_dl = train_repos.intersection(ml_dl_repos)
        train_non_ml = train_repos.intersection(non_ml_repos)
        test_ml_dl = test_repos.intersection(ml_dl_repos)
        test_non_ml = test_repos.intersection(non_ml_repos)

        self._reporter.display_text("Current Split Analysis:")
        self._reporter.display_text(
            f"  Train repos: {len(train_repos)} total ({len(train_ml_dl)} ML/DL, {len(train_non_ml)} Non-ML)"
        )
        self._reporter.display_text(
            f"  Test repos: {len(test_repos)} total ({len(test_ml_dl)} ML/DL, {len(test_non_ml)} Non-ML)"
        )
        self._reporter.display_text(f"  Repository overlap: {len(train_repos.intersection(test_repos))} repos")
        self._reporter.display_text()

        # Evaluation
        if len(train_repos.intersection(test_repos)) == len(train_repos):
            self._reporter.display_text("  INFO: Current split is random record-level (not repository-level)")
            self._reporter.display_text("  INFO: Repository-level split will be used in pipeline to avoid data leakage")
        self._reporter.display_text()

        # Repository-level holdout: explicit in the mapping YAML, shared with the split generator
        all_repos = sorted(train_repos.union(test_repos))
        all_ml_dl = [r for r in all_repos if r in ml_dl_repos]
        all_non_ml = [r for r in all_repos if r in non_ml_repos]

        holdout = load_holdout_repos(self._loader.config.project_root)

        self._reporter.display_text("Configured Repository-Level Holdout (artifacts/mapping/repository_mapping.yaml)")

        suggested_ml_test = [r for r in all_ml_dl if r in holdout["ml_dl"]]
        suggested_ml_train = [r for r in all_ml_dl if r not in holdout["ml_dl"]]
        suggested_non_ml_test = [r for r in all_non_ml if r in holdout["non_ml"]]
        suggested_non_ml_train = [r for r in all_non_ml if r not in holdout["non_ml"]]

        self._reporter.display_text(f"  Test ML/DL repos: {suggested_ml_test}")
        self._reporter.display_text(f"  Test Non-ML repos: {suggested_non_ml_test}")
        self._reporter.display_text(f"  Train ML/DL repos: {suggested_ml_train}")
        self._reporter.display_text(f"  Train Non-ML repos: {suggested_non_ml_train}")
        self._reporter.display_text()

        # Projected counts cover train + val + test of the original files and use the same
        # filtering as the pipeline (freqtrade and empty-content rows removed)
        jit_val = self._loader.load_jit_val()
        generator = DefectorsSplitGenerator(self._loader.config)
        all_data = generator.filter_data(pd.concat([jit_train, jit_test, jit_val], ignore_index=True))
        proj_train_ml = len(all_data[all_data["repo"].isin(suggested_ml_train)])
        proj_test_ml = len(all_data[all_data["repo"].isin(suggested_ml_test)])
        proj_train_non_ml = len(all_data[all_data["repo"].isin(suggested_non_ml_train)])
        proj_test_non_ml = len(all_data[all_data["repo"].isin(suggested_non_ml_test)])

        self._reporter.display_text("Projected Sample Distribution (train + val + test, after filtering):")
        self._reporter.display_text(f"  Train ML/DL: {proj_train_ml:,}")
        self._reporter.display_text(f"  Test ML/DL: {proj_test_ml:,}")
        self._reporter.display_text(f"  Train Non-ML: {proj_train_non_ml:,}")
        self._reporter.display_text(f"  Test Non-ML: {proj_test_non_ml:,}")
        self._reporter.display_text()

        # Save recommendations
        recommendations = {
            "current_split": {
                "train_ml_dl_repos": sorted(train_ml_dl),
                "train_non_ml_repos": sorted(train_non_ml),
                "test_ml_dl_repos": sorted(test_ml_dl),
                "test_non_ml_repos": sorted(test_non_ml),
                "is_repository_level": False,
                "repo_overlap_count": len(train_repos.intersection(test_repos)),
            },
            "configured_repository_level_split": {
                "train_ml_dl_repos": suggested_ml_train,
                "test_ml_dl_repos": suggested_ml_test,
                "train_non_ml_repos": suggested_non_ml_train,
                "test_non_ml_repos": suggested_non_ml_test,
                "projected_train_ml_dl_samples": proj_train_ml,
                "projected_test_ml_dl_samples": proj_test_ml,
                "projected_train_non_ml_samples": proj_train_non_ml,
                "projected_test_non_ml_samples": proj_test_non_ml,
            },
            "issues_identified": [
                "Current split is random record-level, not repository-level",
                "All repositories overlap between train and test",
                "Pipeline will use the repository-level holdout configured in the mapping YAML",
            ],
        }

        json_path = self._reporter.save_json(recommendations, "split_recommendations.json")
        self._reporter.display_text(f"  Split recommendations saved to: {json_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_8": {
                    "current_is_repo_level": False,
                    "current_repo_overlap": len(train_repos.intersection(test_repos)),
                    "configured_test_ml_dl_repos": suggested_ml_test,
                    "configured_test_non_ml_repos": suggested_non_ml_test,
                    "split_recommendations_json": str(json_path),
                }
            }
        )
