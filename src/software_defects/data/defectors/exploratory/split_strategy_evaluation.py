"""Checkpoint 8: Split strategy evaluation and recommendations."""

import json
import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
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

        # Define ML/DL repositories
        ml_dl_repos = {
            "lightning",
            "jax",
            "transformers",
            "ray",
            "scikit-learn",
            "yolov5",
        }

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
        train_non_ml = train_repos - ml_dl_repos
        test_ml_dl = test_repos.intersection(ml_dl_repos)
        test_non_ml = test_repos - ml_dl_repos

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
            self._reporter.display_warning("  WARNING: All repositories overlap between train and test!")
            self._reporter.display_warning("  This is a random record-level split, not repository-level split.")
            self._reporter.display_warning("  This causes potential data leakage.")
        self._reporter.display_text()

        # Improved split suggestions
        all_repos = sorted(train_repos.union(test_repos))
        all_ml_dl = [r for r in all_repos if r in ml_dl_repos]
        all_non_ml = [r for r in all_repos if r not in ml_dl_repos]

        self._reporter.display_text("Recommended Split Strategy: Balanced Repository-Level Split")

        # Suggested test repos
        if len(all_ml_dl) >= 2:
            suggested_ml_test = all_ml_dl[:2]
            suggested_ml_train = all_ml_dl[2:]
        else:
            suggested_ml_test = all_ml_dl[:1] if len(all_ml_dl) > 0 else []
            suggested_ml_train = all_ml_dl[1:] if len(all_ml_dl) > 1 else []

        if len(all_non_ml) >= 4:
            suggested_non_ml_test = all_non_ml[:4]
            suggested_non_ml_train = all_non_ml[4:]
        else:
            suggested_non_ml_test = all_non_ml[:2] if len(all_non_ml) > 0 else []
            suggested_non_ml_train = all_non_ml[2:] if len(all_non_ml) > 2 else []

        self._reporter.display_text(f"  Test ML/DL repos: {suggested_ml_test}")
        self._reporter.display_text(f"  Test Non-ML repos: {suggested_non_ml_test}")
        self._reporter.display_text(f"  Train ML/DL repos: {suggested_ml_train}")
        self._reporter.display_text(f"  Train Non-ML repos: {suggested_non_ml_train}")
        self._reporter.display_text()

        # Calculate projected sample counts
        all_data = pd.concat([train_no_freq, test_no_freq], ignore_index=True)
        proj_train_ml = len(all_data[all_data["repo"].isin(suggested_ml_train)])
        proj_test_ml = len(all_data[all_data["repo"].isin(suggested_ml_test)])
        proj_train_non_ml = len(all_data[all_data["repo"].isin(suggested_non_ml_train)])
        proj_test_non_ml = len(all_data[all_data["repo"].isin(suggested_non_ml_test)])

        self._reporter.display_text("Projected Sample Distribution:")
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
            "suggested_repository_level_split": {
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
                "Potential data leakage from repository-specific patterns",
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
                    "suggested_test_ml_dl_repos": suggested_ml_test,
                    "suggested_test_non_ml_repos": suggested_non_ml_test,
                    "split_recommendations_json": str(json_path),
                }
            }
        )
