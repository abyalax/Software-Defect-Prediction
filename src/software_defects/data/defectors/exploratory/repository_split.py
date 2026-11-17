"""Checkpoint 11: Repository-level split simulation (ML ↔ non-ML)."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class RepositorySplit:
    """Checkpoint 11: Repository-level split simulation (ML ↔ non-ML)."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute repository-level split analysis."""
        self._reporter.display_section("Checkpoint 11: Repository-Level Split Analysis")
        self._reporter.display_text()

        # Load all JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Exclude freqtrade (only 1 record)
        all_jit = all_jit[all_jit["repo"] != "freqtrade"]

        # Define ML/DL repositories
        ml_dl_repos = {
            "lightning",
            "jax",
            "transformers",
            "ray",
            "scikit-learn",
            "yolov5",
        }

        # Calculate binary labels
        all_jit["label"] = all_jit["lines"].apply(
            lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(all_jit["lines"].iloc[0]))) else 0
        )

        # Scenario 1: Train on ML/DL, Test on Non-ML
        ml_dl_df = all_jit[all_jit["repo"].isin(ml_dl_repos)]
        non_ml_df = all_jit[~all_jit["repo"].isin(ml_dl_repos)]

        scenario_1 = {
            "scenario": "ML_to_NonML",
            "train_domain": "ml_dl",
            "test_domain": "non_ml",
            "train_repositories": len(ml_dl_df["repo"].unique()),
            "test_repositories": len(non_ml_df["repo"].unique()),
            "train_samples": len(ml_dl_df),
            "test_samples": len(non_ml_df),
            "train_defective_pct": float(ml_dl_df["label"].mean() * 100) if len(ml_dl_df) > 0 else 0,
            "test_defective_pct": float(non_ml_df["label"].mean() * 100) if len(non_ml_df) > 0 else 0,
        }

        # Scenario 2: Train on Non-ML, Test on ML/DL
        scenario_2 = {
            "scenario": "NonML_to_ML",
            "train_domain": "non_ml",
            "test_domain": "ml_dl",
            "train_repositories": len(non_ml_df["repo"].unique()),
            "test_repositories": len(ml_dl_df["repo"].unique()),
            "train_samples": len(non_ml_df),
            "test_samples": len(ml_dl_df),
            "train_defective_pct": float(non_ml_df["label"].mean() * 100) if len(non_ml_df) > 0 else 0,
            "test_defective_pct": float(ml_dl_df["label"].mean() * 100) if len(ml_dl_df) > 0 else 0,
        }

        # Scenario 3: Repository-level split (holdout some repos from each domain)
        all_repos = sorted(all_jit["repo"].unique())
        ml_dl_repo_list = [r for r in all_repos if r in ml_dl_repos]
        non_ml_repo_list = [r for r in all_repos if r not in ml_dl_repos]

        # Holdout 1 ML/DL repo and 3 Non-ML repos for test
        if len(ml_dl_repo_list) > 1:
            ml_dl_test_repos = [ml_dl_repo_list[0]]
            ml_dl_train_repos = ml_dl_repo_list[1:]
        else:
            ml_dl_test_repos = []
            ml_dl_train_repos = ml_dl_repo_list

        if len(non_ml_repo_list) > 3:
            non_ml_test_repos = non_ml_repo_list[:3]
            non_ml_train_repos = non_ml_repo_list[3:]
        else:
            non_ml_test_repos = non_ml_repo_list[:1] if len(non_ml_repo_list) > 0 else []
            non_ml_train_repos = non_ml_repo_list[1:] if len(non_ml_repo_list) > 1 else []

        train_repos = ml_dl_train_repos + non_ml_train_repos
        test_repos = ml_dl_test_repos + non_ml_test_repos

        train_df = all_jit[all_jit["repo"].isin(train_repos)]
        test_df = all_jit[all_jit["repo"].isin(test_repos)]

        scenario_3 = {
            "scenario": "Repository_Level_Split",
            "train_domain": "mixed",
            "test_domain": "mixed",
            "train_repositories": len(train_repos),
            "test_repositories": len(test_repos),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_defective_pct": float(train_df["label"].mean() * 100) if len(train_df) > 0 else 0,
            "test_defective_pct": float(test_df["label"].mean() * 100) if len(test_df) > 0 else 0,
            "train_repos_list": ", ".join(train_repos),
            "test_repos_list": ", ".join(test_repos),
        }

        # Scenario 4: Within-domain ML/DL split (holdout 1 ML/DL repo)
        if len(ml_dl_repo_list) > 1:
            ml_dl_within_test_repos = [ml_dl_repo_list[0]]
            ml_dl_within_train_repos = ml_dl_repo_list[1:]

            ml_dl_within_train_df = all_jit[all_jit["repo"].isin(ml_dl_within_train_repos)]
            ml_dl_within_test_df = all_jit[all_jit["repo"].isin(ml_dl_within_test_repos)]

            scenario_4 = {
                "scenario": "Within_Domain_ML_DL",
                "train_domain": "ml_dl",
                "test_domain": "ml_dl",
                "train_repositories": len(ml_dl_within_train_repos),
                "test_repositories": len(ml_dl_within_test_repos),
                "train_samples": len(ml_dl_within_train_df),
                "test_samples": len(ml_dl_within_test_df),
                "train_defective_pct": float(ml_dl_within_train_df["label"].mean() * 100)
                if len(ml_dl_within_train_df) > 0
                else 0,
                "test_defective_pct": float(ml_dl_within_test_df["label"].mean() * 100)
                if len(ml_dl_within_test_df) > 0
                else 0,
                "train_repos_list": ", ".join(ml_dl_within_train_repos),
                "test_repos_list": ", ".join(ml_dl_within_test_repos),
            }
        else:
            scenario_4 = None

        # Scenario 5: Within-domain Non-ML split (holdout 3 Non-ML repos)
        if len(non_ml_repo_list) > 3:
            non_ml_within_test_repos = non_ml_repo_list[:3]
            non_ml_within_train_repos = non_ml_repo_list[3:]

            non_ml_within_train_df = all_jit[all_jit["repo"].isin(non_ml_within_train_repos)]
            non_ml_within_test_df = all_jit[all_jit["repo"].isin(non_ml_within_test_repos)]

            scenario_5 = {
                "scenario": "Within_Domain_Non_ML",
                "train_domain": "non_ml",
                "test_domain": "non_ml",
                "train_repositories": len(non_ml_within_train_repos),
                "test_repositories": len(non_ml_within_test_repos),
                "train_samples": len(non_ml_within_train_df),
                "test_samples": len(non_ml_within_test_df),
                "train_defective_pct": float(non_ml_within_train_df["label"].mean() * 100)
                if len(non_ml_within_train_df) > 0
                else 0,
                "test_defective_pct": float(non_ml_within_test_df["label"].mean() * 100)
                if len(non_ml_within_test_df) > 0
                else 0,
                "train_repos_list": ", ".join(non_ml_within_train_repos),
                "test_repos_list": ", ".join(non_ml_within_test_repos),
            }
        else:
            scenario_5 = None

        # Create CSV
        split_data = [scenario_1, scenario_2, scenario_3]
        if scenario_4 is not None:
            split_data.append(scenario_4)
        if scenario_5 is not None:
            split_data.append(scenario_5)
        split_df = pd.DataFrame(split_data)
        csv_path = self._reporter.save_csv(split_df, "split_report.csv")

        self._reporter.display_text(f"Split report saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_11": {
                    "ml_to_non_ml_train_samples": scenario_1["train_samples"],
                    "ml_to_non_ml_test_samples": scenario_1["test_samples"],
                    "non_ml_to_ml_train_samples": scenario_2["train_samples"],
                    "non_ml_to_ml_test_samples": scenario_2["test_samples"],
                    "repo_level_split_train_samples": scenario_3["train_samples"],
                    "repo_level_split_test_samples": scenario_3["test_samples"],
                    "csv_path": str(csv_path),
                }
            }
        )
