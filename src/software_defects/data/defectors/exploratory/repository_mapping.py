"""Checkpoint 10: Repository mapping with classification and justification."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class RepositoryMapping:
    """Checkpoint 10: Repository mapping with classification and justification."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute repository mapping analysis."""
        self._reporter.display_section("Checkpoint 10: Repository Mapping Analysis")
        self._reporter.display_text()

        # Load all JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Define ML/DL repositories with justifications
        repository_classifications = {
            "lightning": {
                "domain": "ml_dl",
                "justification": "PyTorch Lightning - Deep learning framework for PyTorch",
            },
            "jax": {
                "domain": "ml_dl",
                "justification": "JAX - Autograd and XLA for ML research",
            },
            "transformers": {
                "domain": "ml_dl",
                "justification": "Hugging Face Transformers - State-of-the-art NLP models",
            },
            "ray": {
                "domain": "ml_dl",
                "justification": "Ray - Distributed ML and RL framework",
            },
            "scikit-learn": {
                "domain": "ml_dl",
                "justification": "scikit-learn - Classical machine learning library",
            },
            "yolov5": {
                "domain": "ml_dl",
                "justification": "YOLOv5 - Object detection deep learning model",
            },
            "ansible": {
                "domain": "non_ml",
                "justification": "Ansible - IT automation and configuration management",
            },
            "sentry": {
                "domain": "non_ml",
                "justification": "Sentry - Error tracking and monitoring platform",
            },
            "core": {
                "domain": "non_ml",
                "justification": "Core - Likely infrastructure/core utilities",
            },
        }

        # Get all unique repositories, excluding freqtrade (only 1 record)
        all_repos = sorted([r for r in all_jit["repo"].unique() if r != "freqtrade"])

        # Build mapping data
        mapping_data = []
        for repo in all_repos:
            repo_df = all_jit[all_jit["repo"] == repo]
            classification = repository_classifications.get(
                repo,
                {
                    "domain": "non_ml",
                    "justification": "General purpose software repository",
                },
            )

            mapping_data.append(
                {
                    "repository": repo,
                    "domain": classification["domain"],
                    "justification": classification["justification"],
                    "total_samples": len(repo_df),
                    "unique_commits": repo_df["commit"].nunique(),
                }
            )

        mapping_df = pd.DataFrame(mapping_data)
        csv_path = self._reporter.save_csv(mapping_df, "repository_mapping.csv")

        self._reporter.display_text(f"Repository mapping saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_10": {
                    "total_repositories": len(all_repos),
                    "ml_dl_repositories": len(
                        [r for r in all_repos if repository_classifications.get(r, {}).get("domain") == "ml_dl"]
                    ),
                    "non_ml_repositories": len(
                        [r for r in all_repos if repository_classifications.get(r, {}).get("domain") == "non_ml"]
                    ),
                    "csv_path": str(csv_path),
                }
            }
        )
