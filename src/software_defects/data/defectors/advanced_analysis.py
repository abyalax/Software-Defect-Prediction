"""Advanced analysis for Defectors dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsAdvancedAnalysis:
    """Perform advanced analysis on Defectors dataset."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the advanced analysis.

        Args:
            config: Dataset configuration.
        """
        self._config = config
        self._loader = DefectorsDatasetLoader(config)
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="advanced_analysis",
            class_name="DefectorsAdvancedAnalysis",
        )
        self._output_dir = config.artifacts_dir / "report" / "defectors" / "advanced_analysis"

    def run(self) -> None:
        """Run full advanced analysis."""
        self._reporter.display_section("Defectors Advanced Analysis")
        self._reporter.display_text()

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Run all analyses
        self._analyze_label_distribution()
        self._analyze_repository_mapping()
        self._analyze_repository_split()
        self._analyze_token_distribution()

        # Save reports
        self._reporter.save_reports(
            text_filename="advanced_analysis.txt",
            json_filename="advanced_analysis.json",
        )

    def _analyze_label_distribution(self) -> None:
        """Analyze label distribution (global, per domain, per repository)."""
        self._reporter.display_section("Label Distribution Analysis")
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

        # Global distribution
        global_stats = all_jit["label"].value_counts(normalize=True).to_dict()
        global_counts = all_jit["label"].value_counts().to_dict()

        # Per-domain distribution
        ml_dl_df = all_jit[all_jit["repo"].isin(ml_dl_repos)]
        non_ml_df = all_jit[~all_jit["repo"].isin(ml_dl_repos)]

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
            domain = "ml_dl" if repo in ml_dl_repos else "non_ml"
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
        csv_path = self._output_dir / "label_distribution.csv"
        label_dist_df.to_csv(csv_path, index=False)

        self._reporter.display_text(f"Label distribution saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "label_distribution": {
                    "global_defective_pct": float(global_stats.get(1, 0) * 100),
                    "ml_dl_defective_pct": float(domain_stats["ml_dl"].get(1, 0) * 100) if domain_stats["ml_dl"] else 0,
                    "non_ml_defective_pct": float(domain_stats["non_ml"].get(1, 0) * 100)
                    if domain_stats["non_ml"]
                    else 0,
                    "csv_path": str(csv_path),
                }
            }
        )

    def _analyze_repository_mapping(self) -> None:
        """Analyze repository mapping with classification and justification."""
        self._reporter.display_section("Repository Mapping Analysis")
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
        csv_path = self._output_dir / "repository_mapping.csv"
        mapping_df.to_csv(csv_path, index=False)

        self._reporter.display_text(f"Repository mapping saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "repository_mapping": {
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

    def _analyze_repository_split(self) -> None:
        """Analyze repository-level split simulation (ML ↔ non-ML)."""
        self._reporter.display_section("Repository-Level Split Analysis")
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
        csv_path = self._output_dir / "split_report.csv"
        split_df.to_csv(csv_path, index=False)

        self._reporter.display_text(f"Split report saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "repository_split": {
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

    def _analyze_token_distribution(self) -> None:
        """Analyze token distribution for CodeBERT and CodeT5-small."""
        self._reporter.display_section("Token Distribution Analysis")
        self._reporter.display_text()

        # Load JIT train data
        jit_train = self._loader.load_jit_train()

        # Simple token approximation (space-separated tokens)
        # In production, use actual tokenizers from transformers library
        jit_train["approx_tokens"] = jit_train["content"].apply(lambda x: len(str(x).split()) if x is not None else 0)

        # Statistics
        token_stats = {
            "mean": float(jit_train["approx_tokens"].mean()),
            "median": float(jit_train["approx_tokens"].median()),
            "std": float(jit_train["approx_tokens"].std()),
            "min": int(jit_train["approx_tokens"].min()),
            "max": int(jit_train["approx_tokens"].max()),
            "p25": float(jit_train["approx_tokens"].quantile(0.25)),
            "p50": float(jit_train["approx_tokens"].quantile(0.50)),
            "p75": float(jit_train["approx_tokens"].quantile(0.75)),
            "p90": float(jit_train["approx_tokens"].quantile(0.90)),
            "p95": float(jit_train["approx_tokens"].quantile(0.95)),
            "p99": float(jit_train["approx_tokens"].quantile(0.99)),
        }

        # CodeBERT: max 512 tokens
        codebert_overflow = (jit_train["approx_tokens"] > 512).sum()
        codebert_overflow_pct = codebert_overflow / len(jit_train) * 100

        # CodeT5-small: max 512 tokens
        codet5_overflow = (jit_train["approx_tokens"] > 512).sum()
        codet5_overflow_pct = codet5_overflow / len(jit_train) * 100

        # Create CSV
        token_data = [
            {
                "model": "CodeBERT",
                "max_sequence_length": 512,
                "mean_tokens": token_stats["mean"],
                "median_tokens": token_stats["median"],
                "std_tokens": token_stats["std"],
                "min_tokens": token_stats["min"],
                "max_tokens": token_stats["max"],
                "p25_tokens": token_stats["p25"],
                "p50_tokens": token_stats["p50"],
                "p75_tokens": token_stats["p75"],
                "p90_tokens": token_stats["p90"],
                "p95_tokens": token_stats["p95"],
                "p99_tokens": token_stats["p99"],
                "overflow_count": codebert_overflow,
                "overflow_percentage": codebert_overflow_pct,
            },
            {
                "model": "CodeT5-small",
                "max_sequence_length": 512,
                "mean_tokens": token_stats["mean"],
                "median_tokens": token_stats["median"],
                "std_tokens": token_stats["std"],
                "min_tokens": token_stats["min"],
                "max_tokens": token_stats["max"],
                "p25_tokens": token_stats["p25"],
                "p50_tokens": token_stats["p50"],
                "p75_tokens": token_stats["p75"],
                "p90_tokens": token_stats["p90"],
                "p95_tokens": token_stats["p95"],
                "p99_tokens": token_stats["p99"],
                "overflow_count": codet5_overflow,
                "overflow_percentage": codet5_overflow_pct,
            },
        ]

        token_df = pd.DataFrame(token_data)
        csv_path = self._output_dir / "token_analysis.csv"
        token_df.to_csv(csv_path, index=False)

        self._reporter.display_text(f"Token analysis saved to: {csv_path}")
        self._reporter.display_text()
        self._reporter.display_text(f"Mean tokens: {token_stats['mean']:.2f}")
        self._reporter.display_text(f"Median tokens: {token_stats['median']:.2f}")
        self._reporter.display_text(f"P95 tokens: {token_stats['p95']:.2f}")
        self._reporter.display_text(f"P99 tokens: {token_stats['p99']:.2f}")
        self._reporter.display_text(f"CodeBERT overflow (>512): {codebert_overflow:,} ({codebert_overflow_pct:.2f}%)")
        self._reporter.display_text(f"CodeT5-small overflow (>512): {codet5_overflow:,} ({codet5_overflow_pct:.2f}%)")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "token_analysis": {
                    "mean_tokens": token_stats["mean"],
                    "median_tokens": token_stats["median"],
                    "p95_tokens": token_stats["p95"],
                    "p99_tokens": token_stats["p99"],
                    "codebert_overflow_pct": codebert_overflow_pct,
                    "codet5_overflow_pct": codet5_overflow_pct,
                    "csv_path": str(csv_path),
                }
            }
        )
