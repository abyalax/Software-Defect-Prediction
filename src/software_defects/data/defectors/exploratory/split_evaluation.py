"""Experimental evaluation with proper split strategies and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.evaluation import DefectEvaluator
from software_defects.data.defectors.split_generator import DefectorsSplitGenerator, SplitScenario
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    scenario: SplitScenario
    val_ratio: float = 0.1
    equalize_train_size: int | None = None
    seed: int = 42
    n_bootstrap: int = 1000
    combine_target_val_test: bool = True


class SplitExperimentEvaluator:
    """Run experiments with proper split strategies and evaluation metrics."""

    def __init__(
        self,
        config: DatasetConfig,
        loader: DefectorsDatasetLoader,
        reporter: ReportWritter,
    ) -> None:
        """Initialize the experiment evaluator.

        Args:
            config: Dataset configuration.
            loader: Dataset loader.
            reporter: Report writer.
        """
        self._config = config
        self._loader = loader
        self._reporter = reporter
        self._split_generator = DefectorsSplitGenerator(config)
        self._evaluator = DefectEvaluator()

    def run_experiments(self, experiment_configs: list[ExperimentConfig]) -> None:
        """Run multiple experiments with different configurations.

        Args:
            experiment_configs: List of experiment configurations.
        """
        self._reporter.display_section("Split Experiment Evaluation")
        self._reporter.display_text()

        # Load all data
        all_data = self._loader.load_all_jit()

        results = []

        for exp_config in experiment_configs:
            self._reporter.display_text(f"Running scenario: {exp_config.scenario.value}")
            result = self._run_single_experiment(all_data, exp_config)
            results.append(result)
            self._reporter.display_text()

        # Save aggregated results
        self._save_aggregated_results(results)

    def _run_single_experiment(self, df: pd.DataFrame, config: ExperimentConfig) -> dict:
        """Run a single experiment.

        Args:
            df: Full dataset.
            config: Experiment configuration.

        Returns:
            Dictionary with experiment results.
        """
        # Generate split
        split_result = self._split_generator.generate_split(
            df,
            config.scenario,
            val_ratio=config.val_ratio,
            equalize_train_size=config.equalize_train_size,
            seed=config.seed,
        )

        # Check overlap
        overlap_stats = self._split_generator.check_overlap(split_result)

        # For cross-domain, combine target val and test
        if config.combine_target_val_test and split_result.is_cross_domain:
            target_df = pd.concat([split_result.val_df, split_result.test_df], ignore_index=True)
            source_val_df = split_result.val_df
        else:
            target_df = split_result.test_df
            source_val_df = split_result.val_df

        # Placeholder: In real usage, y_scores would come from model predictions
        # For now, use random scores as placeholder
        import numpy as np

        y_val_scores = np.random.rand(len(source_val_df))
        y_test_scores = np.random.rand(len(target_df))

        y_val_true = source_val_df["label"].values
        y_test_true = target_df["label"].values

        # Add scores to target_df for per-repo metrics
        target_df = target_df.copy()
        target_df["score"] = y_test_scores

        # Calculate metrics with threshold from source val
        eval_result = self._evaluator.calculate_metrics_with_val_threshold(
            y_val_true, y_val_scores, y_test_true, y_test_scores
        )

        # Calculate bootstrap CI
        bootstrap_ci = self._evaluator.calculate_bootstrap_ci(
            y_test_true, y_test_scores, n_bootstrap=config.n_bootstrap, threshold=eval_result.threshold
        )

        # Calculate per-repo metrics (only for test, skip small repos)
        per_repo_metrics = self._evaluator.calculate_per_repo_metrics(
            target_df, score_col="score", label_col="label", threshold=eval_result.threshold
        )

        # Filter out small repos (< 100 samples)
        per_repo_metrics_filtered = {
            repo: metrics
            for repo, metrics in per_repo_metrics.items()
            if len(target_df[target_df["repo"] == repo]) >= 100
        }

        result = {
            "scenario": split_result.scenario,
            "is_cross_domain": split_result.is_cross_domain,
            "split_stats": {
                "train_samples": split_result.train_samples,
                "val_samples": split_result.val_samples,
                "test_samples": split_result.test_samples,
                "train_repos": split_result.train_repos,
                "val_repos": split_result.val_repos,
                "test_repos": split_result.test_repos,
                "train_defective_pct": split_result.train_defective_pct,
                "val_defective_pct": split_result.val_defective_pct,
                "test_defective_pct": split_result.test_defective_pct,
            },
            "overlap_stats": overlap_stats,
            "metrics": {
                "pr_auc": eval_result.pr_auc,
                "f1": eval_result.f1,
                "mcc": eval_result.mcc,
                "threshold": eval_result.threshold,
                "prevalence": eval_result.prevalence,
                "prevalence_baseline_pr_auc": eval_result.prevalence_baseline_pr_auc,
            },
            "bootstrap_ci": bootstrap_ci,
            "per_repo_metrics": {
                repo: {
                    "pr_auc": metrics.pr_auc,
                    "f1": metrics.f1,
                    "mcc": metrics.mcc,
                    "n_samples": len(target_df[target_df["repo"] == repo]),
                }
                for repo, metrics in per_repo_metrics_filtered.items()
            },
            "config": {
                "val_ratio": config.val_ratio,
                "equalize_train_size": config.equalize_train_size,
                "seed": config.seed,
                "combine_target_val_test": config.combine_target_val_test,
            },
        }

        # Display summary
        self._reporter.display_text(f"  Train samples: {split_result.train_samples:,}")
        self._reporter.display_text(f"  Val samples: {split_result.val_samples:,}")
        self._reporter.display_text(f"  Test samples: {split_result.test_samples:,}")
        self._reporter.display_text(f"  PR-AUC: {eval_result.pr_auc:.4f}")
        self._reporter.display_text(f"  F1: {eval_result.f1:.4f}")
        self._reporter.display_text(f"  MCC: {eval_result.mcc:.4f}")
        self._reporter.display_text(f"  Prevalence: {eval_result.prevalence:.4f}")
        self._reporter.display_text(f"  Threshold: {eval_result.threshold:.4f}")
        self._reporter.display_text(f"  Per-repo metrics: {len(per_repo_metrics_filtered)} repos")

        return result

    def _save_aggregated_results(self, results: list[dict]) -> None:
        """Save aggregated experiment results.

        Args:
            results: List of experiment results.
        """
        json_path = self._reporter.save_json(results, "experiment_results.json")
        self._reporter.display_text(f"Experiment results saved to: {json_path}")
        self._reporter.display_text()

        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append(
                {
                    "scenario": result["scenario"],
                    "is_cross_domain": result["is_cross_domain"],
                    "train_samples": result["split_stats"]["train_samples"],
                    "test_samples": result["split_stats"]["test_samples"],
                    "pr_auc": result["metrics"]["pr_auc"],
                    "f1": result["metrics"]["f1"],
                    "mcc": result["metrics"]["mcc"],
                    "prevalence": result["metrics"]["prevalence"],
                    "threshold": result["metrics"]["threshold"],
                    "pr_auc_ci_lower": result["bootstrap_ci"]["pr_auc"][0],
                    "pr_auc_ci_upper": result["bootstrap_ci"]["pr_auc"][1],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        csv_path = self._reporter.save_csv(summary_df, "experiment_summary.csv")
        self._reporter.display_text(f"Experiment summary saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "split_experiment_evaluation": {
                    "n_experiments": len(results),
                    "results_json": str(json_path),
                    "summary_csv": str(csv_path),
                }
            }
        )
