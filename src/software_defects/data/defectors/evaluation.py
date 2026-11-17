"""Evaluation metrics for defect prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class EvaluationResult:
    """Result of evaluation metrics calculation."""

    pr_auc: float
    f1: float
    mcc: float
    threshold: float
    prevalence: float
    prevalence_baseline_pr_auc: float
    roc_auc: float = float("nan")


class DefectEvaluator:
    """Calculate evaluation metrics for defect prediction."""

    def __init__(self) -> None:
        """Initialize the evaluator."""

    def calculate_metrics(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
        threshold: float | None = None,
    ) -> EvaluationResult:
        """Calculate PR-AUC, F1, and MCC metrics.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores/probabilities.
            threshold: Threshold for binary classification. If None, optimal threshold is selected.

        Returns:
            EvaluationResult with metrics.
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Calculate prevalence
        prevalence = float(np.mean(y_true))

        # Calculate prevalence baseline PR-AUC (random classifier)
        prevalence_baseline_pr_auc = prevalence

        # Calculate PR-AUC
        pr_auc = float(average_precision_score(y_true, y_scores))

        # ROC-AUC is undefined when only one class is present
        roc_auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else float("nan")

        # Select threshold if not provided
        if threshold is None:
            threshold = self._select_optimal_threshold(y_true, y_scores)

        # Convert scores to binary predictions
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate F1 and MCC
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        return EvaluationResult(
            pr_auc=pr_auc,
            f1=f1,
            roc_auc=roc_auc,
            mcc=mcc,
            threshold=threshold,
            prevalence=prevalence,
            prevalence_baseline_pr_auc=prevalence_baseline_pr_auc,
        )

    def _select_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Select optimal threshold using F1 score.

        Args:
            y_true: True binary labels.
            y_scores: Predicted scores.

        Returns:
            Optimal threshold value.
        """
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_f1 = 0.0

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return float(best_threshold)

    def calculate_metrics_with_val_threshold(
        self,
        y_val_true: np.ndarray | pd.Series,
        y_val_scores: np.ndarray | pd.Series,
        y_test_true: np.ndarray | pd.Series,
        y_test_scores: np.ndarray | pd.Series,
    ) -> EvaluationResult:
        """Calculate metrics using threshold selected from validation set.

        Args:
            y_val_true: True labels for validation set.
            y_val_scores: Predicted scores for validation set.
            y_test_true: True labels for test set.
            y_test_scores: Predicted scores for test set.

        Returns:
            EvaluationResult with metrics calculated on test set using val threshold.
        """
        # Select threshold from validation set
        threshold = self._select_optimal_threshold(np.array(y_val_true), np.array(y_val_scores))

        # Calculate metrics on test set using this threshold
        return self.calculate_metrics(y_test_true, y_test_scores, threshold=threshold)

    def calculate_per_repo_metrics(
        self,
        df: pd.DataFrame,
        score_col: str,
        label_col: str = "label",
        repo_col: str = "repo",
        threshold: float | None = None,
    ) -> dict[str, EvaluationResult]:
        """Calculate metrics per repository.

        Args:
            df: DataFrame with scores and labels.
            score_col: Column name for predicted scores.
            label_col: Column name for true labels.
            repo_col: Column name for repository.
            threshold: Threshold for binary classification. If None, optimal per-repo threshold is used.

        Returns:
            Dictionary mapping repo names to EvaluationResult.
        """
        results = {}

        for repo in df[repo_col].unique():
            repo_df = df[df[repo_col] == repo]
            y_true = repo_df[label_col].values
            y_scores = repo_df[score_col].values

            if len(y_true) == 0:
                continue

            # Use per-repo threshold if not provided
            repo_threshold = threshold
            if repo_threshold is None:
                repo_threshold = self._select_optimal_threshold(y_true, y_scores)

            results[repo] = self.calculate_metrics(y_true, y_scores, threshold=repo_threshold)

        return results

    def calculate_bootstrap_ci(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        threshold: float | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics.

        Args:
            y_true: True binary labels.
            y_scores: Predicted scores.
            n_bootstrap: Number of bootstrap samples.
            ci_level: Confidence level (e.g., 0.95 for 95% CI).
            threshold: Threshold for binary classification.

        Returns:
            Dictionary with CI for each metric (lower, upper).
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        n = len(y_true)

        if threshold is None:
            threshold = self._select_optimal_threshold(y_true, y_scores)

        # Bootstrap samples
        pr_aucs = []
        f1s = []
        mccs = []

        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_true_boot = y_true[idx]
            y_scores_boot = y_scores[idx]

            result = self.calculate_metrics(y_true_boot, y_scores_boot, threshold=threshold)
            pr_aucs.append(result.pr_auc)
            f1s.append(result.f1)
            mccs.append(result.mcc)

        # Calculate CI
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci = {
            "pr_auc": (
                float(np.percentile(pr_aucs, lower_percentile)),
                float(np.percentile(pr_aucs, upper_percentile)),
            ),
            "f1": (
                float(np.percentile(f1s, lower_percentile)),
                float(np.percentile(f1s, upper_percentile)),
            ),
            "mcc": (
                float(np.percentile(mccs, lower_percentile)),
                float(np.percentile(mccs, upper_percentile)),
            ),
        }

        return ci
