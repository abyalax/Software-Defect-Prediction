"""Diff-size-only Logistic Regression baseline: the reference for shortcut signals."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from software_defects.models.tfidf_logreg import BaselineConfig

if TYPE_CHECKING:
    import pandas as pd

# Same definitions as checkpoint 13 (shortcut signal check)
FEATURE_NAMES = ("diff_length", "added_lines", "removed_lines", "num_hunks")


class LengthLogRegBaseline:
    """Logistic Regression over four diff-size counts, with no access to the code itself.

    Checkpoint 13 shows that diff size alone reaches a ROC-AUC of about 0.62-0.63. Any model that is
    supposed to understand code has to beat this baseline clearly to claim more than a size shortcut.
    The interface matches ``TfidfLogRegBaseline`` so both run through ``BaselineExperiment``.
    """

    def __init__(self, config: BaselineConfig | None = None, seed: int = 42) -> None:
        """Initialize the baseline.

        Args:
            config: Baseline hyperparameters; only the ``c_grid`` is used.
            seed: Random seed for the classifier.
        """
        self._config = config or BaselineConfig()
        self._seed = seed
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self.best_c: float | None = None
        self.n_features: int = len(FEATURE_NAMES)
        self.train_seconds: float = 0.0

    @staticmethod
    def extract_features(texts: pd.Series) -> np.ndarray:
        """Compute the diff-size features of a content column.

        Args:
            texts: Diff text of the rows.

        Returns:
            Array of shape (n_rows, 4) with log1p-scaled counts, in ``FEATURE_NAMES`` order.
        """
        text = texts.fillna("").astype(str)
        counts = np.column_stack(
            [
                text.str.len().to_numpy(),
                text.str.count(r"\n\+").to_numpy(),
                text.str.count(r"\n-").to_numpy(),
                text.str.count("@@").to_numpy(),
            ]
        )
        # Counts are heavy-tailed (P99 diff is about 15k characters), so compress them
        return np.log1p(counts.astype(np.float64))

    def fit(
        self,
        train_texts: pd.Series,
        y_train: np.ndarray,
        val_texts: pd.Series,
        y_val: np.ndarray,
    ) -> None:
        """Fit the scaler on train and select C by validation PR-AUC.

        Args:
            train_texts: Diff text of the training rows.
            y_train: Training labels.
            val_texts: Diff text of the validation rows.
            y_val: Validation labels.
        """
        start = time.perf_counter()

        self._scaler = StandardScaler()
        x_train = self._scaler.fit_transform(self.extract_features(train_texts))
        x_val = self._scaler.transform(self.extract_features(val_texts))

        best_score = -1.0
        for c_value in self._config.c_grid:
            model = LogisticRegression(C=c_value, solver="liblinear", max_iter=1000, random_state=self._seed)
            model.fit(x_train, y_train)
            score = float(average_precision_score(y_val, model.predict_proba(x_val)[:, 1]))
            if score > best_score:
                best_score = score
                self._model = model
                self.best_c = c_value

        self.train_seconds = time.perf_counter() - start

    def predict_scores(self, texts: pd.Series) -> np.ndarray:
        """Return the probability of the defective class.

        Args:
            texts: Diff text of the rows to score.

        Returns:
            Array of defect probabilities.
        """
        if self._scaler is None or self._model is None:
            raise RuntimeError("Call fit() before predict_scores().")
        return self._model.predict_proba(self._scaler.transform(self.extract_features(texts)))[:, 1]
