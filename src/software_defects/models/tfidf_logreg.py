"""TF-IDF + Logistic Regression baseline for file-level JIT defect prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    """Hyperparameters of the TF-IDF + Logistic Regression baseline."""

    max_features: int = 100_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    # Cap on diff length in characters (P99 is about 15k); protects memory on huge diffs
    max_chars: int = 100_000
    # Inverse regularization strengths tried on the validation set
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)


class TfidfLogRegBaseline:
    """TF-IDF features over the diff text followed by Logistic Regression."""

    def __init__(self, config: BaselineConfig | None = None, seed: int = 42) -> None:
        """Initialize the baseline.

        Args:
            config: Baseline hyperparameters.
            seed: Random seed for the classifier.
        """
        self._config = config or BaselineConfig()
        self._seed = seed
        self._vectorizer: TfidfVectorizer | None = None
        self._model: LogisticRegression | None = None
        self.best_c: float | None = None
        self.n_features: int = 0
        self.train_seconds: float = 0.0

    def _prepare_texts(self, texts: pd.Series) -> list[str]:
        """Convert a content column to capped strings."""
        return texts.fillna("").astype(str).str.slice(0, self._config.max_chars).tolist()

    def fit(
        self,
        train_texts: pd.Series,
        y_train: np.ndarray,
        val_texts: pd.Series,
        y_val: np.ndarray,
    ) -> None:
        """Fit the vectorizer on train and select C by validation PR-AUC.

        Args:
            train_texts: Diff text of the training rows.
            y_train: Training labels.
            val_texts: Diff text of the validation rows.
            y_val: Validation labels.
        """
        start = time.perf_counter()

        # Identifier-like tokens keep the vocabulary compact on code diffs
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]+",
            lowercase=False,
            ngram_range=self._config.ngram_range,
            min_df=self._config.min_df,
            max_features=self._config.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train = self._vectorizer.fit_transform(self._prepare_texts(train_texts))
        x_val = self._vectorizer.transform(self._prepare_texts(val_texts))
        self.n_features = int(x_train.shape[1])

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
        if self._vectorizer is None or self._model is None:
            raise RuntimeError("Call fit() before predict_scores().")
        return self._model.predict_proba(self._vectorizer.transform(self._prepare_texts(texts)))[:, 1]
