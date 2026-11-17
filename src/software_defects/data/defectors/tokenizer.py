"""Tokenizer integration for Defectors dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsTokenizer:
    """Tokenizer wrapper for Defectors dataset with truncation policy."""

    def __init__(
        self,
        config: DatasetConfig,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 1024,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            config: Dataset configuration.
            model_name: Name of the pre-trained model/tokenizer.
            max_length: Maximum sequence length for truncation.
        """
        self._config = config
        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer = None

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer (lazy loading).

        Returns:
            Loaded tokenizer instance.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
        return self._tokenizer

    def tokenize_batch(
        self,
        texts: list[str],
        truncation: bool = True,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict:
        """Tokenize a batch of texts.

        Args:
            texts: List of text strings to tokenize.
            truncation: Whether to truncate sequences to max_length.
            padding: Whether to pad sequences.
            return_tensors: Format of returned tensors (e.g., 'pt', 'np').

        Returns:
            Dictionary with tokenized inputs.
        """
        tokenizer = self.load_tokenizer()
        return tokenizer(
            texts,
            truncation=truncation,
            max_length=self._max_length if truncation else None,
            padding=padding,
            return_tensors=return_tensors,
        )

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        content_column: str = "content",
        truncation: bool = True,
    ) -> pd.DataFrame:
        """Tokenize content column in a DataFrame.

        Args:
            df: DataFrame with content column.
            content_column: Name of the column containing text to tokenize.
            truncation: Whether to truncate sequences.

        Returns:
            DataFrame with added tokenization metadata.
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")

        # Convert content to strings
        texts = df[content_column].fillna("").astype(str).tolist()

        # Tokenize
        tokenizer = self.load_tokenizer()
        tokenized = tokenizer(
            texts,
            truncation=truncation,
            max_length=self._max_length if truncation else None,
            padding=False,
            return_tensors=None,
        )

        # Add metadata
        df = df.copy()
        df["token_count"] = [len(tokens) for tokens in tokenized["input_ids"]]
        df["truncated"] = df["token_count"] > self._max_length if truncation else False

        return df

    def get_statistics(self, df: pd.DataFrame, token_count_column: str = "token_count") -> dict:
        """Get tokenization statistics from a DataFrame.

        Args:
            df: DataFrame with token_count column.
            token_count_column: Name of the column with token counts.

        Returns:
            Dictionary with statistics.
        """
        if token_count_column not in df.columns:
            raise ValueError(f"Column '{token_count_column}' not found in DataFrame")

        stats = {
            "mean": float(df[token_count_column].mean()),
            "median": float(df[token_count_column].median()),
            "std": float(df[token_count_column].std()),
            "min": int(df[token_count_column].min()),
            "max": int(df[token_count_column].max()),
            "p25": float(df[token_count_column].quantile(0.25)),
            "p50": float(df[token_count_column].quantile(0.50)),
            "p75": float(df[token_count_column].quantile(0.75)),
            "p90": float(df[token_count_column].quantile(0.90)),
            "p95": float(df[token_count_column].quantile(0.95)),
            "p99": float(df[token_count_column].quantile(0.99)),
        }

        if "truncated" in df.columns:
            stats["truncated_count"] = int(df["truncated"].sum())
            stats["truncated_percentage"] = float(df["truncated"].mean() * 100)

        return stats
