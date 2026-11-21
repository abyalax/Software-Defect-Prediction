"""Tokenizer integration for Defectors dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from transformers import AutoTokenizer

from software_defects.data.defectors.diff_input_builder import DiffInput, DiffInputBuilder

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

COUNT_BATCH_SIZE = 2048


class DefectorsTokenizer:
    """Tokenizer wrapper for Defectors dataset: builds ``path + diff`` inputs with hunk-aware truncation."""

    def __init__(
        self,
        config: DatasetConfig,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
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
        self._builder: DiffInputBuilder | None = None

    @property
    def model_name(self) -> str:
        """Name of the pre-trained model/tokenizer."""
        return self._model_name

    @property
    def max_length(self) -> int:
        """Maximum sequence length, special tokens included."""
        return self._max_length

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Count tokens of each text without special tokens and without truncation.

        Args:
            texts: Texts to count.

        Returns:
            Token count per text.
        """
        tokenizer = self.load_tokenizer()
        counts: list[int] = []
        for start in range(0, len(texts), COUNT_BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + COUNT_BATCH_SIZE],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            counts.extend(len(ids) for ids in encoded["input_ids"])
        return counts

    def input_builder(self) -> DiffInputBuilder:
        """Return the (lazily created) builder of ``path + diff`` model inputs."""
        if self._builder is None:
            self._builder = DiffInputBuilder(
                count_tokens=self.count_tokens,
                special_tokens=self.load_tokenizer().num_special_tokens_to_add(pair=False),
                max_length=self._max_length,
                config=self._config.diff_input,
            )
        return self._builder

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
        path_column: str = "filepath",
        truncation: bool = True,
    ) -> pd.DataFrame:
        """Build the model input of every row and tokenize it.

        The input is the file path on the first line followed by the diff, shrunk hunk-aware to the token
        limit. The tokenizer's own truncation stays on as a last safety net and is reported in
        ``fallback_truncation``.

        Args:
            df: DataFrame with the content and path columns.
            content_column: Name of the column containing the diff.
            path_column: Name of the column containing the file path.
            truncation: Whether to shrink inputs to ``max_length``. When False the full text is kept.

        Returns:
            Copy of ``df`` with ``model_input``, ``token_count`` (final, special tokens included),
            ``truncated`` and the preprocessing metadata columns.
        """
        for column in (content_column, path_column):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

        paths = df[path_column].fillna("").astype(str).tolist()
        diffs = df[content_column].fillna("").astype(str).tolist()

        builder = self.input_builder()
        if truncation:
            inputs = builder.build(paths, diffs)
        else:
            inputs = self._untruncated_inputs(paths, diffs)

        tokenizer = self.load_tokenizer()
        token_counts: list[int] = []
        texts = [item.text for item in inputs]
        for start in range(0, len(texts), COUNT_BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + COUNT_BATCH_SIZE],
                truncation=truncation,
                max_length=self._max_length if truncation else None,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            token_counts.extend(len(ids) for ids in encoded["input_ids"])

        df = df.copy()
        df["model_input"] = texts
        df["token_count"] = token_counts
        df["truncated"] = [item.truncated for item in inputs]
        df["pre_cut"] = [item.pre_cut for item in inputs]
        df["fallback_truncation"] = [item.fallback for item in inputs]
        df["path_shortened"] = [item.path_shortened for item in inputs]
        df["full_tokens"] = [item.full_tokens for item in inputs]
        df["hunks_total"] = [item.hunks_total for item in inputs]
        df["hunks_kept"] = [item.hunks_kept for item in inputs]
        df["changed_total"] = [item.changed_total for item in inputs]
        df["changed_kept"] = [item.changed_kept for item in inputs]
        return df

    def _untruncated_inputs(self, paths: list[str], diffs: list[str]) -> list[DiffInput]:
        """Inputs that keep the whole diff (used to measure how long untruncated inputs are)."""
        config = self._config.diff_input
        texts = [f"{path}\n{diff}" if config.include_path else diff for path, diff in zip(paths, diffs, strict=True)]
        counts = self.count_tokens(texts)
        return [
            DiffInput(text, False, False, False, False, count, 0, 0, 0, 0)
            for text, count in zip(texts, counts, strict=True)
        ]

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

        for column in ("truncated", "pre_cut", "fallback_truncation", "path_shortened"):
            if column in df.columns:
                prefix = "truncated" if column == "truncated" else column
                stats[f"{prefix}_count"] = int(df[column].sum())
                stats[f"{prefix}_percentage"] = float(df[column].mean() * 100)

        if {"changed_total", "changed_kept"}.issubset(df.columns):
            total = int(df["changed_total"].sum())
            stats["changed_lines_kept_percentage"] = float(df["changed_kept"].sum() / total * 100) if total else 100.0

        return stats
