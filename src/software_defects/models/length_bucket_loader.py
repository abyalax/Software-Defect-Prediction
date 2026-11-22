"""Batch iterator with dynamic padding and optional length bucketing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch


class LengthBucketLoader:
    """Yield padded batches of pre-tokenized samples.

    Each batch is padded only to its own longest sample (dynamic padding). With bucketing, samples are sorted by
    length inside large random pools, so batches hold similar lengths and waste little compute on padding, while
    the pool shuffle keeps the batch composition random.
    """

    def __init__(
        self,
        input_ids: Sequence[Sequence[int]],
        labels: Sequence[int],
        batch_size: int,
        pad_token_id: int,
        shuffle: bool,
        bucketing: bool = True,
        pool_factor: int = 50,
        seed: int = 42,
    ) -> None:
        """Initialize the loader.

        Args:
            input_ids: Token ids per sample (special tokens included, no padding).
            labels: Binary label per sample.
            batch_size: Samples per batch.
            pad_token_id: Id used for padding.
            shuffle: True for training (random order every epoch); False yields length-sorted batches for evaluation.
            bucketing: Sort by length inside pools of ``batch_size * pool_factor`` samples (training only).
            pool_factor: Pool size in batches.
            seed: Base seed; combined with the epoch set by ``set_epoch``.
        """
        if len(input_ids) != len(labels):
            raise ValueError("input_ids and labels must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._input_ids = input_ids
        self._labels = np.asarray(labels, dtype=np.int64)
        self._lengths = np.fromiter((len(ids) for ids in input_ids), dtype=np.int64, count=len(input_ids))
        self._batch_size = batch_size
        self._pad_token_id = pad_token_id
        self._shuffle = shuffle
        self._bucketing = bucketing
        self._pool_size = batch_size * max(pool_factor, 1)
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return -(-len(self._input_ids) // self._batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Change the shuffle seed so every epoch sees a different order."""
        self._epoch = epoch

    def batch_indices(self) -> list[np.ndarray]:
        """Return the sample indices of every batch for the current epoch."""
        n = len(self._input_ids)
        if not self._shuffle:
            order = np.argsort(self._lengths, kind="stable")
            return [order[start : start + self._batch_size] for start in range(0, n, self._batch_size)]

        rng = np.random.default_rng(self._seed + self._epoch)
        order = rng.permutation(n)
        if not self._bucketing:
            return [order[start : start + self._batch_size] for start in range(0, n, self._batch_size)]

        batches: list[np.ndarray] = []
        for pool_start in range(0, n, self._pool_size):
            pool = order[pool_start : pool_start + self._pool_size]
            pool = pool[np.argsort(self._lengths[pool], kind="stable")]
            batches.extend(pool[start : start + self._batch_size] for start in range(0, len(pool), self._batch_size))
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], np.ndarray]]:
        """Yield ``(batch, sample_indices)``; the batch holds ``input_ids``, ``attention_mask`` and ``labels``."""
        for indices in self.batch_indices():
            yield self._collate(indices), indices

    def _collate(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        """Pad the samples of one batch to their longest member."""
        width = int(self._lengths[indices].max())
        input_ids = torch.full((len(indices), width), self._pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(indices), width), dtype=torch.long)
        for row, index in enumerate(indices):
            ids = self._input_ids[int(index)]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.from_numpy(self._labels[indices]),
        }
