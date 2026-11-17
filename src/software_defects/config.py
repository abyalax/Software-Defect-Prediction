"""Configuration objects for the LSTM Autoencoder research project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """High-level settings of the transaction dataset."""

    project_root: Path
    dataset_name: str
    row_limit: int | None = None
    dtype: dict[str, str] | None = None
    tokenizer_max_length: int = 1024
    tokenizer_models: dict[str, str] = field(
        default_factory=lambda: {
            "codebert": "microsoft/codebert-base",
            "codet5": "Salesforce/codet5-small",
        }
    )

    @property
    def raw_dir(self) -> Path:
        """Directory that holds the untouched dataset files."""
        return self.project_root / "data" / "raw" / self.dataset_name

    @property
    def preprocessed_dir(self) -> Path:
        """Dataset-specific location for reusable preprocessed data."""
        return self.project_root / "data" / "preprocessed" / self.dataset_name

    @property
    def splits_dir(self) -> Path:
        """Root location for persisted dataset splits."""
        return self.project_root / "data" / "splits"

    def split_dir(self, split: str) -> Path:
        """Dataset-specific location for one persisted split."""
        if split not in {"train", "test", "eval"}:
            raise ValueError(f"Unknown split {split!r}; expected train, test, or eval.")
        return self.splits_dir / split / self.dataset_name

    @property
    def evaluations_dir(self) -> Path:
        """Dataset-specific location for evaluation inputs and outputs."""
        return self.project_root / "data" / "evaluations" / self.dataset_name

    @property
    def artifacts_dir(self) -> Path:
        """Directory that receives generated reports."""
        return self.project_root / "artifacts"

    @property
    def processed_dir(self) -> Path:
        """Directory that receives processed parquet files."""
        return self.project_root / "data" / "processed" / "defectors"
