"""Configuration objects for the LSTM Autoencoder research project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DiffInputConfig:
    """Settings of the model input built from a file path and its diff."""

    # False drops the path (used by the input ablation); the default puts the path first in every input
    include_path: bool = True
    # Paths longer than this many tokens are shortened from the left (directories first, filename last)
    path_max_tokens: int = 64
    # Diffs longer than this many characters are pre-cut (hunk-aware) before tokenization to keep the pipeline fast
    pre_cut_chars: int = 16_000
    # Hunk headers may use at most this share of the diff token budget
    header_share: float = 0.5
    # How many times the diff budget is tightened when the assembled text still exceeds the limit
    max_refits: int = 4


@dataclass(frozen=True)
class DatasetConfig:
    """High-level settings of the transaction dataset."""

    project_root: Path
    dataset_name: str
    row_limit: int | None = None
    dtype: dict[str, str] | None = None
    # CodeBERT (RoBERTa) has 512 position embeddings, so 512 is the largest length both models can share
    tokenizer_max_length: int = 512
    diff_input: DiffInputConfig = field(default_factory=DiffInputConfig)
    tokenizer_models: dict[str, str] = field(
        default_factory=lambda: {
            "codebert": "microsoft/codebert-base",
            "codet5": "Salesforce/codet5-small",
        }
    )

    @property
    def model_cache_dir(self) -> Path:
        """Directory that holds the Hugging Face model cache (downloaded by scripts/download_tokenizers.py)."""
        return self.project_root / "data" / "models" / "cache"

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
        """Directory that receives the split parquet files (``data/processed/split/<dataset>``).

        Every scenario (and every Group K-Fold fold) gets its own subdirectory below it.
        """
        return self.project_root / "data" / "processed" / "split" / self.dataset_name

    @property
    def model_dir(self) -> Path:
        """Directory that receives model checkpoints and the run registry (``artifacts/model/<dataset>``)."""
        return self.artifacts_dir / "model" / self.dataset_name

    @property
    def split_manifest_dir(self) -> Path:
        """Directory that holds the row-level split manifests (tracked in git, unlike processed data)."""
        return self.artifacts_dir / "split_manifest" / self.dataset_name

    @property
    def run_registry_path(self) -> Path:
        """Tracked registry of fine-tuning runs (one JSON line per run)."""
        return self.artifacts_dir / "report" / self.dataset_name / "model_training" / "run_registry.jsonl"
