"""Configuration objects of the Transformer classifiers and of their training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

PRECISION_CHOICES = ("auto", "bf16", "fp16", "fp32")
CLASS_WEIGHT_CHOICES = ("none", "balanced")


@dataclass(frozen=True)
class ClassifierConfig:
    """Settings of one classifier (backbone and classification head)."""

    # Hugging Face name of the pre-trained backbone; the CLI picks the right default per model
    pretrained_name: str
    # Dropout of the classification head
    head_dropout: float = 0.1
    # Two logits (not-defective, defective); the defect score is the softmax probability of class 1
    num_labels: int = 2
    # Recompute activations in the backward pass: slower, but needed to fit 512 tokens in 6 GB VRAM
    gradient_checkpointing: bool = False
    # Path to the Hugging Face model cache directory (downloaded by scripts/download_tokenizers.py)
    cache_dir: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Settings of the shared training loop."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    # Number of samples per optimizer micro-step; effective batch = batch_size * gradient_accumulation_steps
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 512
    # Sort samples by length inside large random pools so a batch only pads to its own longest sample
    length_bucketing: bool = True
    # Pool size = batch_size * bucket_pool_factor
    bucket_pool_factor: int = 50
    # ``none`` or ``balanced`` (weights n / (2 * n_class), computed on the training split only)
    class_weight: str = "balanced"
    # ``auto`` picks bf16 on a GPU that supports it (T5 overflows in fp16), else fp16; CPU always runs fp32
    precision: str = "auto"
    seed: int = 42
    # Smoke-test knobs: use only the first N rows of train/val
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    # Evaluation batch size (no gradients, so it can be larger than the training batch)
    eval_batch_size: int = 32
    # Log a line every N optimizer steps
    log_every: int = 50

    def __post_init__(self) -> None:
        """Reject values that would silently break the loop."""
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {self.precision!r}")
        if self.class_weight not in CLASS_WEIGHT_CHOICES:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHT_CHOICES}, got {self.class_weight!r}")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1 or self.epochs < 1:
            raise ValueError("batch_size, gradient_accumulation_steps and epochs must be at least 1")

    @property
    def effective_batch_size(self) -> int:
        """Samples per optimizer update."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class EpochRecord:
    """Metrics of one finished epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_pr_auc: float
    val_roc_auc: float
    seconds: float


@dataclass
class TrainingResult:
    """Outcome and resource usage of one training run (P3.10)."""

    best_epoch: int
    best_val_pr_auc: float
    epochs_run: int
    history: list[EpochRecord] = field(default_factory=list)
    train_seconds: float = 0.0
    # Training samples processed per second, over all epochs run in this process
    throughput_samples_per_second: float = 0.0
    # Peak GPU memory in MiB; None on CPU
    peak_vram_mib: float | None = None
    n_parameters: int = 0
    precision: str = "fp32"
    resumed_from_epoch: int | None = None
    best_checkpoint: str | None = None
    last_checkpoint: str | None = None
