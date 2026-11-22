"""Fine-tune a Transformer classifier on one Defectors split scenario over several seeds."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import nn

from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.data.defectors.tokenizer import DefectorsTokenizer
from software_defects.models import codebert_classifier, codet5_small_classifier
from software_defects.models.classifier_config import ClassifierConfig, TrainingConfig
from software_defects.models.classifier_trainer import ClassifierTrainer
from software_defects.models.codebert_classifier import CodeBertClassifier
from software_defects.models.codet5_small_classifier import CodeT5SmallClassifier
from software_defects.models.length_bucket_loader import LengthBucketLoader
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

# model key -> (classifier class, default pretrained name); keys match the tokenized folders of the pipeline
TRAINABLE_MODELS: dict[str, tuple[type[nn.Module], str]] = {
    "codebert": (CodeBertClassifier, codebert_classifier.DEFAULT_PRETRAINED_NAME),
    "codet5": (CodeT5SmallClassifier, codet5_small_classifier.DEFAULT_PRETRAINED_NAME),
}
REPORT_CLASS_NAMES = {"codebert": "CodeBertClassifier", "codet5": "CodeT5SmallClassifier"}
DEFAULT_SEEDS_PER_RUN = 3
DEFAULT_SEEDS_PER_FOLD = 2
TOKENIZE_CHUNK = 2048

ModelFactory = Callable[[ClassifierConfig], nn.Module]


class TrainingExperiment:
    """Load a tokenized scenario, fine-tune one model per (fold, seed), and record the runs.

    Only train and validation data are used (the model is selected on validation PR-AUC); the test split is left
    for the final evaluation after the hyperparameters are frozen.
    """

    def __init__(
        self,
        config: DatasetConfig,
        model_key: str,
        input_dir: Path,
        training_config: TrainingConfig | None = None,
        classifier_config: ClassifierConfig | None = None,
        model_factory: ModelFactory | None = None,
        tokenizer: DefectorsTokenizer | None = None,
        checkpoint_dir: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        """Initialize the experiment.

        Args:
            config: Dataset configuration.
            model_key: ``codebert`` or ``codet5``.
            input_dir: Directory with the pipeline output (the tokenized parquet files are read from it).
            training_config: Training loop settings.
            classifier_config: Classifier settings; defaults to the pre-trained backbone of ``model_key``.
            model_factory: Builds the classifier from its config (tests inject a tiny random model here).
            tokenizer: Tokenizer wrapper; defaults to the cached pre-trained tokenizer of ``model_key``.
            checkpoint_dir: Directory for checkpoints (default: ``artifacts/model/<dataset>``).
            registry_path: Run registry file (default: ``config.run_registry_path``).

        Raises:
            ValueError: If ``model_key`` is not trainable.
        """
        if model_key not in TRAINABLE_MODELS:
            raise ValueError(f"Unknown model '{model_key}'; expected one of {sorted(TRAINABLE_MODELS)}")
        classifier_class, pretrained_name = TRAINABLE_MODELS[model_key]
        self._config = config
        self._model_key = model_key
        self._input_dir = Path(input_dir)
        self._training_config = training_config or TrainingConfig()
        self._classifier_config = classifier_config or ClassifierConfig(
            pretrained_name=pretrained_name, cache_dir=str(config.model_cache_dir)
        )
        self._model_factory: ModelFactory = model_factory or classifier_class
        self._tokenizer = tokenizer or DefectorsTokenizer(
            config, model_name=config.tokenizer_models[model_key], max_length=self._training_config.max_length
        )
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else config.model_dir
        self._registry_path = Path(registry_path) if registry_path else config.run_registry_path
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="model_training",
            class_name=REPORT_CLASS_NAMES[model_key],
        )

    @staticmethod
    def default_seeds(scenario: SplitScenario) -> list[int]:
        """Seeds of the protocol: 3 for the non-fold scenarios, 2 per fold for Group K-Fold."""
        count = DEFAULT_SEEDS_PER_FOLD if scenario in GROUP_KFOLD_SCENARIOS else DEFAULT_SEEDS_PER_RUN
        return [42 + offset for offset in range(count)]

    def run(
        self,
        scenario: SplitScenario,
        seeds: list[int] | None = None,
        fold: int | None = None,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Train and validate one model per (fold, seed).

        Args:
            scenario: Split scenario to train on.
            seeds: Random seeds; defaults to the protocol seeds of the scenario.
            fold: Specific fold (Group K-Fold only); all folds when None.
            resume: Continue unfinished runs from their last checkpoint.

        Returns:
            One row per (fold, seed) run.
        """
        seeds = seeds or self.default_seeds(scenario)
        if scenario in GROUP_KFOLD_SCENARIOS:
            folds = [fold] if fold is not None else list(range(GROUP_KFOLD_N_FOLDS))
        elif fold is not None:
            raise ValueError("--fold is only valid for the Group_KFold scenarios")
        else:
            folds = [0]

        title = f"{REPORT_CLASS_NAMES[self._model_key]} training ({scenario.value})"
        self._reporter.display_section(title)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._reporter.display_text(f"Device: {device}")
        self._reporter.display_text(f"Training config: {asdict(self._training_config)}")
        self._reporter.display_text()

        rows: list[dict] = []
        for fold_index in folds:
            frames = self._load_frames(scenario, fold_index)
            for seed in seeds:
                rows.append(self._run_once(frames, scenario, fold_index, seed, resume))

        results = pd.DataFrame(rows)
        results_path = self._reporter.save_csv(results, "training_results.csv")
        self._reporter.update_summary(
            {
                "device": device,
                "scenario": scenario.value,
                "model": self._model_key,
                "seeds": seeds,
                "training_config": asdict(self._training_config),
                "classifier_config": asdict(self._classifier_config),
                "n_runs": len(results),
                "results_path": str(results_path),
            }
        )
        self._reporter.save_reports(
            text_filename=f"training_{scenario.value}.txt",
            json_filename=f"training_{scenario.value}.json",
        )
        return results

    def _run_once(
        self,
        frames: dict[str, tuple[list[list[int]], np.ndarray]],
        scenario: SplitScenario,
        fold: int,
        seed: int,
        resume: bool,
    ) -> dict:
        """Train one (fold, seed) run and append it to the run registry."""
        config = replace(self._training_config, seed=seed)
        train_ids, train_labels = frames["train"]
        val_ids, val_labels = frames["val"]
        train_ids, train_labels = self._limit(train_ids, train_labels, config.max_train_samples)
        val_ids, val_labels = self._limit(val_ids, val_labels, config.max_val_samples)

        pad_id = self._tokenizer.load_tokenizer().pad_token_id
        train_loader = LengthBucketLoader(
            train_ids,
            train_labels,
            config.batch_size,
            pad_id,
            shuffle=True,
            bucketing=config.length_bucketing,
            pool_factor=config.bucket_pool_factor,
            seed=seed,
        )
        val_loader = LengthBucketLoader(val_ids, val_labels, config.eval_batch_size, pad_id, False)

        run_name = self._run_name(scenario, fold, seed)
        torch.manual_seed(seed)
        model = self._model_factory(self._classifier_config)
        trainer = ClassifierTrainer(
            model,
            config,
            best_path=self._checkpoint_dir / f"{run_name}_best.pt",
            last_path=self._checkpoint_dir / f"{run_name}_last.pt",
            log=self._reporter.display_text,
        )
        self._reporter.display_text(
            f"Run {run_name}: train={len(train_ids):,} val={len(val_ids):,} "
            f"effective batch={config.effective_batch_size} precision={trainer.precision}"
        )
        result = trainer.fit(train_loader, val_loader, train_labels, resume=resume)
        self._reporter.display_text(
            f"  best epoch {result.best_epoch}: val PR-AUC={result.best_val_pr_auc:.4f} | "
            f"{result.train_seconds:.1f}s, {result.throughput_samples_per_second:.1f} samples/s, "
            f"peak VRAM={'n/a' if result.peak_vram_mib is None else f'{result.peak_vram_mib:.0f} MiB'}"
        )
        self._reporter.display_text()

        row = {
            "scenario": scenario.value,
            "model": self._model_key,
            "fold": fold,
            "seed": seed,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_val_pr_auc": result.best_val_pr_auc,
            "train_seconds": result.train_seconds,
            "throughput_samples_per_second": result.throughput_samples_per_second,
            "peak_vram_mib": result.peak_vram_mib,
            "n_parameters": result.n_parameters,
            "precision": result.precision,
            "effective_batch_size": config.effective_batch_size,
            "best_checkpoint": result.best_checkpoint,
        }
        self._append_registry(row, config, result.history)
        return row

    def _load_frames(self, scenario: SplitScenario, fold: int) -> dict[str, tuple[list[list[int]], np.ndarray]]:
        """Read the tokenized train/val parquet files and encode ``model_input`` to token ids."""
        scenario_dir = DefectorsSplitGenerator.scenario_dir(self._input_dir, scenario, fold)
        fold_hint = f" --fold {fold}" if scenario in GROUP_KFOLD_SCENARIOS else ""
        frames: dict[str, tuple[list[list[int]], np.ndarray]] = {}
        for split in ("train", "val"):
            path = scenario_dir / self._model_key / f"{scenario.value}_{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Tokenized {split} file not found: {path}\n"
                    f"Please run the data pipeline first (with tokenization): uv run python -m software_defects "
                    f"run-pipeline --dataset {self._config.dataset_name} --scenario {scenario.value}{fold_hint}"
                )
            frame = pd.read_parquet(path, columns=["model_input", "label"])
            frames[split] = (self._encode(frame["model_input"].tolist()), frame["label"].to_numpy(dtype=np.int64))
            self._reporter.display_text(f"Loaded {split}: {len(frame):,} rows from {path}")
        self._reporter.display_text()
        return frames

    def _encode(self, texts: list[str]) -> list[list[int]]:
        """Token ids (special tokens included, truncated to ``max_length``) without padding."""
        ids: list[list[int]] = []
        for start in range(0, len(texts), TOKENIZE_CHUNK):
            encoded = self._tokenizer.tokenize_batch(texts[start : start + TOKENIZE_CHUNK], truncation=True)
            ids.extend(encoded["input_ids"])
        return ids

    @staticmethod
    def _limit(ids: list[list[int]], labels: np.ndarray, limit: int | None) -> tuple[list[list[int]], np.ndarray]:
        """Keep a stratified sample of the first ``limit`` samples (smoke tests)."""
        if limit is None or limit >= len(ids):
            return ids, labels

        # Stratified sampling to ensure both classes are represented
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            # If only one class, just take first N
            return ids[:limit], labels[:limit]

        # Sample proportionally from each class
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        indices = np.arange(len(labels))
        sampled_indices = []

        for label, count in zip(unique_labels, counts):
            label_indices = indices[labels == label]
            n_samples = max(1, int(limit * count / len(labels)))  # At least 1 per class
            sampled = rng.choice(label_indices, size=min(n_samples, len(label_indices)), replace=False)
            sampled_indices.append(sampled)

        sampled_indices = np.concatenate(sampled_indices)
        # If we have more than limit, randomly subsample
        if len(sampled_indices) > limit:
            sampled_indices = rng.choice(sampled_indices, size=limit, replace=False)

        # Sort to maintain some order consistency
        sampled_indices = np.sort(sampled_indices)

        sampled_ids = [ids[i] for i in sampled_indices]
        sampled_labels = labels[sampled_indices]
        return sampled_ids, sampled_labels

    def _run_name(self, scenario: SplitScenario, fold: int, seed: int) -> str:
        """Checkpoint stem ``[model]_[scenario][_fold<n>]_seed<s>``."""
        fold_part = f"_fold{fold}" if scenario in GROUP_KFOLD_SCENARIOS else ""
        return f"{self._model_key}_{scenario.value}{fold_part}_seed{seed}"

    def _append_registry(self, row: dict, config: TrainingConfig, history: list) -> None:
        """Append the run to the tracked registry so every number traces back to commit, seed and configuration."""
        entry = {
            **row,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": self._git_commit(),
            "training_config": asdict(config),
            "classifier_config": asdict(self._classifier_config),
            "history": [record.__dict__ for record in history],
        }
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _git_commit(self) -> str | None:
        """Current commit hash of the project, or None outside a git checkout."""
        try:
            output = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._config.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return output.stdout.strip() or None
