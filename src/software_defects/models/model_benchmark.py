"""Local GPU benchmark of the Transformer classifiers and the run-time estimate built on it (P3.7 and P3.8)."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import nn

from software_defects.data.defectors.split_generator import GROUP_KFOLD_SCENARIOS, SplitScenario
from software_defects.models.classifier_config import ClassifierConfig, TrainingConfig
from software_defects.models.classifier_trainer import ClassifierTrainer
from software_defects.models.training_experiment import REPORT_CLASS_NAMES, TRAINABLE_MODELS, ModelFactory
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

BOS_ID, EOS_ID = 0, 2  # the same for the CodeBERT and CodeT5 tokenizers
FOLD_RUNS_PER_SCENARIO = 10  # 5 folds x 2 seeds
SEED_RUNS_PER_SCENARIO = 3


class ModelBenchmark:
    """Measure training/inference throughput and peak VRAM at full length, then project the total run time."""

    def __init__(
        self,
        config: DatasetConfig,
        model_key: str,
        training_config: TrainingConfig | None = None,
        classifier_config: ClassifierConfig | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        """Initialize the benchmark.

        Args:
            config: Dataset configuration.
            model_key: ``codebert`` or ``codet5``.
            training_config: Training settings (precision and sequence length are taken from it).
            classifier_config: Classifier settings (gradient checkpointing is taken from it).
            model_factory: Builds the classifier; tests inject a tiny random model.

        Raises:
            ValueError: If ``model_key`` is not trainable.
        """
        if model_key not in TRAINABLE_MODELS:
            raise ValueError(f"Unknown model '{model_key}'; expected one of {sorted(TRAINABLE_MODELS)}")
        classifier_class, pretrained_name = TRAINABLE_MODELS[model_key]
        self._config = config
        self._model_key = model_key
        self._training_config = training_config or TrainingConfig()
        self._classifier_config = classifier_config or ClassifierConfig(
            pretrained_name=pretrained_name, cache_dir=str(config.model_cache_dir)
        )
        self._model_factory: ModelFactory = model_factory or classifier_class
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="model_benchmark",
            class_name=REPORT_CLASS_NAMES[model_key],
        )

    def run(self, batch_sizes: list[int], seq_len: int = 512, steps: int = 5, epochs: int = 3) -> pd.DataFrame:
        """Benchmark ascending batch sizes until one runs out of memory, then estimate the 35 runs of this model.

        Args:
            batch_sizes: Micro-batch sizes to try, ascending.
            seq_len: Input length of every sample (512 = worst case, no padding savings).
            steps: Timed steps per batch size (after one warm-up step).
            epochs: Epochs used by the run-time estimate.

        Returns:
            One row per batch size with throughput and peak VRAM.
        """
        self._reporter.display_section(f"{REPORT_CLASS_NAMES[self._model_key]} benchmark")
        self._reporter.display_text(f"Device: {self._device} | seq_len={seq_len} | steps={steps}")
        if self._device.type != "cuda":
            self._reporter.display_warning("No CUDA device: numbers below are CPU numbers, not RTX 3050 numbers.")

        rows = []
        for batch_size in batch_sizes:
            row = self._measure(batch_size, seq_len, steps)
            rows.append(row)
            self._reporter.display_text(
                f"  batch {batch_size}: "
                + ("OOM" if row["oom"] else f"{row['train_samples_per_second']:.1f} train samples/s, "
                   f"{row['inference_samples_per_second']:.1f} infer samples/s, "
                   f"peak VRAM {row['peak_vram_mib'] if row['peak_vram_mib'] is not None else 'n/a'} MiB")
            )  # fmt: skip
            if row["oom"]:
                break

        results = pd.DataFrame(rows)
        estimate = self.estimate(results, epochs)
        self._reporter.display_text()
        self._reporter.display_text(f"Estimated hours for the 35 runs of {self._model_key} ({epochs} epochs):")
        self._reporter.display_table(estimate)
        self._reporter.save_csv(results, "benchmark_results.csv")
        self._reporter.save_csv(estimate, "run_time_estimate.csv")
        self._reporter.update_summary(
            {
                "device": str(self._device),
                "model": self._model_key,
                "seq_len": seq_len,
                "precision": self._training_config.precision,
                "gradient_checkpointing": self._classifier_config.gradient_checkpointing,
                "max_batch_size": self.max_batch_size(results),
                "total_hours": float(estimate["hours"].sum()) if len(estimate) else None,
            }
        )
        self._reporter.save_reports("benchmark.txt", "benchmark.json")
        return results

    @staticmethod
    def max_batch_size(results: pd.DataFrame) -> int | None:
        """Largest batch size that did not run out of memory."""
        fitted = results.loc[~results["oom"], "batch_size"]
        return int(fitted.max()) if len(fitted) else None

    def estimate(self, results: pd.DataFrame, epochs: int) -> pd.DataFrame:
        """Project the run time of all runs of this model from the measured throughput.

        The best throughput among the batch sizes that fit is used. Train samples come from the split manifests
        (3 seeds for the five non-fold scenarios, 2 seeds per fold for each Group K-Fold scenario).

        Args:
            results: Output of ``run``.
            epochs: Number of training epochs per run.

        Returns:
            One row per scenario with the number of runs, samples and estimated hours (empty without data).
        """
        fitted = results[~results["oom"]]
        sizes = self._split_sizes()
        if fitted.empty or not sizes:
            return pd.DataFrame(columns=["scenario", "runs", "train_samples", "val_samples", "hours"])
        train_rate = float(fitted["train_samples_per_second"].max())
        infer_rate = float(fitted["inference_samples_per_second"].max())

        rows = []
        for scenario, (train_samples, val_samples, n_manifests) in sizes.items():
            seeds = 2 if SplitScenario(scenario) in GROUP_KFOLD_SCENARIOS else SEED_RUNS_PER_SCENARIO
            runs = n_manifests * seeds
            seconds = runs * (epochs * train_samples / train_rate + epochs * val_samples / infer_rate)
            rows.append(
                {
                    "scenario": scenario,
                    "runs": runs,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "hours": seconds / 3600,
                }
            )
        return pd.DataFrame(rows)

    def _split_sizes(self) -> dict[str, tuple[int, int, int]]:
        """Summed train/val samples and manifest count per scenario, read from the stored split manifests."""
        directory = self._config.split_manifest_dir
        index_path = directory / "index.json"
        if not index_path.exists():
            return {}
        sizes: dict[str, list[int]] = {}
        for entry in json.loads(index_path.read_text())["entries"]:
            splits = json.loads((directory / entry["path"]).read_text())["contract"]["splits"]
            total = sizes.setdefault(entry["scenario"], [0, 0, 0])
            total[0] += splits["train"]["samples"]
            total[1] += splits["val"]["samples"]
            total[2] += 1
        # Folds differ in size, so report the per-run mean to keep ``runs`` = manifests x seeds
        return {name: (round(t / n), round(v / n), n) for name, (t, v, n) in sizes.items()}

    def _measure(self, batch_size: int, seq_len: int, steps: int) -> dict:
        """Time ``steps`` optimizer steps and inference passes at one batch size; flags an out-of-memory failure."""
        row: dict = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "oom": False,
            "train_samples_per_second": None,
            "inference_samples_per_second": None,
            "peak_vram_mib": None,
        }
        model = None
        try:
            model = self._model_factory(self._classifier_config)
            trainer = ClassifierTrainer(model, self._training_config, self._device, log=lambda _: None)
            vocab = _vocab_size(model)
            ids = torch.randint(5, vocab, (batch_size, seq_len))
            ids[:, 0], ids[:, -1] = BOS_ID, EOS_ID
            batch = {
                "input_ids": ids.to(self._device),
                "attention_mask": torch.ones_like(ids).to(self._device),
                "labels": (torch.arange(batch_size) % 2).to(self._device),
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            if self._device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self._device)

            model.train()
            self._train_step(trainer, model, optimizer, batch)  # warm-up (allocator, cuDNN autotune)
            self._sync()
            start = time.perf_counter()
            for _ in range(steps):
                self._train_step(trainer, model, optimizer, batch)
            self._sync()
            row["train_samples_per_second"] = batch_size * steps / (time.perf_counter() - start)
            if self._device.type == "cuda":
                row["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(self._device) / 2**20, 1)

            model.eval()
            with torch.no_grad(), trainer._autocast():
                model(batch["input_ids"], batch["attention_mask"])
                self._sync()
                start = time.perf_counter()
                for _ in range(steps):
                    model(batch["input_ids"], batch["attention_mask"])
                self._sync()
            row["inference_samples_per_second"] = batch_size * steps / (time.perf_counter() - start)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            row["oom"] = True
        finally:
            del model
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
        return row

    def _train_step(
        self, trainer: ClassifierTrainer, model: nn.Module, optimizer: torch.optim.Optimizer, batch: dict
    ) -> None:
        """One forward/backward/update step under the trainer's precision."""
        with trainer._autocast():
            logits = model(batch["input_ids"], batch["attention_mask"])
        loss = nn.functional.cross_entropy(logits.float(), batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _sync(self) -> None:
        """Wait for queued GPU work so wall-clock timings are honest."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)


def _vocab_size(model: nn.Module) -> int:
    """Vocabulary size of the classifier's backbone."""
    return int(model.backbone.config.vocab_size)  # type: ignore[union-attr]
