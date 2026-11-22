"""Shared training loop of the Transformer classifiers (CodeBERT and CodeT5-small)."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import get_linear_schedule_with_warmup

from software_defects.models.classifier_config import EpochRecord, TrainingConfig, TrainingResult
from software_defects.models.length_bucket_loader import LengthBucketLoader

NO_DECAY_MARKERS = ("bias", "LayerNorm.weight", "layer_norm.weight")


class ClassifierTrainer:
    """Fine-tune a binary classifier with mixed precision, gradient accumulation and resumable checkpoints.

    The model must map ``(input_ids, attention_mask)`` to logits ``(batch, 2)``. Model selection uses validation
    PR-AUC only; the test split is never touched here.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device | None = None,
        best_path: Path | None = None,
        last_path: Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Classifier to fine-tune (moved to ``device``).
            config: Training settings.
            device: Device to train on; defaults to CUDA when available, else CPU.
            best_path: File that receives the weights of the best epoch (kept in memory when None).
            last_path: File that receives the full training state after every epoch (and every
                ``checkpoint_every_updates`` updates inside an epoch), used to resume.
            log: Callable that receives progress lines.
        """
        self._config = config
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._best_path = best_path
        self._last_path = last_path
        self._log = log
        self._precision = self._resolve_precision()
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._precision == "fp16")
        self._class_weight: torch.Tensor | None = None
        self._best_state: dict[str, torch.Tensor] | None = None

    @property
    def precision(self) -> str:
        """Precision actually used (``auto`` resolved, CPU fallback applied)."""
        return self._precision

    def fit(
        self,
        train_loader: LengthBucketLoader,
        val_loader: LengthBucketLoader,
        train_labels: np.ndarray,
        resume: bool = False,
    ) -> TrainingResult:
        """Train for ``config.epochs`` epochs and keep the epoch with the best validation PR-AUC.

        Args:
            train_loader: Shuffled training batches.
            val_loader: Length-sorted validation batches.
            train_labels: Training labels, used for the class weights.
            resume: Continue from ``last_path`` when it exists.

        Returns:
            Training outcome with the epoch history and resource usage. The model holds the best weights afterwards.
        """
        config = self._config
        torch.manual_seed(config.seed)
        self._class_weight = self._compute_class_weight(train_labels)

        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.epochs
        optimizer = self._build_optimizer()
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)

        result = TrainingResult(
            best_epoch=0,
            best_val_pr_auc=float("-inf"),
            epochs_run=0,
            precision=self._precision,
            n_parameters=sum(parameter.numel() for parameter in self._model.parameters()),
        )
        start_epoch = 0
        carried_seconds = 0.0
        partial: dict | None = None
        if resume and self._last_path is not None and self._last_path.exists():
            start_epoch, carried_seconds, partial = self._load_last(optimizer, scheduler, result)
            result.resumed_from_epoch = start_epoch
            if partial is not None:
                result.resumed_from_update = partial["updates_done"]
                self._log(
                    f"Resumed from {self._last_path} after epoch {start_epoch} "
                    f"and {partial['updates_done']} updates of epoch {partial['epoch']}"
                )
            else:
                self._log(f"Resumed from {self._last_path} after epoch {start_epoch}")

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        train_seconds = 0.0
        samples_seen = 0
        wall_start = time.perf_counter()
        for epoch in range(start_epoch + 1, config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loader.set_epoch(epoch)
            resume_point = partial if partial is not None and partial["epoch"] == epoch else None
            train_loss, n_samples = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                epoch,
                result,
                lambda: carried_seconds + time.perf_counter() - wall_start,
                resume_point,
            )
            partial = None
            train_seconds += time.perf_counter() - epoch_start
            samples_seen += n_samples

            scores, labels, val_loss = self.predict(val_loader)
            pr_auc = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
            roc_auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
            record = EpochRecord(epoch, train_loss, val_loss, pr_auc, roc_auc, time.perf_counter() - epoch_start)
            result.history.append(record)
            self._log(
                f"  epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_PR-AUC={pr_auc:.4f} val_ROC-AUC={roc_auc:.4f} ({record.seconds:.1f}s)"
            )

            if result.best_epoch == 0 or pr_auc > result.best_val_pr_auc:
                result.best_epoch, result.best_val_pr_auc = epoch, pr_auc
                self._save_best()
            result.epochs_run = epoch
            self._save_last(optimizer, scheduler, result, carried_seconds + time.perf_counter() - wall_start)

        self._load_best()
        result.train_seconds = carried_seconds + time.perf_counter() - wall_start
        result.throughput_samples_per_second = samples_seen / train_seconds if train_seconds > 0 else 0.0
        if self._device.type == "cuda":
            result.peak_vram_mib = torch.cuda.max_memory_allocated(self._device) / 2**20
        result.best_checkpoint = str(self._best_path) if self._best_path else None
        result.last_checkpoint = str(self._last_path) if self._last_path else None
        return result

    @torch.no_grad()
    def predict(self, loader: LengthBucketLoader) -> tuple[np.ndarray, np.ndarray, float]:
        """Score every sample of a loader.

        Returns:
            ``(scores, labels, mean_loss)`` in the original sample order; ``scores`` is P(defective).
        """
        self._model.eval()
        n = sum(len(indices) for indices in loader.batch_indices())
        scores = np.zeros(n, dtype=np.float64)
        labels = np.zeros(n, dtype=np.int64)
        loss_sum = 0.0
        for batch, indices in loader:
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with self._autocast():
                logits = self._model(batch["input_ids"], batch["attention_mask"])
            logits = logits.float()
            loss_sum += float(self._loss(logits, batch["labels"])) * len(indices)
            scores[indices] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels[indices] = batch["labels"].cpu().numpy()
        return scores, labels, loss_sum / max(n, 1)

    def _train_epoch(
        self,
        loader: LengthBucketLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        result: TrainingResult,
        elapsed: Callable[[], float],
        resume_point: dict | None = None,
    ) -> tuple[float, int]:
        """Run one epoch; returns (mean training loss, samples seen in this process).

        Args:
            loader: Training batches of this epoch.
            optimizer: Optimizer to step.
            scheduler: Learning-rate scheduler, stepped once per optimizer update.
            epoch: 1-based epoch number.
            result: Result holder, stored in mid-epoch checkpoints.
            elapsed: Returns the training seconds so far (stored in mid-epoch checkpoints).
            resume_point: Mid-epoch state from ``last.pt``; the batches already done are skipped.
        """
        config = self._config
        self._model.train()
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(loader)
        loss_sum, n_samples, update = 0.0, 0, 0
        first_batch, carried_samples = 0, 0
        if resume_point is not None:
            first_batch = self._check_resume_point(resume_point, n_batches)
            loss_sum, n_samples = resume_point["loss_sum"], resume_point["n_samples"]
            update = resume_point["updates_done"]
            carried_samples = n_samples
            self._restore_rng(resume_point.get("rng"))
        for step, (batch, indices) in enumerate(loader.iter_from(first_batch), start=first_batch + 1):
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with self._autocast():
                logits = self._model(batch["input_ids"], batch["attention_mask"])
            loss = self._loss(logits.float(), batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss became {loss.item()} at epoch {epoch}, step {step} (precision={self._precision}). "
                    "T5 overflows in fp16: use --precision bf16 or fp32, or lower the learning rate."
                )
            self._scaler.scale(loss / config.gradient_accumulation_steps).backward()
            loss_sum += float(loss.detach()) * len(indices)
            n_samples += len(indices)

            if step % config.gradient_accumulation_steps == 0 or step == n_batches:
                self._scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), config.max_grad_norm)
                self._scaler.step(optimizer)
                self._scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if config.log_every and update % config.log_every == 0:
                    self._log(f"    epoch {epoch} update {update}: running loss {loss_sum / n_samples:.4f}")
                every = config.checkpoint_every_updates
                if every and update % every == 0 and step < n_batches:
                    # Right after an update the gradients are zero, so the state below is a clean restart point.
                    # The last step of the epoch is skipped: the end-of-epoch checkpoint covers it.
                    point = {
                        "epoch": epoch,
                        "batches_done": step,
                        "updates_done": update,
                        "loss_sum": loss_sum,
                        "n_samples": n_samples,
                        "batch_size": config.batch_size,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "rng": self._capture_rng(),
                    }
                    self._save_last(optimizer, scheduler, result, elapsed(), partial=point)
        return loss_sum / max(n_samples, 1), n_samples - carried_samples

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy with the optional class weights."""
        weight = self._class_weight.to(logits.device) if self._class_weight is not None else None
        return nn.functional.cross_entropy(logits, labels, weight=weight)

    def _compute_class_weight(self, labels: np.ndarray) -> torch.Tensor | None:
        """Weights ``n / (2 * n_class)`` from the training labels; None when weighting is off."""
        if self._config.class_weight == "none":
            return None
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float64)
        if (counts == 0).any():
            raise ValueError("Class weights need both classes in the training labels")
        return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW without weight decay on biases and layer norms."""
        decay, no_decay = [], []
        for name, parameter in self._model.named_parameters():
            if parameter.requires_grad:
                (no_decay if any(marker in name for marker in NO_DECAY_MARKERS) else decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": self._config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self._config.learning_rate)

    def _resolve_precision(self) -> str:
        """Turn ``auto``/explicit precision into what the device can run."""
        requested = self._config.precision
        if self._device.type != "cuda":
            return "bf16" if requested == "bf16" else "fp32"
        if requested == "auto":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return requested

    def _autocast(self) -> torch.autocast:
        """Autocast context of the resolved precision (a no-op for fp32)."""
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self._precision)
        return torch.autocast(device_type=self._device.type, dtype=dtype, enabled=dtype is not None)

    def _save_best(self) -> None:
        """Keep the weights of the best epoch (file when a path is set, else a CPU copy)."""
        if self._best_path is not None:
            self._atomic_save(self._model.state_dict(), self._best_path)
        else:
            self._best_state = {key: value.detach().cpu().clone() for key, value in self._model.state_dict().items()}

    def _load_best(self) -> None:
        """Load the best weights back into the model."""
        if self._best_path is not None and self._best_path.exists():
            self._model.load_state_dict(torch.load(self._best_path, map_location=self._device))
        elif self._best_state is not None:
            self._model.load_state_dict(self._best_state)

    def _save_last(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        result: TrainingResult,
        elapsed_seconds: float,
        partial: dict | None = None,
    ) -> None:
        """Write the full state needed to resume.

        Without ``partial`` the state is the one after the epoch that just finished; with it, the state is a
        restart point inside the next, unfinished epoch (``epoch`` still counts finished epochs only).
        """
        if self._last_path is None:
            return
        state = {
            "model": self._model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": self._scaler.state_dict(),
            "epoch": result.epochs_run,
            "best_epoch": result.best_epoch,
            "best_val_pr_auc": result.best_val_pr_auc,
            "history": [record.__dict__ for record in result.history],
            "elapsed_seconds": elapsed_seconds,
            "partial": partial,
        }
        self._atomic_save(state, self._last_path)

    def _load_last(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        result: TrainingResult,
    ) -> tuple[int, float, dict | None]:
        """Restore the state written by ``_save_last``.

        Returns:
            ``(finished epochs, elapsed seconds, mid-epoch point or None)``. Checkpoints written before the
            mid-epoch feature have no ``partial`` key and resume at the epoch boundary as before.
        """
        state = torch.load(self._last_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        self._scaler.load_state_dict(state["scaler"])
        result.best_epoch = state["best_epoch"]
        result.best_val_pr_auc = state["best_val_pr_auc"]
        result.epochs_run = state["epoch"]
        result.history = [EpochRecord(**record) for record in state["history"]]
        return state["epoch"], state["elapsed_seconds"], state.get("partial")

    def _check_resume_point(self, point: dict, n_batches: int) -> int:
        """Validate a mid-epoch point against the current setup; returns the number of batches to skip."""
        config = self._config
        if (point["batch_size"], point["gradient_accumulation_steps"]) != (
            config.batch_size,
            config.gradient_accumulation_steps,
        ):
            raise ValueError(
                "Cannot resume inside an epoch: batch size / gradient accumulation differ from the checkpoint "
                f"({point['batch_size']}/{point['gradient_accumulation_steps']} vs "
                f"{config.batch_size}/{config.gradient_accumulation_steps}). Use the original settings."
            )
        if point["batches_done"] >= n_batches:
            raise ValueError("Cannot resume inside an epoch: the checkpoint is past the end of this training set")
        return int(point["batches_done"])

    def _capture_rng(self) -> dict:
        """RNG state (dropout) so a resumed epoch continues the same random stream."""
        state = {"cpu": torch.get_rng_state()}
        if self._device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng(self, state: dict | None) -> None:
        """Restore ``_capture_rng`` output (no-op for checkpoints without it)."""
        if not state:
            return
        torch.set_rng_state(state["cpu"].cpu())
        if "cuda" in state and self._device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

    @staticmethod
    def _atomic_save(obj: object, path: Path) -> None:
        """Write through a temporary file so a crash never leaves a half-written checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, temporary)
        os.replace(temporary, path)
