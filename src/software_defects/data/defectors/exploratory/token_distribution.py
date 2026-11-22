"""Checkpoint 12: Token distribution analysis with the real model tokenizers."""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 2000
TOKENIZER_MODELS = {
    "CodeBERT": "microsoft/codebert-base",
    "CodeT5-small": "Salesforce/codet5-small",
}


def _download_tokenizers(cache_dir: Path | None = None) -> None:
    """Pre-download tokenizers to avoid download errors during analysis."""
    for _, hf_name in TOKENIZER_MODELS.items():
        try:
            AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir, local_files_only=True, use_safetensors=False)
        except Exception:
            pass  # Will be handled during actual analysis


class TokenDistribution:
    """Checkpoint 12: Token distribution analysis with the real model tokenizers."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter, cache_dir: Path | None = None) -> None:
        self._loader = loader
        self._reporter = reporter
        self._cache_dir = cache_dir

    @staticmethod
    def _count_tokens(tokenizer: AutoTokenizer, texts: list[str]) -> np.ndarray:
        """Count tokens per text (special tokens included, no truncation)."""
        counts: list[int] = []
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + BATCH_SIZE],
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            counts.extend(len(ids) for ids in encoded["input_ids"])
        return np.asarray(counts)

    def run(self) -> None:
        """Execute token distribution analysis."""
        self._reporter.display_section("Checkpoint 12: Token Distribution Analysis")
        self._reporter.display_text()

        # Use all splits: the experiments re-pool and re-split the data by repository
        all_jit = self._loader.load_all_jit()
        texts = all_jit["content"].fillna("").astype(str).tolist()

        token_data = []
        summary: dict = {}
        for model_label, hf_name in TOKENIZER_MODELS.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_name, cache_dir=self._cache_dir, local_files_only=True, use_safetensors=False
                )
            except Exception as exc:  # noqa: BLE001 - offline or missing model must not stop the run
                self._reporter.display_warning(f"  Could not load tokenizer {hf_name}: {exc}")
                continue

            counts = self._count_tokens(tokenizer, texts)
            overflow = int((counts > MAX_SEQUENCE_LENGTH).sum())
            overflow_pct = overflow / len(counts) * 100

            token_data.append(
                {
                    "model": model_label,
                    "tokenizer": hf_name,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                    "mean_tokens": float(counts.mean()),
                    "median_tokens": float(np.median(counts)),
                    "std_tokens": float(counts.std(ddof=1)),
                    "min_tokens": int(counts.min()),
                    "max_tokens": int(counts.max()),
                    "p25_tokens": float(np.quantile(counts, 0.25)),
                    "p50_tokens": float(np.quantile(counts, 0.50)),
                    "p75_tokens": float(np.quantile(counts, 0.75)),
                    "p90_tokens": float(np.quantile(counts, 0.90)),
                    "p95_tokens": float(np.quantile(counts, 0.95)),
                    "p99_tokens": float(np.quantile(counts, 0.99)),
                    "overflow_count": overflow,
                    "overflow_percentage": overflow_pct,
                }
            )

            self._reporter.display_text(f"{model_label} ({hf_name}):")
            self._reporter.display_text(f"  Mean tokens: {counts.mean():.2f}")
            self._reporter.display_text(f"  Median tokens: {np.median(counts):.2f}")
            self._reporter.display_text(f"  P95 tokens: {np.quantile(counts, 0.95):.2f}")
            self._reporter.display_text(f"  P99 tokens: {np.quantile(counts, 0.99):.2f}")
            self._reporter.display_text(f"  Overflow (>{MAX_SEQUENCE_LENGTH}): {overflow:,} ({overflow_pct:.2f}%)")
            self._reporter.display_text()
            summary[f"{model_label.lower().replace('-', '_')}_overflow_pct"] = overflow_pct

        if not token_data:
            self._reporter.display_warning("  No tokenizer could be loaded; token analysis skipped")
            return

        csv_path = self._reporter.save_csv(pd.DataFrame(token_data), "token_analysis.csv")
        self._reporter.display_text(f"Token analysis saved to: {csv_path}")
        self._reporter.display_text()

        summary["csv_path"] = str(csv_path)
        self._reporter.update_summary({"checkpoint_12": summary})
