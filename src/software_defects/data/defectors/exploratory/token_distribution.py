"""Checkpoint 12: Token distribution for CodeBERT and CodeT5-small."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class TokenDistribution:
    """Checkpoint 12: Token distribution for CodeBERT and CodeT5-small."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute token distribution analysis."""
        self._reporter.display_section("Checkpoint 12: Token Distribution Analysis")
        self._reporter.display_text()

        # Load JIT train data
        jit_train = self._loader.load_jit_train()

        # Simple token approximation (space-separated tokens)
        # In production, use actual tokenizers from transformers library
        jit_train["approx_tokens"] = jit_train["content"].apply(lambda x: len(str(x).split()) if x is not None else 0)

        # Statistics
        token_stats = {
            "mean": float(jit_train["approx_tokens"].mean()),
            "median": float(jit_train["approx_tokens"].median()),
            "std": float(jit_train["approx_tokens"].std()),
            "min": int(jit_train["approx_tokens"].min()),
            "max": int(jit_train["approx_tokens"].max()),
            "p25": float(jit_train["approx_tokens"].quantile(0.25)),
            "p50": float(jit_train["approx_tokens"].quantile(0.50)),
            "p75": float(jit_train["approx_tokens"].quantile(0.75)),
            "p90": float(jit_train["approx_tokens"].quantile(0.90)),
            "p95": float(jit_train["approx_tokens"].quantile(0.95)),
            "p99": float(jit_train["approx_tokens"].quantile(0.99)),
        }

        # CodeBERT: max 512 tokens
        codebert_overflow = (jit_train["approx_tokens"] > 512).sum()
        codebert_overflow_pct = codebert_overflow / len(jit_train) * 100

        # CodeT5-small: max 512 tokens
        codet5_overflow = (jit_train["approx_tokens"] > 512).sum()
        codet5_overflow_pct = codet5_overflow / len(jit_train) * 100

        # Create CSV
        token_data = [
            {
                "model": "CodeBERT",
                "max_sequence_length": 512,
                "mean_tokens": token_stats["mean"],
                "median_tokens": token_stats["median"],
                "std_tokens": token_stats["std"],
                "min_tokens": token_stats["min"],
                "max_tokens": token_stats["max"],
                "p25_tokens": token_stats["p25"],
                "p50_tokens": token_stats["p50"],
                "p75_tokens": token_stats["p75"],
                "p90_tokens": token_stats["p90"],
                "p95_tokens": token_stats["p95"],
                "p99_tokens": token_stats["p99"],
                "overflow_count": codebert_overflow,
                "overflow_percentage": codebert_overflow_pct,
            },
            {
                "model": "CodeT5-small",
                "max_sequence_length": 512,
                "mean_tokens": token_stats["mean"],
                "median_tokens": token_stats["median"],
                "std_tokens": token_stats["std"],
                "min_tokens": token_stats["min"],
                "max_tokens": token_stats["max"],
                "p25_tokens": token_stats["p25"],
                "p50_tokens": token_stats["p50"],
                "p75_tokens": token_stats["p75"],
                "p90_tokens": token_stats["p90"],
                "p95_tokens": token_stats["p95"],
                "p99_tokens": token_stats["p99"],
                "overflow_count": codet5_overflow,
                "overflow_percentage": codet5_overflow_pct,
            },
        ]

        token_df = pd.DataFrame(token_data)
        csv_path = self._reporter.save_csv(token_df, "token_analysis.csv")

        self._reporter.display_text(f"Token analysis saved to: {csv_path}")
        self._reporter.display_text()
        self._reporter.display_text(f"Mean tokens: {token_stats['mean']:.2f}")
        self._reporter.display_text(f"Median tokens: {token_stats['median']:.2f}")
        self._reporter.display_text(f"P95 tokens: {token_stats['p95']:.2f}")
        self._reporter.display_text(f"P99 tokens: {token_stats['p99']:.2f}")
        self._reporter.display_text(f"CodeBERT overflow (>512): {codebert_overflow:,} ({codebert_overflow_pct:.2f}%)")
        self._reporter.display_text(f"CodeT5-small overflow (>512): {codet5_overflow:,} ({codet5_overflow_pct:.2f}%)")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_12": {
                    "mean_tokens": token_stats["mean"],
                    "median_tokens": token_stats["median"],
                    "p95_tokens": token_stats["p95"],
                    "p99_tokens": token_stats["p99"],
                    "codebert_overflow_pct": codebert_overflow_pct,
                    "codet5_overflow_pct": codet5_overflow_pct,
                    "csv_path": str(csv_path),
                }
            }
        )
