"""Parser for the LSTM Autoencoder research project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

MANIFEST_SCENARIO_CHOICES = [
    "ML_to_NonML",
    "NonML_to_ML",
    "Repository_Level_Split",
    "Within_Domain_ML_DL",
    "Within_Domain_Non_ML",
    "Group_KFold_ML_DL",
    "Group_KFold_Non_ML",
]


class ParserArgs:
    """Parser for CLI arguments."""

    @staticmethod
    def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            help="Project root containing data/raw/.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Dataset folder name in data/raw/ (e.g., financial_fraud_detection).",
        )
        parser.add_argument(
            "--nrows",
            type=int,
            default=None,
            help="Read only the first N rows (quick trial run).",
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            default=None,
            help="Comma-separated list of checkpoint numbers to run (e.g., '1,3,5-7'). Runs all if not specified.",
        )

    @staticmethod
    def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
        ParserArgs._add_dataset_arguments(parser)
        parser.add_argument(
            "--scenario",
            type=str,
            default="Repository_Level_Split",
            choices=[
                "ML_to_NonML",
                "NonML_to_ML",
                "Repository_Level_Split",
                "Within_Domain_ML_DL",
                "Within_Domain_Non_ML",
                "Group_KFold_ML_DL",
                "Group_KFold_Non_ML",
            ],
            help="Split scenario to use (default: Repository_Level_Split).",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory to save processed data (default: data/processed/split/defectors).",
        )
        parser.add_argument(
            "--commit-level",
            action="store_true",
            help="Aggregate data to commit-level instead of file-level.",
        )
        parser.add_argument(
            "--no-tokenize",
            action="store_true",
            help="Skip tokenization step.",
        )
        parser.add_argument(
            "--fold",
            type=int,
            default=None,
            choices=range(5),
            help="Fold index (0-4) for the Group_KFold scenarios. Runs all 5 folds if not specified.",
        )

    @staticmethod
    def _add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
        # No --nrows: a manifest only makes sense for the full dataset
        parser.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            help="Project root containing data/raw/ and artifacts/.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Dataset folder name in data/raw/ (e.g., defectors).",
        )
        parser.add_argument(
            "--scenario",
            type=str,
            default=None,
            choices=MANIFEST_SCENARIO_CHOICES,
            help="Split scenario (default: all seven scenarios).",
        )
        parser.add_argument(
            "--fold",
            type=int,
            default=None,
            choices=range(5),
            help="Fold index (0-4), only together with a Group_KFold scenario. All 5 folds if not specified.",
        )
        parser.add_argument(
            "--manifest-dir",
            type=Path,
            default=None,
            help="Directory of the manifests (default: artifacts/split_manifest/<dataset>).",
        )

    @staticmethod
    def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
        # No --nrows/--checkpoints: training reads the tokenized parquet files written by run-pipeline
        parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root containing data/.")
        parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name (e.g., defectors).")
        parser.add_argument("--scenario", type=str, required=True, choices=MANIFEST_SCENARIO_CHOICES)
        parser.add_argument("--model", type=str, required=True, choices=["codebert", "codet5"])
        parser.add_argument(
            "--input-dir",
            type=Path,
            default=None,
            help="Directory with the pipeline output (default: data/processed/split/<dataset>).",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=Path,
            default=None,
            help="Directory for checkpoints (default: artifacts/model/<dataset>).",
        )
        parser.add_argument(
            "--seeds",
            type=str,
            default=None,
            help="Comma-separated seeds. Default: 42,43,44 (non-fold scenarios) or 42,43 per fold (Group K-Fold).",
        )
        parser.add_argument(
            "--fold",
            type=int,
            default=None,
            choices=range(5),
            help="Fold index (0-4) for the Group_KFold scenarios. Trains all 5 folds if not specified.",
        )
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=8, help="Samples per micro-batch.")
        parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
        parser.add_argument("--learning-rate", type=float, default=2e-5)
        parser.add_argument("--weight-decay", type=float, default=0.01)
        parser.add_argument("--warmup-ratio", type=float, default=0.1)
        parser.add_argument("--max-length", type=int, default=512)
        parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
        parser.add_argument("--class-weight", type=str, default="balanced", choices=["none", "balanced"])
        parser.add_argument(
            "--gradient-checkpointing", action="store_true", help="Trade compute for VRAM (needed for 6 GB GPUs)."
        )
        parser.add_argument(
            "--no-length-bucketing", action="store_true", help="Pad to the longest sample of a random batch."
        )
        parser.add_argument("--resume", action="store_true", help="Continue runs from their last checkpoint.")
        parser.add_argument(
            "--max-train-samples", type=int, default=None, help="Smoke test: use only the first N training rows."
        )
        parser.add_argument(
            "--max-val-samples", type=int, default=None, help="Smoke test: use only the first N validation rows."
        )

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build the command-line argument parser."""
        parser = argparse.ArgumentParser(
            prog="lstm-autoencoder",
            description="Research tools for LSTM Autoencoder.",
        )
        subparsers = parser.add_subparsers(
            dest="command",
            required=True,
        )

        inspect_parser = subparsers.add_parser(
            "inspect-data",
            help="Explore and summarize the input dataset.",
        )
        ParserArgs._add_dataset_arguments(inspect_parser)

        pipeline_parser = subparsers.add_parser(
            "run-pipeline",
            help="Run the data processing pipeline.",
        )
        ParserArgs._add_pipeline_arguments(pipeline_parser)

        baseline_parser = subparsers.add_parser(
            "run-baseline",
            help="Run the TF-IDF + Logistic Regression baseline.",
        )
        ParserArgs._add_dataset_arguments(baseline_parser)
        baseline_parser.add_argument(
            "--scenario",
            type=str,
            required=True,
            choices=[
                "ML_to_NonML",
                "NonML_to_ML",
                "Repository_Level_Split",
                "Within_Domain_ML_DL",
                "Within_Domain_Non_ML",
                "Group_KFold_ML_DL",
                "Group_KFold_Non_ML",
                "Cross_Domain_Default",
            ],
            help="Split scenario to evaluate.",
        )
        baseline_parser.add_argument(
            "--input-dir",
            type=Path,
            default=None,
            help="Processed parquet files from the pipeline (default: data/processed/split/defectors).",
        )
        baseline_parser.add_argument(
            "--seeds",
            type=str,
            default="42",
            help="Comma-separated random seeds, e.g. '42,43,44,45,46'.",
        )
        baseline_parser.add_argument(
            "--model",
            type=str,
            default="tfidf",
            choices=["tfidf", "length"],
            help="Baseline model: TF-IDF + Logistic Regression (tfidf) or diff-size features only (length).",
        )
        baseline_parser.add_argument(
            "--equalize-train-size",
            type=int,
            default=None,
            help="Subsample the training set to this many rows.",
        )
        baseline_parser.add_argument(
            "--exclude-repos",
            type=str,
            default=None,
            help="Comma-separated list of repositories to exclude from the analysis (e.g., 'openpilot').",
        )
        baseline_parser.add_argument(
            "--fold",
            type=int,
            default=None,
            choices=range(5),
            help="Fold index (0-4) for the Group_KFold scenarios. Runs all 5 folds if not specified.",
        )

        verify_parser = subparsers.add_parser(
            "verify-tokenization",
            help="Verify the tokenized model inputs of a scenario (truncation share, special tokens, intact path).",
        )
        ParserArgs._add_dataset_arguments(verify_parser)
        verify_parser.add_argument(
            "--scenario",
            type=str,
            required=True,
            choices=[
                "ML_to_NonML",
                "NonML_to_ML",
                "Repository_Level_Split",
                "Within_Domain_ML_DL",
                "Within_Domain_Non_ML",
                "Group_KFold_ML_DL",
                "Group_KFold_Non_ML",
            ],
            help="Split scenario to verify.",
        )
        verify_parser.add_argument(
            "--input-dir",
            type=Path,
            default=None,
            help="Processed parquet files from the pipeline (default: data/processed/split/defectors).",
        )
        verify_parser.add_argument(
            "--fold",
            type=int,
            default=None,
            choices=range(5),
            help="Fold index (0-4) for the Group_KFold scenarios. Verifies all 5 folds if not specified.",
        )

        build_manifest_parser = subparsers.add_parser(
            "build-split-manifest",
            help="Save the row ids of train/val/test per scenario and fold, with their SHA-256 hashes.",
        )
        ParserArgs._add_manifest_arguments(build_manifest_parser)

        verify_manifest_parser = subparsers.add_parser(
            "verify-split-manifest",
            help="Check the stored split manifests against a fresh regeneration from the raw data.",
        )
        ParserArgs._add_manifest_arguments(verify_manifest_parser)

        train_parser = subparsers.add_parser(
            "train",
            help="Fine-tune CodeBERT or CodeT5-small on a split scenario (train/val only, test stays untouched).",
        )
        ParserArgs._add_train_arguments(train_parser)

        benchmark_parser = subparsers.add_parser(
            "benchmark-model",
            help="Measure throughput and peak VRAM of CodeBERT/CodeT5-small at full length and estimate run hours.",
        )
        benchmark_parser.add_argument("--project-root", type=Path, default=Path.cwd())
        benchmark_parser.add_argument("--dataset", type=str, required=True)
        benchmark_parser.add_argument("--model", type=str, required=True, choices=["codebert", "codet5"])
        benchmark_parser.add_argument(
            "--batch-sizes", type=str, default="1,2,4,8,16", help="Ascending micro-batch sizes; stops at the first OOM."
        )
        benchmark_parser.add_argument("--seq-len", type=int, default=512)
        benchmark_parser.add_argument("--steps", type=int, default=5, help="Timed steps per batch size.")
        benchmark_parser.add_argument("--epochs", type=int, default=3, help="Epochs assumed by the hour estimate.")
        benchmark_parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
        benchmark_parser.add_argument("--gradient-checkpointing", action="store_true")

        return parser
