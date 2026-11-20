"""Parser for the LSTM Autoencoder research project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


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
            help="Directory to save processed data (default: data/processed/defectors).",
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
            help="Directory containing processed parquet files from pipeline (default: data/processed/defectors).",
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

        return parser
