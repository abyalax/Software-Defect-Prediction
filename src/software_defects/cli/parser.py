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

    @staticmethod
    def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
        ParserArgs._add_dataset_arguments(parser)
        parser.add_argument(
            "--scenario",
            type=str,
            required=True,
            choices=[
                "ML_to_NonML",
                "NonML_to_ML",
                "Repository_Level_Split",
                "Within_Domain_ML_DL",
                "Within_Domain_Non_ML",
            ],
            help="Split scenario to use.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory to save processed data.",
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

        advanced_parser = subparsers.add_parser(
            "advanced-analysis",
            help="Perform advanced analysis on the dataset.",
        )
        ParserArgs._add_dataset_arguments(advanced_parser)

        pipeline_parser = subparsers.add_parser(
            "run-pipeline",
            help="Run the data processing pipeline.",
        )
        ParserArgs._add_pipeline_arguments(pipeline_parser)

        return parser
