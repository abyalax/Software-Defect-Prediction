"""Data processing pipeline for Defectors dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.split_generator import DefectorsSplitGenerator, SplitScenario
from software_defects.data.defectors.tokenizer import DefectorsTokenizer
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsDataPipeline:
    """End-to-end data processing pipeline for Defectors dataset."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the data pipeline.

        Args:
            config: Dataset configuration.
        """
        self._config = config
        self._loader = DefectorsDatasetLoader(config)
        self._split_generator = DefectorsSplitGenerator(config)
        self._tokenizer = DefectorsTokenizer(
            config,
            model_name=config.tokenizer_model,
            max_length=config.tokenizer_max_length,
        )
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="data_pipeline",
            class_name="DefectorsDataPipeline",
        )

    def run_pipeline(
        self,
        scenario: SplitScenario,
        output_dir: Path | None = None,
        commit_level: bool = False,
        tokenize: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete data processing pipeline.

        Args:
            scenario: Split scenario to use.
            output_dir: Directory to save processed data. If None, data is not saved.
            commit_level: Whether to aggregate to commit-level.
            tokenize: Whether to apply tokenization.

        Returns:
            Tuple of (train_df, test_df).
        """
        self._reporter.display_section("Defectors Data Pipeline")
        self._reporter.display_text()

        # Step 1: Load all data
        self._reporter.display_text("Step 1: Loading data...")
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_data = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)
        self._reporter.display_text(f"  Loaded {len(all_data):,} records")
        self._reporter.display_text()

        # Update summary with data loading information
        self._reporter.update_summary(
            {
                "total_records": len(all_data),
                "train_records": len(jit_train),
                "test_records": len(jit_test),
                "val_records": len(jit_val),
            }
        )

        # Step 2: Generate split
        self._reporter.display_text(f"Step 2: Generating split ({scenario.value})...")
        split_result = self._split_generator.generate_split(all_data, scenario)
        self._reporter.display_text(
            f"  Train: {split_result.train_samples:,} samples ({split_result.train_defective_pct:.2f}% defective)"
        )
        self._reporter.display_text(
            f"  Test: {split_result.test_samples:,} samples ({split_result.test_defective_pct:.2f}% defective)"
        )
        self._reporter.display_text(f"  Train repos: {len(split_result.train_repos)}")
        self._reporter.display_text(f"  Test repos: {len(split_result.test_repos)}")
        self._reporter.display_text()

        # Update summary with split information
        self._reporter.update_summary(
            {
                "scenario": scenario.value,
                "train_samples": split_result.train_samples,
                "test_samples": split_result.test_samples,
                "train_defective_pct": split_result.train_defective_pct,
                "test_defective_pct": split_result.test_defective_pct,
                "train_repos": split_result.train_repos,
                "test_repos": split_result.test_repos,
                "commit_level": commit_level,
                "tokenize": tokenize,
            }
        )

        train_df = split_result.train_df
        test_df = split_result.test_df

        # Step 3: Aggregate to commit-level if requested
        if commit_level:
            self._reporter.display_text("Step 3: Aggregating to commit-level...")
            train_df = self._split_generator.aggregate_to_commit_level(train_df)
            test_df = self._split_generator.aggregate_to_commit_level(test_df)
            self._reporter.display_text(f"  Train commits: {len(train_df):,}")
            self._reporter.display_text(f"  Test commits: {len(test_df):,}")
            self._reporter.display_text()

            # Update summary with commit-level information
            self._reporter.update_summary(
                {
                    "train_commits": len(train_df),
                    "test_commits": len(test_df),
                }
            )

        # Step 4: Tokenize if requested
        if tokenize:
            self._reporter.display_text(f"Step 4: Tokenizing (max_length={self._config.tokenizer_max_length})...")
            train_df = self._tokenizer.tokenize_dataframe(train_df, truncation=True)
            test_df = self._tokenizer.tokenize_dataframe(test_df, truncation=True)

            train_stats = self._tokenizer.get_statistics(train_df)
            test_stats = self._tokenizer.get_statistics(test_df)

            self._reporter.display_text(
                f"  Train - Mean tokens: {train_stats['mean']:.2f}, Truncated: {train_stats.get('truncated_percentage', 0):.2f}%"
            )
            self._reporter.display_text(
                f"  Test - Mean tokens: {test_stats['mean']:.2f}, Truncated: {test_stats.get('truncated_percentage', 0):.2f}%"
            )
            self._reporter.display_text()

            # Update summary with tokenization statistics
            self._reporter.update_summary(
                {
                    "tokenizer_model": self._config.tokenizer_model,
                    "tokenizer_max_length": self._config.tokenizer_max_length,
                    "train_mean_tokens": train_stats["mean"],
                    "train_median_tokens": train_stats["median"],
                    "train_truncated_pct": train_stats.get("truncated_percentage", 0),
                    "test_mean_tokens": test_stats["mean"],
                    "test_median_tokens": test_stats["median"],
                    "test_truncated_pct": test_stats.get("truncated_percentage", 0),
                }
            )

        # Step 5: Save if output directory provided
        if output_dir is not None:
            self._reporter.display_text(f"Step 5: Saving processed data...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            train_path = output_dir / f"train_{scenario.value}.parquet"
            test_path = output_dir / f"test_{scenario.value}.parquet"

            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)

            self._reporter.display_text(f"  Saved train to: {train_path}")
            self._reporter.display_text(f"  Saved test to: {test_path}")
            self._reporter.display_text()

            # Update summary with output paths
            self._reporter.update_summary(
                {
                    "output_dir": str(output_dir),
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                }
            )

        # Save reports
        self._reporter.save_reports(
            text_filename=f"pipeline_{scenario.value}.txt",
            json_filename=f"pipeline_{scenario.value}.json",
        )

        return train_df, test_df
