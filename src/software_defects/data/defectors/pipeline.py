"""Data processing pipeline for Defectors dataset."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.data.defectors.tokenizer import DefectorsTokenizer
from software_defects.reporting.reporter import ReportWritter
from software_defects.reporting.utils import get_device

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
        self._tokenizers = {
            model_key: DefectorsTokenizer(
                config,
                model_name=model_name,
                max_length=config.tokenizer_max_length,
            )
            for model_key, model_name in config.tokenizer_models.items()
        }
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
        fold: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]:
        """Run the complete data processing pipeline.

        Args:
            scenario: Split scenario to use.
            output_dir: Directory to save processed data. If None, data is not saved.
            commit_level: Whether to aggregate to commit-level.
            tokenize: Whether to apply tokenization.
            fold: Fold index (0 to 4), only used by the Group K-Fold scenarios.

        Returns:
            If tokenize=True: Dict mapping model keys to {train, val, test} DataFrames.
            If tokenize=False: Tuple of (train_df, test_df).
        """
        self._reporter.display_section("Defectors Data Pipeline")
        self._reporter.display_text()

        # Detect and report device
        device = get_device()
        self._reporter.display_text(f"Device: {device}")
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
                "device": device,
                "total_records": len(all_data),
                "train_records": len(jit_train),
                "test_records": len(jit_test),
                "val_records": len(jit_val),
            }
        )

        # Step 2: Generate split
        fold_label = f", fold {fold}" if scenario in GROUP_KFOLD_SCENARIOS else ""
        self._reporter.display_text(f"Step 2: Generating split ({scenario.value}{fold_label})...")
        split_result = self._split_generator.generate_split(all_data, scenario, fold=fold)
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
                "fold": fold if scenario in GROUP_KFOLD_SCENARIOS else None,
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
        val_df = split_result.val_df
        test_df = split_result.test_df

        # Step 3: Aggregate to commit-level if requested
        if commit_level:
            self._reporter.display_text("Step 3: Aggregating to commit-level...")
            train_df = self._split_generator.aggregate_to_commit_level(train_df)
            val_df = self._split_generator.aggregate_to_commit_level(val_df)
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
        tokenized_data = {}
        tokenizer_stats = {}
        if tokenize:
            self._reporter.display_text(
                "Step 4: Building 'path + diff' inputs and tokenizing "
                f"(max_length={self._config.tokenizer_max_length})..."
            )
            for model_key, tokenizer in self._tokenizers.items():
                self._reporter.display_text(f"  Tokenizing with {model_key} ({tokenizer.model_name})...")
                train_df_tok = tokenizer.tokenize_dataframe(train_df.copy(), truncation=True)
                val_df_tok = tokenizer.tokenize_dataframe(val_df.copy(), truncation=True)
                test_df_tok = tokenizer.tokenize_dataframe(test_df.copy(), truncation=True)

                split_stats = {
                    "train": tokenizer.get_statistics(train_df_tok),
                    "val": tokenizer.get_statistics(val_df_tok),
                    "test": tokenizer.get_statistics(test_df_tok),
                }
                for split_name, stats in split_stats.items():
                    self._reporter.display_text(
                        f"    {split_name.capitalize():5} - mean tokens: {stats['mean']:.2f}, "
                        f"truncated: {stats['truncated_percentage']:.2f}%, "
                        f"pre-cut: {stats['pre_cut_percentage']:.2f}%, "
                        f"changed lines kept: {stats['changed_lines_kept_percentage']:.2f}%, "
                        f"fallback: {stats['fallback_truncation_count']}"
                    )
                train_stats = split_stats["train"]
                test_stats = split_stats["test"]

                tokenized_data[model_key] = {
                    "train": train_df_tok,
                    "val": val_df_tok,
                    "test": test_df_tok,
                }
                tokenizer_stats[model_key] = {
                    "tokenizer_model": tokenizer.model_name,
                    "diff_input": asdict(self._config.diff_input),
                    "tokenizer_max_length": self._config.tokenizer_max_length,
                    "train_mean_tokens": train_stats["mean"],
                    "train_median_tokens": train_stats["median"],
                    "train_truncated_pct": train_stats.get("truncated_percentage", 0),
                    "test_mean_tokens": test_stats["mean"],
                    "test_median_tokens": test_stats["median"],
                    "test_truncated_pct": test_stats.get("truncated_percentage", 0),
                    "splits": split_stats,
                }
            self._reporter.display_text()

            # Update summary with tokenization statistics for all models
            self._reporter.update_summary({"tokenizer_stats": tokenizer_stats})

        # Step 5: Save if output directory provided
        if output_dir is not None:
            self._reporter.display_text("Step 5: Saving processed data...")
            output_dir = Path(output_dir)

            # Create scenario-specific subfolder (one per fold for Group K-Fold)
            scenario_dir = self._split_generator.scenario_dir(output_dir, scenario, fold)
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Save non-tokenized data (raw split)
            train_path = scenario_dir / f"{scenario.value}_train.parquet"
            val_path = scenario_dir / f"{scenario.value}_val.parquet"
            test_path = scenario_dir / f"{scenario.value}_test.parquet"

            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            self._reporter.display_text(f"  Saved raw train to: {train_path}")
            self._reporter.display_text(f"  Saved raw val to: {val_path}")
            self._reporter.display_text(f"  Saved raw test to: {test_path}")

            # Save tokenized data for each model
            if tokenize and tokenized_data:
                for model_key, data in tokenized_data.items():
                    model_dir = scenario_dir / model_key
                    model_dir.mkdir(parents=True, exist_ok=True)

                    train_tok_path = model_dir / f"{scenario.value}_train.parquet"
                    val_tok_path = model_dir / f"{scenario.value}_val.parquet"
                    test_tok_path = model_dir / f"{scenario.value}_test.parquet"

                    data["train"].to_parquet(train_tok_path, index=False)
                    data["val"].to_parquet(val_tok_path, index=False)
                    data["test"].to_parquet(test_tok_path, index=False)

                    self._reporter.display_text(f"  Saved {model_key} train to: {train_tok_path}")
                    self._reporter.display_text(f"  Saved {model_key} val to: {val_tok_path}")
                    self._reporter.display_text(f"  Saved {model_key} test to: {test_tok_path}")

            self._reporter.display_text()

            # Update summary with output paths
            output_summary = {
                "output_dir": str(output_dir),
                "scenario_dir": str(scenario_dir),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
            }
            if tokenize and tokenized_data:
                output_summary["model_paths"] = {
                    model_key: {
                        "train": str(scenario_dir / model_key / f"{scenario.value}_train.parquet"),
                        "val": str(scenario_dir / model_key / f"{scenario.value}_val.parquet"),
                        "test": str(scenario_dir / model_key / f"{scenario.value}_test.parquet"),
                    }
                    for model_key in tokenized_data.keys()
                }
            self._reporter.update_summary(output_summary)

        # Save reports
        report_name = f"pipeline_{scenario.value}" + (f"_fold{fold}" if scenario in GROUP_KFOLD_SCENARIOS else "")
        self._reporter.save_reports(
            text_filename=f"{report_name}.txt",
            json_filename=f"{report_name}.json",
        )

        # Return tokenized data if available, otherwise return raw data
        if tokenize and tokenized_data:
            return tokenized_data
        return train_df, test_df
