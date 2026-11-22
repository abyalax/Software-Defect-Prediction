"""Runner CLI for the research project"""

from __future__ import annotations

import argparse
from pathlib import Path


class RunnerCLI:
    """Runner for CLI commands."""

    @staticmethod
    def run_inspect_data(project_root: Path, dataset: str, nrows: int | None, checkpoints: str | None) -> int:
        """Execute dataset exploratory with specified checkpoints."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.dataset_explorer import DefectorsDatasetExplorer

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
            row_limit=nrows,
        )

        # Parse checkpoints argument
        checkpoint_list = None
        if checkpoints is not None:
            checkpoint_list = []
            for part in checkpoints.split(","):
                part = part.strip()
                if "-" in part:
                    # Handle range like "5-7"
                    start, end = map(int, part.split("-"))
                    checkpoint_list.extend(range(start, end + 1))
                else:
                    checkpoint_list.append(int(part))

        # Use the dataset explorer with specified checkpoints
        explorer = DefectorsDatasetExplorer(config=dataset_config, checkpoints=checkpoint_list)
        explorer.run()
        return 0

    @staticmethod
    def run_pipeline(
        project_root: Path,
        dataset: str,
        scenario: str,
        output_dir: Path | None,
        commit_level: bool,
        tokenize: bool,
        fold: int | None = None,
    ) -> int:
        """Execute the data processing pipeline."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.pipeline import DefectorsDataPipeline
        from software_defects.data.defectors.split_generator import (
            GROUP_KFOLD_N_FOLDS,
            GROUP_KFOLD_SCENARIOS,
            SplitScenario,
        )

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
        )

        # Use default output_dir if not provided
        if output_dir is None:
            output_dir = dataset_config.processed_dir

        # Convert scenario string to enum (handle underscore conversion)
        scenario_map = {
            "ML_to_NonML": SplitScenario.ML_TO_NON_ML,
            "NonML_to_ML": SplitScenario.NON_ML_TO_ML,
            "Repository_Level_Split": SplitScenario.REPO_LEVEL_SPLIT,
            "Within_Domain_ML_DL": SplitScenario.WITHIN_DOMAIN_ML_DL,
            "Within_Domain_Non_ML": SplitScenario.WITHIN_DOMAIN_NON_ML,
            "Group_KFold_ML_DL": SplitScenario.GROUP_KFOLD_ML_DL,
            "Group_KFold_Non_ML": SplitScenario.GROUP_KFOLD_NON_ML,
        }
        scenario_enum = scenario_map.get(scenario)
        if scenario_enum is None:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Group K-Fold writes one processed directory per fold; other scenarios have a single split
        if scenario_enum in GROUP_KFOLD_SCENARIOS:
            folds = [fold] if fold is not None else list(range(GROUP_KFOLD_N_FOLDS))
        elif fold is not None:
            raise ValueError("--fold is only valid for the Group_KFold scenarios")
        else:
            folds = [0]

        for fold_index in folds:
            # A fresh pipeline per fold keeps each fold's report separate
            pipeline = DefectorsDataPipeline(config=dataset_config)
            pipeline.run_pipeline(
                scenario=scenario_enum,
                output_dir=output_dir,
                commit_level=commit_level,
                tokenize=tokenize,
                fold=fold_index,
            )
        return 0

    @staticmethod
    def run_baseline(
        project_root: Path,
        dataset: str,
        scenario: str,
        input_dir: Path | None,
        seeds: list[int],
        equalize_train_size: int | None,
        model: str = "tfidf",
        exclude_repos: str | None = None,
        fold: int | None = None,
    ) -> int:
        """Execute a baseline (TF-IDF + Logistic Regression, or diff-size features only)."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.split_generator import SplitScenario
        from software_defects.models.baseline_experiment import BaselineExperiment

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
        )

        # Use default input_dir if not provided
        if input_dir is None:
            input_dir = dataset_config.processed_dir

        # Parse exclude_repos if provided
        exclude_repos_list = None
        if exclude_repos is not None:
            exclude_repos_list = [repo.strip() for repo in exclude_repos.split(",")]

        experiment = BaselineExperiment(config=dataset_config, input_dir=input_dir, model_name=model)
        experiment.run(
            scenario=SplitScenario(scenario),
            seeds=seeds,
            equalize_train_size=equalize_train_size,
            exclude_repos=exclude_repos_list,
            fold=fold,
        )
        return 0

    @staticmethod
    def run_verify_tokenization(
        project_root: Path,
        dataset: str,
        scenario: str,
        input_dir: Path | None,
        fold: int | None = None,
    ) -> int:
        """Verify the tokenized model inputs of one scenario; returns 1 when a check fails."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.split_generator import SplitScenario
        from software_defects.data.defectors.tokenization_verifier import CHECK_COLUMNS, TokenizationVerifier

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
        )

        if input_dir is None:
            input_dir = dataset_config.processed_dir

        report = TokenizationVerifier(config=dataset_config, input_dir=input_dir).run(
            scenario=SplitScenario(scenario),
            fold=fold,
        )
        return 1 if int(report[CHECK_COLUMNS].sum().sum()) else 0

    @staticmethod
    def run_build_split_manifest(
        project_root: Path,
        dataset: str,
        scenario: str | None,
        fold: int | None,
        manifest_dir: Path | None,
    ) -> int:
        """Save the row-level split manifest (ids and hashes) of one or all scenarios."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.split_manifest import DefectorsSplitManifest

        dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset)
        manifest = DefectorsSplitManifest(config=dataset_config, manifest_dir=manifest_dir)
        manifest.build(scenarios=RunnerCLI._manifest_scenarios(scenario), fold=fold)
        return 0

    @staticmethod
    def run_verify_split_manifest(
        project_root: Path,
        dataset: str,
        scenario: str | None,
        fold: int | None,
        manifest_dir: Path | None,
    ) -> int:
        """Verify the stored split manifests against a fresh split; returns 1 when a check fails."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.split_manifest import DefectorsSplitManifest

        dataset_config = DatasetConfig(project_root=project_root, dataset_name=dataset)
        manifest = DefectorsSplitManifest(config=dataset_config, manifest_dir=manifest_dir)
        ok = manifest.verify(scenarios=RunnerCLI._manifest_scenarios(scenario), fold=fold)
        return 0 if ok else 1

    @staticmethod
    def run_train(project_root: Path, arguments: argparse.Namespace) -> int:
        """Fine-tune a Transformer classifier on the train/val split of one scenario."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.split_generator import SplitScenario
        from software_defects.models.classifier_config import ClassifierConfig, TrainingConfig
        from software_defects.models.training_experiment import TRAINABLE_MODELS, TrainingExperiment

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=arguments.dataset,
            tokenizer_max_length=arguments.max_length,
        )
        training_config = TrainingConfig(
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.grad_accum,
            warmup_ratio=arguments.warmup_ratio,
            max_length=arguments.max_length,
            length_bucketing=not arguments.no_length_bucketing,
            class_weight=arguments.class_weight,
            precision=arguments.precision,
            max_train_samples=arguments.max_train_samples,
            max_val_samples=arguments.max_val_samples,
            checkpoint_every_updates=arguments.checkpoint_every,
        )
        classifier_config = ClassifierConfig(
            pretrained_name=TRAINABLE_MODELS[arguments.model][1],
            gradient_checkpointing=arguments.gradient_checkpointing,
            cache_dir=str(dataset_config.model_cache_dir),
        )
        seeds = [int(seed) for seed in arguments.seeds.split(",")] if arguments.seeds else None

        experiment = TrainingExperiment(
            config=dataset_config,
            model_key=arguments.model,
            input_dir=arguments.input_dir or dataset_config.processed_dir,
            training_config=training_config,
            classifier_config=classifier_config,
            checkpoint_dir=arguments.checkpoint_dir,
        )
        experiment.run(
            scenario=SplitScenario(arguments.scenario),
            seeds=seeds,
            fold=arguments.fold,
            resume=arguments.resume,
        )
        return 0

    @staticmethod
    def run_benchmark_model(project_root: Path, arguments: argparse.Namespace) -> int:
        """Benchmark one Transformer classifier at full length and print the run-time estimate."""
        from software_defects.config import DatasetConfig
        from software_defects.models.classifier_config import ClassifierConfig, TrainingConfig
        from software_defects.models.model_benchmark import ModelBenchmark
        from software_defects.models.training_experiment import TRAINABLE_MODELS

        dataset_config = DatasetConfig(project_root=project_root, dataset_name=arguments.dataset)
        benchmark = ModelBenchmark(
            dataset_config,
            arguments.model,
            training_config=TrainingConfig(precision=arguments.precision, max_length=arguments.seq_len),
            classifier_config=ClassifierConfig(
                pretrained_name=TRAINABLE_MODELS[arguments.model][1],
                gradient_checkpointing=arguments.gradient_checkpointing,
                cache_dir=str(dataset_config.model_cache_dir),
            ),
        )
        batch_sizes = [int(size) for size in arguments.batch_sizes.split(",")]
        benchmark.run(batch_sizes, arguments.seq_len, arguments.steps, arguments.epochs)
        return 0

    @staticmethod
    def _manifest_scenarios(scenario: str | None) -> list | None:
        """Convert the optional --scenario value into the list the manifest class expects."""
        from software_defects.data.defectors.split_generator import SplitScenario

        return [SplitScenario(scenario)] if scenario is not None else None
