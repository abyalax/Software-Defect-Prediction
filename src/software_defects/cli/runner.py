"""Runner CLI for the research project"""

from __future__ import annotations

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
    ) -> int:
        """Execute the data processing pipeline."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.pipeline import DefectorsDataPipeline
        from software_defects.data.defectors.split_generator import SplitScenario

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
        )

        # Convert scenario string to enum (handle underscore conversion)
        scenario_map = {
            "ML_to_NonML": SplitScenario.ML_TO_NON_ML,
            "NonML_to_ML": SplitScenario.NON_ML_TO_ML,
            "Repository_Level_Split": SplitScenario.REPO_LEVEL_SPLIT,
            "Within_Domain_ML_DL": SplitScenario.WITHIN_DOMAIN_ML_DL,
            "Within_Domain_Non_ML": SplitScenario.WITHIN_DOMAIN_NON_ML,
        }
        scenario_enum = scenario_map.get(scenario)
        if scenario_enum is None:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Run the pipeline
        pipeline = DefectorsDataPipeline(config=dataset_config)
        pipeline.run_pipeline(
            scenario=scenario_enum,
            output_dir=output_dir,
            commit_level=commit_level,
            tokenize=tokenize,
        )
        return 0
