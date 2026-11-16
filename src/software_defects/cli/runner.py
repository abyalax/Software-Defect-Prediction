"""Runner CLI for the research project"""

from __future__ import annotations

from pathlib import Path


class RunnerCLI:
    """Runner for CLI commands."""

    @staticmethod
    def run_inspect_data(project_root: Path, dataset: str, nrows: int | None) -> int:
        """Execute dataset inspection using modular exploratories."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors import DefectorsDatasetExplorer

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
            row_limit=nrows,
        )

        # Use the modular exploratory
        explorer = DefectorsDatasetExplorer(config=dataset_config)
        explorer.run()
        return 0

    @staticmethod
    def run_advanced_analysis(project_root: Path, dataset: str, nrows: int | None) -> int:
        """Execute advanced analysis on the dataset."""
        from software_defects.config import DatasetConfig
        from software_defects.data.defectors.advanced_analysis import DefectorsAdvancedAnalysis

        dataset_config = DatasetConfig(
            project_root=project_root,
            dataset_name=dataset,
            row_limit=nrows,
        )

        # Use the advanced analysis module
        analyzer = DefectorsAdvancedAnalysis(config=dataset_config)
        analyzer.run()
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
