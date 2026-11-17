"""Dataset explorer for Defectors dataset exploratory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.exploratory import (
    CrossDomainFeasibility,
    DatasetIntegrity,
    JitInputValidity,
    LabelDistribution,
    LabelValidity,
    LeakageUnitAnalysis,
    RecordReconciliation,
    RepositoryDomainValidity,
    RepositoryMapping,
    RepositorySplit,
    RunReport,
    ShortcutSignalCheck,
    SplitStrategyEvaluation,
    TokenDistribution,
)
from software_defects.reporting.reporter import ReportWritter

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class DefectorsDatasetExplorer:
    """Explore and analyze Defectors dataset following TODO.md checkpoints."""

    def __init__(self, config: DatasetConfig, checkpoints: list[int] | None = None) -> None:
        """Initialize the dataset explorer.

        Args:
            config: Dataset configuration.
            checkpoints: List of checkpoint numbers to run (1-14). If None, runs all checkpoints.
        """
        self._config = config
        self._loader = DefectorsDatasetLoader(config)
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="dataset_exploratory",
            class_name="DefectorsDatasetExplorer",
        )
        self._checkpoints = checkpoints

        # Initialize checkpoint classes
        self._checkpoint_classes = {
            1: DatasetIntegrity,
            2: LabelValidity,
            3: JitInputValidity,
            4: RepositoryDomainValidity,
            5: LeakageUnitAnalysis,
            6: CrossDomainFeasibility,
            7: RecordReconciliation,
            8: SplitStrategyEvaluation,
            9: LabelDistribution,
            10: RepositoryMapping,
            11: RepositorySplit,
            12: TokenDistribution,
            13: ShortcutSignalCheck,
            14: RunReport,
        }

    def run(self) -> None:
        """Run dataset exploratory with specified checkpoints or all checkpoints."""
        self._reporter.display_section("Defectors Dataset Exploratory")
        self._reporter.display_text()

        # Determine which checkpoints to run
        if self._checkpoints is None:
            checkpoint_numbers = list(range(1, 15))
        else:
            checkpoint_numbers = self._checkpoints

        self._reporter.display_text(f"Running checkpoints: {checkpoint_numbers}")
        self._reporter.display_text()

        # Run specified checkpoints
        for checkpoint_num in sorted(checkpoint_numbers):
            if checkpoint_num in self._checkpoint_classes:
                checkpoint_class = self._checkpoint_classes[checkpoint_num]
                checkpoint = checkpoint_class(self._loader, self._reporter)
                checkpoint.run()
            else:
                self._reporter.display_warning(f"Checkpoint {checkpoint_num} not found, skipping...")

        # Save reports
        self._reporter.save_reports(
            text_filename="dataset_exploratory.txt",
            json_filename="dataset_exploratory.json",
        )
