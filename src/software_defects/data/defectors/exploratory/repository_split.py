"""Checkpoint 11: Repository-level split simulation (ML ↔ non-ML)."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.split_generator import DefectorsSplitGenerator, SplitResult, SplitScenario
from software_defects.reporting.reporter import ReportWritter

# Scenarios simulated here, with the domain of the train and test side
SIMULATED_SCENARIOS: list[tuple[SplitScenario, str, str]] = [
    (SplitScenario.ML_TO_NON_ML, "ml_dl", "non_ml"),
    (SplitScenario.NON_ML_TO_ML, "non_ml", "ml_dl"),
    (SplitScenario.REPO_LEVEL_SPLIT, "mixed", "mixed"),
    (SplitScenario.WITHIN_DOMAIN_ML_DL, "ml_dl", "ml_dl"),
    (SplitScenario.WITHIN_DOMAIN_NON_ML, "non_ml", "non_ml"),
]


class RepositorySplit:
    """Checkpoint 11: Repository-level split simulation (ML ↔ non-ML).

    The numbers come from the same ``DefectorsSplitGenerator`` the pipeline uses, so the report always
    matches the splits that are actually trained on (holdout from the mapping YAML, freqtrade and
    empty-content rows removed, validation carved out of the training side).
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def _to_row(result: SplitResult, train_domain: str, test_domain: str) -> dict:
        """Convert a split result into one report row."""
        return {
            "scenario": result.scenario,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "train_repositories": len(result.train_repos),
            "test_repositories": len(result.test_repos),
            "train_samples": result.train_samples,
            "val_samples": result.val_samples,
            "test_samples": result.test_samples,
            "train_defective_pct": result.train_defective_pct,
            "val_defective_pct": result.val_defective_pct,
            "test_defective_pct": result.test_defective_pct,
            "train_repos_list": ", ".join(result.train_repos),
            "test_repos_list": ", ".join(result.test_repos),
        }

    def run(self) -> None:
        """Execute repository-level split analysis."""
        self._reporter.display_section("Checkpoint 11: Repository-Level Split Analysis")
        self._reporter.display_text()

        # Raw data of all three original files; the generator applies the pipeline filtering itself
        all_jit = self._loader.load_all_jit()
        generator = DefectorsSplitGenerator(self._loader.config)

        rows = []
        for scenario, train_domain, test_domain in SIMULATED_SCENARIOS:
            result = generator.generate_split(all_jit, scenario)
            rows.append(self._to_row(result, train_domain, test_domain))
            self._reporter.display_text(
                f"  {scenario.value}: train {result.train_samples:,} / val {result.val_samples:,} / "
                f"test {result.test_samples:,} (test repos: {', '.join(result.test_repos)})"
            )
        self._reporter.display_text()

        split_df = pd.DataFrame(rows)
        csv_path = self._reporter.save_csv(split_df, "split_report.csv")

        self._reporter.display_text(f"Split report saved to: {csv_path}")
        self._reporter.display_text()

        by_scenario = {row["scenario"]: row for row in rows}
        ml_to_non_ml = by_scenario[SplitScenario.ML_TO_NON_ML.value]
        non_ml_to_ml = by_scenario[SplitScenario.NON_ML_TO_ML.value]
        repo_level = by_scenario[SplitScenario.REPO_LEVEL_SPLIT.value]

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_11": {
                    "ml_to_non_ml_train_samples": ml_to_non_ml["train_samples"],
                    "ml_to_non_ml_test_samples": ml_to_non_ml["test_samples"],
                    "non_ml_to_ml_train_samples": non_ml_to_ml["train_samples"],
                    "non_ml_to_ml_test_samples": non_ml_to_ml["test_samples"],
                    "repo_level_split_train_samples": repo_level["train_samples"],
                    "repo_level_split_test_samples": repo_level["test_samples"],
                    "csv_path": str(csv_path),
                }
            }
        )
