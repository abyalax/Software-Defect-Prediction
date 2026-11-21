"""Checkpoint 17: Prevalence shift per repository, per scenario and per fold."""

from __future__ import annotations

import math

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets, load_holdout_repos
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.reporting.reporter import ReportWritter

WILSON_Z = 1.96
# A test prevalence this many percentage points away from train is flagged as a prevalence shift
SHIFT_WARNING_PP = 10.0
SCENARIOS = [
    SplitScenario.ML_TO_NON_ML,
    SplitScenario.NON_ML_TO_ML,
    SplitScenario.REPO_LEVEL_SPLIT,
    SplitScenario.WITHIN_DOMAIN_ML_DL,
    SplitScenario.WITHIN_DOMAIN_NON_ML,
    SplitScenario.GROUP_KFOLD_ML_DL,
    SplitScenario.GROUP_KFOLD_NON_ML,
]


class PrevalenceShiftAnalysis:
    """Checkpoint 18: Prevalence shift per repository, per scenario and per fold.

    PR-AUC and F1 depend on prevalence, so scores from test sets with different prevalence are not
    directly comparable. This checkpoint reports the prevalence of every repository (with a Wilson
    interval) and, using the real split generator, the train/val/test prevalence of every scenario and
    every Group K-Fold fold together with the test-minus-train shift.
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def wilson_interval(positives: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
        """Wilson score interval of a proportion.

        Args:
            positives: Number of positive rows.
            total: Number of rows.
            z: Normal quantile of the interval.

        Returns:
            Tuple of (lower, upper) bounds, both NaN when ``total`` is zero.
        """
        if total == 0:
            return float("nan"), float("nan")
        p = positives / total
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        return centre - margin, centre + margin

    def _repo_table(self, df: pd.DataFrame, domains: dict[str, set[str]]) -> pd.DataFrame:
        """Prevalence and Wilson interval per repository."""
        holdout = load_holdout_repos(self._loader.config.project_root)
        holdout_repos = set(holdout["ml_dl"]) | set(holdout["non_ml"])
        rows = []
        for repo, group in df.groupby("repo"):
            positives = int(group["label"].sum())
            lower, upper = self.wilson_interval(positives, len(group))
            rows.append(
                {
                    "repo": repo,
                    "domain": "ml_dl" if repo in domains["ml_dl"] else "non_ml",
                    "in_holdout": repo in holdout_repos,
                    "n_rows": len(group),
                    "n_defective": positives,
                    "prevalence": positives / len(group),
                    "ci_low": lower,
                    "ci_high": upper,
                }
            )
        return pd.DataFrame(rows).sort_values(["domain", "prevalence"], ascending=[True, False]).reset_index(drop=True)

    def _scenario_table(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Train/val/test prevalence of every scenario and fold."""
        generator = DefectorsSplitGenerator(self._loader.config)
        rows = []
        for scenario in SCENARIOS:
            folds = range(GROUP_KFOLD_N_FOLDS) if scenario in GROUP_KFOLD_SCENARIOS else [0]
            for fold in folds:
                result = generator.generate_split(raw, scenario, fold=fold)
                train = result.train_defective_pct
                test = result.test_defective_pct
                rows.append(
                    {
                        "scenario": scenario.value,
                        "fold": fold if scenario in GROUP_KFOLD_SCENARIOS else None,
                        "train_prevalence_pct": train,
                        "val_prevalence_pct": result.val_defective_pct,
                        "test_prevalence_pct": test,
                        "shift_pp": test - train,
                        "test_to_train_ratio": test / train if train > 0 else float("nan"),
                        "train_samples": result.train_samples,
                        "test_samples": result.test_samples,
                        "test_repos": ", ".join(result.test_repos),
                    }
                )
        return pd.DataFrame(rows)

    def run(self) -> None:
        """Execute the prevalence shift analysis."""
        self._reporter.display_section("Checkpoint 17: Prevalence Shift per Repository, Scenario and Fold")
        self._reporter.display_text()

        raw = self._loader.load_all_jit()
        domains = load_domain_sets(self._loader.config.project_root)
        filtered = DefectorsSplitGenerator(self._loader.config).filter_data(raw)
        filtered = filtered[filtered["repo"].isin(domains["ml_dl"] | domains["non_ml"])]

        repo_table = self._repo_table(filtered, domains)
        self._reporter.display_text("Prevalence per repository (pipeline filtering applied):")
        self._reporter.display_table(repo_table.round(4))
        repo_path = self._reporter.save_csv(repo_table, "prevalence_per_repo.csv")
        self._reporter.display_text()

        for domain, group in repo_table.groupby("domain"):
            spread = group["prevalence"].max() - group["prevalence"].min()
            self._reporter.display_text(
                f"  {domain}: prevalence ranges {group['prevalence'].min():.3f} to {group['prevalence'].max():.3f} "
                f"across {len(group)} repositories (spread {spread:.3f})"
            )
        self._reporter.display_text()

        scenario_table = self._scenario_table(raw)
        self._reporter.display_text("Prevalence per scenario and fold (percent):")
        self._reporter.display_table(scenario_table.drop(columns=["test_repos"]).round(3))
        scenario_path = self._reporter.save_csv(scenario_table, "prevalence_shift_scenarios.csv")
        flagged = scenario_table[scenario_table["shift_pp"].abs() >= SHIFT_WARNING_PP]
        for _, row in flagged.iterrows():
            fold = "" if pd.isna(row["fold"]) else f" fold {int(row['fold'])}"
            self._reporter.display_warning(
                f"  Prevalence shift in {row['scenario']}{fold}: train {row['train_prevalence_pct']:.1f}% -> "
                f"test {row['test_prevalence_pct']:.1f}% ({row['shift_pp']:+.1f} pp); compare PR-AUC against prevalence"
            )
        self._reporter.display_text()

        self._reporter.update_summary(
            {
                "checkpoint_17": {
                    "per_repo_csv": str(repo_path),
                    "scenario_csv": str(scenario_path),
                    "flagged_runs": int(len(flagged)),
                    "max_abs_shift_pp": float(scenario_table["shift_pp"].abs().max()),
                }
            }
        )
