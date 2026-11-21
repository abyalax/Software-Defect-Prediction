"""Run a baseline (TF-IDF + Logistic Regression, or diff-size only) on a Defectors split scenario."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.evaluation import DefectEvaluator
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    GROUP_KFOLD_SCENARIOS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.models.length_logreg import LengthLogRegBaseline
from software_defects.models.tfidf_logreg import BaselineConfig, TfidfLogRegBaseline
from software_defects.reporting.reporter import ReportWritter
from software_defects.utils import get_device

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig

BASELINE_MODELS = {"tfidf": TfidfLogRegBaseline, "length": LengthLogRegBaseline}
REPORT_CLASS_NAMES = {"tfidf": "TfidfLogRegBaseline", "length": "LengthLogRegBaseline"}
METRIC_COLUMNS = ["pr_auc", "f1", "mcc", "roc_auc", "train_seconds", "inference_seconds"]


class BaselineExperiment:
    """Train and evaluate a baseline for one split scenario over several seeds."""

    def __init__(
        self,
        config: DatasetConfig,
        input_dir: Path,
        baseline_config: BaselineConfig | None = None,
        model_name: str = "tfidf",
    ) -> None:
        """Initialize the experiment.

        Args:
            config: Dataset configuration.
            input_dir: Directory containing processed parquet files from pipeline.
            baseline_config: Baseline hyperparameters.
            model_name: ``tfidf`` (TF-IDF + Logistic Regression) or ``length`` (diff-size features only).

        Raises:
            ValueError: If ``model_name`` is not a known baseline.
        """
        if model_name not in BASELINE_MODELS:
            raise ValueError(f"Unknown baseline model '{model_name}'; expected one of {sorted(BASELINE_MODELS)}")
        self._config = config
        self._input_dir = Path(input_dir)
        self._baseline_config = baseline_config or BaselineConfig()
        self._model_name = model_name
        self._evaluator = DefectEvaluator()
        self._reporter = ReportWritter(
            config=config,
            domain_data="defectors",
            pipeline="baseline",
            class_name=REPORT_CLASS_NAMES[model_name],
        )

    def run(
        self,
        scenario: SplitScenario,
        seeds: list[int],
        equalize_train_size: int | None = None,
        exclude_repos: list[str] | None = None,
        fold: int | None = None,
    ) -> pd.DataFrame:
        """Run the baseline and save per-run and per-repository results.

        Args:
            scenario: Split scenario to evaluate.
            seeds: Random seeds; each seed gives one run (per fold for Group K-Fold).
            equalize_train_size: If set, subsample the training set to this size.
            exclude_repos: List of repository names to exclude from the analysis.
            fold: Specific fold to run (for Group K-Fold scenarios). If None, runs all folds.

        Returns:
            DataFrame with one row per (fold, seed) run.
        """
        self._reporter.display_section(f"{REPORT_CLASS_NAMES[self._model_name]} ({scenario.value})")

        # Detect and report device
        device = get_device()
        self._reporter.display_text(f"Device: {device}")
        self._reporter.display_text()

        if exclude_repos:
            self._reporter.display_text(f"Excluding repositories: {', '.join(exclude_repos)}")

        # Handle fold selection
        if scenario in GROUP_KFOLD_SCENARIOS:
            folds = [fold] if fold is not None else list(range(GROUP_KFOLD_N_FOLDS))
        else:
            if fold is not None:
                raise ValueError("--fold is only valid for the Group_KFold scenarios")
            folds = [0]

        run_rows: list[dict] = []
        repo_rows: list[dict] = []
        for fold in folds:
            # Each Group K-Fold fold has its own processed split (the pipeline writes one per fold)
            train_df, val_df, test_df = self._load_processed_data(scenario, fold)

            # Filter out excluded repositories
            if exclude_repos:
                train_df = train_df[~train_df["repo"].isin(exclude_repos)].copy()
                val_df = val_df[~val_df["repo"].isin(exclude_repos)].copy()
                test_df = test_df[~test_df["repo"].isin(exclude_repos)].copy()
                self._reporter.display_text(
                    f"After exclusion: Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}"
                )

            for seed in seeds:
                run_row, repo_frame = self._run_once(
                    train_df, val_df, test_df, scenario, seed, fold, equalize_train_size
                )
                run_rows.append(run_row)
                repo_rows.append(repo_frame)
                self._reporter.display_text(
                    f"  fold={fold} seed={seed}: PR-AUC={run_row['pr_auc']:.4f} F1={run_row['f1']:.4f} "
                    f"MCC={run_row['mcc']:.4f} ROC-AUC={run_row['roc_auc']:.4f} "
                    f"(train {run_row['train_seconds']:.1f}s, inference {run_row['inference_seconds']:.1f}s)"
                )

        results = pd.DataFrame(run_rows)
        per_repo = pd.concat(repo_rows, ignore_index=True)

        summary = results[METRIC_COLUMNS].agg(["mean", "std"]).round(4)
        self._reporter.display_text()
        self._reporter.display_text("Mean and standard deviation over runs:")
        self._reporter.display_table(summary)

        results_path = self._reporter.save_csv(results, "baseline_results.csv")
        per_repo_path = self._reporter.save_csv(per_repo, "baseline_per_repo.csv")
        self._reporter.update_summary(
            {
                "device": device,
                "scenario": scenario.value,
                "model": self._model_name,
                "seeds": seeds,
                "equalize_train_size": equalize_train_size,
                "exclude_repos": exclude_repos,
                "n_runs": len(results),
                "mean": summary.loc["mean"].to_dict(),
                "std": summary.loc["std"].fillna(0.0).to_dict(),
                "results_path": str(results_path),
                "per_repo_path": str(per_repo_path),
            }
        )
        self._reporter.save_reports(
            text_filename=f"baseline_{scenario.value}.txt",
            json_filename=f"baseline_{scenario.value}.json",
        )
        return results

    def _load_processed_data(
        self, scenario: SplitScenario, fold: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed parquet files from pipeline output.

        Args:
            scenario: Split scenario to load.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            FileNotFoundError: If processed parquet files don't exist.
        """
        scenario_dir = DefectorsSplitGenerator.scenario_dir(self._input_dir, scenario, fold)
        fold_hint = f" --fold {fold}" if scenario in GROUP_KFOLD_SCENARIOS else ""
        train_path = scenario_dir / f"{scenario.value}_train.parquet"
        val_path = scenario_dir / f"{scenario.value}_val.parquet"
        test_path = scenario_dir / f"{scenario.value}_test.parquet"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Processed train file not found: {train_path}\n"
                f"Please run the data pipeline first: uv run python -m software_defects run-pipeline "
                f"--dataset {self._config.dataset_name} --scenario {scenario.value}"
                f"{fold_hint}"
            )
        if not val_path.exists():
            raise FileNotFoundError(
                f"Processed val file not found: {val_path}\n"
                f"Please run the data pipeline first: uv run python -m software_defects run-pipeline "
                f"--dataset {self._config.dataset_name} --scenario {scenario.value}"
                f"{fold_hint}"
            )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Processed test file not found: {test_path}\n"
                f"Please run the data pipeline first: uv run python -m software_defects run-pipeline "
                f"--dataset {self._config.dataset_name} --scenario {scenario.value}"
                f"{fold_hint}"
            )

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        self._reporter.display_text(f"Loaded processed data from: {scenario_dir}")
        self._reporter.display_text(f"  Train: {len(train_df):,} samples")
        self._reporter.display_text(f"  Val: {len(val_df):,} samples")
        self._reporter.display_text(f"  Test: {len(test_df):,} samples")
        self._reporter.display_text()

        return train_df, val_df, test_df

    def _run_once(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scenario: SplitScenario,
        seed: int,
        fold: int,
        equalize_train_size: int | None,
    ) -> tuple[dict, pd.DataFrame]:
        """Train and evaluate one (fold, seed) run."""
        # Subsample training data if equalize_train_size is set
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        model = BASELINE_MODELS[self._model_name](self._baseline_config, seed=seed)
        model.fit(
            train_df["content"],
            train_df["label"].to_numpy(),
            val_df["content"],
            val_df["label"].to_numpy(),
        )

        val_scores = model.predict_scores(val_df["content"])
        start = time.perf_counter()
        test_scores = model.predict_scores(test_df["content"])
        inference_seconds = time.perf_counter() - start

        # Threshold is selected on validation data only, never on test data
        result = self._evaluator.calculate_metrics_with_val_threshold(
            val_df["label"], val_scores, test_df["label"], test_scores
        )

        scored = test_df[["repo", "label"]].assign(score=test_scores)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_repo = self._evaluator.calculate_per_repo_metrics(scored, "score", threshold=result.threshold)

        repo_frame = pd.DataFrame(
            [
                {
                    "scenario": scenario.value,
                    "fold": fold,
                    "seed": seed,
                    "repo": repo,
                    "n_test": int((scored["repo"] == repo).sum()),
                    "prevalence": metrics.prevalence,
                    "pr_auc": metrics.pr_auc,
                    "f1": metrics.f1,
                    "mcc": metrics.mcc,
                    "roc_auc": metrics.roc_auc,
                }
                for repo, metrics in per_repo.items()
            ]
        )

        train_repos = sorted(train_df["repo"].unique())
        test_repos = sorted(test_df["repo"].unique())

        run_row = {
            "scenario": scenario.value,
            "model": self._model_name,
            "fold": fold,
            "seed": seed,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "train_repos": ";".join(train_repos),
            "test_repos": ";".join(test_repos),
            "train_defective_pct": float(train_df["label"].mean() * 100),
            "test_defective_pct": float(test_df["label"].mean() * 100),
            "pr_auc": result.pr_auc,
            "prevalence_baseline_pr_auc": result.prevalence_baseline_pr_auc,
            "f1": result.f1,
            "mcc": result.mcc,
            "roc_auc": result.roc_auc,
            "threshold": result.threshold,
            "best_c": model.best_c,
            "n_features": model.n_features,
            "train_seconds": model.train_seconds,
            "inference_seconds": inference_seconds,
        }
        return run_row, repo_frame
