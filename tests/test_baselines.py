"""Tests for the diff-size baseline, the per-fold Group K-Fold pipeline and the baseline experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from software_defects.config import DatasetConfig
from software_defects.data.defectors.pipeline import DefectorsDataPipeline
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    SplitScenario,
)
from software_defects.models.baseline_experiment import BaselineExperiment
from software_defects.models.length_logreg import FEATURE_NAMES, LengthLogRegBaseline
from tests.conftest import write_raw_jit


def test_default_tokenizer_length_fits_codebert() -> None:
    assert DatasetConfig(project_root=Path("."), dataset_name="defectors").tokenizer_max_length == 512


def test_length_features_match_checkpoint_13_definitions() -> None:
    texts = pd.Series(["@@ -1 +1 @@\n+a\n+b\n-c\n", "", "x\n+y\n@@ @@", "\n+\n+\n+"])
    features = LengthLogRegBaseline.extract_features(texts)

    def raw(value: str) -> list[int]:
        # Exactly what checkpoint 13 computes
        return [len(value), value.count("\n+"), value.count("\n-"), value.count("@@")]

    assert FEATURE_NAMES == ("diff_length", "added_lines", "removed_lines", "num_hunks")
    expected = np.log1p(np.array([raw(t) for t in texts], dtype=float))
    assert features.shape == (4, 4)
    assert features == pytest.approx(expected)


def test_length_features_treat_missing_diffs_as_empty() -> None:
    features = LengthLogRegBaseline.extract_features(pd.Series([None, ""], dtype=object))
    assert (features == 0).all()


def test_length_baseline_learns_size_signal(synthetic_jit: pd.DataFrame) -> None:
    data = synthetic_jit[synthetic_jit["content"] != ""].copy()
    data["label"] = data["lines"].apply(lambda x: int(len(x) > 0))
    train, test = data.iloc[: len(data) // 2], data.iloc[len(data) // 2 :]

    model = LengthLogRegBaseline(seed=0)
    with pytest.raises(RuntimeError):
        model.predict_scores(test["content"])
    model.fit(train["content"], train["label"].to_numpy(), test["content"], test["label"].to_numpy())

    assert model.n_features == 4
    assert model.best_c is not None
    assert roc_auc_score(test["label"], model.predict_scores(test["content"])) > 0.9


def test_pipeline_writes_one_directory_per_fold(config: DatasetConfig, project_root: Path, synthetic_jit) -> None:
    write_raw_jit(project_root, synthetic_jit)
    out = project_root / "processed"

    test_repos_by_fold = []
    for fold in (0, 2):
        DefectorsDataPipeline(config).run_pipeline(
            SplitScenario.GROUP_KFOLD_ML_DL, output_dir=out, tokenize=False, fold=fold
        )
        fold_dir = out / "Group_KFold_ML_DL" / f"fold_{fold}"
        test_df = pd.read_parquet(fold_dir / "Group_KFold_ML_DL_test.parquet")
        assert (fold_dir / "Group_KFold_ML_DL_train.parquet").exists()
        assert (fold_dir / "Group_KFold_ML_DL_val.parquet").exists()
        test_repos_by_fold.append(set(test_df["repo"]))

    assert test_repos_by_fold[0] != test_repos_by_fold[1]
    assert not test_repos_by_fold[0] & test_repos_by_fold[1]


@pytest.mark.parametrize("model_name", ["tfidf", "length"])
def test_group_kfold_baseline_uses_a_different_split_per_fold(
    config: DatasetConfig, project_root: Path, synthetic_jit: pd.DataFrame, model_name: str
) -> None:
    write_raw_jit(project_root, synthetic_jit)
    out = project_root / "processed"
    for fold in range(GROUP_KFOLD_N_FOLDS):
        DefectorsDataPipeline(config).run_pipeline(
            SplitScenario.GROUP_KFOLD_ML_DL, output_dir=out, tokenize=False, fold=fold
        )

    results = BaselineExperiment(config, out, model_name=model_name).run(SplitScenario.GROUP_KFOLD_ML_DL, seeds=[0])

    assert len(results) == GROUP_KFOLD_N_FOLDS
    assert set(results["model"]) == {model_name}
    # Before the fix all five folds re-used fold 0; now every fold tests different repositories
    assert results["test_repos"].nunique() == GROUP_KFOLD_N_FOLDS


def test_baseline_reports_missing_fold(config: DatasetConfig, project_root: Path) -> None:
    experiment = BaselineExperiment(config, project_root / "processed", model_name="length")
    with pytest.raises(FileNotFoundError, match="--fold 0"):
        experiment.run(SplitScenario.GROUP_KFOLD_NON_ML, seeds=[0])


def test_unknown_baseline_model_is_rejected(config: DatasetConfig, project_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown baseline model"):
        BaselineExperiment(config, project_root, model_name="svm")
