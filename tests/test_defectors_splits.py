"""Tests for the repository holdout, the split generator and the exploratory split checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from software_defects.config import DatasetConfig
from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import (
    MAPPING_RELATIVE_PATH,
    load_domain_sets,
    load_holdout_repos,
)
from software_defects.data.defectors.exploratory.repository_split import RepositorySplit
from software_defects.data.defectors.exploratory.split_strategy_evaluation import SplitStrategyEvaluation
from software_defects.data.defectors.split_generator import (
    GROUP_KFOLD_N_FOLDS,
    DefectorsSplitGenerator,
    SplitScenario,
)
from software_defects.reporting.reporter import ReportWritter
from tests.conftest import write_raw_jit

ML_HOLDOUT = ["jax", "lightning"]
NON_ML_HOLDOUT = ["airflow", "ansible", "black", "celery"]


@pytest.fixture
def generator(config: DatasetConfig) -> DefectorsSplitGenerator:
    """Split generator reading the real mapping YAML."""
    return DefectorsSplitGenerator(config)


def _rewrite_mapping(project_root: Path, mutate) -> None:
    path = project_root / MAPPING_RELATIVE_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _reporter(config: DatasetConfig) -> ReportWritter:
    return ReportWritter(config=config, domain_data="defectors", pipeline="test", class_name="Checkpoint")


class TestHoldoutConfig:
    """Holdout configuration in the mapping YAML."""

    def test_configured_holdout(self, project_root: Path) -> None:
        assert load_holdout_repos(project_root) == {"ml_dl": ML_HOLDOUT, "non_ml": NON_ML_HOLDOUT}

    def test_unknown_repo_is_rejected(self, project_root: Path) -> None:
        _rewrite_mapping(project_root, lambda d: d["holdout_repositories"]["ml_dl"].append("not-a-repo"))
        with pytest.raises(ValueError, match="not mapped"):
            load_holdout_repos(project_root)

    def test_repo_of_other_domain_is_rejected(self, project_root: Path) -> None:
        _rewrite_mapping(project_root, lambda d: d["holdout_repositories"]["ml_dl"].append("django"))
        with pytest.raises(ValueError, match="not mapped"):
            load_holdout_repos(project_root)

    def test_whole_domain_cannot_be_held_out(self, project_root: Path) -> None:
        ml_repos = sorted(load_domain_sets(project_root)["ml_dl"])
        _rewrite_mapping(project_root, lambda d: d["holdout_repositories"].update(ml_dl=ml_repos))
        with pytest.raises(ValueError, match="every repository"):
            load_holdout_repos(project_root)

    def test_missing_section_is_rejected(self, project_root: Path) -> None:
        _rewrite_mapping(project_root, lambda d: d.pop("holdout_repositories"))
        with pytest.raises(ValueError, match="holdout_repositories"):
            load_holdout_repos(project_root)


class TestSplitGenerator:
    """Scenarios of the split generator."""

    def test_filter_drops_freqtrade_and_empty_diffs(
        self, generator: DefectorsSplitGenerator, synthetic_jit: pd.DataFrame
    ) -> None:
        filtered = generator.filter_data(synthetic_jit)
        assert "freqtrade" not in set(filtered["repo"])
        assert (filtered["content"].str.strip() != "").all()
        assert set(filtered["label"].unique()) <= {0, 1}

    def test_repo_level_split_uses_configured_holdout(
        self, generator: DefectorsSplitGenerator, synthetic_jit: pd.DataFrame
    ) -> None:
        result = generator.generate_split(synthetic_jit, SplitScenario.REPO_LEVEL_SPLIT)
        assert sorted(result.test_repos) == sorted(ML_HOLDOUT + NON_ML_HOLDOUT)
        assert not set(result.train_repos) & set(result.test_repos)
        assert set(result.val_repos) <= set(result.train_repos)
        assert not set(result.train_df["commit"]) & set(result.test_df["commit"])
        assert not set(result.train_df["commit"]) & set(result.val_df["commit"])

    def test_within_domain_ml_test_is_not_a_single_tiny_repo(
        self, generator: DefectorsSplitGenerator, synthetic_jit: pd.DataFrame
    ) -> None:
        result = generator.generate_split(synthetic_jit, SplitScenario.WITHIN_DOMAIN_ML_DL)
        ml_repos = load_domain_sets(generator._config.project_root)["ml_dl"]
        assert result.test_repos == ML_HOLDOUT
        assert set(result.train_repos) | set(result.test_repos) == ml_repos
        assert not set(result.train_repos) & set(ML_HOLDOUT)

    def test_within_domain_non_ml(self, generator: DefectorsSplitGenerator, synthetic_jit: pd.DataFrame) -> None:
        result = generator.generate_split(synthetic_jit, SplitScenario.WITHIN_DOMAIN_NON_ML)
        non_ml_repos = load_domain_sets(generator._config.project_root)["non_ml"]
        assert result.test_repos == NON_ML_HOLDOUT
        assert set(result.train_repos) | set(result.test_repos) == non_ml_repos

    @pytest.mark.parametrize(
        ("scenario", "train_domain", "test_domain"),
        [
            (SplitScenario.ML_TO_NON_ML, "ml_dl", "non_ml"),
            (SplitScenario.NON_ML_TO_ML, "non_ml", "ml_dl"),
        ],
    )
    def test_cross_domain_scenarios(
        self,
        generator: DefectorsSplitGenerator,
        synthetic_jit: pd.DataFrame,
        scenario: SplitScenario,
        train_domain: str,
        test_domain: str,
    ) -> None:
        domains = load_domain_sets(generator._config.project_root)
        result = generator.generate_split(synthetic_jit, scenario)
        assert set(result.train_repos) == domains[train_domain]
        assert set(result.test_repos) == domains[test_domain]

    def test_group_kfold_folds_cover_every_repo_once(
        self, generator: DefectorsSplitGenerator, synthetic_jit: pd.DataFrame
    ) -> None:
        ml_repos = load_domain_sets(generator._config.project_root)["ml_dl"]
        test_sets = [
            set(generator.generate_split(synthetic_jit, SplitScenario.GROUP_KFOLD_ML_DL, fold=fold).test_repos)
            for fold in range(GROUP_KFOLD_N_FOLDS)
        ]
        assert len({frozenset(s) for s in test_sets}) == GROUP_KFOLD_N_FOLDS  # folds are different
        assert set().union(*test_sets) == ml_repos
        assert sum(len(s) for s in test_sets) == len(ml_repos)  # no repo is tested twice

    def test_scenario_dir_per_fold(self, tmp_path: Path) -> None:
        assert DefectorsSplitGenerator.scenario_dir(tmp_path, SplitScenario.REPO_LEVEL_SPLIT) == (
            tmp_path / "Repository_Level_Split"
        )
        assert DefectorsSplitGenerator.scenario_dir(tmp_path, SplitScenario.GROUP_KFOLD_ML_DL, 3) == (
            tmp_path / "Group_KFold_ML_DL" / "fold_3"
        )


class TestCheckpoints:
    """Checkpoints 8 and 11 must agree with the generator the pipeline uses."""

    def test_checkpoint_11_matches_generator(
        self, config: DatasetConfig, project_root: Path, synthetic_jit: pd.DataFrame
    ) -> None:
        write_raw_jit(project_root, synthetic_jit)
        reporter = _reporter(config)
        RepositorySplit(DefectorsDatasetLoader(config), reporter).run()

        report = pd.read_csv(reporter.summary["checkpoint_11"]["csv_path"]).set_index("scenario")
        generator = DefectorsSplitGenerator(config)
        for scenario in (
            SplitScenario.REPO_LEVEL_SPLIT,
            SplitScenario.WITHIN_DOMAIN_ML_DL,
            SplitScenario.WITHIN_DOMAIN_NON_ML,
        ):
            expected = generator.generate_split(DefectorsDatasetLoader(config).load_all_jit(), scenario)
            row = report.loc[scenario.value]
            assert row["train_samples"] == expected.train_samples
            assert row["val_samples"] == expected.val_samples
            assert row["test_samples"] == expected.test_samples
            assert sorted(row["test_repos_list"].split(", ")) == sorted(expected.test_repos)

        # The ML/DL test side is no longer the single smallest repository
        within_ml = report.loc[SplitScenario.WITHIN_DOMAIN_ML_DL.value]
        assert within_ml["test_repos_list"] == ", ".join(ML_HOLDOUT)

    def test_checkpoint_8_projection_counts_every_row(
        self, config: DatasetConfig, project_root: Path, synthetic_jit: pd.DataFrame
    ) -> None:
        write_raw_jit(project_root, synthetic_jit)
        reporter = _reporter(config)
        SplitStrategyEvaluation(DefectorsDatasetLoader(config), reporter).run()

        payload = json.loads(Path(reporter.summary["checkpoint_8"]["split_recommendations_json"]).read_text())
        projected = payload["configured_repository_level_split"]
        total = sum(v for k, v in projected.items() if k.startswith("projected_"))

        # val rows must be included: the sum equals every row left after the pipeline filtering
        expected_total = len(DefectorsSplitGenerator(config).filter_data(DefectorsDatasetLoader(config).load_all_jit()))
        assert total == expected_total
        assert projected["test_ml_dl_repos"] == ML_HOLDOUT
        assert projected["test_non_ml_repos"] == NON_ML_HOLDOUT
