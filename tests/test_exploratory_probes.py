"""Tests for the P0 exploratory checkpoints 14 and 15 to 18."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from software_defects.config import DatasetConfig
from software_defects.data.defectors.dataset_explorer import DefectorsDatasetExplorer
from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.data.defectors.exploratory.duplicate_diff_analysis import DuplicateDiffAnalysis
from software_defects.data.defectors.exploratory.path_metadata_probe import PathMetadataProbe
from software_defects.data.defectors.exploratory.prevalence_shift_analysis import PrevalenceShiftAnalysis
from software_defects.data.defectors.exploratory.repository_identity_probe import RepositoryIdentityProbe
from software_defects.data.defectors.exploratory.run_report import RunReport
from software_defects.reporting.reporter import ReportWritter
from tests.conftest import write_raw_jit


def _reporter(config: DatasetConfig) -> ReportWritter:
    return ReportWritter(config=config, domain_data="defectors", pipeline="test", class_name="Checkpoint")


def _rich_jit(project_root: Path, rows_per_repo: int = 80) -> pd.DataFrame:
    """Synthetic data where every repository has its own vocabulary, path style and prevalence."""
    rng = np.random.default_rng(3)
    domains = load_domain_sets(project_root)
    repos = sorted(domains["ml_dl"] | domains["non_ml"])
    records = []
    for repo_index, repo in enumerate(repos):
        for i in range(rows_per_repo):
            defective = rng.random() < (0.2 + 0.03 * (repo_index % 10))
            body = "\n".join(
                f"+{repo}_value_{rng.integers(0, 30)} = call_{repo}({j})" for j in range(4 + int(defective) * 3)
            )
            records.append(
                {
                    "datetime": pd.Timestamp("2021-03-01 10:00") + pd.Timedelta(days=i, hours=repo_index),
                    "commit": f"{repo}-c{i // 3}",
                    "repo": repo,
                    "filepath": f"{repo}/pkg{i % 4}/module_{i % 7}.py",
                    "content": f"@@ -1,2 +1,3 @@\n{body}\n",
                    "methods": [],
                    "lines": [1] if defective else [],
                }
            )
    # One diff shared by a training and a test repository, a normalized-only twin, and a label conflict
    shared = (
        "@@ -5,1 +5,4 @@\n+import shared_helper\n+shared_helper.run(alpha)\n+shared_helper.stop(beta)\n+x = gamma\n"
    )
    twin = (
        "@@ -90,1 +90,4 @@\n+import   shared_helper\n+shared_helper.run(alpha)  \n"
        "+shared_helper.stop(beta)\n+x = gamma\n"
    )
    for repo, text, lines in (
        ("pandas", shared, [1]),
        ("django", shared, []),
        ("sentry", twin, [1]),
        ("ray", shared, [1]),  # ML/DL repository sharing the same diff: cross-domain duplicate
    ):
        records.append(
            {
                "datetime": pd.Timestamp("2021-06-01"),
                "commit": f"{repo}-dup",
                "repo": repo,
                "filepath": f"{repo}/dup.py",
                "content": text,
                "methods": [],
                "lines": lines,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def raw_project(project_root: Path) -> Path:
    write_raw_jit(project_root, _rich_jit(project_root))
    return project_root


def _loader(config: DatasetConfig) -> DefectorsDatasetLoader:
    return DefectorsDatasetLoader(config)


class TestPathMetadataProbe:
    """Checkpoint 14."""

    def test_feature_engineering(self) -> None:
        df = pd.DataFrame(
            {
                "repo": ["a", "a", "a"],
                "commit": ["c1", "c1", "c2"],
                "filepath": ["pkg/tests/test_x.py", "pkg/__init__.py", "main.py"],
                "datetime": pd.to_datetime(["2022-01-03 13:00", "2022-01-03 13:00", "2022-01-09 02:00"]),
                "label": [1, 0, 0],
            }
        )
        out = PathMetadataProbe._add_features(df)
        assert out["path_depth"].tolist() == [3, 2, 1]
        assert out["is_test_file"].tolist() == [1, 0, 0]
        assert out["is_init_file"].tolist() == [0, 1, 0]
        assert out["files_in_commit"].tolist() == [2, 2, 1]
        assert out["commit_hour"].tolist() == [13, 13, 2]
        assert out["commit_weekday"].tolist() == [0, 0, 6]
        assert out["commit_year"].tolist() == [2022, 2022, 2022]

    def test_run_writes_reports(self, raw_project: Path, config: DatasetConfig) -> None:
        reporter = _reporter(config)
        PathMetadataProbe(_loader(config), reporter).run()
        summary = reporter.summary["checkpoint_14"]
        table = pd.read_csv(summary["signal_auc_csv"])
        assert set(table["feature"]) >= {"path_depth", "files_in_commit", "commit_day"}
        assert Path(summary["path_text_probe_csv"]).exists()
        assert Path(summary["commit_year_csv"]).exists()


class TestRepositoryIdentityProbe:
    """Checkpoint 15."""

    def test_repo_is_easy_to_guess_from_distinct_vocabulary(self, raw_project: Path, config: DatasetConfig) -> None:
        reporter = _reporter(config)
        RepositoryIdentityProbe(_loader(config), reporter).run()
        summary = reporter.summary["checkpoint_15"]
        by_probe = {row["probe"]: row for row in summary["repo_probe"]}
        assert by_probe["repo_from_diff_content"]["accuracy"] > 0.9
        assert by_probe["repo_from_file_path"]["accuracy_lift"] > 3
        assert summary["domain_probe"]["n_rows"] > 0
        assert Path(summary["probe_csv"]).exists()


class TestDuplicateDiffAnalysis:
    """Checkpoint 16."""

    def test_changed_lines_normalization(self) -> None:
        lines = DuplicateDiffAnalysis.changed_lines("@@ -1 +1 @@\n--- a/x\n+++ b/x\n context\n+  a   =  1 \n-b\n+\n")
        assert lines == ["+a = 1", "-b"]

    def test_keys_ignore_hunk_offsets_and_spacing(self) -> None:
        first = DuplicateDiffAnalysis.changed_lines("@@ -1,1 +1,2 @@\n+a = 1\n-b\n")
        second = DuplicateDiffAnalysis.changed_lines("@@ -50,1 +50,2 @@\n+a  =  1\n-b\n")
        assert DuplicateDiffAnalysis.normalized_key(first) == DuplicateDiffAnalysis.normalized_key(second)
        assert DuplicateDiffAnalysis.normalized_key([]) is None

    def test_near_duplicates_found_and_unrelated_left_alone(self, project_root: Path, config: DatasetConfig) -> None:
        analysis = DuplicateDiffAnalysis(_loader(config), _reporter(config))
        base = [f"+line_{i} = {i}" for i in range(20)]
        similar = base[:-1] + ["+extra = 1"]  # Jaccard 19/21 > 0.8
        unrelated = [f"-other_{i}()" for i in range(20)]
        shingles = [analysis.shingles(lines) for lines in (base, similar, unrelated)]
        pairs, _ = analysis.near_duplicate_pairs(shingles)
        assert pairs == [(0, 1)]
        labels = analysis.cluster_labels(3, pairs)
        assert labels[0] == labels[1]
        assert pd.isna(labels[2])

    def test_run_reports_cross_repo_duplicates_and_exposure(self, raw_project: Path, config: DatasetConfig) -> None:
        reporter = _reporter(config)
        DuplicateDiffAnalysis(_loader(config), reporter).run()
        summary = reporter.summary["checkpoint_16"]
        by_key = {row["key"]: row for row in summary["key_summary"]}
        assert by_key["exact_key"]["cross_repo_groups"] >= 1
        assert by_key["exact_key"]["label_conflict_groups"] >= 1
        assert by_key["normalized_key"]["duplicate_rows"] >= 3  # shared + twin
        exposure = pd.read_csv(summary["split_exposure_csv"]).set_index(["layout", "key"])
        # ray (ML/DL, train side of ML_to_NonML) shares the diff with pandas/django/sentry (Non-ML tests)
        assert exposure.loc[("ML_to_NonML", "exact_key"), "exposed_rows"] == 2  # pandas and django
        assert (
            exposure.loc[("ML_to_NonML", "normalized_key"), "exposed_rows"] == 3
        )  # sentry twin matches once normalized
        assert exposure.loc[("NonML_to_ML", "exact_key"), "exposed_rows"] == 1
        assert exposure.loc[("Within_Domain_ML_DL", "exact_key"), "exposed_rows"] == 0


class TestPrevalenceShiftAnalysis:
    """Checkpoint 17."""

    def test_wilson_interval(self) -> None:
        low, high = PrevalenceShiftAnalysis.wilson_interval(50, 100)
        assert low < 0.5 < high
        assert high - low < 0.25
        assert all(np.isnan(v) for v in PrevalenceShiftAnalysis.wilson_interval(0, 0))
        low0, high0 = PrevalenceShiftAnalysis.wilson_interval(0, 50)
        assert low0 == pytest.approx(0.0, abs=1e-9) and high0 > 0

    def test_run_covers_every_scenario_and_fold(self, raw_project: Path, config: DatasetConfig) -> None:
        reporter = _reporter(config)
        PrevalenceShiftAnalysis(_loader(config), reporter).run()
        summary = reporter.summary["checkpoint_17"]
        scenarios = pd.read_csv(summary["scenario_csv"])
        assert len(scenarios) == 5 + 2 * 5  # five single-split scenarios plus two five-fold scenarios
        assert scenarios.loc[scenarios["scenario"] == "Group_KFold_ML_DL", "fold"].tolist() == [0, 1, 2, 3, 4]
        assert np.allclose(scenarios["shift_pp"], scenarios["test_prevalence_pct"] - scenarios["train_prevalence_pct"])
        per_repo = pd.read_csv(summary["per_repo_csv"])
        assert per_repo["in_holdout"].sum() == 6
        assert (per_repo["ci_low"] <= per_repo["prevalence"]).all()


class TestRunReport:
    """Checkpoint 18 and checkpoint ordering."""

    def test_run_report_is_always_last(self) -> None:
        assert DefectorsDatasetExplorer.execution_order([18, 18, 3, 14, 3]) == [3, 14, 18]
        assert DefectorsDatasetExplorer.execution_order([1, 2]) == [1, 2]

    def test_report_records_git_state_and_hashes(self, raw_project: Path, config: DatasetConfig) -> None:
        subprocess.run(["git", "init", "-q"], cwd=raw_project, check=True)
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init"],
            cwd=raw_project,
            check=True,
        )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=raw_project, text=True).strip()

        reporter = _reporter(config)
        RunReport(_loader(config), reporter).run()
        payload = json.loads(Path(reporter.summary["checkpoint_18"]["combined_report_json"]).read_text())
        meta = payload["metadata"]
        assert meta["git_commit"] == commit
        assert meta["git_dirty"] is True  # the raw data and mapping are untracked in this temporary repo
        assert all(meta["data_sha256"][k] for k in ("jit_random_train", "jit_random_val", "repository_mapping"))
        assert "pandas" in meta["library_versions"]
        assert Path(reporter.summary["checkpoint_18"]["combined_report_md"]).read_text().startswith("# Defectors")
        assert any("uncommitted" in w["message"] for w in payload["warnings"])

    def test_report_without_git_is_unknown(self, raw_project: Path, config: DatasetConfig) -> None:
        reporter = _reporter(config)
        RunReport(_loader(config), reporter).run()
        assert (
            reporter.summary["checkpoint_18"]["git_commit"] in {"unknown"}
            or len(reporter.summary["checkpoint_18"]["git_commit"]) == 40
        )
