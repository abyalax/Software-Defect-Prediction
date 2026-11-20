"""Shared fixtures: a synthetic Defectors-like dataset built on the real repository mapping."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from software_defects.config import DatasetConfig
from software_defects.data.defectors.domain_mapping import load_domain_sets

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_synthetic_jit(rows_per_repo: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a JIT-style frame covering every mapped repository plus freqtrade.

    Defective diffs are longer on average so a size-only model has some signal. A few rows have an
    empty diff (always defect-free, like the real data).
    """
    rng = np.random.default_rng(seed)
    domain_sets = load_domain_sets(REPO_ROOT)
    repos = sorted(domain_sets["ml_dl"] | domain_sets["non_ml"]) + ["freqtrade"]

    records = []
    for repo in repos:
        for i in range(rows_per_repo):
            defective = rng.random() < 0.45
            n_lines = int(rng.integers(2, 12) + (10 if defective else 0))
            body = "\n".join(f"+x{j} = {j}" if j % 2 == 0 else f"-y{j} = {j}" for j in range(n_lines))
            content = f"@@ -1,3 +1,4 @@\n{body}\n"
            if i % 25 == 0 and repo != "freqtrade":
                content = ""
                defective = False
            records.append(
                {
                    "datetime": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i),
                    "commit": f"{repo}-c{i // 2}",
                    "repo": repo,
                    "filepath": f"{repo}/file{i % 2}.py",
                    "content": content,
                    "methods": [],
                    "lines": [1, 2] if defective else [],
                }
            )
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def synthetic_jit() -> pd.DataFrame:
    """Synthetic JIT data shared by the tests."""
    return make_synthetic_jit()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Temporary project root that already contains the real repository mapping YAML."""
    mapping_dir = tmp_path / "artifacts" / "mapping"
    mapping_dir.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "artifacts" / "mapping" / "repository_mapping.yaml", mapping_dir)
    return tmp_path


@pytest.fixture
def config(project_root: Path) -> DatasetConfig:
    """Dataset config pointing at the temporary project root."""
    return DatasetConfig(project_root=project_root, dataset_name="defectors")


def write_raw_jit(project_root: Path, df: pd.DataFrame) -> None:
    """Write the frame as the three original JIT parquet files (random record-level split)."""
    raw = project_root / "data" / "raw" / "defectors" / "defectors" / "jit_bug_prediction_splits" / "random"
    raw.mkdir(parents=True, exist_ok=True)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    n = len(shuffled)
    parts = {
        "train": shuffled.iloc[: int(n * 0.8)],
        "test": shuffled.iloc[int(n * 0.8) : int(n * 0.9)],
        "val": shuffled.iloc[int(n * 0.9) :],
    }
    for name, part in parts.items():
        part.to_parquet(raw / f"{name}.parquet.gzip", compression="gzip", index=False)
