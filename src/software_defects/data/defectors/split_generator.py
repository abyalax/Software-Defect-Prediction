"""Dataset split generator for Defectors dataset based on validated scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class SplitScenario(Enum):
    """Experiment split scenarios."""

    ML_TO_NON_ML = "ML_to_NonML"
    NON_ML_TO_ML = "NonML_to_ML"
    REPO_LEVEL_SPLIT = "Repository_Level_Split"
    WITHIN_DOMAIN_ML_DL = "Within_Domain_ML_DL"
    WITHIN_DOMAIN_NON_ML = "Within_Domain_Non_ML"


@dataclass
class SplitResult:
    """Result of a dataset split operation."""

    scenario: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_repos: list[str]
    test_repos: list[str]
    train_samples: int
    test_samples: int
    train_defective_pct: float
    test_defective_pct: float


class DefectorsSplitGenerator:
    """Generate dataset splits based on validated scenarios."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the split generator.

        Args:
            config: Dataset configuration.
        """
        self._config = config
        self._domain_mapping = self._load_domain_mapping()

        # Define ML/DL repositories
        self._ml_dl_repos = set(self._domain_mapping.get("ml_dl", []))
        self._non_ml_repos = set(self._domain_mapping.get("non_ml", []))

    def _load_domain_mapping(self) -> dict:
        """Load domain mapping from YAML file.

        Returns:
            Dictionary with domain mappings.
        """
        mapping_path = self._config.raw_dir / "domain_mapping.yaml"

        if not mapping_path.exists():
            raise FileNotFoundError(f"Domain mapping file not found: {mapping_path}")

        with open(mapping_path) as f:
            mapping = yaml.safe_load(f)

        # Extract repo names
        ml_dl_repos = [item["name"] for item in mapping.get("ml_dl", [])]
        non_ml_repos = [item["name"] for item in mapping.get("non_ml", [])]

        return {"ml_dl": ml_dl_repos, "non_ml": non_ml_repos}

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to exclude freqtrade and apply basic cleaning.

        Args:
            df: Input DataFrame.

        Returns:
            Filtered DataFrame.
        """
        # Exclude freqtrade (only 1 record)
        df = df[df["repo"] != "freqtrade"].copy()

        # Calculate binary labels
        df["label"] = df["lines"].apply(
            lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(df["lines"].iloc[0]))) else 0
        )

        return df

    def generate_split(self, df: pd.DataFrame, scenario: SplitScenario) -> SplitResult:
        """Generate a dataset split based on the specified scenario.

        Args:
            df: Input DataFrame with all data.
            scenario: Split scenario to generate.

        Returns:
            SplitResult with train/test DataFrames and metadata.
        """
        df = self._filter_data(df)

        if scenario == SplitScenario.ML_TO_NON_ML:
            return self._split_ml_to_non_ml(df)
        elif scenario == SplitScenario.NON_ML_TO_ML:
            return self._split_non_ml_to_ml(df)
        elif scenario == SplitScenario.REPO_LEVEL_SPLIT:
            return self._split_repo_level(df)
        elif scenario == SplitScenario.WITHIN_DOMAIN_ML_DL:
            return self._split_within_domain_ml_dl(df)
        elif scenario == SplitScenario.WITHIN_DOMAIN_NON_ML:
            return self._split_within_domain_non_ml(df)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _split_ml_to_non_ml(self, df: pd.DataFrame) -> SplitResult:
        """Split: Train on ML/DL, Test on Non-ML."""
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)]
        non_ml_df = df[~df["repo"].isin(self._ml_dl_repos)]

        return SplitResult(
            scenario=SplitScenario.ML_TO_NON_ML.value,
            train_df=ml_dl_df,
            test_df=non_ml_df,
            train_repos=sorted(ml_dl_df["repo"].unique().tolist()),
            test_repos=sorted(non_ml_df["repo"].unique().tolist()),
            train_samples=len(ml_dl_df),
            test_samples=len(non_ml_df),
            train_defective_pct=float(ml_dl_df["label"].mean() * 100),
            test_defective_pct=float(non_ml_df["label"].mean() * 100),
        )

    def _split_non_ml_to_ml(self, df: pd.DataFrame) -> SplitResult:
        """Split: Train on Non-ML, Test on ML/DL."""
        non_ml_df = df[~df["repo"].isin(self._ml_dl_repos)]
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)]

        return SplitResult(
            scenario=SplitScenario.NON_ML_TO_ML.value,
            train_df=non_ml_df,
            test_df=ml_dl_df,
            train_repos=sorted(non_ml_df["repo"].unique().tolist()),
            test_repos=sorted(ml_dl_df["repo"].unique().tolist()),
            train_samples=len(non_ml_df),
            test_samples=len(ml_dl_df),
            train_defective_pct=float(non_ml_df["label"].mean() * 100),
            test_defective_pct=float(ml_dl_df["label"].mean() * 100),
        )

    def _split_repo_level(self, df: pd.DataFrame) -> SplitResult:
        """Split: Repository-level holdout (mixed domains)."""
        all_repos = sorted(df["repo"].unique().tolist())
        ml_dl_repo_list = [r for r in all_repos if r in self._ml_dl_repos]
        non_ml_repo_list = [r for r in all_repos if r not in self._ml_dl_repos]

        # Holdout 1 ML/DL repo and 3 Non-ML repos for test
        if len(ml_dl_repo_list) > 1:
            ml_dl_test_repos = [ml_dl_repo_list[0]]
            ml_dl_train_repos = ml_dl_repo_list[1:]
        else:
            ml_dl_test_repos = []
            ml_dl_train_repos = ml_dl_repo_list

        if len(non_ml_repo_list) > 3:
            non_ml_test_repos = non_ml_repo_list[:3]
            non_ml_train_repos = non_ml_repo_list[3:]
        else:
            non_ml_test_repos = non_ml_repo_list[:1] if len(non_ml_repo_list) > 0 else []
            non_ml_train_repos = non_ml_repo_list[1:] if len(non_ml_repo_list) > 1 else []

        train_repos = ml_dl_train_repos + non_ml_train_repos
        test_repos = ml_dl_test_repos + non_ml_test_repos

        train_df = df[df["repo"].isin(train_repos)]
        test_df = df[df["repo"].isin(test_repos)]

        return SplitResult(
            scenario=SplitScenario.REPO_LEVEL_SPLIT.value,
            train_df=train_df,
            test_df=test_df,
            train_repos=train_repos,
            test_repos=test_repos,
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
        )

    def _split_within_domain_ml_dl(self, df: pd.DataFrame) -> SplitResult:
        """Split: Within-domain ML/DL holdout."""
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)]
        ml_dl_repos = sorted(ml_dl_df["repo"].unique().tolist())

        if len(ml_dl_repos) > 1:
            test_repos = [ml_dl_repos[0]]
            train_repos = ml_dl_repos[1:]
        else:
            test_repos = []
            train_repos = ml_dl_repos

        train_df = ml_dl_df[ml_dl_df["repo"].isin(train_repos)]
        test_df = ml_dl_df[ml_dl_df["repo"].isin(test_repos)]

        return SplitResult(
            scenario=SplitScenario.WITHIN_DOMAIN_ML_DL.value,
            train_df=train_df,
            test_df=test_df,
            train_repos=train_repos,
            test_repos=test_repos,
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
        )

    def _split_within_domain_non_ml(self, df: pd.DataFrame) -> SplitResult:
        """Split: Within-domain Non-ML holdout."""
        non_ml_df = df[~df["repo"].isin(self._ml_dl_repos)]
        non_ml_repos = sorted(non_ml_df["repo"].unique().tolist())

        if len(non_ml_repos) > 3:
            test_repos = non_ml_repos[:3]
            train_repos = non_ml_repos[3:]
        else:
            test_repos = non_ml_repos[:1] if len(non_ml_repos) > 0 else []
            train_repos = non_ml_repos[1:] if len(non_ml_repos) > 1 else []

        train_df = non_ml_df[non_ml_df["repo"].isin(train_repos)]
        test_df = non_ml_df[non_ml_df["repo"].isin(test_repos)]

        return SplitResult(
            scenario=SplitScenario.WITHIN_DOMAIN_NON_ML.value,
            train_df=train_df,
            test_df=test_df,
            train_repos=train_repos,
            test_repos=test_repos,
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
        )

    def aggregate_to_commit_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate file-level data to commit-level.

        Args:
            df: File-level DataFrame.

        Returns:
            Commit-level DataFrame.
        """
        # Group by commit and aggregate
        commit_stats = (
            df.groupby("commit")
            .agg(
                {
                    "label": "max",  # Commit is defective if any file is defective
                    "content": lambda x: "\n".join(x.fillna("").astype(str)),  # Concatenate diffs
                    "repo": "first",
                    "lines": lambda x: [line for sublist in x for line in sublist],  # Merge line lists
                }
            )
            .reset_index()
        )

        commit_stats.columns = ["commit", "label", "content", "repo", "lines"]

        return commit_stats
