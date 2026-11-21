"""Dataset split generator for Defectors dataset based on validated scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from software_defects.data.defectors.domain_mapping import load_domain_sets, load_holdout_repos

if TYPE_CHECKING:
    from software_defects.config import DatasetConfig


class SplitScenario(Enum):
    """Experiment split scenarios."""

    ML_TO_NON_ML = "ML_to_NonML"
    NON_ML_TO_ML = "NonML_to_ML"
    REPO_LEVEL_SPLIT = "Repository_Level_Split"
    WITHIN_DOMAIN_ML_DL = "Within_Domain_ML_DL"
    WITHIN_DOMAIN_NON_ML = "Within_Domain_Non_ML"
    GROUP_KFOLD_ML_DL = "Group_KFold_ML_DL"
    GROUP_KFOLD_NON_ML = "Group_KFold_Non_ML"
    CROSS_DOMAIN_DEFAULT = "Cross_Domain_Default"


GROUP_KFOLD_SCENARIOS = frozenset({SplitScenario.GROUP_KFOLD_ML_DL, SplitScenario.GROUP_KFOLD_NON_ML})
GROUP_KFOLD_N_FOLDS = 5


@dataclass
class SplitResult:
    """Result of a dataset split operation."""

    scenario: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_repos: list[str]
    val_repos: list[str]
    test_repos: list[str]
    train_samples: int
    val_samples: int
    test_samples: int
    train_defective_pct: float
    val_defective_pct: float
    test_defective_pct: float
    is_cross_domain: bool = False


class DefectorsSplitGenerator:
    """Generate dataset splits based on validated scenarios."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the split generator.

        Args:
            config: Dataset configuration.
        """
        self._config = config

        # Domains come from the shared YAML; ambiguous/unmapped repos are never used in a split
        domain_sets = load_domain_sets(config.project_root)
        self._ml_dl_repos = domain_sets["ml_dl"]
        self._non_ml_repos = domain_sets["non_ml"]

        # Test holdout is explicit in the same YAML (never derived from alphabetical order)
        self._holdout = load_holdout_repos(config.project_root)

    @staticmethod
    def scenario_dir(base_dir: Path, scenario: SplitScenario, fold: int = 0) -> Path:
        """Return the directory that holds the processed files of one scenario.

        Group K-Fold scenarios store each fold in its own ``fold_<n>`` subdirectory so that folds
        never overwrite each other.

        Args:
            base_dir: Root directory of the processed data.
            scenario: Split scenario.
            fold: Fold index, only used by the Group K-Fold scenarios.

        Returns:
            Directory of the scenario (and fold).
        """
        scenario_path = Path(base_dir) / scenario.value
        if scenario in GROUP_KFOLD_SCENARIOS:
            return scenario_path / f"fold_{fold}"
        return scenario_path

    def _holdout_split(self, domain: str, available_repos: list[str]) -> tuple[list[str], list[str]]:
        """Split the repositories of one domain into train and test using the configured holdout.

        Args:
            domain: ``ml_dl`` or ``non_ml``.
            available_repos: Sorted repositories of that domain present in the data.

        Returns:
            Tuple of (train_repos, test_repos).

        Raises:
            ValueError: If no holdout repository is present in the data, or no training repository remains.
        """
        holdout = set(self._holdout[domain])
        test_repos = [repo for repo in available_repos if repo in holdout]
        train_repos = [repo for repo in available_repos if repo not in holdout]
        if not test_repos or not train_repos:
            raise ValueError(
                f"Holdout for domain '{domain}' leaves an empty side "
                f"(train={train_repos}, test={test_repos}); check holdout_repositories in the mapping YAML"
            )
        return train_repos, test_repos

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to exclude freqtrade and apply basic cleaning.

        Args:
            df: Input DataFrame.

        Returns:
            Filtered DataFrame.
        """
        # Exclude freqtrade (only 1 record)
        df = df[df["repo"] != "freqtrade"].copy()

        # Drop rows with an empty diff: all of them are defect-free, so they would leak the label
        df = df[df["content"].fillna("").str.strip() != ""].copy()

        # Calculate binary labels
        df["label"] = df["lines"].apply(lambda x: int(len(x) > 0))

        return df

    def generate_split(
        self,
        df: pd.DataFrame,
        scenario: SplitScenario,
        val_ratio: float = 0.1,
        equalize_train_size: int | None = None,
        seed: int = 42,
        fold: int = 0,
    ) -> SplitResult:
        """Generate a dataset split based on the specified scenario.

        Args:
            df: Input DataFrame with all data.
            scenario: Split scenario to generate.
            val_ratio: Ratio of validation set from train (default 0.1).
            equalize_train_size: If set, limit training samples to this size.
            seed: Random seed for reproducibility.
            fold: Fold index used by the Group K-Fold scenarios (0 to 4).

        Returns:
            SplitResult with train/val/test DataFrames and metadata.
        """
        df = self.filter_data(df)

        if scenario == SplitScenario.ML_TO_NON_ML:
            return self._split_ml_to_non_ml(df, val_ratio, equalize_train_size, seed)
        elif scenario == SplitScenario.NON_ML_TO_ML:
            return self._split_non_ml_to_ml(df, val_ratio, equalize_train_size, seed)
        elif scenario == SplitScenario.REPO_LEVEL_SPLIT:
            return self._split_repo_level(df, val_ratio, equalize_train_size, seed)
        elif scenario == SplitScenario.WITHIN_DOMAIN_ML_DL:
            return self._split_within_domain_ml_dl(df, val_ratio, equalize_train_size, seed)
        elif scenario == SplitScenario.WITHIN_DOMAIN_NON_ML:
            return self._split_within_domain_non_ml(df, val_ratio, equalize_train_size, seed)
        elif scenario == SplitScenario.GROUP_KFOLD_ML_DL:
            return self._split_group_kfold_ml_dl(df, val_ratio, equalize_train_size, seed, fold=fold)
        elif scenario == SplitScenario.GROUP_KFOLD_NON_ML:
            return self._split_group_kfold_non_ml(df, val_ratio, equalize_train_size, seed, fold=fold)
        elif scenario == SplitScenario.CROSS_DOMAIN_DEFAULT:
            return self._split_cross_domain_default(df, val_ratio, equalize_train_size, seed)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _split_ml_to_non_ml(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Train on ML/DL, Test on Non-ML."""
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)].copy()
        non_ml_df = df[df["repo"].isin(self._non_ml_repos)].copy()

        # Split train into train/val
        train_df, val_df = self._split_train_val_by_commit(ml_dl_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.ML_TO_NON_ML.value,
            train_df=train_df,
            val_df=val_df,
            test_df=non_ml_df,
            train_repos=sorted(train_df["repo"].unique().tolist()),
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=sorted(non_ml_df["repo"].unique().tolist()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(non_ml_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(non_ml_df["label"].mean() * 100),
            is_cross_domain=True,
        )

    def _split_non_ml_to_ml(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Train on Non-ML, Test on ML/DL."""
        non_ml_df = df[df["repo"].isin(self._non_ml_repos)].copy()
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)].copy()

        # Split train into train/val
        train_df, val_df = self._split_train_val_by_commit(non_ml_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.NON_ML_TO_ML.value,
            train_df=train_df,
            val_df=val_df,
            test_df=ml_dl_df,
            train_repos=sorted(train_df["repo"].unique().tolist()),
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=sorted(ml_dl_df["repo"].unique().tolist()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(ml_dl_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(ml_dl_df["label"].mean() * 100),
            is_cross_domain=True,
        )

    def _split_repo_level(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Repository-level holdout (mixed domains)."""
        all_repos = sorted(df["repo"].unique().tolist())
        ml_dl_repo_list = [r for r in all_repos if r in self._ml_dl_repos]
        non_ml_repo_list = [r for r in all_repos if r in self._non_ml_repos]

        # Holdout repositories come from the mapping YAML
        ml_dl_train_repos, ml_dl_test_repos = self._holdout_split("ml_dl", ml_dl_repo_list)
        non_ml_train_repos, non_ml_test_repos = self._holdout_split("non_ml", non_ml_repo_list)

        train_repos = ml_dl_train_repos + non_ml_train_repos
        test_repos = ml_dl_test_repos + non_ml_test_repos

        train_df = df[df["repo"].isin(train_repos)].copy()
        test_df = df[df["repo"].isin(test_repos)].copy()

        # Split train into train/val
        train_df, val_df = self._split_train_val_by_commit(train_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.REPO_LEVEL_SPLIT.value,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_repos=train_repos,
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=test_repos,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
            is_cross_domain=False,
        )

    def _split_within_domain_ml_dl(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Within-domain ML/DL holdout."""
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)].copy()
        ml_dl_repos = sorted(ml_dl_df["repo"].unique().tolist())

        train_repos, test_repos = self._holdout_split("ml_dl", ml_dl_repos)

        train_df = ml_dl_df[ml_dl_df["repo"].isin(train_repos)].copy()
        test_df = ml_dl_df[ml_dl_df["repo"].isin(test_repos)].copy()

        # Split train into train/val
        train_df, val_df = self._split_train_val_by_commit(train_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.WITHIN_DOMAIN_ML_DL.value,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_repos=train_repos,
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=test_repos,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
            is_cross_domain=False,
        )

    def _split_within_domain_non_ml(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Within-domain Non-ML holdout."""
        non_ml_df = df[df["repo"].isin(self._non_ml_repos)].copy()
        non_ml_repos = sorted(non_ml_df["repo"].unique().tolist())

        train_repos, test_repos = self._holdout_split("non_ml", non_ml_repos)

        train_df = non_ml_df[non_ml_df["repo"].isin(train_repos)].copy()
        test_df = non_ml_df[non_ml_df["repo"].isin(test_repos)].copy()

        # Split train into train/val
        train_df, val_df = self._split_train_val_by_commit(train_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.WITHIN_DOMAIN_NON_ML.value,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_repos=train_repos,
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=test_repos,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
            is_cross_domain=False,
        )

    def _split_group_kfold_ml_dl(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
        n_folds: int = GROUP_KFOLD_N_FOLDS,
        fold: int = 0,
    ) -> SplitResult:
        """Split: Group K-Fold for ML/DL domain (groups by repo)."""
        from sklearn.model_selection import GroupKFold

        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)].copy()

        if len(ml_dl_df["repo"].unique()) < n_folds:
            raise ValueError(f"Need at least {n_folds} ML/DL repos for {n_folds}-fold split")

        gkf = GroupKFold(n_splits=n_folds)
        groups = ml_dl_df["repo"].values

        # Select the requested fold as the test fold
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
        train_val_idx, test_idx = list(gkf.split(ml_dl_df, groups=groups))[fold]
        train_val_df = ml_dl_df.iloc[train_val_idx].copy()
        test_df = ml_dl_df.iloc[test_idx].copy()

        # Split train_val into train/val by commit
        train_df, val_df = self._split_train_val_by_commit(train_val_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.GROUP_KFOLD_ML_DL.value,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_repos=sorted(train_df["repo"].unique().tolist()),
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=sorted(test_df["repo"].unique().tolist()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
            is_cross_domain=False,
        )

    def _split_group_kfold_non_ml(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
        n_folds: int = GROUP_KFOLD_N_FOLDS,
        fold: int = 0,
    ) -> SplitResult:
        """Split: Group K-Fold for Non-ML domain (groups by repo)."""
        from sklearn.model_selection import GroupKFold

        non_ml_df = df[df["repo"].isin(self._non_ml_repos)].copy()

        if len(non_ml_df["repo"].unique()) < n_folds:
            raise ValueError(f"Need at least {n_folds} Non-ML repos for {n_folds}-fold split")

        gkf = GroupKFold(n_splits=n_folds)
        groups = non_ml_df["repo"].values

        # Select the requested fold as the test fold
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
        train_val_idx, test_idx = list(gkf.split(non_ml_df, groups=groups))[fold]
        train_val_df = non_ml_df.iloc[train_val_idx].copy()
        test_df = non_ml_df.iloc[test_idx].copy()

        # Split train_val into train/val by commit
        train_df, val_df = self._split_train_val_by_commit(train_val_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.GROUP_KFOLD_NON_ML.value,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_repos=sorted(train_df["repo"].unique().tolist()),
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=sorted(test_df["repo"].unique().tolist()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(test_df["label"].mean() * 100),
            is_cross_domain=False,
        )

    def _split_cross_domain_default(
        self,
        df: pd.DataFrame,
        val_ratio: float,
        equalize_train_size: int | None,
        seed: int,
    ) -> SplitResult:
        """Split: Cross-domain using default random split (train/val/test)."""
        ml_dl_df = df[df["repo"].isin(self._ml_dl_repos)].copy()
        non_ml_df = df[df["repo"].isin(self._non_ml_repos)].copy()

        # Use ML/DL as source, Non-ML as target
        source_df = ml_dl_df
        target_df = non_ml_df

        # Split source into train/val
        train_df, val_df = self._split_train_val_by_commit(source_df, val_ratio, seed)

        # Equalize train size if requested
        if equalize_train_size is not None and len(train_df) > equalize_train_size:
            train_df = train_df.sample(n=equalize_train_size, random_state=seed)

        return SplitResult(
            scenario=SplitScenario.CROSS_DOMAIN_DEFAULT.value,
            train_df=train_df,
            val_df=val_df,
            test_df=target_df,
            train_repos=sorted(train_df["repo"].unique().tolist()),
            val_repos=sorted(val_df["repo"].unique().tolist()),
            test_repos=sorted(target_df["repo"].unique().tolist()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(target_df),
            train_defective_pct=float(train_df["label"].mean() * 100),
            val_defective_pct=float(val_df["label"].mean() * 100),
            test_defective_pct=float(target_df["label"].mean() * 100),
            is_cross_domain=True,
        )

    def _split_train_val_by_commit(
        self, df: pd.DataFrame, val_ratio: float, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train/val by commit to avoid leakage.

        Args:
            df: Input DataFrame.
            val_ratio: Ratio of validation set.
            seed: Random seed.

        Returns:
            Tuple of (train_df, val_df).
        """
        # Get unique commits
        unique_commits = df["commit"].unique()
        n_val = max(1, int(len(unique_commits) * val_ratio))

        # Sample commits for val
        val_commits = set(pd.Series(unique_commits).sample(n=n_val, random_state=seed))

        # Split by commit
        val_df = df[df["commit"].isin(val_commits)].copy()
        train_df = df[~df["commit"].isin(val_commits)].copy()

        return train_df, val_df

    def check_overlap(self, split_result: SplitResult) -> dict:
        """Check for overlap between train, val, and test splits.

        Args:
            split_result: SplitResult to check.

        Returns:
            Dictionary with overlap statistics.
        """
        train_commits = set(split_result.train_df["commit"].unique())
        val_commits = set(split_result.val_df["commit"].unique())
        test_commits = set(split_result.test_df["commit"].unique())

        train_repos = set(split_result.train_df["repo"].unique())
        val_repos = set(split_result.val_df["repo"].unique())
        test_repos = set(split_result.test_df["repo"].unique())

        overlap_stats = {
            "commit_overlap": {
                "train_val": len(train_commits.intersection(val_commits)),
                "train_test": len(train_commits.intersection(test_commits)),
                "val_test": len(val_commits.intersection(test_commits)),
            },
            "repo_overlap": {
                "train_val": len(train_repos.intersection(val_repos)),
                "train_test": len(train_repos.intersection(test_repos)),
                "val_test": len(val_repos.intersection(test_repos)),
            },
        }

        return overlap_stats

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
