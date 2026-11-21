"""Checkpoint 15: How easily the origin repository can be guessed from the diff."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.reporting.reporter import ReportWritter

N_FOLDS = 5
MAX_ROWS_PER_REPO = 3000
MAX_CHARS = 4000
MAX_FEATURES = 50000
CODE_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]+"
PATH_TOKEN_PATTERN = r"[A-Za-z][A-Za-z0-9]+"
SAMPLE_SEED = 42
# Repository accuracy this many times above the majority-class rate marks the origin as easy to guess
EASY_LIFT = 3.0


class RepositoryIdentityProbe:
    """Checkpoint 15: How easily the origin repository can be guessed from the diff.

    A model that can name the repository from the diff text can also pick up repository-specific
    style, which makes a cross-domain gap hard to separate from a repository-recognition effect.
    Two probes are run on a per-repository capped sample:

    * repository identification, grouped by commit so the files of one commit never straddle
      train and test;
    * domain identification (ML/DL vs non-ML), grouped by repository so every test repository
      is unseen. High accuracy means the domain boundary is visible in the text itself.
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        """Cap the rows per repository so large repositories do not dominate the probe."""
        parts = [
            group.sample(n=min(len(group), MAX_ROWS_PER_REPO), random_state=SAMPLE_SEED)
            for _, group in df.groupby("repo")
        ]
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

    @staticmethod
    def _vectorizer(token_pattern: str) -> TfidfVectorizer:
        return TfidfVectorizer(
            token_pattern=token_pattern,
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=MAX_FEATURES,
        )

    def _predict_repo(
        self, texts: pd.Series, repos: np.ndarray, commit_groups: np.ndarray, token_pattern: str
    ) -> np.ndarray:
        """Out-of-fold repository predictions, grouped by commit."""
        n_folds = min(N_FOLDS, len(np.unique(commit_groups)))
        predictions = np.empty(len(texts), dtype=object)
        for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(texts, repos, commit_groups):
            vectorizer = self._vectorizer(token_pattern)
            train_x = vectorizer.fit_transform(texts.iloc[train_idx])
            if len(np.unique(repos[train_idx])) < 2:
                predictions[test_idx] = repos[train_idx][0]
                continue
            model = LinearSVC(C=0.5)
            model.fit(train_x, repos[train_idx])
            predictions[test_idx] = model.predict(vectorizer.transform(texts.iloc[test_idx]))
        return predictions

    def _repo_probe(self, df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
        """Repository identification from the diff text and from the file path."""
        repos = df["repo"].to_numpy()
        commit_groups = (df["repo"] + "::" + df["commit"].astype(str)).to_numpy()
        majority = float(pd.Series(repos).value_counts(normalize=True).iloc[0])
        inputs = {
            "diff_content": (df["content"].fillna("").astype(str).str.slice(0, MAX_CHARS), CODE_TOKEN_PATTERN),
            "file_path": (df["filepath"].fillna("").astype(str), PATH_TOKEN_PATTERN),
        }

        rows: list[dict] = []
        recall_frames: list[pd.DataFrame] = []
        for name, (texts, pattern) in inputs.items():
            predicted = self._predict_repo(texts, repos, commit_groups, pattern)
            accuracy = float(accuracy_score(repos, predicted))
            rows.append(
                {
                    "probe": f"repo_from_{name}",
                    "n_rows": len(df),
                    "n_classes": int(len(np.unique(repos))),
                    "majority_rate": majority,
                    "accuracy": accuracy,
                    "accuracy_lift": accuracy / majority,
                    "macro_f1": float(f1_score(repos, predicted, average="macro", zero_division=0)),
                }
            )
            recall = (
                pd.DataFrame({"repo": repos, "hit": predicted == repos})
                .groupby("repo")["hit"]
                .agg(n_rows="count", recall="mean")
                .reset_index()
                .assign(probe=f"repo_from_{name}")
            )
            recall_frames.append(recall)
        return rows, pd.concat(recall_frames, ignore_index=True)

    def _domain_probe(self, df: pd.DataFrame, domains: dict[str, set[str]]) -> dict:
        """Domain identification from the diff text on repositories never seen in training."""
        domain_label = df["repo"].isin(domains["ml_dl"]).astype(int).to_numpy()
        repos = df["repo"].to_numpy()
        texts = df["content"].fillna("").astype(str).str.slice(0, MAX_CHARS)
        n_folds = min(N_FOLDS, len(np.unique(repos)))
        scores = np.full(len(df), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(texts, domain_label, repos):
            if len(np.unique(domain_label[train_idx])) < 2:
                continue
            vectorizer = self._vectorizer(CODE_TOKEN_PATTERN)
            model = LinearSVC(C=0.5, class_weight="balanced")
            model.fit(vectorizer.fit_transform(texts.iloc[train_idx]), domain_label[train_idx])
            scores[test_idx] = model.decision_function(vectorizer.transform(texts.iloc[test_idx]))

        valid = ~np.isnan(scores)
        if valid.sum() == 0 or len(np.unique(domain_label[valid])) < 2:
            return {}
        predicted = (scores[valid] > 0).astype(int)
        truth = domain_label[valid]
        recall_ml = float((predicted[truth == 1] == 1).mean())
        recall_non_ml = float((predicted[truth == 0] == 0).mean())
        return {
            "probe": "domain_from_diff_content_unseen_repos",
            "n_rows": int(valid.sum()),
            "ml_dl_share": float(truth.mean()),
            "roc_auc": float(roc_auc_score(truth, scores[valid])),
            "balanced_accuracy": (recall_ml + recall_non_ml) / 2,
            "recall_ml_dl": recall_ml,
            "recall_non_ml": recall_non_ml,
        }

    def run(self) -> None:
        """Execute the repository and domain identity probes."""
        self._reporter.display_section("Checkpoint 15: Repository Identity Probe")
        self._reporter.display_text()

        df = self._loader.load_filtered_jit()
        domains = load_domain_sets(self._loader.config.project_root)
        df = df[df["repo"].isin(domains["ml_dl"] | domains["non_ml"])]
        sample = self._sample(df)
        self._reporter.display_text(
            f"Sample: {len(sample):,} rows from {sample['repo'].nunique()} repositories "
            f"(cap {MAX_ROWS_PER_REPO:,} per repo)"
        )
        self._reporter.display_text()

        rows, recall_table = self._repo_probe(sample)
        repo_table = pd.DataFrame(rows)
        self._reporter.display_text(f"Repository identification ({N_FOLDS}-fold grouped by commit):")
        self._reporter.display_table(repo_table.round(4))
        for _, row in repo_table.iterrows():
            if row["accuracy_lift"] >= EASY_LIFT:
                self._reporter.display_warning(
                    f"  Repository is easy to guess ({row['probe']}): accuracy {row['accuracy']:.3f} vs majority "
                    f"{row['majority_rate']:.3f}"
                )
        self._reporter.display_text()

        domain_row = self._domain_probe(sample, domains)
        domain_table = pd.DataFrame([domain_row]) if domain_row else pd.DataFrame()
        self._reporter.display_text(f"Domain identification on unseen repositories ({N_FOLDS}-fold grouped by repo):")
        if domain_table.empty:
            self._reporter.display_warning("  Domain probe could not be evaluated")
        else:
            self._reporter.display_table(domain_table.round(4))
            if domain_row["roc_auc"] >= 0.85:
                self._reporter.display_warning(
                    f"  Domain is visible in the diff text itself (ROC-AUC {domain_row['roc_auc']:.3f}); "
                    "a cross-domain gap may reflect style differences, not only defect semantics"
                )
        self._reporter.display_text()

        probe_path = self._reporter.save_csv(
            pd.concat([repo_table, domain_table], ignore_index=True), "repository_identity_probe.csv"
        )
        recall_path = self._reporter.save_csv(recall_table, "repository_identity_recall.csv")
        self._reporter.display_text(f"Identity probe saved to: {probe_path}")
        self._reporter.display_text()

        self._reporter.update_summary(
            {
                "checkpoint_15": {
                    "sample_rows": len(sample),
                    "repo_probe": rows,
                    "domain_probe": domain_row,
                    "probe_csv": str(probe_path),
                    "recall_csv": str(recall_path),
                }
            }
        )
