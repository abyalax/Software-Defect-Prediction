"""Checkpoint 14: Shortcut probing on file path, files per commit and commit time."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.reporting.reporter import ReportWritter

N_FOLDS = 5
MIN_ROWS_PER_REPO = 50
PATH_TOKEN_PATTERN = r"[A-Za-z][A-Za-z0-9]+"
# A single feature with a larger edge than this is reported as a shortcut candidate
SHORTCUT_AUC_EDGE = 0.10
NUMERIC_FEATURES = [
    "path_depth",
    "path_length",
    "filename_length",
    "is_test_file",
    "is_init_file",
    "files_in_commit",
    "commit_day",
    "commit_hour",
    "commit_weekday",
]


class PathMetadataProbe:
    """Checkpoint 14: Shortcut probing on file path, files per commit and commit time.

    The path is part of the Transformer input, so its predictive power must be known before the
    models are trained. Two views are reported for every signal: the pooled ROC-AUC and the
    within-repository ROC-AUC. The pooled value mixes in the prevalence difference between
    repositories; the within-repository value isolates the signal itself.
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add path, files-per-commit and commit-time features."""
        df = df.copy()
        paths = df["filepath"].fillna("").astype(str)
        names = paths.str.rsplit("/", n=1).str[-1]
        lowered = paths.str.lower()

        df["path_depth"] = paths.str.count("/") + 1
        df["path_length"] = paths.str.len()
        df["filename_length"] = names.str.len()
        df["is_test_file"] = (
            lowered.str.contains(r"(?:^|/)tests?/|(?:^|/)test_|_test\.py$|conftest", regex=True)
        ).astype(int)
        df["is_init_file"] = (names == "__init__.py").astype(int)
        df["files_in_commit"] = df.groupby(["repo", "commit"])["filepath"].transform("count")

        moment = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["commit_day"] = (moment - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / 86400.0
        df["commit_hour"] = moment.dt.hour
        df["commit_weekday"] = moment.dt.day_of_week
        df["commit_year"] = moment.dt.year
        return df

    @staticmethod
    def _safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
        """ROC-AUC, or NaN when only one class (or no valid score) is present."""
        labels = np.asarray(labels)
        scores = np.asarray(scores, dtype=float)
        valid = ~np.isnan(scores)
        if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
            return float("nan")
        return float(roc_auc_score(labels[valid], scores[valid]))

    def _within_repo_auc(self, df: pd.DataFrame, score_column: str) -> float:
        """Row-weighted mean of the per-repository ROC-AUC (repositories with both classes only)."""
        values: list[float] = []
        weights: list[int] = []
        for _, group in df.groupby("repo"):
            if len(group) < MIN_ROWS_PER_REPO:
                continue
            auc = self._safe_auc(group["label"], group[score_column])
            if not np.isnan(auc):
                values.append(auc)
                weights.append(len(group))
        if not values:
            return float("nan")
        return float(np.average(values, weights=weights))

    def _numeric_feature_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pooled and within-repository ROC-AUC of every numeric signal."""
        rows = []
        for feature in NUMERIC_FEATURES:
            pooled = self._safe_auc(df["label"], df[feature])
            within = self._within_repo_auc(df, feature)
            rows.append(
                {
                    "feature": feature,
                    "pooled_auc": pooled,
                    "within_repo_auc": within,
                    "within_repo_edge": abs(within - 0.5) if not np.isnan(within) else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def _out_of_fold_text_scores(self, df: pd.DataFrame, text: pd.Series) -> np.ndarray:
        """Cross-repository out-of-fold scores of a TF-IDF + Logistic Regression text probe."""
        groups = df["repo"].to_numpy()
        labels = df["label"].to_numpy()
        n_folds = min(N_FOLDS, len(np.unique(groups)))
        scores = np.full(len(df), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(df, labels, groups):
            vectorizer = TfidfVectorizer(token_pattern=PATH_TOKEN_PATTERN, lowercase=True, sublinear_tf=True)
            train_x = vectorizer.fit_transform(text.iloc[train_idx])
            if len(np.unique(labels[train_idx])) < 2:
                continue
            model = LogisticRegression(max_iter=300, class_weight="balanced")
            model.fit(train_x, labels[train_idx])
            scores[test_idx] = model.predict_proba(vectorizer.transform(text.iloc[test_idx]))[:, 1]
        return scores

    def _text_probe_rows(self, df: pd.DataFrame) -> list[dict]:
        """Run the path-only and filename-only probes under repository-grouped cross-validation."""
        filenames = df["filepath"].fillna("").astype(str).str.rsplit("/", n=1).str[-1]
        probes = {"path_tokens": df["filepath"].fillna("").astype(str), "filename_tokens": filenames}
        rows = []
        for name, text in probes.items():
            scores = self._out_of_fold_text_scores(df, text)
            scored = df.assign(probe_score=scores).dropna(subset=["probe_score"])
            if scored.empty:
                continue
            prevalence = float(scored["label"].mean())
            pr_auc = float(average_precision_score(scored["label"], scored["probe_score"]))
            rows.append(
                {
                    "probe": name,
                    "n_rows": len(scored),
                    "prevalence": prevalence,
                    "pr_auc": pr_auc,
                    "pr_auc_lift": pr_auc / prevalence if prevalence > 0 else float("nan"),
                    "pooled_auc": self._safe_auc(scored["label"], scored["probe_score"]),
                    "within_repo_auc": self._within_repo_auc(scored, "probe_score"),
                }
            )
        return rows

    def _prevalence_by_year(self, df: pd.DataFrame, domains: dict[str, set[str]]) -> pd.DataFrame:
        """Label prevalence per commit year and domain, to expose temporal drift."""
        domain_of = {repo: "ml_dl" for repo in domains["ml_dl"]} | {repo: "non_ml" for repo in domains["non_ml"]}
        frame = df.assign(domain=df["repo"].map(domain_of)).dropna(subset=["commit_year", "domain"])
        table = frame.groupby(["domain", "commit_year"])["label"].agg(n_rows="count", prevalence="mean").reset_index()
        table["commit_year"] = table["commit_year"].astype(int)
        return table

    def _files_in_commit_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label prevalence per bucket of files changed in the same commit."""
        bins = [0, 1, 2, 3, 5, 10, 25, np.inf]
        labels = ["1", "2", "3", "4-5", "6-10", "11-25", ">25"]
        bucket = pd.cut(df["files_in_commit"], bins=bins, labels=labels)
        table = df.groupby(bucket, observed=True)["label"].agg(n_rows="count", prevalence="mean").reset_index()
        return table.rename(columns={"files_in_commit": "files_in_commit_bucket"})

    def run(self) -> None:
        """Execute the path, file-count and commit-time probes."""
        self._reporter.display_section("Checkpoint 14: Path, Files-per-Commit and Commit-Time Shortcut Probe")
        self._reporter.display_text()

        df = self._add_features(self._loader.load_filtered_jit())
        domains = load_domain_sets(self._loader.config.project_root)
        df = df[df["repo"].isin(domains["ml_dl"] | domains["non_ml"])].copy()
        self._reporter.display_text(f"Rows analysed (pipeline filtering applied): {len(df):,}")
        self._reporter.display_text(f"Overall prevalence: {df['label'].mean():.4f}")
        self._reporter.display_text()

        feature_table = self._numeric_feature_table(df)
        self._reporter.display_text("Single-signal ROC-AUC (0.5 = no signal):")
        self._reporter.display_table(feature_table.round(4))
        feature_path = self._reporter.save_csv(feature_table, "path_metadata_signal_auc.csv")
        candidates = feature_table[feature_table["within_repo_edge"] >= SHORTCUT_AUC_EDGE]
        for _, row in candidates.iterrows():
            self._reporter.display_warning(
                f"  Shortcut candidate: {row['feature']} within-repo AUC {row['within_repo_auc']:.3f}"
            )
        self._reporter.display_text()

        text_table = pd.DataFrame(self._text_probe_rows(df))
        self._reporter.display_text(f"Path text probes (TF-IDF + Logistic Regression, {N_FOLDS}-fold grouped by repo):")
        if text_table.empty:
            self._reporter.display_warning("  No text probe could be evaluated")
        else:
            self._reporter.display_table(text_table.round(4))
            for _, row in text_table.iterrows():
                if row["pr_auc_lift"] >= 1.25:
                    self._reporter.display_warning(
                        f"  Path signal: {row['probe']} reaches PR-AUC {row['pr_auc']:.3f} "
                        f"({row['pr_auc_lift']:.2f}x prevalence) on unseen repositories"
                    )
        text_path = self._reporter.save_csv(text_table, "path_text_probe.csv")
        self._reporter.display_text()

        bins_table = self._files_in_commit_bins(df)
        self._reporter.display_text("Prevalence by files changed in the same commit:")
        self._reporter.display_table(bins_table.round(4))
        bins_path = self._reporter.save_csv(bins_table, "files_in_commit_prevalence.csv")
        self._reporter.display_text()

        year_table = self._prevalence_by_year(df, domains)
        self._reporter.display_text("Prevalence by commit year and domain:")
        self._reporter.display_table(year_table.round(4))
        year_path = self._reporter.save_csv(year_table, "commit_year_prevalence.csv")
        self._reporter.display_text()

        self._reporter.update_summary(
            {
                "checkpoint_14": {
                    "n_rows": len(df),
                    "shortcut_candidates": candidates["feature"].tolist(),
                    "signal_auc_csv": str(feature_path),
                    "path_text_probe_csv": str(text_path),
                    "files_in_commit_csv": str(bins_path),
                    "commit_year_csv": str(year_path),
                }
            }
        )
