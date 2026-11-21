"""Checkpoint 16: Exact and near-duplicate diffs across repositories."""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets, load_holdout_repos
from software_defects.reporting.reporter import ReportWritter

N_PERMUTATIONS = 128
N_BANDS = 16
ROWS_PER_BAND = N_PERMUTATIONS // N_BANDS
NEAR_JACCARD = 0.8
MIN_CHANGED_LINES = 3
MAX_SHINGLES = 300
MAX_BUCKET_SIZE = 50
MINHASH_PRIME = (1 << 31) - 1
MINHASH_SEED = 7
_WHITESPACE = re.compile(r"\s+")


class DuplicateDiffAnalysis:
    """Checkpoint 16: Exact and near-duplicate diffs across repositories.

    Three keys are compared. ``exact`` hashes the raw diff. ``normalized`` hashes only the added and
    removed lines with whitespace collapsed, so hunk offsets and surrounding context no longer matter.
    ``near`` uses MinHash with banding over the set of changed lines and verifies candidates with the
    exact Jaccard similarity. A duplicate shared between a training and a test repository inflates
    cross-repository scores, so the exposure is reported for every train/test layout used later.
    """

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    @staticmethod
    def changed_lines(diff: str) -> list[str]:
        """Return the normalized added/removed lines of a diff (headers and context dropped)."""
        lines = []
        for raw in str(diff).splitlines():
            if raw.startswith(("+++", "---")) or not raw or raw[0] not in "+-":
                continue
            text = _WHITESPACE.sub(" ", raw[1:]).strip()
            if text:
                lines.append(raw[0] + text)
        return lines

    @staticmethod
    def exact_key(diff: str) -> str:
        """Hash of the raw diff text."""
        return hashlib.sha1(str(diff).encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def normalized_key(lines: list[str]) -> str | None:
        """Hash of the normalized changed lines, or None when the diff has none."""
        if not lines:
            return None
        return hashlib.sha1("\n".join(lines).encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def shingles(lines: list[str]) -> np.ndarray:
        """Bottom-k sketch of the 32-bit hashes of the distinct changed lines."""
        hashes = {zlib.crc32(line.encode("utf-8", errors="ignore")) % MINHASH_PRIME for line in lines}
        return np.array(sorted(hashes)[:MAX_SHINGLES], dtype=np.uint64)

    @staticmethod
    def jaccard(left: np.ndarray, right: np.ndarray) -> float:
        """Exact Jaccard similarity of two shingle arrays."""
        if len(left) == 0 or len(right) == 0:
            return 0.0
        inter = len(np.intersect1d(left, right, assume_unique=True))
        return inter / (len(left) + len(right) - inter)

    @staticmethod
    def cluster_labels(n_rows: int, pairs: list[tuple[int, int]]) -> pd.Series:
        """Connected components of the near-duplicate pairs (union-find).

        Args:
            n_rows: Number of rows.
            pairs: Verified near-duplicate pairs of row positions.

        Returns:
            String cluster id per row, ``<NA>`` for rows without a near-duplicate.
        """
        parent = list(range(n_rows))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for left, right in pairs:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

        in_pair = {node for pair in pairs for node in pair}
        labels = [f"n{find(i)}" if i in in_pair else None for i in range(n_rows)]
        return pd.Series(labels, dtype="string")

    @staticmethod
    def _signature(shingle_array: np.ndarray, coef_a: np.ndarray, coef_b: np.ndarray) -> np.ndarray:
        """MinHash signature of one shingle array."""
        permuted = (shingle_array[:, None] * coef_a[None, :] + coef_b[None, :]) % MINHASH_PRIME
        return permuted.min(axis=0)

    def near_duplicate_pairs(self, shingle_list: list[np.ndarray]) -> tuple[list[tuple[int, int]], int]:
        """Find verified near-duplicate pairs (Jaccard >= NEAR_JACCARD) with MinHash banding.

        Args:
            shingle_list: One shingle array per row (empty arrays are skipped).

        Returns:
            Tuple of (verified pairs of row positions, number of skipped oversized buckets).
        """
        rng = np.random.default_rng(MINHASH_SEED)
        coef_a = rng.integers(1, MINHASH_PRIME, size=N_PERMUTATIONS, dtype=np.uint64)
        coef_b = rng.integers(0, MINHASH_PRIME, size=N_PERMUTATIONS, dtype=np.uint64)

        buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        for position, array in enumerate(shingle_list):
            if len(array) < MIN_CHANGED_LINES:
                continue
            signature = self._signature(array, coef_a, coef_b)
            for band in range(N_BANDS):
                chunk = signature[band * ROWS_PER_BAND : (band + 1) * ROWS_PER_BAND]
                buckets[(band, chunk.tobytes())].append(position)

        pairs: set[tuple[int, int]] = set()
        skipped = 0
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > MAX_BUCKET_SIZE:
                skipped += 1
                continue
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    if (left, right) not in pairs and self.jaccard(
                        shingle_list[left], shingle_list[right]
                    ) >= NEAR_JACCARD:
                        pairs.add((left, right))
        return sorted(pairs), skipped

    @staticmethod
    def _key_summary(df: pd.DataFrame, key_column: str, domain_of: dict[str, str]) -> dict:
        """Group statistics of one duplicate key."""
        keyed = df.dropna(subset=[key_column])
        sizes = keyed.groupby(key_column)["repo"].transform("size")
        dup = keyed[sizes > 1]
        if dup.empty:
            return {"key": key_column, "rows": len(keyed), "duplicate_rows": 0, "duplicate_pct": 0.0}
        grouped = dup.groupby(key_column)
        repos_per_group = grouped["repo"].nunique()
        domains_per_group = grouped["repo"].apply(lambda s: s.map(domain_of).nunique())
        labels_per_group = grouped["label"].nunique()
        return {
            "key": key_column,
            "rows": len(keyed),
            "duplicate_groups": int(grouped.ngroups),
            "duplicate_rows": len(dup),
            "duplicate_pct": len(dup) / len(keyed) * 100,
            "cross_repo_groups": int((repos_per_group > 1).sum()),
            "cross_domain_groups": int((domains_per_group > 1).sum()),
            "label_conflict_groups": int((labels_per_group > 1).sum()),
            "largest_group": int(grouped.size().max()),
        }

    @staticmethod
    def _layouts(project_root: Path, domains: dict[str, set[str]]) -> dict[str, tuple[set[str], set[str]]]:
        """Train/test repository layouts of the experiments (train side, test side)."""
        holdout = load_holdout_repos(project_root)
        ml, non_ml = domains["ml_dl"], domains["non_ml"]
        ml_test, non_ml_test = set(holdout["ml_dl"]), set(holdout["non_ml"])
        return {
            "ML_to_NonML": (ml, non_ml),
            "NonML_to_ML": (non_ml, ml),
            "Repository_Level_Split": ((ml - ml_test) | (non_ml - non_ml_test), ml_test | non_ml_test),
            "Within_Domain_ML_DL": (ml - ml_test, ml_test),
            "Within_Domain_Non_ML": (non_ml - non_ml_test, non_ml_test),
        }

    @staticmethod
    def _exposure_row(df: pd.DataFrame, layout: str, train_repos: set[str], test_repos: set[str], key: str) -> dict:
        """Share of test rows whose key also occurs on the training side."""
        test = df[df["repo"].isin(test_repos)].dropna(subset=[key])
        train_keys = set(df[df["repo"].isin(train_repos)][key].dropna())
        exposed = test[test[key].isin(train_keys)]
        return {
            "layout": layout,
            "key": key,
            "test_rows": len(test),
            "exposed_rows": len(exposed),
            "exposed_pct": len(exposed) / len(test) * 100 if len(test) else 0.0,
            "exposed_defective_pct": float(exposed["label"].mean() * 100) if len(exposed) else float("nan"),
        }

    def run(self) -> None:
        """Execute the duplicate and near-duplicate analysis."""
        self._reporter.display_section("Checkpoint 16: Duplicate and Near-Duplicate Diffs")
        self._reporter.display_text()

        project_root = self._loader.config.project_root
        domains = load_domain_sets(project_root)
        domain_of = {repo: "ml_dl" for repo in domains["ml_dl"]} | {repo: "non_ml" for repo in domains["non_ml"]}

        df = self._loader.load_filtered_jit()
        df = df[df["repo"].isin(domain_of)].reset_index(drop=True)
        self._reporter.display_text(f"Rows analysed (pipeline filtering applied): {len(df):,}")

        line_lists = [self.changed_lines(text) for text in df["content"]]
        df["exact_key"] = [self.exact_key(text) for text in df["content"]]
        df["normalized_key"] = [self.normalized_key(lines) for lines in line_lists]

        summaries = [self._key_summary(df, key, domain_of) for key in ("exact_key", "normalized_key")]

        shingle_list = [self.shingles(lines) for lines in line_lists]
        pairs, skipped = self.near_duplicate_pairs(shingle_list)
        df["near_key"] = self.cluster_labels(len(df), pairs)
        summaries.append(self._key_summary(df, "near_key", domain_of))

        summary_table = pd.DataFrame(summaries)
        self._reporter.display_text(
            f"Near-duplicate pairs (Jaccard >= {NEAR_JACCARD}): {len(pairs):,}; oversized buckets skipped: {skipped:,}"
        )
        self._reporter.display_text()
        self._reporter.display_text("Duplicate groups by key:")
        self._reporter.display_table(summary_table.round(3))
        self._reporter.display_text()

        layouts = self._layouts(project_root, domains)
        exposure_rows = [
            self._exposure_row(df, name, train_repos, test_repos, key)
            for name, (train_repos, test_repos) in layouts.items()
            for key in ("exact_key", "normalized_key", "near_key")
        ]
        exposure_table = pd.DataFrame(exposure_rows)
        self._reporter.display_text("Test rows that also occur on the training side (by experiment layout):")
        self._reporter.display_table(exposure_table.round(3))
        for _, row in exposure_table.iterrows():
            if row["exposed_pct"] >= 1.0:
                self._reporter.display_warning(
                    f"  {row['layout']}: {row['exposed_pct']:.2f}% of test rows have a {row['key']} twin in train"
                )
        self._reporter.display_text()

        summary_path = self._reporter.save_csv(summary_table, "duplicate_key_summary.csv")
        exposure_path = self._reporter.save_csv(exposure_table, "duplicate_split_exposure.csv")
        self._reporter.update_summary(
            {
                "checkpoint_16": {
                    "n_rows": len(df),
                    "near_pairs": len(pairs),
                    "skipped_buckets": skipped,
                    "key_summary": summaries,
                    "key_summary_csv": str(summary_path),
                    "split_exposure_csv": str(exposure_path),
                }
            }
        )
