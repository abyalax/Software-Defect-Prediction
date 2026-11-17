"""Checkpoint 13: Shortcut signal check ."""

import pandas as pd
from sklearn.metrics import roc_auc_score

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import load_domain_sets
from software_defects.reporting.reporter import ReportWritter


class ShortcutSignalCheck:
    """Checkpoint 13: Shortcut signal check ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute shortcut signal check."""
        self._reporter.display_section("Checkpoint 13: Shortcut Signal Check")
        self._reporter.display_text()

        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()

        # Calculate labels
        if "lines" in jit_train.columns:
            sample_lines = jit_train["lines"].iloc[0]
            for df in [jit_train, jit_test, jit_val]:
                df["label"] = df["lines"].apply(
                    lambda x: 1 if len(x) > 0 else 0 if isinstance(x, (list, type(sample_lines))) else 0
                )

        # Calculate features
        for df in [jit_train, jit_test, jit_val]:
            df["diff_length"] = df["content"].apply(lambda x: len(str(x)) if x is not None else 0)
            df["added_lines"] = df["content"].apply(lambda x: str(x).count("\n+") if x is not None else 0)
            df["removed_lines"] = df["content"].apply(lambda x: str(x).count("\n-") if x is not None else 0)
            df["num_hunks"] = df["content"].apply(lambda x: str(x).count("@@") if x is not None else 0)

        # AUC per feature per split
        self._reporter.display_text("AUC for Single Features (Predicting Label):")

        features = ["diff_length", "added_lines", "removed_lines", "num_hunks"]
        auc_results = []

        for split_name, df in [("Train", jit_train), ("Test", jit_test), ("Val", jit_val)]:
            self._reporter.display_text(f"  {split_name}:")
            for feature in features:
                try:
                    auc = roc_auc_score(df["label"], df[feature])
                    self._reporter.display_text(f"    {feature}: {auc:.4f}")
                    auc_results.append({"split": split_name, "feature": feature, "auc": float(auc)})
                except Exception as e:
                    self._reporter.display_text(f"    {feature}: Error ({e})")
            self._reporter.display_text()

        # Save AUC results
        auc_df = pd.DataFrame(auc_results)
        csv_path = self._reporter.save_csv(auc_df, "shortcut_signal_auc.csv")
        self._reporter.display_text(f"AUC results saved to: {csv_path}")
        self._reporter.display_text()

        # AUC per domain
        if "repo" in jit_train.columns:
            self._reporter.display_text("AUC per Domain (Train):")

            # Load domain mapping from the shared YAML (single source of truth)
            domain_sets = load_domain_sets(self._loader.config.project_root)
            ml_dl_repos = domain_sets["ml_dl"]
            non_ml_repos = domain_sets["non_ml"]

            ml_dl_df = jit_train[jit_train["repo"].isin(ml_dl_repos)]
            non_ml_df = jit_train[jit_train["repo"].isin(non_ml_repos)]

            for domain_name, domain_df in [("ML/DL", ml_dl_df), ("Non-ML", non_ml_df)]:
                if len(domain_df) > 0:
                    self._reporter.display_text(f"  {domain_name}:")
                    for feature in features:
                        try:
                            auc = roc_auc_score(domain_df["label"], domain_df[feature])
                            self._reporter.display_text(f"    {feature}: {auc:.4f}")
                        except Exception:
                            pass
            self._reporter.display_text()

        # File extension impact on prevalence
        if "filepath" in jit_train.columns:
            self._reporter.display_text("Label Prevalence by File Extension (Train):")

            def get_extension(filepath: str) -> str:
                if not filepath:
                    return "unknown"
                parts = filepath.rsplit(".", 1)
                return parts[-1].lower() if len(parts) > 1 else "no_ext"

            jit_train["extension"] = jit_train["filepath"].apply(get_extension)
            ext_stats = (
                jit_train.groupby("extension")["label"].agg(["mean", "count"]).sort_values("mean", ascending=False)
            )
            ext_stats.columns = ["prevalence", "count"]

            for ext, row in ext_stats.head(10).iterrows():
                self._reporter.display_text(f"  .{ext}: {row['prevalence']:.4f} (n={row['count']:,})")
            self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_13": {
                    "feature_auc_csv": str(csv_path),
                    "max_diff_length_auc": max([r["auc"] for r in auc_results if r["feature"] == "diff_length"])
                    if auc_results
                    else 0,
                }
            }
        )
