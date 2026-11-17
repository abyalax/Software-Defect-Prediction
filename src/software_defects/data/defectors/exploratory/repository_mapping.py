"""Checkpoint 10: Repository mapping with classification and justification."""

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.data.defectors.domain_mapping import SECTION_TO_DOMAIN, load_repository_mapping
from software_defects.reporting.reporter import ReportWritter


class RepositoryMapping:
    """Checkpoint 10: Repository mapping with classification and justification."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute repository mapping analysis."""
        self._reporter.display_section("Checkpoint 10: Repository Mapping Analysis")
        self._reporter.display_text()

        # Load all JIT data
        jit_train = self._loader.load_jit_train()
        jit_test = self._loader.load_jit_test()
        jit_val = self._loader.load_jit_val()
        all_jit = pd.concat([jit_train, jit_test, jit_val], ignore_index=True)

        # Classifications come from the shared YAML (single source of truth)
        raw_mapping = load_repository_mapping(self._loader.config.project_root)
        repository_classifications = {
            repo: {"domain": domain, "justification": meta.get("criteria", "")}
            for section, domain in SECTION_TO_DOMAIN.items()
            for repo, meta in (raw_mapping.get(section) or {}).items()
        }

        # Get all unique repositories, excluding freqtrade (only 1 record)
        all_repos = sorted([r for r in all_jit["repo"].unique() if r != "freqtrade"])

        # Build mapping data
        mapping_data = []
        for repo in all_repos:
            repo_df = all_jit[all_jit["repo"] == repo]
            classification = repository_classifications.get(
                repo,
                {
                    "domain": "unmapped",
                    "justification": "Not listed in repository_mapping.yaml",
                },
            )

            mapping_data.append(
                {
                    "repository": repo,
                    "domain": classification["domain"],
                    "justification": classification["justification"],
                    "total_samples": len(repo_df),
                    "unique_commits": repo_df["commit"].nunique(),
                }
            )

        mapping_df = pd.DataFrame(mapping_data)
        csv_path = self._reporter.save_csv(mapping_df, "repository_mapping.csv")

        self._reporter.display_text(f"Repository mapping saved to: {csv_path}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_10": {
                    "total_repositories": len(all_repos),
                    "ml_dl_repositories": len(
                        [r for r in all_repos if repository_classifications.get(r, {}).get("domain") == "ml_dl"]
                    ),
                    "non_ml_repositories": len(
                        [r for r in all_repos if repository_classifications.get(r, {}).get("domain") == "non_ml"]
                    ),
                    "csv_path": str(csv_path),
                }
            }
        )
