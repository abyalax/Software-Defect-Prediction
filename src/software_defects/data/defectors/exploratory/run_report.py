"""Checkpoint 14: Run report ."""

import subprocess
from datetime import datetime

import pandas as pd

from software_defects.data.defectors.dataset_loader import DefectorsDatasetLoader
from software_defects.reporting.reporter import ReportWritter


class RunReport:
    """Checkpoint 14: Run report ."""

    def __init__(self, loader: DefectorsDatasetLoader, reporter: ReportWritter) -> None:
        self._loader = loader
        self._reporter = reporter

    def run(self) -> None:
        """Execute run report generation."""
        self._reporter.display_section("Checkpoint 14: Run Report")
        self._reporter.display_text()

        # Compile warnings from all checkpoints
        warnings = self._reporter.get_warnings()

        self._reporter.display_text(f"Total warnings collected: {len(warnings)}")
        self._reporter.display_text()

        # Structured warnings
        structured_warnings = []
        for i, warning in enumerate(warnings, 1):
            structured_warnings.append(
                {
                    "id": f"W{i:03d}",
                    "severity": "warning",
                    "message": warning,
                }
            )

        # Metadata
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_commit = "unknown"

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "library_versions": {
                "pandas": pd.__version__,
                "numpy": "unknown",  # Would need to import numpy
            },
            "total_warnings": len(warnings),
        }

        # Combined summary
        combined_report = {
            "metadata": metadata,
            "warnings": structured_warnings,
            "checkpoint_summary": self._reporter.summary,
        }

        # Save as JSON
        json_path = self._reporter.save_json(combined_report, "combined_run_report.json")
        self._reporter.display_text(f"Combined run report saved to: {json_path}")
        self._reporter.display_text()

        # Generate Markdown summary
        markdown_lines = [
            "# Defectors Dataset Validation Report",
            "",
            f"Generated: {metadata['timestamp']}",
            f"Git Commit: {git_commit}",
            "",
            "## Summary",
            "",
            f"- Total Warnings: {len(warnings)}",
            "",
            "## Warnings",
            "",
        ]

        for warning in structured_warnings:
            markdown_lines.append(f"- **{warning['id']}**: {warning['message']}")

        markdown_lines.extend(
            [
                "",
                "## Checkpoint Summary",
                "",
                "See the JSON report for detailed checkpoint results.",
            ]
        )

        markdown_content = "\n".join(markdown_lines)
        md_path = self._reporter.save_json(markdown_content, "combined_run_report.md")
        # Save as markdown (using save_json but with .md extension)
        md_path_actual = str(md_path).replace(".json", ".md")
        with open(md_path_actual, "w") as f:
            f.write(markdown_content)

        self._reporter.display_text(f"Markdown report saved to: {md_path_actual}")
        self._reporter.display_text()

        # Update summary
        self._reporter.update_summary(
            {
                "checkpoint_14": {
                    "total_warnings": len(warnings),
                    "combined_report_json": str(json_path),
                    "combined_report_md": md_path_actual,
                }
            }
        )
