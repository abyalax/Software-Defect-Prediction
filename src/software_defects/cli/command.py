"""Command CLI for the research project"""

from __future__ import annotations

from collections.abc import Sequence

from software_defects.cli.parser import ParserArgs
from software_defects.cli.runner import RunnerCLI


class CommandCLI:
    """Command dispatcher for CLI."""

    @staticmethod
    def execute(argv: Sequence[str] | None = None) -> int:
        """Execute the selected CLI command."""
        parser = ParserArgs.build_parser()
        arguments = parser.parse_args(argv)

        project_root = arguments.project_root.resolve()

        commands = {
            "inspect-data": lambda: RunnerCLI.run_inspect_data(
                project_root, arguments.dataset, arguments.nrows, arguments.checkpoints
            ),
            "run-pipeline": lambda: RunnerCLI.run_pipeline(
                project_root,
                arguments.dataset,
                arguments.scenario,
                arguments.output_dir,
                arguments.commit_level,
                not arguments.no_tokenize,
                arguments.fold,
            ),
            "run-baseline": lambda: RunnerCLI.run_baseline(
                project_root,
                arguments.dataset,
                arguments.scenario,
                arguments.input_dir,
                [int(seed) for seed in arguments.seeds.split(",")],
                arguments.equalize_train_size,
                arguments.model,
                arguments.exclude_repos,
                arguments.fold,
            ),
            "verify-tokenization": lambda: RunnerCLI.run_verify_tokenization(
                project_root,
                arguments.dataset,
                arguments.scenario,
                arguments.input_dir,
                arguments.fold,
            ),
            "build-split-manifest": lambda: RunnerCLI.run_build_split_manifest(
                project_root,
                arguments.dataset,
                arguments.scenario,
                arguments.fold,
                arguments.manifest_dir,
            ),
            "verify-split-manifest": lambda: RunnerCLI.run_verify_split_manifest(
                project_root,
                arguments.dataset,
                arguments.scenario,
                arguments.fold,
                arguments.manifest_dir,
            ),
        }

        try:
            handler = commands.get(arguments.command)

            if handler is None:
                parser.error(f"Unknown command: {arguments.command}")

            return handler()

        except (OSError, ValueError) as error:
            parser.error(str(error))

        return 2
