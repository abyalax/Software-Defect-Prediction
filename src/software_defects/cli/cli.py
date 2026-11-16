"""Command-line entry points for the LSTM Autoencoder research project."""

from __future__ import annotations

from collections.abc import Sequence

from software_defects.cli.command import CommandCLI


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected CLI command."""
    return CommandCLI.execute(argv)
