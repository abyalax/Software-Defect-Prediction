"""Module entry point. Allow ``python -m software_defects`` to run the command-line interface."""

from software_defects.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
