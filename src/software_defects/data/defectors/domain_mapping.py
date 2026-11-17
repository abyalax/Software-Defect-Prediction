"""Single source of truth for the repository -> domain mapping."""

from __future__ import annotations

from pathlib import Path

import yaml

MAPPING_RELATIVE_PATH = Path("artifacts") / "mapping" / "repository_mapping.yaml"

SECTION_TO_DOMAIN = {
    "ml_dl_repositories": "ml_dl",
    "non_ml_repositories": "non_ml",
    "ambiguous_repositories": "ambiguous",
    "unmapped_repositories": "unmapped",
}


def load_repository_mapping(project_root: Path) -> dict:
    """Load the raw repository mapping YAML.

    Args:
        project_root: Project root that contains the ``artifacts`` directory.

    Returns:
        Parsed YAML as a dictionary (section name -> repo name -> metadata).

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    path = Path(project_root) / MAPPING_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Repository mapping not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_domain_sets(project_root: Path) -> dict[str, set[str]]:
    """Return repository names grouped by domain.

    Args:
        project_root: Project root that contains the ``artifacts`` directory.

    Returns:
        Dictionary with the keys ``ml_dl``, ``non_ml``, ``ambiguous`` and ``unmapped``.
    """
    raw = load_repository_mapping(project_root)
    return {domain: set((raw.get(section) or {}).keys()) for section, domain in SECTION_TO_DOMAIN.items()}
