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


def load_holdout_repos(project_root: Path) -> dict[str, list[str]]:
    """Return the explicit test-holdout repositories of each domain.

    Args:
        project_root: Project root that contains the ``artifacts`` directory.

    Returns:
        Dictionary with the keys ``ml_dl`` and ``non_ml`` mapping to sorted repository names.

    Raises:
        ValueError: If the ``holdout_repositories`` section is missing, a domain has no holdout,
            a holdout repository is not mapped to that domain, or a whole domain is held out.
    """
    raw = load_repository_mapping(project_root)
    holdout = raw.get("holdout_repositories")
    if not holdout:
        raise ValueError(f"Section 'holdout_repositories' is missing in {MAPPING_RELATIVE_PATH}")

    holdout_repos: dict[str, list[str]] = {}
    for section, domain in SECTION_TO_DOMAIN.items():
        if domain not in ("ml_dl", "non_ml"):
            continue
        mapped = set((raw.get(section) or {}).keys())
        repos = sorted(set(holdout.get(domain) or []))
        if not repos:
            raise ValueError(f"No holdout repositories configured for domain '{domain}'")
        unknown = sorted(set(repos) - mapped)
        if unknown:
            raise ValueError(f"Holdout repositories not mapped to domain '{domain}': {unknown}")
        if set(repos) == mapped:
            raise ValueError(f"Holdout cannot contain every repository of domain '{domain}'")
        holdout_repos[domain] = repos
    return holdout_repos
