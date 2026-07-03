"""Provenance helpers for feature calculator outputs.

This module records both Python package versions and editable-install git state
for the ``ibleatools`` repository. The metadata is intended to be attached to
newly computed per-feature parquet files, so different feature families in the
same snippet directory can retain different provenance.

Functions
---------
collect_ibleatools_provenance
    Build a JSON-friendly provenance dictionary for one calculator run.
log_reproduction_command
    Log the git command that reproduces the recorded editable commit.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import ephysatlas

LOGGER = logging.getLogger(__name__)


def _run_git(repo_path: Path, args: list[str]) -> str | None:
    """Run a read-only git command in ``repo_path``.

    Args:
        repo_path (Path): Repository path.
        args (list[str]): Git arguments after ``git``.

    Returns:
        str | None: Stripped command output, or ``None`` when git is not
            available or the command fails.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        LOGGER.debug("Failed to run git %s in %s", " ".join(args), repo_path)
        return None
    return completed.stdout.strip()


def _editable_path_from_direct_url(direct_url_text: str | None) -> Path | None:
    """Extract the editable source path from ``direct_url.json`` text.

    Args:
        direct_url_text (str | None): Contents of an installed distribution's
            ``direct_url.json`` metadata file.

    Returns:
        Path | None: Editable local path, or ``None`` when the distribution is
            not editable or cannot be resolved to a local path.
    """
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    if not direct_url.get("dir_info", {}).get("editable", False):
        return None

    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    return None


def _git_provenance(repo_path: Path) -> dict[str, Any]:
    """Return git metadata for an editable checkout.

    Args:
        repo_path (Path): Local editable repository path.

    Returns:
        dict[str, Any]: Git commit, branch, and dirty-worktree metadata.
    """
    status = _run_git(repo_path, ["status", "--porcelain"])
    return {
        "ibleatools_git_repo_path": repo_path.as_posix(),
        "ibleatools_git_commit_hash": _run_git(repo_path, ["rev-parse", "HEAD"]),
        "ibleatools_git_branch": _run_git(repo_path, ["branch", "--show-current"]),
        "ibleatools_git_is_dirty": bool(status),
    }


def collect_ibleatools_provenance(
    calculator_name: str,
    feature_names: tuple[str, ...],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect version and git provenance for one calculator run.

    Args:
        calculator_name (str): Concrete calculator class name.
        feature_names (tuple[str, ...]): Feature families requested for this
            computation.
        extra (Mapping[str, Any], optional): Additional JSON-friendly metadata to
            include in the result.

    Returns:
        dict[str, Any]: Provenance metadata suitable for ``DataFrame.attrs``.

    Note:
        Editable install detection uses the package ``direct_url.json`` metadata
        defined by PEP 610. When the install is editable and local, git metadata
        is collected with read-only git commands.
    """
    provenance: dict[str, Any] = {
        "feature_calculator_class": calculator_name,
        "feature_calculator_features": list(feature_names),
        "feature_calculator_created_utc": datetime.now(timezone.utc).isoformat(),
        "ephysatlas_version": ephysatlas.__version__,
    }

    try:
        dist = metadata.distribution("ibleatools")
        provenance["ibleatools_distribution_version"] = dist.version
        direct_url_text = dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        provenance["ibleatools_distribution_version"] = None
        direct_url_text = None

    editable_path = _editable_path_from_direct_url(direct_url_text)
    provenance["ibleatools_is_editable_install"] = editable_path is not None
    if editable_path is not None:
        provenance.update(_git_provenance(editable_path))

    if extra:
        provenance.update(dict(extra))

    return provenance


def log_reproduction_command(provenance: Mapping[str, Any]) -> None:
    """Log a command that recreates the recorded editable checkout.

    Args:
        provenance (Mapping[str, Any]): Provenance dictionary returned by
            :func:`collect_ibleatools_provenance`.

    Note:
        When the editable checkout is dirty, the commit hash is still useful but
        does not fully reproduce uncommitted changes. The log message makes that
        limitation explicit.
    """
    commit_hash = provenance.get("ibleatools_git_commit_hash")
    repo_path = provenance.get("ibleatools_git_repo_path")
    if not commit_hash or not repo_path:
        LOGGER.info(
            "Feature provenance did not include an editable git checkout; "
            "reproduce by installing ibleatools version %s",
            provenance.get("ibleatools_distribution_version"),
        )
        return

    LOGGER.info(
        "Reproduce ibleatools code state with: cd %s && git checkout %s && "
        "python -m pip install -e .",
        repo_path,
        commit_hash,
    )
    if provenance.get("ibleatools_git_is_dirty"):
        LOGGER.warning(
            "The editable ibleatools checkout had uncommitted changes; commit %s "
            "alone will not fully reproduce the calculated features.",
            commit_hash,
        )
