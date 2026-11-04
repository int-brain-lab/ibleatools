"""Collect small raw histology files for storage alongside recording snippets."""

import logging
from collections.abc import Sequence
from pathlib import Path

# Keep optional-input messages under the public extractor's logger.
LOGGER = logging.getLogger("ephysatlas.snippet_extractor")
MAX_HISTOLOGY_BYTES = 100 * 1024**2


def _collect_histology_files(paths: Sequence[str | Path]) -> dict[str, Path]:
    """Resolve raw histology paths, checking stored names and the total size limit."""
    histology_files = {}  # name stored under histology/raw/ -> resolved source path
    for path in map(Path, paths):
        if not path.exists():
            raise FileNotFoundError(f"Histology path not found: {path}")
        if path.is_dir():
            # path.name is empty for "." and "..", so those use the resolved name.
            folder_name = path.name or path.resolve().name
            members = {
                f"{folder_name}/{p.relative_to(path).as_posix()}": p
                for p in sorted(path.rglob("*"))
                if p.is_file()
            }
            if not members:
                LOGGER.warning("Histology folder %s contains no files", path)
        else:
            members = {path.name: path}
        for hist_name, member in members.items():
            member = member.resolve()
            # Only a path given twice is skipped. A file that is also inside a listed folder
            # is stored twice, under two names.
            if hist_name in histology_files and histology_files[hist_name] != member:
                raise ValueError(
                    f"Two histology files would both be stored as histology/raw/{hist_name}: "
                    f"{histology_files[hist_name]} and {member}. Pass only one of them."
                )
            histology_files[hist_name] = member
    histology_sizes = {name: p.stat().st_size for name, p in histology_files.items()}
    histology_total = sum(histology_sizes.values())
    if histology_files:
        LOGGER.info(
            "Storing %d histology files (%.2f MB): %s",
            len(histology_files),
            histology_total / 1024**2,
            ", ".join(histology_files),
        )
    if histology_total > MAX_HISTOLOGY_BYTES:
        largest = sorted(histology_sizes, key=histology_sizes.get, reverse=True)[:3]
        raise ValueError(
            f"The histology files total {histology_total / 1024**2:.0f} MB, "
            f"over the {MAX_HISTOLOGY_BYTES // 1024**2} MB limit. Largest: "
            + ", ".join(
                f"{name} ({histology_sizes[name] / 1024**2:.0f} MB)" for name in largest
            )
            + ". Leave out image volumes and pass only the small channel location, "
            "pick and track files."
        )
    return histology_files
