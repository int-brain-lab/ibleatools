"""Write the QC labels at one level as the package fixture ``ephysatlas.fixtures.pid_qc_filter``."""

from __future__ import annotations

import datetime
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd

from .levels import BAD_GROUPS, LABELS, LEVELS

BAD, GOOD = "bad", "good"
WIDTH = 99  # of the generated docstring

# The paragraphs of the generated module's docstring (refilled to WIDTH).
DOCSTRING = (
    """Quality-control filter of the Ephys Atlas insertions -- GENERATED, do not edit by hand.""",
    """Built by ``scripts/build_pid_qc_filter.py`` ({generated}, ibleatools {commit}) from feature
    release {vintage}, at level {level} of ``ephysatlas.qc.levels``: {n_total} insertions, {n_bad}
    with at least one bad component ({by_type}), {n_review} to review.""",
    """An insertion is bad at {level} when it fails at least {rule}. The failure cases are
    ``ephysatlas.qc.levels.TESTS``: each compares the probe's recording with what the other probes
    record around the same positions, or its signal with the signals of all probes; no model,
    auto-alignment or manual label decides. Each component -- ``alignment``, ``lf``, ``ap``,
    ``waveform`` -- is ``bad`` when the failures at {level} point to it, {review_rule}, else
    ``good``. Every insertion that is not good on all four is listed with its labels and the
    failure cases that fired ({checks_rule}). Unlisted insertions are good.""",
    """Use :func:`excluded_pids` for the insertions to leave out of a dataset, e.g.
    ``excluded_pids()`` (bad on any component) for the channel-level model, or
    ``excluded_pids(("alignment", "ap", "waveform"))`` for the unit-level model, and pass them as
    ``exclude_pids`` to the data loaders (``spatial_encoder.utils.LoadInsertionData``,
    ``alignment.ChannelDataset.load``,
    ``unit_level_encoder.prepare_data.prepare_latest_cells_encoder_data``), which use exactly
    these by default.""",
)

TEMPLATE = '''"""{doc}
"""

VINTAGE = "{vintage}"
LEVEL = "{level}"
COMPONENTS = ("alignment", "lf", "ap", "waveform")

# pid: (alignment, lf, ap, waveform, failure cases that fired)
pid_qc_filter = {{
{entries}
}}


def qc_labels(pid: str) -> dict:
    """``{{component: label}}`` of an insertion (all ``good`` when unlisted) and ``checks``."""
    entry = pid_qc_filter.get(str(pid))
    if entry is None:
        return {{**{{c: "good" for c in COMPONENTS}}, "checks": ""}}
    return {{**dict(zip(COMPONENTS, entry[:4])), "checks": entry[4]}}


def excluded_pids(components=COMPONENTS, include_review: bool = False) -> list:
    """Insertions bad (or, with ``include_review``, also to review) on any of ``components``."""
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {{sorted(unknown)}}; choose from {{COMPONENTS}}")
    drop = ("bad", "review") if include_review else ("bad",)
    idx = [COMPONENTS.index(c) for c in components]
    return sorted(p for p, e in pid_qc_filter.items() if any(e[i] in drop for i in idx))
'''


def _next_level(level: str) -> Optional[str]:
    names = [lv.name for lv in LEVELS]
    if level not in names:
        raise ValueError(f"unknown level {level!r}; choose from {names}")
    k = names.index(level)
    return names[k + 1] if k + 1 < len(names) else None


def fixture_labels(table: pd.DataFrame, level: str = "L3") -> pd.DataFrame:
    """The insertions of a :func:`~.levels.level_table` that are not good on every component at
    ``level``, sorted: the label of each component (``bad`` at the level, ``review`` when bad
    only at the next level, else ``good``) and ``checks``, the failure cases that fired at the
    level (for an insertion only to review, at the next level)."""
    nxt = _next_level(level)
    labels = table[[f"{level}_{c}_label" for c in LABELS]].set_axis(
        list(LABELS), axis=1
    )
    listed = labels.ne(GOOD).any(axis=1)
    checks = table[f"{level}_failed"].fillna("").astype(str)
    if nxt is not None:
        review_only = ~table[f"{level}_bad"].astype(bool)
        checks = checks.where(
            ~review_only, table[f"{nxt}_failed"].fillna("").astype(str)
        )
    out = labels[listed].assign(checks=checks[listed])
    out.index = out.index.astype(str)
    return out.sort_index()


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
    except Exception:
        return "unknown"


def write_fixture(
    table: pd.DataFrame, vintage: str, level: str = "L3", path: Optional[Path] = None
) -> Path:
    """Write the labels of ``table`` (a :func:`~.levels.level_table`, e.g. ``qc_levels.pqt``) at
    ``level`` as the fixture module (by default ``ephysatlas/fixtures/pid_qc_filter.py``), see
    :func:`fixture_labels`."""
    path = Path(path or Path(__file__).parents[1] / "fixtures" / "pid_qc_filter.py")
    rules = {lv.name: lv.text for lv in LEVELS}
    nxt = _next_level(level)
    listed = fixture_labels(table, level)
    bad = listed[list(LABELS)].eq(BAD).any(axis=1)
    kinds = table.loc[table[f"{level}_bad"].astype(bool), f"{level}_bad_type"]
    groups = {name: int(kinds.isin(types).sum()) for name, types in BAD_GROUPS.items()}
    groups["several bands"] = len(kinds) - sum(groups.values())
    fields = dict(
        generated=datetime.date.today().isoformat(),
        commit=_commit(),
        vintage=vintage,
        level=level,
        rule=rules[level],
        review_rule=(
            f"``review`` when it is bad only at the next, more sensitive level {nxt} "
            f"({rules[nxt]})"
            if nxt
            else "never ``review`` (there is no more sensitive level)"
        ),
        checks_rule=(
            f"at {level}; for an insertion only to review, at {nxt}"
            if nxt
            else f"at {level}"
        ),
        n_total=len(table),
        n_bad=int(bad.sum()),
        by_type=", ".join(f"{n} {name}" for name, n in groups.items() if n) or "none",
        n_review=int((~bad).sum()),
    )
    doc = "\n\n".join(
        textwrap.fill(
            " ".join(p.format(**fields).split()),
            WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for p in DOCSTRING
    )
    entries = []
    for pid, row in listed.iterrows():
        labels = ", ".join(f'"{row[c]}"' for c in LABELS)
        entries.append(f'    "{pid}": ({labels}, "{row["checks"]}"),')
    text = TEMPLATE.format(
        doc=doc, vintage=vintage, level=level, entries="\n".join(entries)
    )
    path.write_text(text, encoding="utf-8")
    # The repository's CI checks the formatting; format the generated module when ruff is there.
    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", str(path)],
            check=False,
            capture_output=True,
        )
    except OSError:
        pass
    return path
