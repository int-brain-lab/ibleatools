"""Insertion QC at five sensitivity levels: the tests, their z-scores and the labels.

**Only the data decide.** Every test compares a probe's recording, at the positions its alignment
gives it, with what the other probes record in the same brain regions (:mod:`.regions`,
:mod:`.measures`). No model, no auto-alignment and no manual label enters a decision, so a model
later trained without the flagged probes is not judged by itself, and nothing is calibrated on a
selection made by hand.

**Tests**, in three groups (:data:`GROUPS`):

- *unusual for its regions* (:data:`TESTS`): for the LF band, the AP band and the waveforms, the
  ``level`` of the band and its ``shape`` along the probe;
- *more typical elsewhere*: the recording fits better at another depth along the probe, or sideways
  (:data:`TESTS`), or, in a two-probe session, on the other probe's track (:data:`FIXED_TESTS`);
- *dead or noisy channels*: at least a fifth of the channels (:data:`FIXED_TESTS`).

**z-scores.** Each measurement of :data:`TESTS` becomes ``z = (x - median) / (1.4826 MAD)`` over every
probe, without any label or split. The tests of :data:`FIXED_TESTS` have a natural limit and need no
calibration: nearly every pair of probes fits far worse swapped, and a dead channel is dead.

**Levels** (:data:`LEVELS`). A probe is *bad* at a level when a band test is beyond the level's limit
``k``, a position test (:data:`POSITION_TESTS`) beyond ``k_position``, or a test of
:data:`FIXED_TESTS` fails. The position tests have their own limit because their z-scores have short
tails: the same limit would leave them little to say. A probe is *review* when it is bad at the next
level only. The levels are nested. What is bad: **alignment** when the depth, the sideways or
the swap test fails; **lf** when an LF test fails; **ap** when an AP test fails or the channels are
dead or noisy; **waveform** when a waveform test fails, or the AP band is bad.

**Tables.** :func:`level_table` adds the z-scores and every level's labels to the measurements of the
pipeline (one row per insertion or planted problem), :func:`bad_list` and :func:`summary` report
them; :func:`~.fixture.write_fixture` publishes one level as ``ephysatlas.fixtures.pid_qc_filter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

LABELS = ("alignment", "lf", "ap", "waveform")
MAD_TO_SD = 1.4826

# The groups of tests, as reported: {group: description}.
GROUPS = {
    "regions": "unusual for its regions",
    "elsewhere": "more typical elsewhere",
    "channels": "dead or noisy channels",
}
# name -> (group, component that is bad, description). The measurement of a test has the same name.
TESTS = {
    "lf_level": ("regions", "lf", "LF band unlike the other probes in its regions"),
    "lf_shape": ("regions", "lf", "LF band along the probe unlike those regions"),
    "ap_level": ("regions", "ap", "AP band unlike the other probes in its regions"),
    "ap_shape": ("regions", "ap", "AP band along the probe unlike those regions"),
    "wf_level": ("regions", "waveform", "waveforms unlike those of its regions"),
    "wf_shape": ("regions", "waveform", "waveforms along the probe unlike its regions"),
    "depth": ("elsewhere", "alignment", "more typical deeper or shallower"),
    "sideways": ("elsewhere", "alignment", "more typical sideways"),
}
# name -> (group, component, measurement, limit, description); beyond the limit at every level.
FIXED_TESTS = {
    "swap": (
        "elsewhere",
        "alignment",
        "swap_gain",
        0.1,
        "more typical on the other probe's track",
    ),
    "channels": (
        "channels",
        "ap",
        "dead_fraction",
        0.2,
        "a fifth of the channels dead or noisy",
    ),
}
ALL_TESTS = tuple(TESTS) + tuple(FIXED_TESTS)
POSITION_TESTS = ("depth", "sideways")


@dataclass(frozen=True)
class Level:
    """Bad when a band test is beyond ``k`` robust SDs or a position test beyond ``k_position``."""

    name: str
    k: float
    k_position: float

    @property
    def text(self) -> str:
        return f"a band test beyond {self.k:g} SD or a position test beyond {self.k_position:g} SD"


LEVELS = (
    Level("L1", 7.0, 5.0),
    Level("L2", 6.0, 4.0),
    Level("L3", 5.0, 3.5),
    Level("L4", 4.5, 3.0),
    Level("L5", 4.0, 2.5),
)


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return (
        pd.to_numeric(df[c], errors="coerce")
        if c in df
        else df.index.to_series() * 0.0 + np.nan
    )


def calibrate(df: pd.DataFrame) -> dict:
    """``{test: {"median", "sd"}}`` of every test of :data:`TESTS`, on every probe with a value: no
    label, no split."""
    out = {}
    for test in TESTS:
        x = _col(df, test).to_numpy(float)
        med = float(np.nanmedian(x))
        out[test] = dict(
            median=med, sd=MAD_TO_SD * float(np.nanmedian(np.abs(x - med)))
        )
    return out


def zscores(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """z of every test of :data:`TESTS`; larger is worse."""
    return pd.DataFrame(
        {t: (_col(df, t) - cal[t]["median"]) / cal[t]["sd"] for t in TESTS},
        index=df.index,
    )


def failures(df: pd.DataFrame, z: pd.DataFrame, level: Level) -> pd.DataFrame:
    """Boolean table of the tests beyond their limit at ``level`` and of the fixed tests."""
    limits = pd.Series(
        {t: level.k_position if t in POSITION_TESTS else level.k for t in TESTS}
    )
    fail = z[list(TESTS)].gt(limits)
    for test, (_, _, column, limit, _) in FIXED_TESTS.items():
        fail[test] = _col(df, column).gt(limit)
    return fail.fillna(False).astype(bool)


def label(df: pd.DataFrame, z: pd.DataFrame, level: Level) -> pd.DataFrame:
    """Failed tests, whether each component is bad, and the bad type, at one level."""
    fail = failures(df, z, level)
    component = {t: spec[1] for t, spec in TESTS.items()}
    component.update({t: spec[1] for t, spec in FIXED_TESTS.items()})
    points_to = {
        c: fail[[t for t in fail if component[t] == c]].any(axis=1) for c in LABELS
    }
    out = pd.DataFrame(
        {
            "n_fail": fail.sum(axis=1),
            "bad": fail.any(axis=1),
            "alignment": points_to["alignment"],
            "lf": points_to["lf"],
            "ap": points_to["ap"],
            "waveform": points_to["waveform"] | points_to["ap"],
        },
        index=df.index,
    )
    out["failed"] = [
        ";".join(c for c, v in zip(fail.columns, row) if v) for row in fail.to_numpy()
    ]
    out["bad_type"] = [_bad_type(r) for r in out[list(LABELS)].to_numpy()]
    return pd.concat([out, fail.add_prefix("fail_")], axis=1)


def _bad_type(row) -> str:
    bad = [k for k, v in zip(LABELS, row) if v]
    if not bad:
        return "good"
    if "alignment" in bad:
        return "alignment"
    if bad == ["ap", "waveform"]:
        return "ap+waveform"
    return "+".join(bad)


def all_levels(df: pd.DataFrame, cal: dict, levels=LEVELS) -> tuple[dict, pd.DataFrame]:
    """``({level name: label table}, z-scores)``; ``review`` = bad at the next more sensitive
    level."""
    z = zscores(df, cal)
    out = {lv.name: label(df, z, lv) for lv in levels}
    names = [lv.name for lv in levels]
    for a, b in zip(names, names[1:] + [None]):
        nxt = out[b]["bad"] if b else pd.Series(False, index=df.index)
        out[a]["review"] = ~out[a]["bad"] & nxt
        for comp in LABELS:
            nxt_c = out[b][comp] if b else pd.Series(False, index=df.index)
            out[a][f"{comp}_label"] = np.where(
                out[a][comp], "bad", np.where(nxt_c, "review", "good")
            )
    return out, z


# -- tables --------------------------------------------------------------------------------------
BAD_GROUPS = {  # what is bad, as summarised (any other combination: several bands)
    "alignment": ("alignment",),
    "lf only": ("lf",),
    "ap/waveform": ("ap", "waveform", "ap+waveform"),
}


def level_table(df: pd.DataFrame, cal: dict, levels=LEVELS) -> pd.DataFrame:
    """``df`` with the z-scores (``z_<test>``), every level's labels (``<level>_<column>``: the
    columns of :func:`label`, ``review`` and ``<component>_label``, see :func:`all_levels`) and
    ``first_level``, the first level that marks the row bad (empty when none does)."""
    tables, z = all_levels(df, cal, levels)
    first = pd.Series("", index=df.index, dtype=object)
    for lv in reversed(levels):
        first = first.where(~tables[lv.name]["bad"], lv.name)
    return pd.concat(
        [df, z.add_prefix("z_")]
        + [t.add_prefix(f"{name}_") for name, t in tables.items()]
        + [first.rename("first_level")],
        axis=1,
    )


def group_failures(table: pd.DataFrame, level: str) -> pd.DataFrame:
    """Per row of a :func:`level_table`: whether it fails a test of each of :data:`GROUPS` at
    ``level``."""
    group = {t: spec[0] for t, spec in TESTS.items()}
    group.update({t: spec[0] for t, spec in FIXED_TESTS.items()})
    return pd.DataFrame(
        {
            g: table[[f"{level}_fail_{t}" for t in ALL_TESTS if group[t] == g]].any(
                axis=1
            )
            for g in GROUPS
        },
        index=table.index,
    )


def bad_list(table: pd.DataFrame, level: str) -> pd.DataFrame:
    """The insertions of a :func:`level_table` bad at ``level``: the first level that marks them,
    what is bad (``bad_type`` and the label of each component), the failed tests and the z-scores;
    the first marked first."""
    cols = {
        "first_level": "first_level",
        f"{level}_bad_type": "bad_type",
        f"{level}_n_fail": "n_tests_failed",
        f"{level}_failed": "tests_failed",
    }
    cols.update({f"{level}_{c}_label": c for c in LABELS})
    keep = list(cols) + [f"z_{t}" for t in TESTS]
    keep += [spec[2] for spec in FIXED_TESTS.values()]
    bad = table.loc[table[f"{level}_bad"], [c for c in keep if c in table]]
    return bad.rename(columns=cols).sort_values(
        ["first_level", "n_tests_failed"], ascending=[True, False]
    )


def detection(
    table: pd.DataFrame, planted: pd.DataFrame, levels=LEVELS
) -> pd.DataFrame:
    """The share each level marks bad of the insertions as recorded and of each kind of planted
    problem, on the same insertions and with the same tests.

    One row per level and kind of problem (``planted``, a :func:`level_table` of the problems;
    ``as recorded``: their source insertions in ``table``, a :func:`level_table` of the insertions):
    ``n`` insertions, ``detected`` of them bad and ``percent``."""
    sources = pd.Index(planted["source_pid"].astype(str).unique())
    rows = []
    for level in levels:
        n = level.name
        bad = table[f"{n}_bad"].astype(bool).reindex(sources).fillna(False)
        rows.append(
            dict(level=n, fake="as recorded", n=len(sources), detected=int(bad.sum()))
        )
        for fake, g in planted.groupby("fake", sort=False):
            rows.append(
                dict(
                    level=n,
                    fake=fake,
                    n=len(g),
                    detected=int(g[f"{n}_bad"].astype(bool).sum()),
                )
            )
    out = pd.DataFrame(rows)
    out["percent"] = (100.0 * out["detected"] / out["n"].where(out["n"] > 0)).round(1)
    return out


def summary(
    table: pd.DataFrame, planted: Optional[pd.DataFrame] = None, levels=LEVELS
) -> pd.DataFrame:
    """Per level: the rule, the insertions bad and to review, what is bad (:data:`BAD_GROUPS`), the
    insertions failing a test of each of :data:`GROUPS` and, with the planted problems ``planted``,
    the share marked bad of their source insertions as recorded and of each kind of problem
    (:func:`detection`)."""
    rows = []
    for level in levels:
        n = level.name
        bad = table[f"{n}_bad"].astype(bool)
        kind = table.loc[bad, f"{n}_bad_type"]
        row = {
            "level": n,
            "rule": level.text,
            "bad": int(bad.sum()),
            "bad %": round(100 * float(bad.mean()), 1),
            "review": int(table[f"{n}_review"].sum()),
        }
        for name, types in BAD_GROUPS.items():
            row[name] = int(kind.isin(types).sum())
        row["several bands"] = len(kind) - sum(row[name] for name in BAD_GROUPS)
        row.update(
            {GROUPS[g]: int(v) for g, v in group_failures(table, n).sum().items()}
        )
        rows.append(row)
    out = pd.DataFrame(rows).set_index("level")
    if planted is not None:
        det = detection(table, planted, levels)
        det["fake"] = det["fake"] + " %"
        wide = det.pivot(index="level", columns="fake", values="percent")
        order = ["as recorded %"] + [
            f"{k} %" for k in planted["fake"].drop_duplicates()
        ]
        out = out.join(wide[order])
    return out
