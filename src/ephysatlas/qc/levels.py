"""Insertion QC at five sensitivity levels: the failure cases, their calibration and the labels.

**Only the data decide.** Every failure case compares a probe's recording, at the positions its
alignment gives it, with the recordings of the other probes: with what they record around the same
positions -- their recordings, not a model's prediction (:class:`~.modelfree.LocalReference`) --
or, for the signal, with the signals of all probes (:mod:`.signal`). No model, no auto-alignment
and no manual label enters a decision, so a model later trained without the flagged probes is not
judged by itself, and nothing is calibrated on a selection made by hand.

**Failure cases** (:data:`TESTS`), eight in three groups: how unusual each band is for where the
probe is placed (LF, AP, waveform; medians over features and over channels, each feature scaled by
how much it typically disagrees with its neighbours, so one noisy feature or a few odd channels do
not flag a band); whether each band's signal is unusual wherever the probe is (robust fingerprints,
the LF 1/f fit, a fifth of the channels dead or noisy); and whether the recording is more typical
elsewhere (another depth along the probe, the other probe's track in a two-probe session).

**z-scores.** Each measurement is robustly centred and scaled, Yeo-Johnson transformed (exponent
fitted on the central 90% of the probes) and turned into a robust z on the transformed values, in
two passes so that bad probes do not widen the limits. The reference is every probe, without any
label or split.

**Levels** (:data:`LEVELS`). A probe is *bad* when at least ``m`` tests fail beyond ``k`` SDs, or
(when given) at least ``m2`` tests fail beyond the lower ``k2``: L1, one test beyond 3 SD; L2, one
beyond 2.5 SD; L3, L4 and L5, as L2 or two tests beyond 1.5, 1.25 and 1 SD. It is *review* when it
is bad at the next level only. The levels are nested. What is bad: **alignment** when the depth or
the swap test fails, or the LF and the spike band are both unusual (the probe does not belong where
it is placed); **lf** on an LF signal failure or when the LF alone is unusual; **ap** (and
**waveform**) on an AP signal failure or when the AP alone is unusual; **waveform** also on its
own.

**Tables.** :func:`insertion_table` gathers the outputs of the pipeline's measuring stages (one
row per insertion), :func:`level_table` adds the z-scores and every level's labels,
:func:`bad_list` and :func:`summary` report them; :func:`~.fixture.write_fixture` publishes one
level as ``ephysatlas.fixtures.pid_qc_filter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

LABELS = ("alignment", "lf", "ap", "waveform")
COMPONENTS = ("lf", "ap", "waveform")

# name -> (group, [(column, worse)], description); worse: +1 high is bad, -1 low is bad.
TESTS = {
    "lf_outlier": (
        "lf",
        [("h_lc_lf", 1)],
        "LF unlike what nearby probes record",
    ),
    "ap_outlier": (
        "spike",
        [("h_lc_ap", 1)],
        "AP unlike what nearby probes record",
    ),
    "wf_outlier": (
        "spike",
        [("h_lc_waveform", 1)],
        "waveforms unlike those of nearby probes",
    ),
    "lf_signal": (
        "lf_signal",
        [("sig_fpr_lf", 1), ("sig_lf_fit_r2", -1)],
        "LF signal unusual anywhere (fingerprint or 1/f fit)",
    ),
    "ap_signal": (
        "ap_signal",
        [("sig_fpr_ap", 1)],
        "AP signal unusual anywhere (fingerprint or dead channels)",
    ),
    "wf_signal": (
        "wf_signal",
        [("sig_fpr_waveform", 1)],
        "waveforms unusual anywhere (fingerprint)",
    ),
    "depth": (
        "position",
        [("depth_gain", 1)],
        "more typical at another depth along the probe",
    ),
    "swap": (
        "position",
        [("swap_gain_mf", 1)],
        "more typical on the other probe's track",
    ),
}
BAD_CHANNEL_FRAC = 0.2  # a fifth of the channels dead or noisy fails the AP signal test
# Measurements of QC version 4 at the auto-aligned positions, which cached ``mf.csv`` tables may
# still carry: never used, dropped by :func:`insertion_table`.
LEGACY_COLUMNS = r"^(a_lc_.*|auto_dist_um)$"


@dataclass(frozen=True)
class Level:
    """Bad when at least ``m`` tests fail beyond ``k`` SDs, or (when given) at least ``m2`` tests
    fail beyond ``k2`` SDs: one strong sign, or several moderate ones."""

    name: str
    k: float
    m: int
    k2: Optional[float] = None
    m2: int = 2

    @property
    def text(self) -> str:
        def one(m, k):
            return f"{m} test{'s' if m > 1 else ''} beyond {k:g} SD"

        if self.k2 is None:
            return one(self.m, self.k)
        return f"{one(self.m, self.k)}, or {one(self.m2, self.k2)}"


LEVELS = (
    Level("L1", 3.0, 1),
    Level("L2", 2.5, 1),
    Level("L3", 2.5, 1, 1.5, 2),
    Level("L4", 2.5, 1, 1.25, 2),
    Level("L5", 2.5, 1, 1.0, 2),
)


def measurement_columns() -> list:
    """The measurements the failure cases read, sorted."""
    return sorted({c for spec in TESTS.values() for c, _ in spec[1]})


def _robust(x: np.ndarray, trim: float = 3.0) -> tuple[float, float]:
    """Median and robust SD, in two passes (the second without the values beyond ``trim``)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    med = np.median(x)
    sd = 1.4826 * np.median(np.abs(x - med)) or np.std(x) or 1.0
    keep = np.abs(x - med) <= trim * sd
    med = np.median(x[keep])
    sd = 1.4826 * np.median(np.abs(x[keep] - med)) or np.std(x[keep]) or 1.0
    return float(med), float(sd)


def _fit(x) -> dict:
    """Normalizing transform of one measurement on reference values ``x`` (see the module doc)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return dict(med=np.nan, sd=np.nan, lam=1.0, m2=np.nan, s2=np.nan)
    med, sd = _robust(x)
    u = (x - med) / sd
    lo, hi = np.quantile(u, [0.05, 0.95])
    lam = float(yeojohnson_normmax(u[(u >= lo) & (u <= hi)]))
    m2, s2 = _robust(yeojohnson(u, lam))
    return dict(med=med, sd=sd, lam=lam, m2=m2, s2=s2)


def _z(x: pd.Series, p: dict) -> pd.Series:
    u = ((pd.to_numeric(x, errors="coerce") - p["med"]) / p["sd"]).to_numpy(float)
    t = np.full(len(u), np.nan)
    ok = np.isfinite(u)
    t[ok] = yeojohnson(u[ok], p["lam"])
    return pd.Series((t - p["m2"]) / p["s2"], index=x.index)


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df else pd.Series(np.nan, index=df.index)


def calibrate(df: pd.DataFrame) -> dict:
    """``{column: transform}`` of every measurement the failure cases read
    (:func:`measurement_columns`), fitted on every probe with a value: no label, no split."""
    return {c: _fit(_col(df, c)) for c in measurement_columns()}


def zscores(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """z of every test, oriented so that larger is worse (the max over a test's columns)."""
    z = pd.DataFrame(index=df.index)
    for name, (_, cols, _) in TESTS.items():
        parts = [worse * _z(_col(df, c), cal[c]) for c, worse in cols]
        z[name] = pd.concat(parts, axis=1).max(axis=1)
    return z


def failures(df: pd.DataFrame, z: pd.DataFrame, k: float) -> pd.DataFrame:
    """Boolean table of the failure cases at ``k`` SDs (dead channels fail the AP signal test)."""
    fail = z[list(TESTS)].gt(k)
    bad_ch = pd.to_numeric(_col(df, "sig_bad_channel_frac"), errors="coerce")
    fail["ap_signal"] = fail["ap_signal"] | (bad_ch > BAD_CHANNEL_FRAC).to_numpy()
    return fail.fillna(False).astype(bool)


def label(df: pd.DataFrame, z: pd.DataFrame, level: Level) -> pd.DataFrame:
    """Failed tests, whether each component is bad, and the bad type, at one level.

    The failed tests (which decide what is bad) are those of the rule that marked the probe: the
    tests beyond ``k`` when the first rule holds, plus those beyond ``k2`` when the second does.
    """
    fail1 = failures(df, z, level.k)
    rule1 = fail1.sum(axis=1) >= level.m
    fail = fail1 & rule1.to_numpy()[:, None]
    rule2 = pd.Series(False, index=df.index)
    if level.k2 is not None:
        fail2 = failures(df, z, level.k2)
        rule2 = fail2.sum(axis=1) >= level.m2
        fail = fail | (fail2 & rule2.to_numpy()[:, None])
    group = {n: TESTS[n][0] for n in TESTS}
    by = {
        g: fail[[n for n in fail if group[n] == g]].any(axis=1)
        for g in set(group.values())
    }
    bad = rule1 | rule2
    lf_mis = fail["lf_outlier"]
    sp_mis = fail[["ap_outlier", "wf_outlier"]].any(axis=1)
    align = bad & (by["position"] | (lf_mis & sp_mis))
    lf = bad & (by["lf_signal"] | (lf_mis & ~sp_mis & ~align))
    ap = bad & (by["ap_signal"] | (fail["ap_outlier"] & ~lf_mis & ~align))
    wf = bad & (ap | by["wf_signal"] | (sp_mis & ~lf_mis & ~align))
    out = pd.DataFrame(
        {
            "n_fail": fail.sum(axis=1),
            "bad": bad,
            "alignment": align,
            "lf": lf,
            "ap": ap,
            "waveform": wf,
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


def insertion_table(mf: pd.DataFrame, swap: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per insertion (index ``pid``, sorted): its model-free metrics (``mf``, the last row
    of an insertion scored twice; the :data:`LEGACY_COLUMNS` dropped) and ``swap_gain_mf``, its
    largest probe-swap gain over the pairs ``swap`` puts it in (NaN in none)."""
    mf = mf.assign(pid=mf["pid"].astype(str)).drop_duplicates("pid", keep="last")
    df = mf.set_index("pid").sort_index()
    df = df.drop(columns=df.filter(regex=LEGACY_COLUMNS).columns)
    df["swap_gain_mf"] = np.nan
    if swap is not None and len(swap):
        pairs = pd.concat(
            [
                swap[[end, "swap_gain_mf"]].rename(columns={end: "pid"})
                for end in ("pid_a", "pid_b")
            ]
        )
        best = pairs.assign(pid=pairs["pid"].astype(str)).groupby("pid")["swap_gain_mf"]
        df["swap_gain_mf"] = best.max().reindex(df.index)
    return df


def fakes_table(fakes: pd.DataFrame) -> pd.DataFrame:
    """The planted problems that could be scored (rows without an ``error``; the
    :data:`LEGACY_COLUMNS` dropped)."""
    error = fakes["error"] if "error" in fakes else pd.Series("", index=fakes.index)
    out = fakes[error.fillna("").eq("")].reset_index(drop=True)
    return out.drop(columns=out.filter(regex=LEGACY_COLUMNS).columns)


def level_table(df: pd.DataFrame, cal: dict, levels=LEVELS) -> pd.DataFrame:
    """``df`` with the z-scores (``z_<test>``) and every level's labels (``<level>_<column>``: the
    columns of :func:`label`, ``review`` and ``<component>_label``, see :func:`all_levels`)."""
    tables, z = all_levels(df, cal, levels)
    return pd.concat(
        [df, z.add_prefix("z_")]
        + [t.add_prefix(f"{name}_") for name, t in tables.items()],
        axis=1,
    )


def bad_list(table: pd.DataFrame, level: str) -> pd.DataFrame:
    """The insertions of a :func:`level_table` bad at ``level``: what is bad (``bad_type`` and
    the label of each component), the failed tests and the z-scores; the most failed tests
    first."""
    cols = {
        f"{level}_bad_type": "bad_type",
        f"{level}_n_fail": "n_tests_failed",
        f"{level}_failed": "tests_failed",
    }
    cols.update({f"{level}_{c}_label": c for c in LABELS})
    keep = list(cols) + [f"z_{t}" for t in TESTS]
    bad = table.loc[table[f"{level}_bad"], [c for c in keep if c in table]]
    return bad.rename(columns=cols).sort_values(
        ["n_tests_failed", "bad_type"], ascending=[False, True]
    )


def detection(table: pd.DataFrame, fakes: pd.DataFrame, levels=LEVELS) -> pd.DataFrame:
    """What each level catches of the planted problems ``fakes`` (a :func:`level_table` of
    :func:`fakes_table`), with ``table`` the level table of the insertions they were planted in.

    One row per level and kind of planted problem (``fake``), plus ``unmodified``, the source
    insertions themselves: ``n`` problems counted, ``detected`` of them bad at the level and
    ``percent``. A problem counts only when its source insertion is not bad at the level itself
    (otherwise it would be caught whatever was planted); for ``unmodified``, every source counts
    and ``detected`` is the number bad (the flagged share of a random sample of the insertions).
    """
    sources = pd.Index(fakes["source_pid"].astype(str).unique())
    rows = []
    for level in levels:
        n = level.name
        bad = table[f"{n}_bad"].astype(bool)
        src_bad = bad.reindex(sources).fillna(False).astype(bool)
        rows.append(
            dict(
                level=n, fake="unmodified", n=len(sources), detected=int(src_bad.sum())
            )
        )
        for fake, g in fakes.groupby("fake", sort=False):
            clean = ~g["source_pid"].astype(str).map(bad).eq(True)
            rows.append(
                dict(
                    level=n,
                    fake=fake,
                    n=int(clean.sum()),
                    detected=int(g.loc[clean, f"{n}_bad"].astype(bool).sum()),
                )
            )
    out = pd.DataFrame(rows)
    out["percent"] = (100.0 * out["detected"] / out["n"].where(out["n"] > 0)).round(1)
    return out


def summary(
    table: pd.DataFrame, fakes: Optional[pd.DataFrame] = None, levels=LEVELS
) -> pd.DataFrame:
    """Per level: the rule, the insertions bad and to review, what is bad (:data:`BAD_GROUPS`)
    and, with the planted problems ``fakes``, the flagged share of their unmodified source
    insertions (``sources flagged %``) and the share of each kind of problem marked bad, among
    those planted in insertions that are not bad themselves (:func:`detection`)."""
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
        rows.append(row)
    out = pd.DataFrame(rows).set_index("level")
    if fakes is not None:
        det = detection(table, fakes, levels)
        det["fake"] = det["fake"].replace({"unmodified": "sources flagged"}) + " %"
        wide = det.pivot(index="level", columns="fake", values="percent")
        order = ["sources flagged %"] + [
            f"{k} %" for k in fakes["fake"].drop_duplicates()
        ]
        out = out.join(wide[order])
    return out
