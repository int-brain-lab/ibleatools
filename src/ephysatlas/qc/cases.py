"""Example insertions of the QC: which insertions to show, and a figure that shows why each is marked.

A figure has one row per test the insertion fails (the strongest first, one row per band), and each
row shows the evidence for that test, from the top of the probe to its tip:

- a band test: what the probe records against what the other insertions record in the same regions
  (their median and spread), for the features furthest from it, and how far every feature of the band
  is from it;
- ``depth``, ``sideways``: the same with what the other insertions record at the position that fits
  best, the regions at both positions, and the fit at every position tried;
- ``swap``: the same on the track of the session's other probe, and the fit of both probes on both
  tracks;
- ``channels``: which channels are dead or noisy, and the signal of every channel.

An insertion no level marks shows the test closest to its limit, so a good fit can be seen too.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from .levels import FIXED_TESTS, LEVELS, POSITION_TESTS, TESTS  # noqa: E402
from .measures import (  # noqa: E402
    SHIFTS_UM,
    SIDEWAYS_UM,
    bin_cost,
    moved_positions,
)
from .regions import BIN_CHANNELS, parcel_labels, region_labels  # noqa: E402

PER_LEVEL = 18  # insertions shown of each first level
NOT_MARKED = 10  # insertions shown that no level marks
MAX_ROWS = 3  # tests shown per insertion
N_FEATURES = 3  # features shown per test
ROW_INCHES = 2.0
BIN_MM = 0.1
RECORDED, EXPECTED, BETTER = "#0b0b0b", "#8a8985", "#1baf7a"
GROUP_COLORS = {"regions": "#2a78d6", "elsewhere": "#14866a", "channels": "#c9501f"}
BAND_NAMES = {"lf": "LF band", "ap": "AP band", "wf": "Waveforms"}
CHANNEL_LABELS = {
    0: ("good", "#e4e4e0"),
    1: ("dead", "#eb6834"),
    2: ("noisy", "#8c2d04"),
}
CHANNEL_OUTSIDE = ("outside the brain", "#ffffff")
LEGEND = (
    "Recorded",
    "Expected at the recorded position",
    "Spread of the other insertions",
    "Expected at the better position",
)


def pick_examples(table: pd.DataFrame, seed: int = 0) -> list:
    """About a hundred insertions of a :func:`~.levels.level_table`, ordered by the first level that
    marks them: at most ``PER_LEVEL`` of each level, drawn at random with ``seed``, then
    ``NOT_MARKED`` that no level marks."""
    rng = np.random.default_rng(seed)
    out = []
    for level in [lv.name for lv in LEVELS] + [""]:
        pids = table.index[table["first_level"].fillna("").eq(level)].astype(str)
        n = min(len(pids), NOT_MARKED if level == "" else PER_LEVEL)
        out += sorted(rng.choice(np.asarray(pids), size=n, replace=False).tolist())
    return out


def tests_to_show(row: pd.Series) -> list:
    """The tests an insertion's figure shows: those it fails at its first level, the fixed tests
    first, then the others by how far they are beyond their limit, one per band, at most
    ``MAX_ROWS``. For an insertion no level marks, the test closest to its limit."""
    level = {lv.name: lv for lv in LEVELS}.get(row["first_level"] or "", LEVELS[-1])

    def beyond(test):
        limit = level.k_position if test in POSITION_TESTS else level.k
        return row[f"z_{test}"] / limit

    failed = (
        str(row[f"{level.name}_failed"] or "").split(";") if row["first_level"] else []
    )
    failed = [t for t in failed if t]
    if not failed:
        return [max(TESTS, key=lambda t: np.nan_to_num(beyond(t), nan=-9.0))]
    fixed = [t for t in failed if t in FIXED_TESTS]
    bands = {}
    for t in sorted((t for t in failed if t in TESTS), key=beyond, reverse=True):
        bands.setdefault(t.split("_")[0], t)
    return (fixed + list(bands.values()))[:MAX_ROWS]


def _pretty(name: str) -> tuple[str, float]:
    """A feature's label with its unit, and the factor to that unit."""
    label = name.replace("_", " ")
    if name.startswith(("rms_", "psd_")) and "residual" not in name:
        return f"{label} (dB)", 1.0
    if name.endswith("_secs"):
        return label.replace(" secs", " (ms)"), 1e3
    if name.endswith("_um"):
        return label.replace(" um", " (µm)"), 1.0
    if name in ("spike_amplitude", "tip_val"):
        return f"{label} (µV)", 1e6
    return label, 1.0


def _regions(ax, xyz, brain_atlas, title, ylabel=False):
    """The probe's regions, top to tip: Cosmos colors, Beryl names."""
    regions = brain_atlas.regions
    cosmos = region_labels(xyz, brain_atlas, "Cosmos")
    beryl = region_labels(xyz, brain_atlas, "Beryl")
    rows = np.zeros(len(cosmos), dtype=int)
    for r in np.unique(cosmos):
        rows[cosmos == r] = np.flatnonzero(regions.id == r)[0]
    rgb = regions.rgb[rows][:, None, :] / 255.0
    rgb[cosmos == 0] = 1.0
    extent = [0, 1, len(cosmos) * BIN_MM, 0]
    ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=extent)
    start = 0
    for b in range(1, len(beryl) + 1):
        if b == len(beryl) or beryl[b] != beryl[start]:
            if beryl[start] != 0:
                name = regions.acronym[np.flatnonzero(regions.id == beryl[start])[0]]
                ax.text(
                    0.5,
                    (start + b) / 2 * BIN_MM,
                    name,
                    fontsize=3.8,
                    ha="center",
                    va="center",
                    color="0.1",
                    clip_on=True,
                )
            if b < len(beryl):
                ax.axhline(b * BIN_MM, color="white", lw=0.3)
            start = b
    ax.set_xticks([])
    ax.tick_params(labelsize=5, length=2)
    ax.set_title(title, fontsize=5, pad=2)
    if ylabel:
        ax.set_ylabel("Along the probe, top to tip (mm)", fontsize=5.5)
    else:
        ax.set_yticks([])


def _profile(ax, depth, recorded, expected, spread, better=None):
    """A feature along the probe: recorded, and expected from the other insertions."""
    ax.fill_betweenx(
        depth,
        expected - spread,
        expected + spread,
        color=EXPECTED,
        alpha=0.28,
        lw=0,
        step="mid",
        label="Spread of the other insertions",
    )
    ax.step(
        expected,
        depth,
        where="mid",
        color=EXPECTED,
        lw=1.1,
        label="Expected at the recorded position",
    )
    if better is not None:
        ax.step(
            better,
            depth,
            where="mid",
            color=BETTER,
            lw=1.2,
            label="Expected at the better position",
        )
    ax.plot(recorded, depth, color=RECORDED, lw=1.0, label="Recorded")
    ax.set_ylim(depth[-1] + BIN_MM / 2, depth[0] - BIN_MM / 2)
    ax.set_yticks([])
    ax.tick_params(labelsize=5, length=2)
    ax.locator_params(axis="x", nbins=4)
    ax.spines[["top", "right"]].set_visible(False)


class _Probe:
    """One insertion and what is expected of it, for the panels of its figure."""

    def __init__(self, pid, ds, bins, reference, brain_atlas):
        self.i = int(np.flatnonzero(bins.pids.astype(str) == str(pid))[0])
        self.ds, self.bins = ds, bins
        self.reference, self.brain_atlas = reference, brain_atlas
        self.z, self.xyz = bins.z[self.i], bins.xyz[self.i]
        self.depth = (np.arange(len(self.z)) + 0.5) * BIN_MM
        self.labels = parcel_labels(self.xyz, brain_atlas)
        self.expected, self.spread = reference.expected(self.labels, [self.i])
        self.e = (self.z - self.expected) / self.spread
        self.names = ds.feature_names

    def at(self, xyz):
        """Expected values and residuals of this probe's recording placed at ``xyz``."""
        labels = parcel_labels(xyz, self.brain_atlas)
        expected, spread = self.reference.expected(labels, [self.i])
        return expected, (self.z - expected) / spread

    def native(self, values, f):
        """Feature ``f``, from robust units to its own unit."""
        factor = _pretty(self.names[f])[1]
        return (values * self.bins.scale[f] + self.bins.centre[f]) * factor

    def width(self, values, f):
        return values * self.bins.scale[f] * _pretty(self.names[f])[1]


def _feature_panels(p, axes, features, there=None):
    """One recorded-vs-expected panel per feature; ``there``: expected at the better position."""
    for ax, f in zip(axes, features):
        _profile(
            ax,
            p.depth,
            p.native(p.z[:, f], f),
            p.native(p.expected[:, f], f),
            p.width(p.spread[:, f], f),
            None if there is None else p.native(there[:, f], f),
        )
        ax.set_title(_pretty(p.names[f])[0], fontsize=5.5, pad=2)


def _title(ax, text, color):
    ax.text(
        0.0,
        1.17,
        text,
        transform=ax.transAxes,
        fontsize=6.6,
        fontweight="bold",
        color=color,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _row_band(p, cells, test, row, table, bands):
    band, kind = test.split("_")
    cols = np.asarray(bands[band])
    e = p.e[:, cols]
    if kind == "shape":
        e = e - np.nanmedian(e, axis=0, keepdims=True)
    distance = np.nanmedian(np.abs(e), axis=0)
    order = np.argsort(-np.nan_to_num(distance))
    _regions(cells[0], p.xyz, p.brain_atlas, "Regions", ylabel=True)
    _feature_panels(p, cells[1:4], cols[order[:N_FEATURES]])
    what = "shape along the probe" if kind == "shape" else "level"
    z = row[f"z_{test}"]
    if row["first_level"]:
        text = (
            f"{BAND_NAMES[band]}: {what} unlike the other insertions in these regions"
        )
    else:
        text = (
            f"{BAND_NAMES[band]}: {what} within what the other insertions record here"
        )
    _title(cells[0], f"{text} (z = {z:.1f})", GROUP_COLORS["regions"])
    ax = cells[4]
    rank = np.arange(1, len(cols) + 1)
    ax.plot(
        rank,
        distance[order],
        "o-",
        color=RECORDED,
        ms=1.8,
        lw=0.7,
        label="This insertion",
    )
    ax.axhline(
        float(table[test].median()),
        color=EXPECTED,
        lw=0.9,
        ls="--",
        label="A typical insertion, all features",
    )
    ax.set_title(f"All {len(cols)} features of the band", fontsize=5.5, pad=2)
    ax.set_xlabel("Features, furthest first", fontsize=5)
    ax.set_ylabel("Distance from expected (MADs)", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=4.3, loc="best", handlelength=1.3)
    ax.spines[["top", "right"]].set_visible(False)


def _best_features(p, residuals_there):
    """The features that fit best after the move, by the mean improvement over the bins."""
    gain = np.minimum(np.abs(p.e), 3.0) - np.minimum(np.abs(residuals_there), 3.0)
    with np.errstate(all="ignore"):
        mean = np.nansum(gain, axis=0) / np.maximum(np.isfinite(gain).sum(axis=0), 1)
    return np.argsort(-mean)[:N_FEATURES]


def _row_move(p, cells, test, row, scans, k):
    moves = moved_positions(p.xyz)[test]
    gains = np.asarray(scans[test][k], dtype=float)
    best = int(np.nanargmax(gains))
    um, xyz = moves[best]
    if test == "depth":
        where = f"{abs(um):.0f} µm {'deeper' if um > 0 else 'shallower'}"
    else:
        side = best // len(SIDEWAYS_UM)
        words = (("left", "right"), ("back", "front"))[side]
        where = f"{abs(um):.0f} µm to the {words[int(um > 0)]}"
    there, e_there = p.at(xyz)
    _regions(cells[0], p.xyz, p.brain_atlas, "Regions\nrecorded", ylabel=True)
    _regions(cells[1], xyz, p.brain_atlas, "Regions\nmoved")
    _feature_panels(p, cells[2:5], _best_features(p, e_there), there)
    z = row[f"z_{test}"]
    text = (
        "The recording fits better"
        if row["first_level"]
        else "Within the limit: its best move is"
    )
    _title(cells[0], f"{text} {where} (z = {z:.1f})", GROUP_COLORS["elsewhere"])
    ax = cells[5]
    ax.axhline(0, color=EXPECTED, lw=0.6)
    if test == "depth":
        ax.plot(SHIFTS_UM, gains, "o-", color=RECORDED, ms=2, lw=0.8)
        ax.set_xlabel("Shallower ← move (µm) → deeper", fontsize=5)
    else:
        n = len(SIDEWAYS_UM)
        ax.plot(
            SIDEWAYS_UM,
            gains[:n],
            "o-",
            color=RECORDED,
            ms=2,
            lw=0.8,
            label="left – right",
        )
        ax.plot(
            SIDEWAYS_UM,
            gains[n:],
            "o--",
            color=EXPECTED,
            ms=2,
            lw=0.8,
            label="back – front",
        )
        ax.legend(frameon=False, fontsize=4.3, loc="best", handlelength=1.6)
        ax.set_xlabel("Move (µm)", fontsize=5)
    ax.plot([um], [gains[best]], "o", color=BETTER, ms=4, zorder=5)
    ax.set_title("Better fit at each position", fontsize=5.5, pad=2)
    ax.set_ylabel("Gain over the recorded position", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.spines[["top", "right"]].set_visible(False)


def _row_swap(p, cells, row, pairs):
    bins, brain_atlas = p.bins, p.brain_atlas

    def fit(z, xyz, exclude):
        c = bin_cost(p.reference.residuals(z, parcel_labels(xyz, brain_atlas), exclude))
        return float(np.mean(c[np.isfinite(c)]))

    best, costs = None, None
    for b in pairs.get(p.i, []):
        ex = (p.i, b)
        c = [
            fit(p.z, p.xyz, ex),
            fit(p.z, bins.xyz[b], ex),
            fit(bins.z[b], bins.xyz[b], ex),
            fit(bins.z[b], p.xyz, ex),
        ]
        gain = c[0] + c[2] - c[1] - c[3]
        if best is None or gain > costs[0] + costs[2] - costs[1] - costs[3]:
            best, costs = b, c
    other = str(bins.pids[best])[:8]
    there, e_there = p.at(bins.xyz[best])
    _regions(cells[0], p.xyz, brain_atlas, "Regions\nrecorded", ylabel=True)
    _regions(cells[1], bins.xyz[best], brain_atlas, f"Regions\n{other}")
    _feature_panels(p, cells[2:5], _best_features(p, e_there), there)
    _title(
        cells[0],
        f"The recording fits better on the track of the session's other probe, {other} "
        f"(gain {row['swap_gain']:.2f})",
        GROUP_COLORS["elsewhere"],
    )
    ax = cells[5]
    x = np.array([0, 1, 2.6, 3.6])
    ax.bar(x, costs, width=0.8, color=[EXPECTED, BETTER, EXPECTED, BETTER])
    ax.set_xticks(x, ["own", "other", "own", "other"], fontsize=5)
    ax.set_xlabel(f"Recording of this probe  |  of {other}", fontsize=5)
    ax.set_title("Misfit on each track", fontsize=5.5, pad=2)
    ax.set_ylabel("Mean distance from expected (MADs)", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.spines[["top", "right"]].set_visible(False)


def _row_channels(p, cells, row):
    labels = p.ds.channel_labels[p.i]
    features = p.ds.features[p.i]
    n = len(labels)
    per_row = 2
    y = (np.arange(n) // per_row * per_row + 1) / BIN_CHANNELS * BIN_MM
    dead = np.isin(labels, (1, 2))
    ax = cells[0]
    for lab, (name, color) in {**CHANNEL_LABELS, 3: CHANNEL_OUTSIDE}.items():
        s = labels == lab
        ax.scatter(
            (np.arange(n) % per_row)[s],
            y[s],
            s=2.2,
            marker="s",
            color=color,
            edgecolors="0.6" if lab == 3 else "none",
            linewidths=0.2,
            label=f"{name} ({int(s.sum())})",
        )
    ax.set_xlim(-1, 2)
    ax.set_ylim(y[-1] + 0.05, y[0] - 0.05)
    ax.set_xticks([])
    ax.tick_params(labelsize=5, length=2)
    ax.set_title("Channels", fontsize=5, pad=2)
    ax.set_ylabel("Along the probe, top to tip (mm)", fontsize=5.5)
    cells[1].axis("off")
    cells[1].legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        fontsize=4.3,
        loc="center left",
        handletextpad=0.1,
        borderaxespad=0,
        markerscale=1.6,
    )
    for ax, name in zip(cells[2:5], ("rms_ap", "rms_lf", "spike_count")):
        f = p.names.index(name)
        v = features[:, f].astype(float) * _pretty(name)[1]
        ok = np.isfinite(v) & (v != 0)
        ax.plot(v[ok & ~dead], y[ok & ~dead], ".", color=RECORDED, ms=1.2, mew=0)
        ax.plot(
            v[ok & dead], y[ok & dead], ".", color=CHANNEL_LABELS[1][1], ms=2.4, mew=0
        )
        ax.set_ylim(y[-1] + 0.05, y[0] - 0.05)
        ax.set_yticks([])
        ax.tick_params(labelsize=5, length=2)
        ax.locator_params(axis="x", nbins=4)
        ax.set_title(f"{_pretty(name)[0]}, per channel", fontsize=5.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    cells[5].axis("off")
    _title(
        cells[0],
        f"{100 * row['dead_fraction']:.0f}% of the channels are dead or noisy (the limit is 20%)",
        GROUP_COLORS["channels"],
    )


def figure_case(pid, ds, bins, table, scans, reference, brain_atlas, bands, pairs):
    """The figure of one insertion: one row of evidence per test it fails."""
    p = _Probe(pid, ds, bins, reference, brain_atlas)
    row = table.loc[str(pid)]
    tests = tests_to_show(row)
    k = int(np.flatnonzero(scans["pids"].astype(str) == str(pid))[0])
    n = len(tests)
    height = 0.98 + n * ROW_INCHES
    fig = plt.figure(figsize=(7.6, height))
    grid = GridSpec(
        n,
        7,
        figure=fig,
        width_ratios=[0.34, 0.34, 0.12, 1.0, 1.0, 1.0, 1.15],
        left=0.06,
        right=0.99,
        top=1 - 0.6 / height,
        bottom=0.62 / height,
        hspace=0.62,
        wspace=0.34,
    )
    for r, test in enumerate(tests):
        if test in POSITION_TESTS or test in FIXED_TESTS:
            cells = [fig.add_subplot(grid[r, c]) for c in (0, 1, 3, 4, 5, 6)]
        else:
            cells = [fig.add_subplot(grid[r, 0:2])]
            cells += [fig.add_subplot(grid[r, c]) for c in (3, 4, 5, 6)]
        if test == "channels":
            _row_channels(p, cells, row)
        elif test == "swap":
            _row_swap(p, cells, row, pairs)
        elif test in POSITION_TESTS:
            _row_move(p, cells, test, row, scans, k)
        else:
            _row_band(p, cells, test, row, table, bands)
    first = row["first_level"]
    head = (
        f"first marked at {first}"
        if first
        else "not marked; the test closest to a limit"
    )
    fig.text(0.06, 1 - 0.06 / height, f"{pid}: {head}", fontsize=7, va="top")
    handles = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in LEGEND:
                handles.setdefault(label, handle)
    if handles:
        order = [label for label in LEGEND if label in handles]
        fig.legend(
            [handles[label] for label in order],
            order,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(order),
            frameon=False,
            fontsize=5.5,
            handlelength=1.8,
            columnspacing=1.6,
        )
    return fig


def render_examples(
    pids,
    ds,
    bins,
    table,
    scans,
    reference,
    brain_atlas,
    bands,
    pairs,
    out_dir: Path,
    overwrite=False,
) -> dict:
    """``{pid: path}`` of the figures of the insertions ``pids`` in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for pid in pids:
        path = out_dir / f"{pid}.png"
        if overwrite or not path.exists():
            fig = figure_case(
                pid, ds, bins, table, scans, reference, brain_atlas, bands, pairs
            )
            fig.savefig(path, dpi=135, facecolor="white")
            plt.close(fig)
        out[str(pid)] = path
    return out
