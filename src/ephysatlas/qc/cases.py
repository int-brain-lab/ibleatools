"""Example insertions of the QC: which insertions to show, and a figure that shows why each is marked.

Every figure shows the same features (:data:`FEATURES`), whatever is wrong with the insertion, one row
per band, from the top of the probe to its tip: what the probe records against what the other
insertions record in the same regions (their median and spread), then the feature of the band
furthest from it if it is not one of those, and how far every feature of the band is.

A row follows for each test of position or of the channels that the insertion fails:

- ``depth``, ``sideways``: the regions at both positions and the fit at every position tried;
- ``track``: the regions on the track it follows best and its fit on every track;
- ``swap``: the regions on the track of the other probe and the fit of both probes on both tracks;
- ``channels``: which channels are dead or noisy, and the signal of every channel.

What is expected at the better position is then drawn on the rows of the bands. An insertion no level
marks shows the test of position closest to its limit, so a good fit can be seen too.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from .levels import FIXED_TESTS, LEVELS, POSITION_TESTS, TRACK_TEST  # noqa: E402
from .measures import (  # noqa: E402
    SHIFTS_UM,
    SIDEWAYS_UM,
    fit,
    moved_positions,
)
from .regions import BIN_CHANNELS, parcel_labels, region_labels  # noqa: E402

PER_LEVEL = 18  # insertions shown of each first level
NOT_MARKED = 10  # insertions shown that no level marks
MAX_ROWS = 2  # tests shown per insertion, beyond the bands
# The features of every figure, per band.
FEATURES = {
    "lf": ("rms_lf", "psd_theta", "psd_theta_bpr"),
    "ap": ("rms_ap", "alpha_mean", "spike_count"),
    "wf": ("spike_width_secs", "peak_to_trough_ratio_log", "spike_amplitude"),
}
ROW_INCHES = 1.75
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
    """The tests an insertion's figure shows, on top of the bands: those of position and of the
    channels it fails at its first level, the fixed tests first, then the others by how far they are
    beyond their limit, at most ``MAX_ROWS``. Without any, the test of position closest to its
    limit."""
    level = {lv.name: lv for lv in LEVELS}.get(row["first_level"] or "", LEVELS[-1])

    def beyond(test):
        if test in TRACK_TEST:
            return row[TRACK_TEST[test][2]] / level.track
        return row[f"z_{test}"] / level.k_position

    failed = (
        str(row[f"{level.name}_failed"] or "").split(";") if row["first_level"] else []
    )
    fixed = [t for t in failed if t in FIXED_TESTS]
    moves = (*POSITION_TESTS, *TRACK_TEST)
    others = sorted((t for t in failed if t in moves), key=beyond, reverse=True)
    return (fixed + others)[:MAX_ROWS] or [
        max(moves, key=lambda t: np.nan_to_num(beyond(t), nan=-9.0))
    ]


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
    rgb = regions.get(cosmos).rgb[:, None, :] / 255.0
    rgb[cosmos == 0] = 1.0
    extent = [0, 1, len(cosmos) * BIN_MM, 0]
    ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=extent)
    start = 0
    for b in range(1, len(beryl) + 1):
        if b == len(beryl) or beryl[b] != beryl[start]:
            if beryl[start] != 0:
                name = regions.id2acronym(beryl[start])[0]
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


def _row_band(p, cells, band, row, table, cols, level, there):
    """A band: the features of every figure, the one furthest from expected, every feature."""
    fixed = [p.names.index(name) for name in FEATURES[band]]
    distance = np.nanmedian(np.abs(p.e[:, cols]), axis=0)
    order = np.argsort(-np.nan_to_num(distance))
    other = next(int(cols[k]) for k in order if int(cols[k]) not in fixed)
    _regions(cells[0], p.xyz, p.brain_atlas, "Regions", ylabel=True)
    _feature_panels(p, cells[1:5], fixed + [other], there)
    cells[4].set_title(
        f"Furthest off: {_pretty(p.names[other])[0]}", fontsize=5.5, pad=2
    )
    scores = {kind: row[f"z_{band}_{kind}"] for kind in ("level", "shape")}
    failed = any(z > level.k for z in scores.values())
    _title(
        cells[0],
        f"{BAND_NAMES[band]}: level z = {scores['level']:.1f}, "
        f"shape z = {scores['shape']:.1f}",
        GROUP_COLORS["regions"] if failed else EXPECTED,
    )
    ax = cells[5]
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
        float(table[f"{band}_level"].median()),
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


def _best_move(p, test, gains):
    """``(words, µm, positions)`` of the best move of a scan."""
    moves = moved_positions(p.xyz)[test]
    best = int(np.nanargmax(gains))
    um, xyz = moves[best]
    if test == "depth":
        return f"{abs(um):.0f} µm {'deeper' if um > 0 else 'shallower'}", um, xyz
    words = (("left", "right"), ("back", "front"))[best // len(SIDEWAYS_UM)]
    return f"{abs(um):.0f} µm to the {words[int(um > 0)]}", um, xyz


def _row_move(p, cells, test, row, gains):
    where, um, xyz = _best_move(p, test, gains)
    best = float(np.nanmax(gains))
    _regions(cells[0], p.xyz, p.brain_atlas, "Regions\nrecorded", ylabel=True)
    _regions(cells[1], xyz, p.brain_atlas, "Regions\nmoved")
    z = row[f"z_{test}"]
    text = (
        "The recording fits better"
        if row["first_level"]
        else "Within the limit: its best move is"
    )
    _title(cells[0], f"{text} {where} (z = {z:.1f})", GROUP_COLORS["elsewhere"])
    ax = cells[2]
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
    ax.plot([um], [best], "o", color=BETTER, ms=4, zorder=5)
    ax.set_title("Better fit at each position", fontsize=5.5, pad=2)
    ax.set_ylabel("Gain over the recorded position", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    return xyz


def _row_track(p, cells, row, tracks):
    fits = tracks.fits(p.z, [p.i])
    best = int(np.nanargmax(fits))
    other = str(p.bins.pids[best])[:8]
    _regions(cells[0], p.xyz, p.brain_atlas, "Regions\nrecorded", ylabel=True)
    _regions(cells[1], p.bins.xyz[best], p.brain_atlas, f"Regions\n{other}")
    rank = row["track_rank"]
    text = f"{100 * rank:.0f}% of the other tracks fit the recording as well as its own"
    _title(
        cells[0],
        text if row["first_level"] else f"Within the limit: {text}",
        GROUP_COLORS["elsewhere"],
    )
    ax = cells[2]
    ax.hist(fits[np.isfinite(fits)], bins=30, color=EXPECTED)
    ax.axvline(row["fit"], color=RECORDED, lw=1.2, label="Its own track")
    ax.axvline(fits[best], color=BETTER, lw=1.2, label="The track it follows best")
    ax.legend(frameon=False, fontsize=4.3, loc="best", handlelength=1.3)
    ax.set_title("Fit on the tracks of the other insertions", fontsize=5.5, pad=2)
    ax.set_xlabel("Fit (correlation along the probe)", fontsize=5)
    ax.set_ylabel("Tracks", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    return p.bins.xyz[best]


def _row_swap(p, cells, row, bands):
    bins, brain_atlas = p.bins, p.brain_atlas
    partner = int(np.flatnonzero(bins.pids.astype(str) == row["swap_partner"])[0])
    other = str(bins.pids[partner])[:8]
    ex = (p.i, partner)

    def on(z, xyz):
        expected = p.reference.expected(parcel_labels(xyz, brain_atlas), ex)[0]
        return fit(z, expected, bands)[0]

    fits = [
        on(p.z, p.xyz),
        on(p.z, bins.xyz[partner]),
        on(bins.z[partner], bins.xyz[partner]),
        on(bins.z[partner], p.xyz),
    ]
    _regions(cells[0], p.xyz, brain_atlas, "Regions\nrecorded", ylabel=True)
    _regions(cells[1], bins.xyz[partner], brain_atlas, f"Regions\n{other}")
    _title(
        cells[0],
        f"This probe and {other}, from the same mouse, both fit better swapped "
        f"(gain {row['swap_gain']:.2f})",
        GROUP_COLORS["elsewhere"],
    )
    ax = cells[2]
    x = np.array([0, 1, 2.6, 3.6])
    ax.bar(x, fits, width=0.8, color=[EXPECTED, BETTER, EXPECTED, BETTER])
    ax.axhline(0, color=EXPECTED, lw=0.6)
    ax.set_xticks(x, ["own", "other", "own", "other"], fontsize=5)
    ax.set_xlabel(f"Recording of this probe  |  of {other}", fontsize=5)
    ax.set_title("Fit on each track", fontsize=5.5, pad=2)
    ax.set_ylabel("Fit (correlation along the probe)", fontsize=5)
    ax.tick_params(labelsize=5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    return bins.xyz[partner]


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
    for ax, name in zip(cells[2:], ("rms_ap", "rms_lf", "spike_count")):
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
    _title(
        cells[0],
        f"{100 * row['dead_fraction']:.0f}% of the channels are dead or noisy (the limit is 20%)",
        GROUP_COLORS["channels"],
    )


def figure_case(pid, ds, bins, table, scans, reference, brain_atlas, bands, tracks):
    """The figure of one insertion: a row per band, then one per test of position or channels."""
    p = _Probe(pid, ds, bins, reference, brain_atlas)
    row = table.loc[str(pid)]
    level = {lv.name: lv for lv in LEVELS}.get(row["first_level"] or "", LEVELS[-1])
    tests = tests_to_show(row)
    k = int(np.flatnonzero(scans["pids"].astype(str) == str(pid))[0])
    n = len(bands) + len(tests)
    height = 0.98 + n * ROW_INCHES
    fig = plt.figure(figsize=(8.6, height))
    grid = GridSpec(
        n,
        8,
        figure=fig,
        width_ratios=[0.34, 0.34, 0.12, 1.0, 1.0, 1.0, 1.0, 1.15],
        left=0.05,
        right=0.99,
        top=1 - 0.6 / height,
        bottom=0.62 / height,
        hspace=0.62,
        wspace=0.34,
    )
    there = None
    for r, test in enumerate(tests, start=len(bands)):
        if test == "channels":
            cells = [fig.add_subplot(grid[r, c]) for c in (0, 1, 3, 4, 5)]
            _row_channels(p, cells, row)
            continue
        cells = [fig.add_subplot(grid[r, c]) for c in (0, 1)]
        cells.append(fig.add_subplot(grid[r, 3:5]))
        if test == "swap":
            xyz = _row_swap(p, cells, row, bands)
        elif test in TRACK_TEST:
            xyz = _row_track(p, cells, row, tracks)
        else:
            xyz = _row_move(
                p, cells, test, row, np.asarray(scans[test][k], dtype=float)
            )
        if there is None:
            there = p.at(xyz)[0]
    for r, (band, cols) in enumerate(bands.items()):
        cells = [fig.add_subplot(grid[r, 0:2])]
        cells += [fig.add_subplot(grid[r, c]) for c in (3, 4, 5, 6, 7)]
        _row_band(p, cells, band, row, table, np.asarray(cols), level, there)
    first = row["first_level"]
    head = f"first marked at {first}" if first else "not marked at any level"
    fig.text(0.05, 1 - 0.06 / height, f"{pid}: {head}", fontsize=7, va="top")
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
    tracks,
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
                pid, ds, bins, table, scans, reference, brain_atlas, bands, tracks
            )
            fig.savefig(path, dpi=135, facecolor="white")
            plt.close(fig)
        out[str(pid)] = path
    return out
