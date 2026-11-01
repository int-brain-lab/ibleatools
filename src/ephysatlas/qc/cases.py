"""Example insertions of the QC: which insertions to show, and one figure per insertion.

Each figure shows a probe from top to tip: its regions, the residuals of its LF band, AP band and
waveforms against the region references (in MADs of each feature in each region; a band that is
fine is pale), and how much more typical the recording would be at another depth or sideways. The
title gives the first level that marks the insertion and the tests it fails.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .levels import LEVELS  # noqa: E402
from .measures import SHIFTS_UM, SIDEWAYS_UM  # noqa: E402
from .regions import parcel_labels, region_labels  # noqa: E402

CLIM = 4.0  # MADs, the limits of the residual images
PER_LEVEL = 18  # insertions shown of each first level
NOT_MARKED = 10  # insertions shown that no level marks


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


def _region_stripe(ax, xyz, brain_atlas):
    """The Cosmos region of each bin, in the atlas colors."""
    lab = region_labels(xyz, brain_atlas, "Cosmos")
    regions = brain_atlas.regions
    rows = np.zeros(len(lab), dtype=int)
    for r in np.unique(lab):
        rows[lab == r] = np.flatnonzero(regions.id == r)[0]
    rgb = regions.rgb[rows] / 255.0
    rgb[lab == 0] = 1.0
    ax.imshow(rgb[:, None, :], aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_ylabel("Depth bin (top to tip)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title("Regions", fontsize=6)
    for b in np.flatnonzero(np.diff(lab) != 0):
        ax.axhline(b + 0.5, color="white", lw=0.4)
    for k in range(len(lab)):
        if lab[k] != 0 and (k == 0 or lab[k] != lab[k - 1]):
            name = regions.acronym[rows[k]]
            ax.text(0, k, name, fontsize=4.5, va="top", ha="center", color="0.15")


def figure_case(pid, bins, table, scans, reference, brain_atlas, bands):
    """The figure of one insertion."""
    i = int(np.flatnonzero(bins.pids.astype(str) == str(pid))[0])
    row = table.loc[str(pid)]
    e = reference.residuals(bins.z[i], parcel_labels(bins.xyz[i], brain_atlas), [i])
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(7.0, 2.9),
        gridspec_kw=dict(width_ratios=[0.35, 1.5, 0.6, 0.9, 1.0], wspace=0.28),
    )
    _region_stripe(axes[0], bins.xyz[i], brain_atlas)
    for ax, (name, band) in zip(axes[1:4], bands.items()):
        im = ax.imshow(
            e[:, band],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-CLIM,
            vmax=CLIM,
            interpolation="nearest",
        )
        ax.set_title(
            f"{ {'lf': 'LF', 'ap': 'AP', 'wf': 'Waveforms'}[name] }: "
            f"z {row[f'z_{name}_level']:.1f} / {row[f'z_{name}_shape']:.1f}",
            fontsize=6,
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("Features", fontsize=5.5)
    k = np.flatnonzero(scans["pids"].astype(str) == str(pid))[0]
    ax = axes[4]
    ax.axhline(0, color="0.7", lw=0.5)
    ax.plot(
        SHIFTS_UM,
        scans["depth"][k],
        "o-",
        ms=2,
        lw=0.8,
        color="0.15",
        label="Along the probe",
    )
    n = len(SIDEWAYS_UM)
    side = scans["sideways"][k]
    for j, (axis, color) in enumerate(
        (("left-right", "#8a3f9e"), ("front-back", "#b98cc6"))
    ):
        ax.plot(
            SIDEWAYS_UM,
            side[j * n : (j + 1) * n],
            "o-",
            ms=2,
            lw=0.8,
            color=color,
            label=axis,
        )
    ax.set_xlabel("Move (µm)", fontsize=5.5)
    ax.set_title(
        f"Gain elsewhere: z {row['z_depth']:.1f} / {row['z_sideways']:.1f}", fontsize=6
    )
    ax.tick_params(labelsize=5)
    ax.legend(frameon=False, fontsize=4.5, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(im, ax=axes[1:4], fraction=0.02, pad=0.015)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Residual (MADs)", fontsize=5)
    first = row["first_level"] if isinstance(row["first_level"], str) else ""
    failed = row[f"{first}_failed"] if first else ""
    swap = (
        ""
        if not np.isfinite(row["swap_gain"])
        else f", swap gain {row['swap_gain']:.2f}"
    )
    fig.suptitle(
        f"{pid}: {'first marked at ' + first if first else 'not marked at any level'}"
        f"{'  (' + failed.replace(';', ', ') + ')' if failed else ''}"
        f"; dead or noisy channels {100 * row['dead_fraction']:.0f}%{swap}",
        fontsize=6.5,
        y=1.0,
    )
    return fig


def render_examples(
    pids,
    bins,
    table,
    scans,
    reference,
    brain_atlas,
    bands,
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
            fig = figure_case(pid, bins, table, scans, reference, brain_atlas, bands)
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        out[str(pid)] = path
    return out
