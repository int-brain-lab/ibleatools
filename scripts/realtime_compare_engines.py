"""Compare two scored replay runs: the Python pipeline (SpikeGLX tools) and the C++ engine (Open Ephys plugin).

Reads ``update_metrics.csv``, ``final_metrics.csv`` and ``timings.csv`` written by
``realtime_evaluate.py`` in each run directory, prints the paired comparison over the shared
insertions (Wilcoxon signed-rank on the last update, and on the seed means when present), and
saves ``engines.png`` in the first run's parent directory: accuracy against recording time for
both engines with the planned trajectory as reference, and the per-snippet compute time.

Usage::

    python scripts/realtime_compare_engines.py speedmax_cadence10 cpp_speedmax_cadence10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings ------------------------------------------------------------------------------
REPLAY_DIR = DEFAULT_RESULTS_DIR / "realtime" / "replay"
LABELS = {
    "speedmax_cadence10": "Python (SpikeGLX)",
    "cpp_speedmax_cadence10": "C++ (Open Ephys plugin)",
}
METRICS = (
    ("mean_distance_um", "Mean channel distance\nto histology (µm)"),
    ("cosmos_acc", "Cosmos accuracy"),
    ("beryl_acc", "Beryl accuracy"),
)


def paired(a: pd.Series, b: pd.Series) -> str:
    from scipy.stats import wilcoxon

    d = (b - a).dropna()
    p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
    return f"{a.mean():8.3f} {b.mean():8.3f}  diff {d.mean():+8.3f}  p={p:.2f}"


def plot(runs: list, updates: dict, finals: dict, timings: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    colors = ["#2a78d6", "#eb6834"]
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 3.8),
        facecolor=surface,
        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]),
    )
    for ax, (metric, ylabel) in zip(axes[:3], METRICS):
        for run, color in zip(runs, colors):
            u = updates[run]
            per_t = u.groupby("acquired_s")[metric]
            mean = per_t.mean()
            sem = per_t.std() / np.sqrt(per_t.count())
            t = mean.index.to_numpy() / 60.0
            ax.fill_between(
                t,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(t, mean, color=color, linewidth=2, label=LABELS.get(run, run))
        plan = finals[runs[0]][f"planned_{metric}"].mean()
        ax.axhline(
            plan,
            color=ink2,
            linewidth=1.2,
            linestyle="--",
            zorder=0,
            label="planned trajectory",
        )
        ax.set_xlabel("Recording time used (min)", color=ink, fontsize=10)
        ax.set_ylabel(ylabel, color=ink, fontsize=10)
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper right")

    ax = axes[3]
    names = [LABELS.get(r, r).replace(" (", "\n(") for r in runs]
    compute = [timings[r]["compute_s_mean"].mean() for r in runs]
    bars = ax.bar(names, compute, color=colors[: len(runs)], width=0.55)
    ax.axhline(5.0, color=ink2, linewidth=1.2, linestyle="--", zorder=0)
    ax.annotate(
        "snippet length (5 s)",
        (1.0, 5.0),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=ink2,
    )
    for b, v in zip(bars, compute):
        ax.annotate(
            f"{v:.1f} s",
            (b.get_x() + b.get_width() / 2, v / 2),
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )
    ax.set_ylabel("Feature compute per 5 s snippet (s)", color=ink, fontsize=10)
    ax.set_title(
        "Python: worker pool, 2 threads each\nC++: one snippet at a time, all cores",
        color=ink2,
        fontsize=8.5,
    )
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9)
    n = len(set.intersection(*[set(finals[r]["pid"]) for r in runs]))
    fig.suptitle(
        f"Real-time ephys-only localization: Python and C++ engines on {n} held-out recordings "
        "(5 s snippet every 10 s, max-speed replay; mean ± 95% CI over recordings, one seed per update)",
        color=ink,
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "runs", nargs=2, help="run names under REPLAY_DIR (reference first)"
    )
    args = parser.parse_args()
    runs = list(args.runs)
    updates, finals, timings = {}, {}, {}
    for run in runs:
        d = REPLAY_DIR / run
        updates[run] = pd.read_csv(d / "update_metrics.csv")
        finals[run] = pd.read_csv(d / "final_metrics.csv")
        timings[run] = pd.read_csv(d / "timings.csv")
    shared = sorted(set(finals[runs[0]]["pid"]) & set(finals[runs[1]]["pid"]))
    a = finals[runs[0]].set_index("pid").loc[shared]
    b = finals[runs[1]].set_index("pid").loc[shared]
    print(
        f"{len(shared)} shared insertions: {runs[0]} vs {runs[1]} (means, paired Wilcoxon)"
    )
    for prefix in ("online", "online_seedmean", "offline_release_seedmean", "planned"):
        for metric, _ in METRICS:
            col = f"{prefix}_{metric}"
            if col in a and col in b:
                print(f"  {col:42s} {paired(a[col], b[col])}")
    for run in runs:
        t = timings[run]
        print(
            f"  {run}: compute {t['compute_s_mean'].mean():.1f} s/snippet, localize "
            f"{t['localize_s_mean'].mean():.2f} s, dropped {int(t['dropped'].sum())}"
        )
    path = REPLAY_DIR / "engines.png"
    plot(runs, updates, finals, timings, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
