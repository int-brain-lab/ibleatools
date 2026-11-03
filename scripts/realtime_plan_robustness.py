"""How good must the planned trajectory be? Localize the test recordings from perturbed plans.

The ephys-only localizer starts its search from the plan and its prior is centred on it
(fitted on how far real insertions end up from their plans: a few hundred µm, ~4.5 deg). Here
each test recording's features (the last update of a replay run: 5 min of data) are localized
from plans that are

- shifted by ``SHIFTS_MM`` in a random 3-D direction, or
- tilted by ``TILTS_DEG`` about their centre, around a random axis perpendicular to the probe,

with ``N_DRAWS`` random draws each (draw k uses localizer seed k), and the unperturbed plan with
the same seeds as the baseline. Every row of ``robustness.csv`` holds the scores of the estimate
and of the perturbed plan against the human alignment; the summary gives, per perturbation, the
mean scores and the fraction of cases where the estimate is closer than the perturbed plan.

Usage::

    python scripts/realtime_plan_robustness.py
    python scripts/realtime_plan_robustness.py --kinds none shift_mm
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings ------------------------------------------------------------------------------
RUN_DIR = DEFAULT_RESULTS_DIR / "realtime" / "replay" / "speedmax_cadence10"
TRUTH_DIR = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "truth"
OUT_DIR = DEFAULT_RESULTS_DIR / "realtime" / "plan_robustness"
SHIFTS_MM = (0.25, 0.5, 1.0, 2.0)
TILTS_DEG = (5.0, 10.0, 20.0)
N_DRAWS = 4
FEATURE_SET = "lf_ap_no_detection"
METRICS = ("cosmos_acc", "beryl_acc", "mean_distance_um")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def shift_plan(xyz: np.ndarray, mm: float, rng: np.random.Generator) -> np.ndarray:
    """The plan translated by ``mm`` in a random direction."""
    return xyz + mm * 1e-3 * _unit(rng.normal(size=3))[None, :]


def tilt_plan(xyz: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    """The plan rotated by ``deg`` about its centre, around a random axis perpendicular to it."""
    ok = np.isfinite(xyz).all(axis=1)
    c = xyz[ok].mean(axis=0)
    axis = np.linalg.svd(xyz[ok] - c, full_matrices=False)[2][0]
    k = rng.normal(size=3)
    k = _unit(k - (k @ axis) * axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.deg2rad(deg)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return (xyz - c) @ R.T + c


def _bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def plot_summary(df: pd.DataFrame, path: Path, title: str = "") -> None:
    """Small multiples: distance / Cosmos / Beryl (rows) against plan shift and tilt (columns),
    perturbed plan vs estimate (mean over recordings of the per-recording mean over draws,
    95% bootstrap band), the unperturbed estimate as a dotted reference."""
    import matplotlib.pyplot as plt

    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    series = {
        "plan": ("perturbed plan", "#eb6834"),
        "estimate": ("localizer estimate", "#2a78d6"),
    }
    rows = (
        ("mean_distance_um", "Mean channel distance\nto histology (µm)"),
        ("cosmos_acc", "Cosmos accuracy"),
        ("beryl_acc", "Beryl accuracy"),
    )
    cols = (
        ("shift_mm", "Plan shift (mm, random direction)"),
        ("tilt_deg", "Plan tilt (deg, about its centre)"),
    )
    per_rec = df.groupby(["kind", "size", "pid"]).mean(numeric_only=True).reset_index()
    base = per_rec[per_rec["kind"] == "none"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9.5), sharey="row", facecolor=surface)
    for c, (kind, xlabel) in enumerate(cols):
        sub = pd.concat([base.assign(kind=kind), per_rec[per_rec["kind"] == kind]])
        sizes = sorted(sub["size"].unique())
        for r, (metric, ylabel) in enumerate(rows):
            ax = axes[r, c]
            ax.set_facecolor(surface)
            ends = []
            for key, (label, color) in series.items():
                col = f"{key}_{metric}"
                mean = np.array([sub.loc[sub["size"] == s, col].mean() for s in sizes])
                ci = np.array(
                    [
                        _bootstrap_ci(sub.loc[sub["size"] == s, col].to_numpy())
                        for s in sizes
                    ]
                )
                ax.fill_between(
                    sizes, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0
                )
                ax.plot(
                    sizes,
                    mean,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=7,
                    markeredgecolor=surface,
                    markeredgewidth=1.5,
                    label=label,
                )
                ends.append((mean[-1], label))
            if (
                c == 1
            ):  # direct labels at the line ends, pushed apart when they would collide
                lo, hi = ax.get_ylim()
                ys = sorted(((y - lo) / (hi - lo), label) for y, label in ends)
                if ys[1][0] - ys[0][0] < 0.08:
                    mid = 0.5 * (ys[0][0] + ys[1][0])
                    ys = [(mid - 0.04, ys[0][1]), (mid + 0.04, ys[1][1])]
                for y, label in ys:
                    ax.annotate(
                        label,
                        (1.0, y),
                        xycoords="axes fraction",
                        xytext=(8, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=9,
                        color=ink2,
                        annotation_clip=False,
                    )
            ref = base[f"estimate_{metric}"].mean()
            ax.axhline(ref, color=ink2, linewidth=1, linestyle=":", zorder=0)
            ax.set_xticks(sizes)
            ax.set_xticklabels([("0" if s == 0 else f"{s:g}") for s in sizes])
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.tick_params(colors=ink2, labelsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, color=ink, fontsize=10)
            if r == 2:
                ax.set_xlabel(xlabel, color=ink, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper left")
    n = per_rec["pid"].nunique()
    fig.suptitle(
        (title or "Ephys-only localization from perturbed plans")
        + f"\n{n} held-out recordings, 5 min of online data; mean over recordings ± 95% bootstrap CI; "
        "dotted: unperturbed estimate",
        color=ink,
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--plot-only", action="store_true", help="only redraw the summary figure"
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["none", "shift_mm", "tilt_deg"],
        choices=["none", "shift_mm", "tilt_deg"],
    )
    args = parser.parse_args()
    if args.plot_only:
        df = pd.read_csv(Path(args.out_dir) / "robustness.csv")
        plot_summary(df, Path(args.out_dir) / "plan_robustness.png")
        print(f"[robustness] figure -> {Path(args.out_dir) / 'plan_robustness.png'}")
        return

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer, alignment_metrics
    from ephysatlas.alignment.ephys_only import stable_seed

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "robustness.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["pid"], r["kind"], r["size"], r["draw"]) for r in rows}
    ba = AllenAtlas()
    cm = ChannelModel()
    localizers = [
        EphysOnlyLocalizer(cm, ba, feature_set=FEATURE_SET, seed=k)
        for k in range(N_DRAWS)
    ]
    cases = (
        [("none", 0.0)]
        + [("shift_mm", s) for s in SHIFTS_MM]
        + [("tilt_deg", t) for t in TILTS_DEG]
    )
    cases = [case for case in cases if case[0] in args.kinds]
    pids = sorted(
        p.name for p in Path(args.run_dir).iterdir() if (p / "updates.npz").exists()
    )
    for i, pid in enumerate(pids, 1):
        features = np.load(Path(args.run_dir) / pid / "updates.npz")["features"][-1]
        truth = np.load(TRUTH_DIR / f"{pid}.npz")
        human, plan = truth["human_xyz"], truth["planned_xyz"]
        print(f"[robustness] {i}/{len(pids)} {pid}", flush=True)
        for kind, size in cases:
            for draw in range(N_DRAWS):
                if (pid, kind, size, draw) in done:
                    continue
                rng = np.random.default_rng(
                    stable_seed("plan_robustness", pid, kind, size, draw)
                )
                if kind == "shift_mm":
                    p = shift_plan(plan, size, rng)
                elif kind == "tilt_deg":
                    p = tilt_plan(plan, size, rng)
                else:
                    p = plan
                res = localizers[draw].localize(features, p, pid="")
                est = alignment_metrics(human, res.channel_xyz, ba, valid=res.valid)
                pm = alignment_metrics(human, p, ba)
                rows.append(
                    dict(
                        pid=pid,
                        kind=kind,
                        size=size,
                        draw=draw,
                        **{f"estimate_{m}": est[m] for m in METRICS},
                        **{f"plan_{m}": pm[m] for m in METRICS},
                    )
                )
        pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    df["recovered"] = df["estimate_mean_distance_um"] < df["plan_mean_distance_um"]
    summary = (
        df.groupby(["kind", "size"])
        .agg(
            n=("pid", "size"),
            plan_distance_um=("plan_mean_distance_um", "mean"),
            estimate_distance_um=("estimate_mean_distance_um", "mean"),
            estimate_distance_median=("estimate_mean_distance_um", "median"),
            plan_cosmos=("plan_cosmos_acc", "mean"),
            estimate_cosmos=("estimate_cosmos_acc", "mean"),
            plan_beryl=("plan_beryl_acc", "mean"),
            estimate_beryl=("estimate_beryl_acc", "mean"),
            closer_than_plan=("recovered", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    plot_summary(df, out / "plan_robustness.png")


if __name__ == "__main__":
    main()
