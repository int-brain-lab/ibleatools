"""The alignment result figure: where the probe went, how the method decided, and how well the
recorded features match the model there.

Three blocks, left to right:

1. **Trajectories** on a coronal and a sagittal slice through the probe: the human (reference)
   alignment, the planned trajectory, the histology trace (histology methods) and the inferred
   channel positions.
2. **Decision**: the channel x trace cost matrix with the warping path (histology methods), or the
   score of every candidate trajectory against its distance from the inferred trajectory
   (ephys-only).
3. **Stripes along the probe** (top at the top): human vs predicted Cosmos and Beryl regions, the
   probe-confidence p(good) of each channel, then recorded vs predicted features.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .geometry import region_ids, valid_xyz_mask

# Recorded vs predicted feature stripes: (feature, label).
DEFAULT_STRIPE_FEATURES = (
    ("rms_lf", "RMS LF"),
    ("psd_alpha", "PSD alpha"),
    ("psd_gamma", "PSD gamma"),
    ("rms_ap", "RMS AP"),
    ("alpha_mean", "Alpha mean"),
    ("spike_width_secs", "Spike width"),
)
TRACE_STYLES = {
    "human": dict(color="tab:blue", lw=2.2, label="Human alignment"),
    "planned": dict(color="tab:orange", lw=1.6, ls="--", label="Planned"),
    "histology": dict(color="0.35", lw=1.0, ls=":", label="Histology trace"),
    "inferred": dict(color="tab:red", lw=1.8, label="Inferred"),
}


def region_rgb(brain_atlas, ids: np.ndarray) -> np.ndarray:
    """``[n, 3]`` Allen colour (0-1) of each region id; black outside the brain."""
    ids = np.asarray(ids).reshape(-1).astype(int)
    rgb = np.asarray(brain_atlas._label2rgb(ids), dtype=float)[:, :3]
    if rgb.size and np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb[ids == 0] = 0.0
    return rgb


def _region_stripe(ax, brain_atlas, ids, title):
    rgb = region_rgb(brain_atlas, ids)
    ax.imshow(np.repeat(rgb[:, None, :], 8, axis=1), aspect="auto", interpolation="nearest")
    _stripe_frame(ax, title)


def _value_stripe(ax, values, title, *, cmap="viridis", vmin=None, vmax=None):
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    im = ax.imshow(np.repeat(v, 8, axis=1), aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    _stripe_frame(ax, title)
    return im


def _stripe_frame(ax, title):
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def _group_label(ax, label, pair=True):
    """A bold label centred over a stripe, or over it and its right neighbour (``pair``)."""
    ax.text(1.06 if pair else 0.5, 1.07, label, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, fontweight="bold")


def _limits(*arrays):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [2, 98])
    return (float(lo), float(hi)) if hi > lo else (float(lo) - 1, float(hi) + 1)


def _slice_panel(ax, brain_atlas, traces: dict, view: str):
    """Traces (µm) over the atlas slice through their centre (``view`` coronal or sagittal)."""
    from iblatlas.plots import plot_points_on_slice

    pts = np.concatenate([t for t in traces.values() if t is not None and len(t)])
    centre_um = np.nanmean(pts, axis=0) * 1e6
    axis = 1 if view == "coronal" else 0
    plot_points_on_slice(np.zeros((0, 3)), coord=float(centre_um[axis]), slice=view, ax=ax,
                         brain_atlas=brain_atlas, background="boundary", show_cbar=False)
    h = 0 if view == "coronal" else 1
    for name, xyz in traces.items():
        if xyz is None or not len(xyz):
            continue
        xyz_um = np.asarray(xyz, dtype=float) * 1e6
        ax.plot(xyz_um[:, h], xyz_um[:, 2], **TRACE_STYLES[name])
    ax.set_title(f"{view.capitalize()} ({'AP' if view == 'coronal' else 'ML'} "
                 f"{centre_um[axis]:.0f} µm)", fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def _cost_panel(ax, result):
    cost = np.asarray(result.cost_matrix, dtype=float)
    finite = cost[np.isfinite(cost)]
    vmax = np.percentile(finite, 95) if finite.size else 1.0
    ax.imshow(cost, aspect="auto", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
    if result.path is not None and len(result.path):
        ax.plot(result.path[:, 1], result.path[:, 0], color="white", lw=1.2)
    ax.set_xlabel("Histology trace sample (top → bottom)", fontsize=8)
    ax.set_ylabel("Recorded channel (top → bottom)", fontsize=8)
    title = "Alignment cost matrix"
    if result.method == "histology_unit":
        title = "Fused channel + unit cost matrix"
    if result.diagnostics.get("rigid_fallback"):
        title += " (rigid fallback)"
    ax.set_title(title, fontsize=8, pad=2)


def _score_panel(ax, result):
    d = result.diagnostics
    dist = np.asarray(d.get("candidate_distance_um", []), dtype=float)
    score = np.asarray(d.get("candidate_score", []), dtype=float)
    ok = np.isfinite(dist) & np.isfinite(score)
    if ok.any():
        stage = np.asarray(d.get("candidate_stage", np.zeros(len(dist))), dtype=float)
        sc = ax.scatter(dist[ok], score[ok], c=stage[ok], s=6, cmap="viridis", alpha=0.7, lw=0)
        best = int(np.nanargmin(np.where(ok, score, np.inf)))
        ax.scatter([dist[best]], [score[best]], marker="*", s=90, color="tab:red", zorder=5,
                   label="Selected")
        if np.ptp(stage[ok]) > 0:
            cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
            cbar.set_label("Search stage", fontsize=7)
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.text(0.5, 0.5, "No candidate scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Distance from the inferred trajectory (µm)", fontsize=8)
    ax.set_ylabel("Score (lower is better)", fontsize=8)
    ax.set_title("Candidate trajectories", fontsize=8, pad=2)


def plot_alignment_result(
    result,
    brain_atlas,
    *,
    human_xyz: Optional[np.ndarray] = None,
    planned_xyz: Optional[np.ndarray] = None,
    histology_trace_xyz: Optional[np.ndarray] = None,
    features: Sequence = DEFAULT_STRIPE_FEATURES,
    metrics: Optional[dict] = None,
    fig=None,
):
    """Draw an :class:`~.result.AlignmentResult` (see the module doc for the layout).

    Args:
        result: The alignment result.
        brain_atlas: ``iblatlas.atlas.AllenAtlas``.
        human_xyz: ``[C, 3]`` reference channel positions, row order (optional).
        planned_xyz: ``[C, 3]`` planned channel positions (optional).
        histology_trace_xyz: The histology trace; defaults to the result's trace for the
            histology methods.
        features: (name, label) pairs of the recorded vs predicted stripes.
        metrics: Scores to print in the title (e.g. from :func:`~.metrics.alignment_metrics`).
        fig: A matplotlib Figure to draw into (the GUI passes its canvas figure).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    if fig is None:
        fig = plt.figure(figsize=(17, 8.5))
    features = [(name, label) for name, label in features if name in result.feature_names]
    has_human = human_xyz is not None and valid_xyz_mask(human_xyz).any()
    if histology_trace_xyz is None and result.method.startswith("histology"):
        histology_trace_xyz = result.trace_xyz
    inferred = np.where(valid_xyz_mask(result.channel_xyz)[:, None], result.channel_xyz, np.nan)
    traces = {
        "histology": histology_trace_xyz,
        "planned": None if planned_xyz is None else np.asarray(planned_xyz)[valid_xyz_mask(planned_xyz)],
        "human": None if not has_human else np.asarray(human_xyz)[valid_xyz_mask(human_xyz)],
        "inferred": inferred[np.isfinite(inferred).all(axis=1)],
    }

    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1.25, 2.6], wspace=0.22,
                     left=0.04, right=0.99, top=0.86, bottom=0.08)
    slices = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.18)
    for k, view in enumerate(("coronal", "sagittal")):
        _slice_panel(fig.add_subplot(slices[k]), brain_atlas, traces, view)
    handles = [Line2D([], [], **TRACE_STYLES[k]) for k, v in traces.items() if v is not None and len(v)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.955), ncol=2,
               frameon=False, fontsize=8)

    ax_decision = fig.add_subplot(outer[1])
    if result.cost_matrix is not None:
        _cost_panel(ax_decision, result)
    else:
        _score_panel(ax_decision, result)

    n_region = 4 if has_human else 2
    n_cols = n_region + 1 + 2 * len(features)
    stripes = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[2], wspace=0.12)
    col = 0
    for mapping in ("Cosmos", "Beryl"):
        first = ax = fig.add_subplot(stripes[col])
        if has_human:
            _region_stripe(ax, brain_atlas, region_ids(brain_atlas, human_xyz, mapping), "human")
            col += 1
            ax = fig.add_subplot(stripes[col])
        _region_stripe(ax, brain_atlas,
                       region_ids(brain_atlas, np.nan_to_num(result.channel_xyz), mapping), "pred.")
        _group_label(first, mapping, pair=has_human)
        col += 1
    ax_conf = fig.add_subplot(stripes[col])
    _value_stripe(ax_conf, result.p_good, "p(good)", cmap="RdYlGn", vmin=0, vmax=1)
    _group_label(ax_conf, "Conf.", pair=False)
    col += 1
    # Recorded and predicted share one colour scale per feature, in model (standardised) units.
    for name, label in features:
        j = result.feature_names.index(name)
        rec_j = np.where(result.valid, result.recorded_std[:, j], np.nan)
        pred_j = result.predicted_std[:, j]
        lo, hi = _limits(rec_j, pred_j)
        ax_rec = fig.add_subplot(stripes[col])
        _value_stripe(ax_rec, rec_j, "rec.", vmin=lo, vmax=hi)
        _value_stripe(fig.add_subplot(stripes[col + 1]), pred_j, "pred.", vmin=lo, vmax=hi)
        _group_label(ax_rec, label)
        col += 2

    title = f"{result.label}" + (f" -- {result.pid}" if result.pid else "")
    if metrics:
        parts = []
        for key, fmt in (("cosmos_acc", "Cosmos {:.0%}"), ("beryl_acc", "Beryl {:.0%}"),
                         ("mean_distance_um", "distance {:.0f} µm"),
                         ("cosmos_acc_high_conf", "high-confidence Cosmos {:.0%}")):
            if key in metrics and np.isfinite(metrics[key]):
                parts.append(fmt.format(metrics[key]))
        if parts:
            title += "\n" + ", ".join(parts)
    fig.suptitle(title, fontsize=10, y=0.995)
    return fig
