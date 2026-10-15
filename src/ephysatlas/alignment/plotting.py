"""The alignment result figure: where the probe went, how the method decided, and how well the
recorded features match the model there.

It shows only what a user aligning a new recording has: the histology trace, the planned
trajectory (when known) and the inferred alignment -- no reference alignment or region accuracy.
Three blocks, left to right:

1. **Trajectories** on a coronal and a sagittal slice through the probe, drawn in equal square
   windows one above the other: the histology trace (top to bottom of the brain), the planned
   trajectory and the inferred channel positions, as thick semi-transparent lines.
2. **Decision**: the channel x trace cost matrix with the warping path, drawn with square cells so
   its shape follows the number of channels and trace samples (histology methods); or the score
   of every candidate trajectory against its distance from the inferred one (ephys-only).
3. **Stripes along the histology trace** (top of the brain at the top): the atlas regions of the
   trace, then the probe-confidence p(good) and the recorded vs predicted features. The predicted
   features cover the whole trace; the recorded features and the confidence sit only where the
   alignment placed the channels, so their extent and offset show the inferred probe position.
   (Ephys-only results, which have no histology trace, are drawn per channel instead.)

Offline evaluations, which know a reference alignment, can still pass it (``human_xyz``) to
draw it on the slices, and ``metrics`` to print scores in the title.
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
# Thick semi-transparent lines, drawn in this order (widest underneath).
TRACE_STYLES = {
    "histology": dict(color="#4c72b0", lw=10, alpha=0.45, solid_capstyle="round",
                      label="Histology trace"),
    "planned": dict(color="#dd8452", lw=7, alpha=0.55, solid_capstyle="round",
                    label="Planned trajectory"),
    "human": dict(color="#55a868", lw=6, alpha=0.6, solid_capstyle="round",
                  label="Reference alignment"),
    "inferred": dict(color="#c44e52", lw=5, alpha=0.9, solid_capstyle="butt",
                     label="Inferred alignment"),
}
SLICE_MARGIN_UM = 600.0
STRIPE_BACKGROUND = "0.92"  # where a stripe has no value (outside the inferred probe)


def region_rgb(brain_atlas, ids: np.ndarray) -> np.ndarray:
    """``[n, 3]`` Allen colour (0-1) of each region id; black outside the brain."""
    ids = np.asarray(ids).reshape(-1).astype(int)
    rgb = np.asarray(brain_atlas._label2rgb(ids), dtype=float)[:, :3]
    if rgb.size and np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb[ids == 0] = 0.0
    return rgb


def trace_depth_mm(trace_xyz: np.ndarray) -> np.ndarray:
    """``[L]`` arc length along the trace from its first (top) sample, in mm."""
    xyz = np.asarray(trace_xyz, dtype=float)
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1e3
    return np.r_[0.0, np.cumsum(seg)]


def values_on_trace(values: np.ndarray, channel_to_trace: np.ndarray, valid: np.ndarray,
                    n_samples: int) -> np.ndarray:
    """Per-channel ``values`` placed on the trace samples the alignment mapped them to.

    Samples several channels map to get their mean; samples skipped between two aligned channels
    (the warp stretched the probe there) take the nearest aligned value; samples outside the
    aligned segment are NaN.
    """
    values = np.asarray(values, dtype=float)
    j = np.asarray(channel_to_trace, dtype=int)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(values)
    out = np.full(n_samples, np.nan)
    if not keep.any():
        return out
    total = np.bincount(j[keep], weights=values[keep], minlength=n_samples)
    count = np.bincount(j[keep], minlength=n_samples)
    hit = count > 0
    out[hit] = total[hit] / count[hit]
    filled = np.flatnonzero(hit)
    inside = np.arange(filled.min(), filled.max() + 1)
    nearest = filled[np.clip(np.searchsorted(filled, inside), 0, len(filled) - 1)]
    left = filled[np.clip(np.searchsorted(filled, inside) - 1, 0, len(filled) - 1)]
    nearest = np.where(np.abs(inside - left) <= np.abs(nearest - inside), left, nearest)
    out[inside] = out[nearest]
    return out


def _stripe_frame(ax, title):
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(STRIPE_BACKGROUND)
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def _image_stripe(ax, image, depth_mm, title, **kwargs):
    """A stripe image whose rows span ``depth_mm`` (top of the brain at the top)."""
    extent = (0.0, 1.0, float(depth_mm[-1]), float(depth_mm[0])) if depth_mm is not None else None
    im = ax.imshow(image, aspect="auto", interpolation="nearest", extent=extent, **kwargs)
    _stripe_frame(ax, title)
    return im


def _region_stripe(ax, brain_atlas, ids, title, depth_mm=None):
    rgb = region_rgb(brain_atlas, ids)
    _image_stripe(ax, np.repeat(rgb[:, None, :], 8, axis=1), depth_mm, title)


def _value_stripe(ax, values, title, *, depth_mm=None, cmap="viridis", vmin=None, vmax=None):
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    return _image_stripe(ax, np.repeat(v, 8, axis=1), depth_mm, title, cmap=cmap, vmin=vmin, vmax=vmax)


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


def _slice_window(traces: dict) -> tuple[np.ndarray, float]:
    """Centre (µm) of the traces and the half-width of the square window both slices share."""
    pts = np.concatenate([t for t in traces.values() if t is not None and len(t)]) * 1e6
    centre = np.nanmean(pts, axis=0)
    half = np.nanmax(np.abs(pts - centre)) + SLICE_MARGIN_UM
    return centre, float(half)


def _slice_panel(ax, brain_atlas, traces: dict, view: str, centre_um: np.ndarray, half_um: float):
    """Traces over the atlas slice through ``centre_um`` (``view`` coronal or sagittal), in a
    square window of half-width ``half_um`` so both views have the same size."""
    from iblatlas.plots import plot_points_on_slice

    axis = 1 if view == "coronal" else 0
    plot_points_on_slice(np.zeros((0, 3)), coord=float(centre_um[axis]), slice=view, ax=ax,
                         brain_atlas=brain_atlas, background="boundary", show_cbar=False)
    h = 0 if view == "coronal" else 1
    for name in TRACE_STYLES:
        xyz = traces.get(name)
        if xyz is None or not len(xyz):
            continue
        xyz_um = np.asarray(xyz, dtype=float) * 1e6
        ax.plot(xyz_um[:, h], xyz_um[:, 2], **TRACE_STYLES[name])
    ax.set_xlim(centre_um[h] - half_um, centre_um[h] + half_um)
    ax.set_ylim(centre_um[2] - half_um, centre_um[2] + half_um)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{view.capitalize()} ({'AP' if view == 'coronal' else 'ML'} "
                 f"{centre_um[axis]:.0f} µm)", fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


def _cost_panel(ax, result):
    """The cost matrix with square cells (its shape follows channels x trace samples)."""
    cost = np.asarray(result.cost_matrix, dtype=float)
    finite = cost[np.isfinite(cost)]
    vmax = np.percentile(finite, 95) if finite.size else 1.0
    im = ax.imshow(cost, aspect="equal", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
    if result.path is not None and len(result.path):
        ax.plot(result.path[:, 1], result.path[:, 0], color="#4dd2ff", lw=3, alpha=0.85,
                solid_capstyle="round")
    ax.set_xlabel("Histology trace sample (top → bottom)", fontsize=8)
    ax.set_ylabel("Recorded channel (top → bottom)", fontsize=8)
    title = "Alignment cost matrix"
    if result.method == "histology_unit":
        title = "Fused channel + unit cost matrix"
    title += " and path (blue)"
    if result.diagnostics.get("rigid_fallback"):
        title += ", rigid fallback"
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(labelsize=7)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.5)
    cbar.set_label("Cost (low = match)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


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


def _trace_stripes(fig, spec, result, brain_atlas, features):
    """Stripes along the histology trace: atlas regions, then confidence and recorded features
    where the channels were aligned, predicted features everywhere."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    d = result.diagnostics
    trace = np.asarray(result.trace_xyz, dtype=float)
    n = len(trace)
    depth = trace_depth_mm(trace)
    j_map = np.asarray(d["channel_to_trace"], dtype=int)
    predicted_trace = np.asarray(d["predicted_trace_std"], dtype=float)
    aligned = j_map[np.asarray(result.valid, dtype=bool)]
    top_mm, tip_mm = depth[aligned.min()], depth[aligned.max()]

    # Columns: atlas, a gap for its labels, confidence, then a (recorded, predicted) pair per feature.
    widths = [1.0, 0.9, 1.0] + [1.0] * (2 * len(features))
    stripes = GridSpecFromSubplotSpec(1, len(widths), subplot_spec=spec, wspace=0.12, width_ratios=widths)
    axes = []

    ax_atlas = fig.add_subplot(stripes[0])
    ids = region_ids(brain_atlas, trace, "Beryl")
    _region_stripe(ax_atlas, brain_atlas, ids, "Beryl", depth)
    _group_label(ax_atlas, "Atlas", pair=False)
    # Region acronyms (right) for the segments long enough to label, depth along the trace (left).
    change = np.flatnonzero(np.diff(ids)) + 1
    starts, stops = np.r_[0, change], np.r_[change, n]
    ticks, labels = [], []
    for a, b in zip(starts, stops):
        if ids[a] != 0 and depth[b - 1] - depth[a] >= 0.25:
            ticks.append(0.5 * (depth[a] + depth[b - 1]))
            labels.append(brain_atlas.regions.acronym[ids[a]])
    ax_right = ax_atlas.twinx()
    ax_right.set_ylim(ax_atlas.get_ylim())
    ax_right.set_yticks(ticks, labels, fontsize=5.5)
    ax_right.tick_params(length=1.5, pad=1)
    for s in ax_right.spines.values():
        s.set_visible(False)
    ax_atlas.set_yticks(np.arange(0.0, depth[-1] + 1e-9, 1.0))
    ax_atlas.tick_params(labelsize=6.5)
    ax_atlas.set_ylabel("Depth along the histology trace (mm)", fontsize=7)
    axes.append(ax_atlas)

    ax_conf = fig.add_subplot(stripes[2])
    p_good = values_on_trace(result.p_good, j_map, result.valid, n)
    _value_stripe(ax_conf, p_good, "p(good)", depth_mm=depth, cmap="RdYlGn", vmin=0, vmax=1)
    _group_label(ax_conf, "Conf.", pair=False)
    axes.append(ax_conf)

    # Recorded and predicted share one colour scale per feature, in model (standardised) units.
    col = 3
    for name, label in features:
        j = result.feature_names.index(name)
        rec_j = values_on_trace(result.recorded_std[:, j], j_map, result.valid, n)
        pred_j = predicted_trace[:, j]
        lo, hi = _limits(rec_j, pred_j)
        ax_rec = fig.add_subplot(stripes[col])
        _value_stripe(ax_rec, rec_j, "rec.", depth_mm=depth, vmin=lo, vmax=hi)
        ax_pred = fig.add_subplot(stripes[col + 1])
        _value_stripe(ax_pred, pred_j, "pred.", depth_mm=depth, vmin=lo, vmax=hi)
        _group_label(ax_rec, label)
        axes += [ax_rec, ax_pred]
        col += 2

    for ax in axes:
        for y in (top_mm, tip_mm):
            ax.axhline(y, color="k", lw=0.7, ls="--")
    axes[-1].annotate("inferred\nprobe", xy=(1.0, 0.5 * (top_mm + tip_mm)),
                      xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                      fontsize=6.5, va="center", ha="left")


def _channel_stripes(fig, spec, result, brain_atlas, features):
    """Per-channel stripes (ephys-only results, which have no histology trace)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    n_cols = 2 + 2 * len(features)
    stripes = GridSpecFromSubplotSpec(1, n_cols, subplot_spec=spec, wspace=0.12)
    ax = fig.add_subplot(stripes[0])
    _region_stripe(ax, brain_atlas, region_ids(brain_atlas, np.nan_to_num(result.channel_xyz), "Beryl"),
                   "Beryl")
    _group_label(ax, "Inferred", pair=False)
    ax_conf = fig.add_subplot(stripes[1])
    _value_stripe(ax_conf, result.p_good, "p(good)", cmap="RdYlGn", vmin=0, vmax=1)
    _group_label(ax_conf, "Conf.", pair=False)
    col = 2
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


def plot_alignment_result(
    result,
    brain_atlas,
    *,
    planned_xyz: Optional[np.ndarray] = None,
    histology_trace_xyz: Optional[np.ndarray] = None,
    features: Sequence = DEFAULT_STRIPE_FEATURES,
    human_xyz: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
    fig=None,
):
    """Draw an :class:`~.result.AlignmentResult` (see the module doc for the layout).

    Args:
        result: The alignment result.
        brain_atlas: ``iblatlas.atlas.AllenAtlas``.
        planned_xyz: ``[C, 3]`` planned channel positions (optional).
        histology_trace_xyz: The histology trace; defaults to the result's trace for the
            histology methods.
        features: (name, label) pairs of the recorded vs predicted stripes.
        human_xyz: ``[C, 3]`` reference channel positions, drawn on the slices (offline
            evaluations only -- a new recording has none).
        metrics: Scores to print in the title (offline evaluations only, e.g. from
            :func:`~.metrics.alignment_metrics`).
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
    if histology_trace_xyz is None and result.method.startswith("histology"):
        histology_trace_xyz = result.trace_xyz
    inferred = np.where(valid_xyz_mask(result.channel_xyz)[:, None], result.channel_xyz, np.nan)
    traces = {
        "histology": histology_trace_xyz,
        "planned": None if planned_xyz is None else np.asarray(planned_xyz)[valid_xyz_mask(planned_xyz)],
        "human": None if human_xyz is None else np.asarray(human_xyz)[valid_xyz_mask(human_xyz)],
        "inferred": inferred[np.isfinite(inferred).all(axis=1)],
    }

    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.55, 2.25], wspace=0.2,
                     left=0.03, right=0.97, top=0.87, bottom=0.07)
    slices = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.12)
    centre, half = _slice_window(traces)
    for k, view in enumerate(("coronal", "sagittal")):
        _slice_panel(fig.add_subplot(slices[k]), brain_atlas, traces, view, centre, half)
    handles = [Line2D([], [], **TRACE_STYLES[k]) for k in TRACE_STYLES
               if traces.get(k) is not None and len(traces[k])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.965), ncol=2,
               frameon=False, fontsize=8, handlelength=2.5)

    ax_decision = fig.add_subplot(outer[1])
    if result.cost_matrix is not None:
        _cost_panel(ax_decision, result)
    else:
        _score_panel(ax_decision, result)

    on_trace = "channel_to_trace" in result.diagnostics and "predicted_trace_std" in result.diagnostics
    if on_trace:
        _trace_stripes(fig, outer[2], result, brain_atlas, features)
    else:
        _channel_stripes(fig, outer[2], result, brain_atlas, features)

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
