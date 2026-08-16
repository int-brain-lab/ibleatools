# Figure 2: Ephys Atlas interpolation summary
# Updated for the vintage-tagged Hugging Face/local model registry.
#
# Key changes:
#   1. Loads the released channel model, PCA volumes, split manifest, feature
#      ordering, and preprocessing statistics from the model registry.
#   2. Evaluation uses the exact test PIDs stored with the released vintage.
#   3. Panel e shows exactly three mean R² points per interpolation method:
#         green  = AP features
#         purple = LF features
#         blue   = spike-related features
#
# The rest of the Figure 2 layout is intentionally kept close to the
# previous implementation.

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig

from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    FEATURE_LIST,
    get_device,
)
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    evaluate_r2_per_feature,
)
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
)


# =============================================================================
# Feature groups
# =============================================================================

# These groups are defined by feature NAME, not array slices. This avoids
# silently changing the plotted group means if FEATURE_LIST ordering changes.
#
# LF = channel-level LF / PSD / CSD / spectral features up to cor_ratio.
# AP = AP-band RMS and alpha summary features.
# Spike = spike count and waveform/spike-detection features.
LF_FEATURES = (
    "rms_lf",
    "psd_lfp",
    "psd_alpha",
    "psd_beta",
    "psd_gamma",
    "psd_delta",
    "psd_theta",
    "psd_lfp_csd_diff1",
    "psd_alpha_csd_diff1",
    "psd_beta_csd_diff1",
    "psd_gamma_csd_diff1",
    "psd_delta_csd_diff1",
    "psd_theta_csd_diff1",
    "rms_lf_csd_diff1",
    "psd_residual_lfp",
    "psd_residual_alpha",
    "psd_residual_beta",
    "psd_residual_gamma",
    "psd_residual_delta",
    "psd_residual_theta",
    "decay_fit_error",
    "decay_fit_r_squared",
    "decay_n_peaks",
    "aperiodic_exponent",
    "aperiodic_offset",
    "cor_ratio",
)

AP_FEATURES = (
    "rms_ap",
    "alpha_mean",
    "alpha_std",
)

SPIKE_FEATURES = (
    "spike_count",
    "tip_time_secs",
    "recovery_time_secs",
    "peak_time_secs",
    "trough_time_secs",
    "trough_val",
    "tip_val",
    "peak_val",
    "recovery_slope",
    "depolarisation_slope",
    "repolarisation_slope",
    "polarity",
)

# User-requested panel-e colors:
GROUP_COLORS = {
    "LF": "#1b5e20",     # green
    "AP": "#7b1fa2",     # purple
    "Spike": "#1565c0",  # blue
}


# =============================================================================
# Release loading
# =============================================================================

def load_channel_release(
    *,
    vintage: str,
    device: torch.device,
    hf_repo_id: Optional[str],
    registry_root: Path = DEFAULT_REGISTRY_ROOT,
    hf_token: Optional[str] = None,
):
    """
    Resolve and load a released Ephys Atlas channel model.

    Resolution order:
      1. local registry
      2. Hugging Face repo at revision=<vintage>

    Returns a dictionary containing the model plus the frozen release artifacts
    needed to rebuild the exact evaluation data.
    """
    registry = EphysAtlasReleaseRegistry(registry_root)

    release_dir = registry.resolve_release(
        vintage,
        repo_id=hf_repo_id,
        token=hf_token,
        require_weights=True,
    )
    registry.verify_checksums(vintage)
    registry.validate_feature_order(vintage, FEATURE_LIST)

    release_features = registry.load_features(vintage)
    release_config = registry.load_config(vintage)
    split_manifest = registry.load_split(vintage)
    preprocessing_stats = registry.load_channel_preprocessing_stats(vintage)

    ckpt_path = release_dir / "models" / "channel" / "spatial_encoder.pt"
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    arch = checkpoint.get("architecture", {})
    required = (
        "f_ctx",
        "f_ephys",
        "f_out",
        "d_model",
        "nhead",
        "depth",
        "drop",
    )
    missing = [name for name in required if name not in arch]
    if missing:
        raise RuntimeError(
            f"Released checkpoint {ckpt_path} is missing architecture fields: {missing}"
        )

    def stat_tensor(name: str) -> torch.Tensor:
        if name not in preprocessing_stats:
            raise RuntimeError(
                f"Release preprocessing statistics are missing {name!r}."
            )
        return torch.as_tensor(
            preprocessing_stats[name],
            dtype=torch.float32,
        )

    model = NeighborInpaintingModel(
        f_ctx=int(arch["f_ctx"]),
        f_ephys=int(arch["f_ephys"]),
        f_out=int(arch["f_out"]),
        e_mean=stat_tensor("e_mean"),
        e_std=stat_tensor("e_std"),
        ctx_mean=stat_tensor("ctx_mean"),
        ctx_std=stat_tensor("ctx_std"),
        d_model=int(arch["d_model"]),
        nhead=int(arch["nhead"]),
        depth=int(arch["depth"]),
        drop=float(arch["drop"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    print(f"[Figure 2] loaded release vintage={vintage}")
    print(f"[Figure 2] release directory: {release_dir}")

    return {
        "registry": registry,
        "release_dir": release_dir,
        "features": release_features,
        "config": release_config,
        "split_manifest": split_manifest,
        "preprocessing_stats": preprocessing_stats,
        "model": model,
    }


# =============================================================================
# Figure-specific helpers
# =============================================================================

def _context_pc_images(context_manager, slice_xyz_um, rid_mask, shape):
    pack = context_manager.sample_context_numpy_m(
        slice_xyz_um.astype(np.float32) / 1e6,
        mode="clip",
    )
    cell = np.full(shape, np.nan, dtype=float)
    gene = np.full(shape, np.nan, dtype=float)
    cell[rid_mask] = pack["cell_pc"][:, 0]
    gene[rid_mask] = pack["gene_pc"][:, 0]
    return cell, gene


def _region_mean_image(
    train_xyz_m,
    train_y,
    query_xyz_um,
    rid_mask,
    shape,
    brain_atlas,
    fidx,
):
    from ephysatlas.spatial_encoder.utils import region_ids_from_xyz

    train_rid = region_ids_from_xyz(
        brain_atlas,
        train_xyz_m,
        mapping="Cosmos",
        mode="clip",
    )
    query_rid = region_ids_from_xyz(
        brain_atlas,
        query_xyz_um / 1e6,
        mapping="Cosmos",
        mode="clip",
    )

    global_mean = np.nanmean(train_y[:, fidx])
    means = {}
    for rid in np.unique(train_rid):
        m = train_rid == rid
        vals = train_y[m, fidx]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            means[int(rid)] = float(vals.mean())

    pred = np.array(
        [means.get(int(rid), global_mean) for rid in query_rid],
        dtype=float,
    )
    img = np.full(shape, np.nan)
    img[rid_mask] = pred
    return img


def _draw_symbol(ax, symbol, fontsize=12):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        symbol,
        ha="center",
        va="center",
        color="black",
        fontsize=fontsize,
        fontweight="bold",
    )


def _draw_empty_vector(ax, xy, width, height, color, n=7, lw=1.0):
    x0, y0 = xy
    cell_w = width / n
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (x0 + i * cell_w, y0),
                cell_w,
                height,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=color,
                lw=lw,
            )
        )


def _draw_empty_matrix(ax, xy, width, height, color, rows=4, cols=7, lw=0.8):
    x0, y0 = xy
    cw, ch = width / cols, height / rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x0 + c * cw, y0 + r * ch),
                    cw,
                    ch,
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor=color,
                    lw=lw,
                )
            )


def _read_ablation_scores(path, feature_list):
    """Read per-feature held-out R² values while preserving feature order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the interpolation ablation analysis first."
        )

    # CSV files created/saved on Windows or through Excel may be cp1252
    # rather than UTF-8. Try the common encodings in a sensible order.
    rows = None
    used_encoding = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            with path.open(
                "r",
                newline="",
                encoding=encoding,
            ) as f:
                rows = list(csv.DictReader(f))

            used_encoding = encoding
            break

        except UnicodeDecodeError:
            continue

    if rows is None:
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"Could not decode {path} using UTF-8, cp1252, or latin1.",
        )

    print(
        f"[Figure 2] loaded ablation scores from {path} "
        f"using encoding={used_encoding}"
    )

    required = {"method", "feature", "r2"}

    if not rows:
        raise ValueError(
            f"{path} is empty."
        )

    missing_columns = required.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(
            f"{path} must contain columns: method, feature, r2. "
            f"Missing: {sorted(missing_columns)}. "
            f"Found: {list(rows[0].keys())}"
        )

    methods = []
    by_method = {}

    for row in rows:
        method = str(row["method"]).strip()
        feature = str(row["feature"]).strip()

        if method not in methods:
            methods.append(method)

        try:
            score = float(row["r2"])
        except (TypeError, ValueError):
            score = np.nan

        by_method.setdefault(method, {})[feature] = score

    scores = np.full(
        (len(methods), len(feature_list)),
        np.nan,
        dtype=float,
    )

    for mi, method in enumerate(methods):
        for fi, feature in enumerate(feature_list):
            if feature in by_method[method]:
                scores[mi, fi] = by_method[method][feature]

    return methods, scores


def _feature_group_indices(feature_list, names, group_name):
    missing = [name for name in names if name not in feature_list]
    if missing:
        raise ValueError(
            f"{group_name} feature group contains features absent from "
            f"the released FEATURE_LIST: {missing}"
        )
    return np.asarray(
        [feature_list.index(name) for name in names],
        dtype=int,
    )


def _r2_stats_by_feature_group(scores, feature_list):
    """
    Convert [n_methods, n_features] per-feature R² scores into
    mean and standard deviation for AP, LF, and Spike groups.
    """
    scores = np.asarray(scores, dtype=float)

    group_names = ("AP", "LF", "Spike")
    group_features = (
        AP_FEATURES,
        LF_FEATURES,
        SPIKE_FEATURES,
    )

    means = {}
    stds = {}

    for group_name, names in zip(
        group_names,
        group_features,
    ):
        idx = _feature_group_indices(
            feature_list,
            names,
            group_name,
        )

        vals = scores[:, idx]

        group_means = np.full(
            scores.shape[0],
            np.nan,
            dtype=float,
        )
        group_stds = np.full(
            scores.shape[0],
            np.nan,
            dtype=float,
        )

        for mi in range(scores.shape[0]):
            finite_vals = vals[mi][
                np.isfinite(vals[mi])
            ]

            if finite_vals.size:
                group_means[mi] = float(
                    np.mean(finite_vals)
                )
                group_stds[mi] = float(
                    np.std(
                        finite_vals,
                        ddof=0,
                    )
                )

        means[group_name] = group_means
        stds[group_name] = group_stds

    return means, stds


def _plot_panel_d_group_bars(
    ax,
    *,
    methods,
    method_scores,
    feature_list,
):
    """
    Panel d: mean held-out R² for AP, LF, and Spike features
    for each interpolation method.

    Bar height = mean R² across features in each group.

    Colors:
        AP    = purple
        LF    = green
        Spike = blue
    """
    group_means, _ = _r2_stats_by_feature_group(
        method_scores,
        feature_list,
    )

    x = np.arange(
        len(methods),
        dtype=float,
    )

    width = 0.24

    group_order = (
        "AP",
        "LF",
        "Spike",
    )

    offsets = {
        "AP": -width,
        "LF": 0.0,
        "Spike": width,
    }

    for group_name in group_order:
        means = group_means[group_name]

        ax.bar(
            x + offsets[group_name],
            means,
            width=width,
            color=GROUP_COLORS[group_name],
            edgecolor="black",
            linewidth=0.4,
            label=group_name,
            zorder=2,
        )

    ax.axhline(
        0,
        color="black",
        lw=0.8,
        zorder=1,
    )

    ax.set_xlim(
        -0.5,
        len(methods) - 0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        methods,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        r"Mean held-out $R^2$"
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
        fontsize=6,
        handlelength=1.0,
        handletextpad=0.4,
    )

    return group_means


def plot_ephys_atlas_interpolation_summary(
    *,
    ephys,
    probe_positions,
    feature_list,
    predict_fn,
    context_manager,
    train_xyz_m,
    train_ephys,
    r2,
    ablation_scores_csv="interpolation_model_ablation_scores.csv",
    brain_atlas=None,
    coord_um=-1200,
    voxel_size_um=200,
    slice_thickness_um=200,
    observed_slice_thickness_um=200,
    features=("psd_delta", "rms_ap", "recovery_time_secs"),
    feature_titles=(
        r"PSD delta [$\mu V^2$/Hz]",
        "RMS AP [µV]",
        "Recovery time [s]",
    ),
    feature_scales=None,
    cmap="inferno",
    neighbor_radius_um=500,
    n_neighbors=8,
    seed=0,
    save_path=None,
    dpi=600,
):
    figure_style()

    brain_atlas = AllenAtlas() if brain_atlas is None else brain_atlas
    feature_scales = feature_scales or {
        "psd_delta": 1.0,
        "rms_ap": 1.0,
        "recovery_time_secs": 1.0,
    }

    rng = np.random.default_rng(seed)
    ephys = np.asarray(ephys)
    probe_positions_um = _to_um(probe_positions)

    # ------------------------------------------------------------------
    # Build coronal slice
    # ------------------------------------------------------------------
    grid = _build_coronal_slice_grid(
        brain_atlas=brain_atlas,
        coord_um=coord_um,
        voxel_size_um=voxel_size_um,
        slice_thickness_um=slice_thickness_um,
    )

    x_grid = grid["x_grid"]
    z_grid = grid["z_grid"]
    rid_mask = grid["rid_mask"]
    slice_xyz_um = grid["slice_xyz_um"]
    coord_um_actual = grid["coord_um_actual"]

    pred = np.asarray(predict_fn(slice_xyz_um))
    feat_indices = [
        feature_list.index(feature)
        for feature in features
    ]

    pred_imgs = []
    obs_imgs = []
    baseline_imgs = []

    for feature, fidx in zip(features, feat_indices):
        scale = feature_scales.get(feature, 1.0)

        pred_img = np.full(rid_mask.shape, np.nan)
        pred_img[rid_mask] = pred[:, fidx] * scale
        pred_imgs.append(pred_img)

        obs_imgs.append(
            _observed_voxel_average_image(
                ephys=ephys,
                probe_positions_um=probe_positions_um,
                feature_list=feature_list,
                feature=feature,
                x_grid=x_grid,
                z_grid=z_grid,
                coord_um_actual=coord_um_actual,
                observed_slice_thickness_um=observed_slice_thickness_um,
            )
            * scale
        )

        baseline_imgs.append(
            _region_mean_image(
                np.asarray(train_xyz_m),
                np.asarray(train_ephys),
                slice_xyz_um,
                rid_mask,
                rid_mask.shape,
                brain_atlas,
                fidx,
            )
            * scale
        )

    merfish_img, agea_img = _context_pc_images(
        context_manager,
        slice_xyz_um,
        rid_mask,
        rid_mask.shape,
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0], 9.35)

    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.05, 1.12, 2.75, 0.95],
        hspace=0.24,
    )

    # ------------------------------------------------------------------
    # Panel a
    # ------------------------------------------------------------------
    gs = GridSpecFromSubplotSpec(
        1,
        7,
        subplot_spec=outer[0],
        width_ratios=[1, 0.08, 1, 0.08, 1, 0.12, 1],
        wspace=0.015,
    )
    a_axes = [
        fig.add_subplot(gs[0, i])
        for i in range(7)
    ]

    rms_vals = np.concatenate(
        [
            obs_imgs[1][np.isfinite(obs_imgs[1])],
            pred_imgs[1][np.isfinite(pred_imgs[1])],
        ]
    )
    rvmin, rvmax = _safe_percentile_limits(rms_vals)

    _draw_slice_heatmap(
        fig=fig,
        ax=a_axes[0],
        img=obs_imgs[1],
        x_grid=x_grid,
        z_grid=z_grid,
        coord_um_actual=coord_um_actual,
        voxel_size_um=voxel_size_um,
        title="Sparse ephys dataset",
        cmap=cmap,
        vmin=rvmin,
        vmax=rvmax,
    )

    _draw_symbol(a_axes[1], "+")

    vmin, vmax = _safe_percentile_limits(merfish_img)
    _draw_slice_heatmap(
        fig=fig,
        ax=a_axes[2],
        img=merfish_img,
        x_grid=x_grid,
        z_grid=z_grid,
        coord_um_actual=coord_um_actual,
        voxel_size_um=voxel_size_um,
        title="MERFISH cellular densities",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    _draw_symbol(a_axes[3], "+")

    vmin, vmax = _safe_percentile_limits(agea_img)
    _draw_slice_heatmap(
        fig=fig,
        ax=a_axes[4],
        img=agea_img,
        x_grid=x_grid,
        z_grid=z_grid,
        coord_um_actual=coord_um_actual,
        voxel_size_um=voxel_size_um,
        title="AGEA molecular densities",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    _draw_symbol(a_axes[5], "→", fontsize=16)

    _draw_slice_heatmap(
        fig=fig,
        ax=a_axes[6],
        img=pred_imgs[1],
        x_grid=x_grid,
        z_grid=z_grid,
        coord_um_actual=coord_um_actual,
        voxel_size_um=voxel_size_um,
        title="Dense ephys map",
        cmap=cmap,
        vmin=rvmin,
        vmax=rvmax,
    )

    _panel_label(a_axes[0], "a")

    # ------------------------------------------------------------------
    # Panel b
    # ------------------------------------------------------------------
    gs = GridSpecFromSubplotSpec(
        1,
        5,
        subplot_spec=outer[1],
        width_ratios=[1.45, 0.86, 0.16, 0.72, 0.92],
        wspace=0.10,
    )

    ax_slice, ax_in, ax_arr, ax_model, ax_out = [
        fig.add_subplot(gs[0, i])
        for i in range(5)
    ]

    _draw_neighbor_cartoon_slice(
        ax=ax_slice,
        probe_positions_um=probe_positions_um,
        ephys=ephys,
        brain_atlas=brain_atlas,
        neighbor_radius_um=neighbor_radius_um,
        n_neighbors=n_neighbors,
        rng=rng,
    )

    ax_in.axis("off")
    ax_in.text(
        0.5,
        0.78,
        "AGEA/MERFISH PCs",
        ha="center",
        va="center",
        color="red",
    )
    _draw_empty_vector(
        ax_in,
        (0.17, 0.61),
        0.66,
        0.10,
        "red",
    )

    ax_in.text(
        0.5,
        0.43,
        "Recorded ephys from\nneighboring probes",
        ha="center",
        va="center",
        color="green",
    )
    _draw_empty_matrix(
        ax_in,
        (0.16, 0.10),
        0.68,
        0.20,
        "green",
    )

    ax_arr.axis("off")

    ax_model.axis("off")
    ax_model.add_patch(
        Rectangle(
            (0.06, 0.28),
            0.88,
            0.44,
            transform=ax_model.transAxes,
            facecolor="black",
        )
    )
    ax_model.text(
        0.5,
        0.5,
        "Interpolation\nModel",
        color="white",
        ha="center",
        va="center",
    )

    ax_out.axis("off")
    ax_out.text(
        0.64,
        0.72,
        "Predicted ephys\nat the query voxel",
        ha="center",
        va="center",
        color="red",
    )

    for y_src, y_dst in (
        (0.70, 0.60),
        (0.30, 0.40),
    ):
        fig.add_artist(
            ConnectionPatch(
                xyA=(0.92, y_src),
                coordsA=ax_in.transAxes,
                xyB=(0.06, y_dst),
                coordsB=ax_model.transAxes,
                arrowstyle="->",
                lw=1.25,
                color="black",
                mutation_scale=10,
            )
        )

    fig.add_artist(
        ConnectionPatch(
            xyA=(0.94, 0.50),
            coordsA=ax_model.transAxes,
            xyB=(0.30, 0.50),
            coordsB=ax_out.transAxes,
            arrowstyle="->",
            lw=1.25,
            color="black",
            mutation_scale=10,
        )
    )

    _draw_empty_vector(
        ax_out,
        (0.38, 0.42),
        0.52,
        0.10,
        "red",
    )

    _panel_label(ax_slice, "b")

    # ------------------------------------------------------------------
    # Panel c
    # ------------------------------------------------------------------
    gs_c = GridSpecFromSubplotSpec(
        len(features),
        4,
        subplot_spec=outer[2],
        width_ratios=[1, 1, 1, 0.045],
        hspace=0.04,
        wspace=0.025,
    )

    col_titles = [
        "Observed data",
        "Region mean baseline",
        "Interpolation model",
    ]

    for i, (feat_title, imgs) in enumerate(
            zip(
                feature_titles,
                zip(
                    obs_imgs,
                    baseline_imgs,
                    pred_imgs,
                ),
            )
    ):
        finite_arrays = [
            img[np.isfinite(img)]
            for img in imgs
            if np.isfinite(img).any()
        ]

        vals = np.concatenate(finite_arrays)
        vmin, vmax = _safe_percentile_limits(vals)

        last_im = None

        for j, img in enumerate(imgs):
            ax = fig.add_subplot(
                gs_c[i, j]
            )

            last_im = _draw_slice_heatmap(
                fig=fig,
                ax=ax,
                img=img,
                x_grid=x_grid,
                z_grid=z_grid,
                coord_um_actual=coord_um_actual,
                voxel_size_um=voxel_size_um,
                title=col_titles[j] if i == 0 else "",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            if i == 0:
                ax.set_title(
                    col_titles[j],
                    pad=1.0,
                )

            if j == 0:
                ax.text(
                    -0.035,
                    0.5,
                    feat_title,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                )

                if i == 0:
                    _panel_label(
                        ax,
                        "c",
                    )

        cax = fig.add_subplot(
            gs_c[i, 3]
        )

        cb = fig.colorbar(
            last_im,
            cax=cax,
        )

        cb.ax.tick_params(
            length=2,
            pad=1,
        )

    # ------------------------------------------------------------------
    # Panel d
    # ------------------------------------------------------------------
    # Full-width comparison of interpolation methods.
    #
    # Each method has three bars:
    #   green  = AP
    #   purple = LF
    #   blue   = Spike
    #
    # Bar height = mean R² across features in that group.
    # Error bar  = standard deviation across those feature R² values.
    gs_d = GridSpecFromSubplotSpec(
        1,
        1,
        subplot_spec=outer[3],
    )

    ax_d = fig.add_subplot(
        gs_d[0, 0]
    )

    methods, method_scores = _read_ablation_scores(
        ablation_scores_csv,
        feature_list,
    )

    group_means = _plot_panel_d_group_bars(
        ax_d,
        methods=methods,
        method_scores=method_scores,
        feature_list=feature_list,
    )

    _panel_label(
        ax_d,
        "d",
    )

    print(
        "\n[Figure 2 panel d] mean R² by method:"
    )

    for mi, method in enumerate(methods):
        print(
            f"  {method}: "
            f"AP={group_means['AP'][mi]:.4f}, "
            f"LF={group_means['LF'][mi]:.4f}, "
            f"Spike={group_means['Spike'][mi]:.4f}"
        )

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.055,
    )

    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

        return save_path

    return fig

# =============================================================================
# General plotting helpers
# =============================================================================

def _to_um(x):
    x = np.asarray(x, dtype=float)
    if x.size and np.nanmax(np.abs(x)) < 50:
        x = x * 1e6
    return x


def _nearest_indices(axis, values):
    axis = np.asarray(axis)
    values = np.asarray(values)

    order = np.argsort(axis)
    axis_s = axis[order]

    idx = np.searchsorted(axis_s, values)
    idx = np.clip(
        idx,
        1,
        len(axis_s) - 1,
    )

    left = axis_s[idx - 1]
    right = axis_s[idx]

    idx_s = np.where(
        np.abs(values - left) <= np.abs(values - right),
        idx - 1,
        idx,
    )

    return order[idx_s]


def _safe_percentile_limits(vals, p=(1, 99)):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.nanpercentile(
        vals,
        p,
    )

    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or vmin == vmax
    ):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if vmin == vmax:
        vmax = vmin + 1.0

    return float(vmin), float(vmax)


def _build_coronal_slice_grid(
    *,
    brain_atlas,
    coord_um,
    voxel_size_um,
    slice_thickness_um,
):
    x_atlas = _to_um(brain_atlas.bc.xscale)
    y_atlas = _to_um(brain_atlas.bc.yscale)
    z_atlas = _to_um(brain_atlas.bc.zscale)

    label_raw = brain_atlas.label

    if label_raw.shape != (
        len(y_atlas),
        len(x_atlas),
        len(z_atlas),
    ):
        raise ValueError(
            "Expected brain_atlas.label layout [Y, X, Z]. "
            f"Got {label_raw.shape}."
        )

    x_order = np.argsort(x_atlas)
    y_order = np.argsort(y_atlas)
    z_order = np.argsort(z_atlas)

    x_atlas = x_atlas[x_order]
    y_atlas = y_atlas[y_order]
    z_atlas = z_atlas[z_order]

    label = label_raw[
        np.ix_(
            y_order,
            x_order,
            z_order,
        )
    ]

    y_inds = np.where(
        np.abs(y_atlas - coord_um)
        <= slice_thickness_um / 2
    )[0]

    if len(y_inds) == 0:
        y_inds = np.array(
            [
                np.argmin(
                    np.abs(
                        y_atlas - coord_um
                    )
                )
            ]
        )

    coord_um_actual = float(
        np.mean(
            y_atlas[y_inds]
        )
    )

    x_grid = np.arange(
        np.floor(
            x_atlas.min()
            / voxel_size_um
        )
        * voxel_size_um,
        np.ceil(
            x_atlas.max()
            / voxel_size_um
        )
        * voxel_size_um
        + voxel_size_um,
        voxel_size_um,
    )

    z_grid = np.arange(
        np.floor(
            z_atlas.min()
            / voxel_size_um
        )
        * voxel_size_um,
        np.ceil(
            z_atlas.max()
            / voxel_size_um
        )
        * voxel_size_um
        + voxel_size_um,
        voxel_size_um,
    )

    Xg, Zg = np.meshgrid(
        x_grid,
        z_grid,
        indexing="ij",
    )

    xi_atlas = _nearest_indices(
        x_atlas,
        Xg.ravel(),
    )
    zi_atlas = _nearest_indices(
        z_atlas,
        Zg.ravel(),
    )

    label_slab = label[
        y_inds,
        :,
        :,
    ]

    rid_grid = np.any(
        label_slab[
            :,
            xi_atlas,
            zi_atlas,
        ]
        > 0,
        axis=0,
    )

    rid_mask = rid_grid.reshape(
        Xg.shape
    )

    slice_xyz_um = np.column_stack(
        [
            Xg[rid_mask],
            np.full(
                np.sum(rid_mask),
                coord_um_actual,
            ),
            Zg[rid_mask],
        ]
    )

    return {
        "x_grid": x_grid,
        "z_grid": z_grid,
        "rid_mask": rid_mask,
        "slice_xyz_um": slice_xyz_um,
        "coord_um_actual": coord_um_actual,
    }


def _observed_voxel_average_image(
    *,
    ephys,
    probe_positions_um,
    feature_list,
    feature,
    x_grid,
    z_grid,
    coord_um_actual,
    observed_slice_thickness_um,
):
    xyz_obs_um = probe_positions_um.reshape(
        -1,
        3,
    )
    ephys_obs = ephys.reshape(
        -1,
        ephys.shape[-1],
    )

    # Ignore zero-filled invalid channel positions.
    valid_xyz = ~np.all(
        xyz_obs_um == 0.0,
        axis=1,
    )
    finite_xyz = np.isfinite(
        xyz_obs_um
    ).all(axis=1)

    in_slice = (
        valid_xyz
        & finite_xyz
        & (
            np.abs(
                xyz_obs_um[:, 1]
                - coord_um_actual
            )
            <= observed_slice_thickness_um / 2
        )
    )

    xyz_obs_um = xyz_obs_um[
        in_slice
    ]
    ephys_obs = ephys_obs[
        in_slice
    ]

    obs_sum = np.zeros(
        (
            len(x_grid),
            len(z_grid),
        ),
        dtype=float,
    )
    obs_count = np.zeros_like(
        obs_sum
    )

    if len(xyz_obs_um) == 0:
        return np.full_like(
            obs_sum,
            np.nan,
        )

    xi = _nearest_indices(
        x_grid,
        xyz_obs_um[:, 0],
    )
    zi = _nearest_indices(
        z_grid,
        xyz_obs_um[:, 2],
    )

    fidx = feature_list.index(
        feature
    )
    vals = ephys_obs[
        :,
        fidx,
    ]
    good = np.isfinite(vals)

    np.add.at(
        obs_sum,
        (
            xi[good],
            zi[good],
        ),
        vals[good],
    )
    np.add.at(
        obs_count,
        (
            xi[good],
            zi[good],
        ),
        1,
    )

    obs_img = np.full_like(
        obs_sum,
        np.nan,
    )
    occupied = obs_count > 0
    obs_img[occupied] = (
        obs_sum[occupied]
        / obs_count[occupied]
    )

    return obs_img


def _draw_slice_heatmap(
    *,
    fig,
    ax,
    img,
    x_grid,
    z_grid,
    coord_um_actual,
    voxel_size_um,
    title,
    cmap,
    vmin,
    vmax,
    show_colorbar=False,
    colorbar_label=None,
    pad_um=0,
):
    empty_pts = np.zeros(
        (0, 3),
        dtype=float,
    )

    plot_points_on_slice(
        empty_pts,
        coord=int(coord_um_actual),
        slice="coronal",
        ax=ax,
        cmap="Greys",
    )

    im = ax.imshow(
        img.T,
        origin="lower",
        extent=[
            x_grid.min()
            - voxel_size_um / 2,
            x_grid.max()
            + voxel_size_um / 2,
            z_grid.min()
            - voxel_size_um / 2,
            z_grid.max()
            + voxel_size_um / 2,
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(
        title,
        pad=2,
    )

    _clean_slice_axis(ax)
    ax.margins(0)
    ax.set_anchor("C")

    ax.set_xlim(
        x_grid.min()
        - voxel_size_um / 2
        - pad_um,
        x_grid.max()
        + voxel_size_um / 2,
    )

    ax.set_ylim(
        z_grid.min()
        - voxel_size_um / 2,
        z_grid.max()
        + voxel_size_um / 2,
    )

    if show_colorbar:
        cb = fig.colorbar(
            im,
            ax=ax,
            shrink=0.5,
            fraction=0.026,
            pad=0.004,
        )

        if colorbar_label is not None:
            cb.set_label(
                colorbar_label,
                labelpad=2,
            )

        cb.ax.tick_params(
            length=2,
            pad=1,
        )

    return im


def _clean_slice_axis(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


def _panel_label(ax, label):
    ax.text(
        -0.06,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _draw_neighbor_cartoon_slice(
    *,
    ax,
    probe_positions_um,
    ephys,
    brain_atlas=None,
    neighbor_radius_um=500,
    n_neighbors=8,
    rng=None,
    zoom_half_width_um=500,
):
    if rng is None:
        rng = np.random.default_rng(0)

    probe_xyz = np.asarray(
        probe_positions_um,
        dtype=float,
    )

    n_probes, n_channels, _ = probe_xyz.shape

    valid_per_channel = (
        np.isfinite(
            probe_xyz
        ).all(axis=-1)
        & ~np.all(
            probe_xyz == 0.0,
            axis=-1,
        )
    )

    valid_probe_ids = np.where(
        valid_per_channel.sum(
            axis=1
        )
        > 20
    )[0]

    if len(valid_probe_ids) == 0:
        raise ValueError(
            "No valid probe trajectories found."
        )

    query_pid = int(
        rng.choice(
            valid_probe_ids
        )
    )

    query_trace = probe_xyz[
        query_pid
    ]

    query_valid_ch = np.where(
        valid_per_channel[
            query_pid
        ]
    )[0]

    query_ch = int(
        rng.choice(
            query_valid_ch
        )
    )

    query_xyz = query_trace[
        query_ch
    ]

    all_xyz = probe_xyz.reshape(
        -1,
        3,
    )
    all_pid = np.repeat(
        np.arange(n_probes),
        n_channels,
    )

    good = (
        np.isfinite(
            all_xyz
        ).all(axis=1)
        & ~np.all(
            all_xyz == 0.0,
            axis=1,
        )
        & (
            all_pid
            != query_pid
        )
    )

    candidate_xyz = all_xyz[
        good
    ]
    candidate_pid = all_pid[
        good
    ]

    if len(candidate_xyz) == 0:
        raise ValueError(
            "No valid neighboring probe channels found."
        )

    d = np.linalg.norm(
        candidate_xyz
        - query_xyz[None, :],
        axis=1,
    )

    within = np.where(
        d
        <= neighbor_radius_um
    )[0]

    if len(within) > 0:
        order = within[
            np.argsort(
                d[within]
            )[:n_neighbors]
        ]
    else:
        order = np.argsort(
            d
        )[:n_neighbors]

    nn_xyz = candidate_xyz[
        order
    ]
    nn_pids = np.unique(
        candidate_pid[
            order
        ]
    )

    for pid in nn_pids:
        tr = probe_xyz[
            pid
        ]

        good_tr = (
            np.isfinite(
                tr
            ).all(axis=1)
            & ~np.all(
                tr == 0.0,
                axis=1,
            )
        )

        if good_tr.sum() < 2:
            continue

        ax.plot(
            tr[good_tr, 0],
            tr[good_tr, 2],
            color="0.55",
            lw=0.6,
            alpha=0.8,
            zorder=2,
        )

    good_q = (
        np.isfinite(
            query_trace
        ).all(axis=1)
        & ~np.all(
            query_trace == 0.0,
            axis=1,
        )
    )

    ax.plot(
        query_trace[good_q, 0],
        query_trace[good_q, 2],
        color="black",
        lw=0.9,
        alpha=0.95,
        zorder=3,
        clip_on=True,
    )

    ax.scatter(
        nn_xyz[:, 0],
        nn_xyz[:, 2],
        s=6,
        color="green",
        edgecolor="none",
        alpha=0.9,
        zorder=5,
        clip_on=True,
    )

    ax.scatter(
        [query_xyz[0]],
        [query_xyz[2]],
        s=12,
        color="red",
        edgecolor="none",
        zorder=6,
        clip_on=True,
    )

    ax.set_xlim(
        query_xyz[0]
        - zoom_half_width_um,
        query_xyz[0]
        + zoom_half_width_um,
    )
    ax.set_ylim(
        query_xyz[2]
        - zoom_half_width_um,
        query_xyz[2]
        + zoom_half_width_um,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_anchor("C")

    ax.set_xlabel(
        "ML (µm)"
    )
    ax.set_ylabel(
        "DV (µm)"
    )
    ax.set_title(
        "Query voxel and neighbors",
        pad=2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="red",
            markersize=3.5,
            label="query voxel",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="green",
            markersize=3.5,
            label="neighboring voxels",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=0.9,
            label="query probe",
        ),
        Line2D(
            [0],
            [0],
            color="0.55",
            lw=0.9,
            label="neighboring probes",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower left",
        fontsize=5.5,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.1,
        labelspacing=0.25,
    )


# =============================================================================
# Model inference
# =============================================================================

@torch.no_grad()
def make_ephys_atlas_predict_fn(
    *,
    model,
    context_manager,
    ephys_mean,
    ephys_std,
    device,
    batch_size=4096,
):
    """
    Predict from anatomical context only (no neighboring observations), matching
    the previous Figure 2 dense-map visualization behavior.
    """
    model.eval()

    ephys_mean = torch.as_tensor(
        ephys_mean,
        dtype=torch.float32,
        device=device,
    )
    ephys_std = torch.as_tensor(
        ephys_std,
        dtype=torch.float32,
        device=device,
    )

    use_amp = device.type == "cuda"

    def predict_fn(xyz_um):
        xyz_um = np.asarray(
            xyz_um,
            dtype=np.float32,
        )

        out = []

        for i0 in range(
            0,
            len(xyz_um),
            batch_size,
        ):
            xyz_m = (
                xyz_um[
                    i0 : i0 + batch_size
                ]
                / 1e6
            )

            pack = context_manager.sample_context_numpy_m(
                xyz_m,
                mode="clip",
            )
            ctx = np.concatenate(
                [
                    pack["cell_pc"],
                    pack["gene_pc"],
                ],
                axis=-1,
            )

            # ContextAtlasManager returns raw PCA context. The released model
            # expects standardized context.
            ctx_mean = (
                model.ctx_mean
                .detach()
                .cpu()
                .numpy()
            )
            ctx_std = (
                model.ctx_std
                .detach()
                .cpu()
                .numpy()
            )

            has_ctx = (
                np.sum(
                    np.abs(ctx),
                    axis=1,
                )
                != 0
            )

            ctx_std_np = np.zeros_like(
                ctx,
                dtype=np.float32,
            )
            ctx_std_np[has_ctx] = (
                ctx[has_ctx]
                - ctx_mean
            ) / (
                ctx_std
                + 1e-8
            )

            ctx_q = torch.as_tensor(
                ctx_std_np,
                dtype=torch.float32,
                device=device,
            )
            p_q = torch.as_tensor(
                xyz_m,
                dtype=torch.float32,
                device=device,
            )

            B = len(xyz_m)
            e_n = torch.zeros(
                (
                    B,
                    1,
                    ephys_mean.numel(),
                ),
                dtype=torch.float32,
                device=device,
            )
            p_n = p_q[
                :,
                None,
                :,
            ].clone()
            mask = torch.zeros(
                (
                    B,
                    1,
                ),
                dtype=torch.bool,
                device=device,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                _, mu_std = model(
                    ctx_q,
                    p_q,
                    e_n,
                    p_n,
                    mask,
                )

            out.append(
                (
                    mu_std.float()
                    * ephys_std
                    + ephys_mean
                )
                .detach()
                .cpu()
                .numpy()
            )

        return np.concatenate(
            out,
            axis=0,
        )

    return predict_fn


def _raw_train_bank(
    train_loader,
    e_mean,
    e_std,
):
    """
    Recover raw-scale training-bank features for the Region mean baseline.
    """
    collate = train_loader.collate_fn

    xyz = np.asarray(
        collate.bank_xyz,
        dtype=np.float32,
    )
    y_std = np.asarray(
        collate.bank_feat,
        dtype=np.float32,
    )

    mean = (
        torch.as_tensor(
            e_mean
        )
        .detach()
        .cpu()
        .numpy()[None]
    )
    std = (
        torch.as_tensor(
            e_std
        )
        .detach()
        .cpu()
        .numpy()[None]
    )

    return (
        xyz,
        y_std * std + mean,
    )


# =============================================================================
# Config / main
# =============================================================================

@dataclass
class RunConfig:
    data_dir: Path = Path("../")

    # Vintage = Hugging Face revision/tag.
    vintage: str = "2026_W26"

    # Set this to your actual Hugging Face model repository.
    # Example:
    #   "alonsaguy/ephys-atlas-models"
    # or an IBL organization repo.
    hf_repo_id: Optional[str] = None

    # Leave None after running `hf auth login`.
    hf_token: Optional[str] = None

    # Separate local model registry; not your Git project directory.
    registry_root: Path = DEFAULT_REGISTRY_ROOT

    batch_size_train: int = 1024
    batch_size_eval: int = 1024

    device: torch.device = get_device()
    seed: int = 0

    coord_um: int = -1200

    # Per-feature R² values for ALL interpolation/baseline methods.
    ablation_scores_csv: Path = Path(
        "interpolation_model_ablation_scores.csv"
    )

    save_path: Path = Path(
        "figure2_ephys_atlas_interpolation.pdf"
    )


def main():
    cfg = RunConfig()

    torch.manual_seed(
        cfg.seed
    )
    np.random.seed(
        cfg.seed
    )

    print(
        f"Using device: {cfg.device}"
    )

    # ------------------------------------------------------------------
    # Resolve the released model and all model-defining artifacts.
    # ------------------------------------------------------------------
    release = load_channel_release(
        vintage=cfg.vintage,
        device=cfg.device,
        hf_repo_id=cfg.hf_repo_id,
        registry_root=cfg.registry_root,
        hf_token=cfg.hf_token,
    )

    release_dir = release[
        "release_dir"
    ]
    release_config = release[
        "config"
    ]
    split_manifest = release[
        "split_manifest"
    ]
    preprocessing_stats = release[
        "preprocessing_stats"
    ]
    feature_list = release[
        "features"
    ]
    model = release[
        "model"
    ]

    # ------------------------------------------------------------------
    # Take data/preprocessing identifiers from the release itself.
    # ------------------------------------------------------------------
    data_cfg = release_config.get(
        "data",
        {},
    )
    context_cfg = release_config.get(
        "context",
        {},
    )
    neighbors_cfg = release_config.get(
        "neighbors",
        {},
    )

    project = str(
        data_cfg.get(
            "project",
            "ea_active",
        )
    )
    agg = str(
        data_cfg.get(
            "agg",
            "agg_full",
        )
    )
    saved_vintage = str(
        data_cfg.get(
            "vintage",
            cfg.vintage,
        )
    )

    if saved_vintage != cfg.vintage:
        raise RuntimeError(
            f"Requested vintage={cfg.vintage}, but release config "
            f"records data vintage={saved_vintage}."
        )

    n_cell_pcs = int(
        context_cfg.get(
            "n_cell_pcs",
            50,
        )
    )
    n_gene_pcs = int(
        context_cfg.get(
            "n_gene_pcs",
            50,
        )
    )

    radius_um = int(
        neighbors_cfg.get(
            "radius_um",
            500,
        )
    )
    m_max = int(
        neighbors_cfg.get(
            "m_max",
            8,
        )
    )

    # ------------------------------------------------------------------
    # PCA volumes come directly from the released vintage.
    # ------------------------------------------------------------------
    ctx_manager = ContextAtlasManager(
        AtlasPCAConfig(
            n_cell_pcs=n_cell_pcs,
            n_gene_pcs=n_gene_pcs,
        ),
        regenerate_context=False,
        output_dir=release_dir / "context",
    )

    # ------------------------------------------------------------------
    # Load the source ephys data for the same vintage.
    # ------------------------------------------------------------------
    (
        pid_names,
        ephys,
        probe_positions,
        _,
    ) = LoadInsertionData(
        project=project,
        agg=agg,
        VINTAGE=saved_vintage,
        path_data=cfg.data_dir,
    )

    pid_names = [
        str(x)
        for x in pid_names
    ]

    # ------------------------------------------------------------------
    # Rebuild loaders using the EXACT saved PID split and train-time
    # preprocessing statistics.
    # ------------------------------------------------------------------
    loaders = (
        build_channels_plus_emptyvoxels_with_neighbors(
            ctx_manager,
            ephys,
            probe_positions,
            RADIUS_UM=radius_um,
            M_MAX=m_max,
            pid_names=pid_names,
            batch_size_train=cfg.batch_size_train,
            batch_size_eval=cfg.batch_size_eval,
            seed=cfg.seed,
            split_manifest=split_manifest,
            preprocessing_stats=preprocessing_stats,
        )
    )

    (
        train_loader,
        _,
        _,
        test_loader,
        e_mean,
        e_std,
        ctx_mean,
        ctx_std,
        split_info,
    ) = loaders

    print(
        "[Figure 2] evaluation split:",
        f"train={len(split_info['p_tr_names'])},",
        f"val={len(split_info['p_va_names'])},",
        f"test={len(split_info['p_te_names'])}",
    )

    # Sanity checks against the released model.
    if int(e_mean.numel()) != len(feature_list):
        raise RuntimeError(
            f"Release has {len(feature_list)} features but rebuilt "
            f"evaluation preprocessing has {int(e_mean.numel())}."
        )

    if int(model.e_mean.numel()) != len(feature_list):
        raise RuntimeError(
            "Loaded model output dimensionality does not match features.json."
        )

    # ------------------------------------------------------------------
    # Evaluate the exact released model on its exact saved test PIDs.
    # ------------------------------------------------------------------
    r2 = evaluate_r2_per_feature(
        model,
        test_loader,
        e_mean,
        e_std,
        device=cfg.device,
    )

    r2_np = np.asarray(
        r2.detach().cpu(),
        dtype=float,
    )

    print(
        "[Figure 2] released model mean test R²:",
        float(
            np.nanmean(
                r2_np
            )
        ),
    )

    # ------------------------------------------------------------------
    # Region-mean baseline in panel c uses ONLY the released training bank.
    # ------------------------------------------------------------------
    train_xyz_m, train_ephys = _raw_train_bank(
        train_loader,
        e_mean,
        e_std,
    )

    predict_fn = make_ephys_atlas_predict_fn(
        model=model,
        context_manager=ctx_manager,
        ephys_mean=e_mean,
        ephys_std=e_std,
        device=cfg.device,
    )

    plot_ephys_atlas_interpolation_summary(
        ephys=ephys,
        probe_positions=probe_positions,
        feature_list=feature_list,
        predict_fn=predict_fn,
        context_manager=ctx_manager,
        train_xyz_m=train_xyz_m,
        train_ephys=train_ephys,
        r2=r2_np,
        ablation_scores_csv=cfg.ablation_scores_csv,
        brain_atlas=AllenAtlas(),
        coord_um=cfg.coord_um,
        save_path=cfg.save_path,
        seed=9,
    )

    print(
        f"saved {cfg.save_path}"
    )


if __name__ == "__main__":
    main()
