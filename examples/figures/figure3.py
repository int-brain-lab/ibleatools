from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from skimage.measure import marching_cubes

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig
from iblatlas.atlas import AllenAtlas
from iblatlas.genomics import merfish

# Loader + atlas helpers swapped to this branch's shim; the figure body below is unchanged.
from _release import (
    cosmos_ids_for_xyz,
    region_color,
    region_id,
    unit_release,
)
from ephysatlas.unit_level_encoder.gmm_models import move


@dataclass
class FigureConfig:
    repo_id: str = "int-brain-lab/ea-encoder-unit"
    vintage: str = "2026_W26"
    token: Optional[str] = None
    cache_dir: Optional[Path] = None
    save_path: Path = Path("figure3_unit_level_atlas.pdf")
    dpi: int = 600
    seed: int = 0

    # Panel a: individual negative-dominant units on the brain.
    panel_a_max_units: int = 30_000
    panel_a_mesh_stride: int = 4
    panel_a_clip_quantiles: tuple[float, float] = (0.01, 0.99)
    panel_a_elev: float = 16.0
    # Rotate 90 degrees in the xy plane so the camera faces the hemisphere
    # containing the mirrored observations.
    panel_a_azim: float = 135.0
    panel_a_zoom: float = 1.6
    panel_a_brain_alpha: float = 0.25

    # Panel b: manual-feature space.
    panel_b_regions: tuple[str, ...] = ("Isocortex", "HB")
    panel_b_max_units_per_region: int = 6_000
    panel_b_elev: float = 18.0
    panel_b_azim: float = -55.0
    panel_b_axis_quantiles: tuple[float, float] = (0.005, 0.995)

    # Panel c: MERFISH class vs putative GMM component co-occurrence.
    panel_c_voxel_um: int = 200
    panel_c_min_units_per_voxel: int = 3
    panel_c_max_cell_types: Optional[int] = 30

    # Panel e.
    panel_d_regions: tuple[str, ...] = ("Isocortex", "TH", "MB", "CB")
    panel_d_samples_per_voxel: int = 96


def _panel_label(ax, label):
    """Add a panel label that works for both 2-D and 3-D axes."""

    kwargs = dict(
        transform=ax.transAxes,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    if hasattr(ax, "text2D"):
        # Axes3D
        ax.text2D(
            -0.08,
            1.04,
            label,
            **kwargs,
        )
    else:
        # Standard 2-D Axes
        ax.text(
            -0.08,
            1.04,
            label,
            **kwargs,
        )


def _clean_3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.grid(False)
    try:
        ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
    except Exception:
        pass


def _brain_surface_mesh(
    ba: AllenAtlas,
    *,
    stride: int = 4,
):
    """
    Build a coherent outer-brain triangular mesh from the IBL Allen annotation.

    IBL's ``ba.label`` is a 3-D annotation volume in AP x ML x DV order.
    Rather than plotting thousands of independent ``ba.surface`` points, this
    function downsamples the brain mask and runs marching cubes to obtain an
    actual surface.  The result is much easier to read as a brain volume.

    Returns
    -------
    vertices_xyz_m : (N, 3)
        Mesh vertices in IBL xyz coordinates (metres).
    faces : (M, 3)
        Triangle indices.
    """
    label = np.asarray(ba.label)

    # In the IBL atlas, non-zero label indices correspond to annotated brain.
    brain = label != 0

    stride = max(int(stride), 1)
    brain_ds = brain[::stride, ::stride, ::stride]

    # marching_cubes returns continuous coordinates in array order:
    # AP, ML, DV.
    verts_ap_ml_dv, faces, _, _ = marching_cubes(
        brain_ds.astype(np.float32),
        level=0.5,
    )

    # Undo the downsampling to recover coordinates in original voxel-index units.
    verts_ap_ml_dv *= float(stride)

    # BrainCoordinates.i2xyz expects ML, AP, DV index order.
    idx_ml_ap_dv = np.c_[
        verts_ap_ml_dv[:, 1],
        verts_ap_ml_dv[:, 0],
        verts_ap_ml_dv[:, 2],
    ]

    vertices_xyz_m = ba.bc.i2xyz(idx_ml_ap_dv)

    return (
        np.asarray(vertices_xyz_m, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def _bin_xyz_feature(
    xyz_m: np.ndarray,
    values: np.ndarray,
    *,
    bin_um: int = 200,
):
    """
    Average a unit-level feature within regular 3-D spatial bins.

    Plotting one point per 200-um occupied bin produces a substantially clearer
    brain-wide map than drawing every 25-um atlas voxel or every individual
    neuron, while matching the anatomical-context scale used by the atlas.
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(values)
    xyz = xyz[valid]
    values = values[valid]

    if len(xyz) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)

    bin_m = float(bin_um) * 1e-6

    # Use a fixed origin aligned to the bin width so the binning is deterministic.
    origin = np.floor(np.nanmin(xyz, axis=0) / bin_m) * bin_m
    ijk = np.floor((xyz - origin) / bin_m).astype(np.int64)

    unique_ijk, inv = np.unique(
        ijk,
        axis=0,
        return_inverse=True,
    )

    sums = np.bincount(
        inv,
        weights=values,
        minlength=len(unique_ijk),
    )
    counts = np.bincount(
        inv,
        minlength=len(unique_ijk),
    )

    mean_values = sums / np.maximum(counts, 1)

    # Plot at the mean observed xyz within each occupied bin rather than at the
    # geometric bin center; this keeps the points tied to real sampled locations.
    xyz_sum = np.column_stack(
        [
            np.bincount(
                inv,
                weights=xyz[:, dim],
                minlength=len(unique_ijk),
            )
            for dim in range(3)
        ]
    )
    xyz_mean = xyz_sum / counts[:, None]

    return (
        xyz_mean.astype(np.float32),
        mean_values.astype(np.float32),
        counts.astype(int),
    )


def _subsample_feature_bins(
    xyz,
    values,
    counts,
    *,
    max_points: int,
    seed: int,
):
    """
    Deterministically subsample occupied feature bins only when necessary.

    Preferentially retain better-sampled bins by sampling with probability
    proportional to sqrt(unit count).
    """
    if len(xyz) <= max_points:
        return xyz, values, counts

    rng = np.random.default_rng(seed)
    p = np.sqrt(np.maximum(counts, 1)).astype(float)
    p /= p.sum()

    keep = rng.choice(
        len(xyz),
        size=max_points,
        replace=False,
        p=p,
    )

    return xyz[keep], values[keep], counts[keep]


def _raw_feature_limits(values: np.ndarray):
    """
    Robust display limits in ORIGINAL feature units.

    A sequential scale is used unless the distribution actually spans zero,
    in which case a diverging scale centered on zero is more meaningful.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return 0.0, 1.0, "viridis"

    lo, hi = np.quantile(v, [0.02, 0.98])

    if lo < 0 < hi:
        lim = max(abs(lo), abs(hi))
        return -lim, lim, "coolwarm"

    if hi <= lo:
        hi = lo + 1e-8

    return float(lo), float(hi), "viridis"


def _set_equal_3d_limits(ax, xyz_m, *, zoom: float = 1.0):
    """
    Give all three axes an equal physical scale so the brain is not distorted.

    `zoom` changes only the rendered camera framing, not the physical xyz limits.
    """
    xyz_um = np.asarray(xyz_m, dtype=float) * 1e6
    mins = np.nanmin(xyz_um, axis=0)
    maxs = np.nanmax(xyz_um, axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        # Matplotlib >= 3.6 supports the zoom keyword directly.
        ax.set_box_aspect((1, 1, 1), zoom=float(zoom))
    except TypeError:
        # Older Matplotlib fallback.
        try:
            ax.set_box_aspect((1, 1, 1))
            ax.dist = 10.0 / max(float(zoom), 1e-6)
        except Exception:
            pass
    except Exception:
        pass


def negative_dominant_mask(waveforms: np.ndarray) -> np.ndarray:
    """
    Keep units whose max-absolute-amplitude channel is negative-dominant.

    The representative channel definition is identical to the waveform-feature
    extractor used throughout this figure.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    keep = np.zeros(len(waveforms), dtype=bool)

    for i, waveform in enumerate(waveforms):
        channel = int(
            np.unravel_index(
                np.argmax(np.abs(waveform)),
                waveform.shape,
            )[0]
        )
        trace = waveform[channel]
        keep[i] = abs(float(np.min(trace))) >= abs(float(np.max(trace)))

    return keep


def _parabolic_extremum_offset(trace: np.ndarray, index: int) -> float:
    """
    Sub-sample extremum position using a 3-point quadratic interpolation.

    Returns an offset in samples relative to `index`, clipped to ±0.5 samples.
    At waveform boundaries the discrete position is retained.
    """
    if index <= 0 or index >= len(trace) - 1:
        return 0.0

    ym1 = float(trace[index - 1])
    y0 = float(trace[index])
    yp1 = float(trace[index + 1])

    denom = ym1 - 2.0 * y0 + yp1
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return 0.0

    offset = 0.5 * (ym1 - yp1) / denom
    return float(np.clip(offset, -0.5, 0.5))


def extract_three_waveform_features_continuous(
    waveforms: np.ndarray,
    sampling_rate_hz: float,
):
    """
    Pre-peak value, post-trough peak value, and continuous trough-to-peak duration.

    Amplitudes are read from the observed normalized waveform. Duration is made
    continuous by quadratic interpolation around both the trough and the
    post-trough peak.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    out = np.full((len(waveforms), 3), np.nan, dtype=np.float32)
    dt_ms = 1000.0 / float(sampling_rate_hz)

    for i, waveform in enumerate(waveforms):
        channel = int(
            np.unravel_index(
                np.argmax(np.abs(waveform)),
                waveform.shape,
            )[0]
        )
        trace = waveform[channel]

        trough = int(np.argmin(trace))
        pre_peak = int(np.argmax(trace[: trough + 1]))
        post_peak = trough + int(np.argmax(trace[trough:]))

        trough_sub = trough + _parabolic_extremum_offset(trace, trough)
        post_peak_sub = post_peak + _parabolic_extremum_offset(trace, post_peak)

        duration_ms = max(
            (post_peak_sub - trough_sub) * dt_ms,
            0.0,
        )

        out[i] = (
            float(trace[pre_peak]),
            float(trace[post_peak]),
            float(duration_ms),
        )

    return out, (
        "Pre-peak value",
        "Peak value",
        "Duration (ms)",
    )


def _robust_limits(values, quantiles=(0.01, 0.99)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(values, quantiles)
    if hi <= lo:
        hi = lo + 1e-8
    return float(lo), float(hi)


def draw_panel_a(fig, spec, data, cfg, fig_cfg, ba, negative_mask):
    """
    Panel a: individual negative-dominant units over the Allen brain surface.

    No spatial binning is applied. This matches the exploratory visualization
    that revealed the smooth anatomical structure in the manual features.
    """
    features, names = extract_three_waveform_features_continuous(
        data.waveforms,
        cfg.waveform_sampling_rate_hz,
    )

    # Display order matches the older qualitative figure:
    # pre-peak, duration, peak.
    feature_order = (0, 2, 1)

    valid = (
        np.asarray(negative_mask, dtype=bool)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(data.xyz_m).all(axis=1)
    )
    ids = np.flatnonzero(valid)

    if len(ids) > fig_cfg.panel_a_max_units:
        rng = np.random.default_rng(fig_cfg.seed + 11)
        ids = rng.choice(
            ids,
            size=fig_cfg.panel_a_max_units,
            replace=False,
        )

    mesh_xyz, mesh_faces = _brain_surface_mesh(
        ba,
        stride=fig_cfg.panel_a_mesh_stride,
    )
    vertices_um = mesh_xyz * 1e6
    triangles = vertices_um[mesh_faces]

    gs = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=spec,
        wspace=0.02,
    )

    axes = []

    for column, feature_index in enumerate(feature_order):
        name = names[feature_index]
        ax = fig.add_subplot(
            gs[0, column],
            projection="3d",
        )
        axes.append(ax)

        # Very light anatomical envelope.
        mesh = Poly3DCollection(
            triangles,
            facecolor="0.82",
            edgecolor="none",
            alpha=fig_cfg.panel_a_brain_alpha,
            rasterized=True,
        )
        ax.add_collection3d(mesh)

        values_all = features[valid, feature_index]
        vmin, vmax = _robust_limits(
            values_all,
            fig_cfg.panel_a_clip_quantiles,
        )

        xyz = np.asarray(data.xyz_m[ids], dtype=float)
        values = features[ids, feature_index]

        sc = ax.scatter(
            xyz[:, 0] * 1e6,
            xyz[:, 1] * 1e6,
            xyz[:, 2] * 1e6,
            c=values,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            s=0.1,
            alpha=0.58,
            depthshade=False,
            rasterized=True,
        )

        # Above the horizontal plane, viewed from the data-containing
        # hemisphere after a 90° azimuth rotation relative to the prior view.
        ax.view_init(
            elev=fig_cfg.panel_a_elev,
            azim=fig_cfg.panel_a_azim,
        )
        _set_equal_3d_limits(
            ax,
            mesh_xyz,
            zoom=fig_cfg.panel_a_zoom,
        )

        ax.set_axis_off()

        ax.set_title(name, pad=1)
        _clean_3d(ax)

        cb = fig.colorbar(
            sc,
            ax=ax,
            shrink=0.48,
            pad=0.00,
        )
        cb.set_label(name, fontsize=6)
        cb.ax.tick_params(labelsize=5)

    _panel_label(axes[0], "a")



def _region_rgb_string(color):
    """Convert an IBL region color into a Plotly-compatible rgb() string."""
    arr = np.asarray(color).reshape(-1)

    if len(arr) < 3:
        return "rgb(80,80,80)"

    rgb = arr[:3].astype(float)

    # region_color may return [0,1] floats or [0,255] integers.
    if np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0

    rgb = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _panel_b_raw_data(
    data,
    cfg,
    fig_cfg,
    ba,
    negative_mask,
):
    """
    Return continuous manual-feature values for Isocortex and HB using only
    negative-dominant units.
    """
    features, names = extract_three_waveform_features_continuous(
        data.waveforms,
        cfg.waveform_sampling_rate_hz,
    )
    cosmos = np.asarray(
        cosmos_ids_for_xyz(
            ba,
            data.xyz_m,
        )
    )

    rng = np.random.default_rng(fig_cfg.seed + 17)
    out = {}

    for acronym in fig_cfg.panel_b_regions:
        rid = region_id(ba, acronym)
        ids = np.flatnonzero(
            negative_mask
            & np.isfinite(features).all(axis=1)
            & (cosmos == rid)
        )

        if len(ids) > fig_cfg.panel_b_max_units_per_region:
            ids = rng.choice(
                ids,
                fig_cfg.panel_b_max_units_per_region,
                replace=False,
            )

        # Required axis order:
        # x = pre-peak, y = duration, z = peak.
        points = np.column_stack(
            [
                features[ids, 0],
                features[ids, 2],
                features[ids, 1],
            ]
        ).astype(np.float32)

        out[acronym] = {
            "ids": ids,
            "points": points,
            "color": region_color(ba, acronym),
        }

    axis_names = (
        names[0],
        names[2],
        names[1],
    )
    return out, axis_names


def draw_panel_b(fig, spec, data, cfg, fig_cfg, ba, negative_mask):
    """
    Panel b: Isocortex vs HB in manual waveform-feature space.

    Duration is placed on the y-axis and the box aspect/camera are chosen to
    make between-region variation along that dimension visually prominent.
    """
    ax = fig.add_subplot(
        spec,
        projection="3d",
    )

    region_data, names = _panel_b_raw_data(
        data,
        cfg,
        fig_cfg,
        ba,
        negative_mask,
    )

    pooled = []

    for acronym in fig_cfg.panel_b_regions:
        pts = region_data[acronym]["points"]
        pooled.append(pts)
        color = region_data[acronym]["color"]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=0.80,
            alpha=0.20,
            color=color,
            label=acronym,
            rasterized=True,
            depthshade=False,
        )

    pooled = np.concatenate(pooled, axis=0)

    ax.set_xlabel(names[0], labelpad=2)
    ax.set_ylabel(names[1], labelpad=2)
    ax.set_zlabel(names[2], labelpad=2)

    # Robust limits prevent a small number of tails from collapsing the cloud.
    qlo, qhi = fig_cfg.panel_b_axis_quantiles
    for setter, dim in (
        (ax.set_xlim, 0),
        (ax.set_ylim, 1),
        (ax.set_zlim, 2),
    ):
        lo, hi = np.quantile(
            pooled[:, dim][np.isfinite(pooled[:, dim])],
            [qlo, qhi],
        )
        setter(float(lo), float(hi))

    # Elongating y makes the dominant duration variation easier to see.
    try:
        ax.set_box_aspect((1.0, 1.75, 1.0))
    except Exception:
        pass

    ax.view_init(
        elev=fig_cfg.panel_b_elev,
        azim=fig_cfg.panel_b_azim,
    )

    legend = ax.legend(
        frameon=False,
        loc="upper left",
    )
    for txt in legend.get_texts():
        label = txt.get_text()
        if label in region_data:
            txt.set_color(region_data[label]["color"])

    _panel_label(ax, "b")



def _box(ax, xy, width, height, text, *, facecolor="black", textcolor="white", fontsize=7):
    ax.add_patch(FancyBboxPatch(xy, width, height, transform=ax.transAxes,
                               boxstyle="round,pad=0.012,rounding_size=0.015",
                               facecolor=facecolor, edgecolor="black", lw=1.0))
    ax.text(xy[0] + width/2, xy[1] + height/2, text, transform=ax.transAxes,
            ha="center", va="center", color=textcolor, fontsize=fontsize)


def _arrow(ax, start, end, color="black"):
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, mutation_scale=9))


def _gmm_component_assignments(
    standardized_latents: np.ndarray,
    model_gmm,
):
    """
    Hard-assign observed unit latents to the global Gaussian component with
    highest diagonal-Gaussian likelihood.
    """
    z = np.asarray(standardized_latents, dtype=np.float32)
    mu = model_gmm.means.detach().cpu().numpy().astype(np.float32)
    log_var = model_gmm.log_var.detach().cpu().numpy().astype(np.float32)

    if log_var.ndim == 1:
        if log_var.shape[0] == mu.shape[0]:
            log_var = np.repeat(
                log_var[:, None],
                mu.shape[1],
                axis=1,
            )
        else:
            log_var = np.broadcast_to(
                log_var[None, :],
                mu.shape,
            )

    var = np.exp(log_var)
    diff = z[:, None, :] - mu[None, :, :]

    log_likelihood = -0.5 * np.sum(
        diff * diff / np.maximum(var[None, :, :], 1e-8)
        + log_var[None, :, :],
        axis=-1,
    )
    return np.argmax(log_likelihood, axis=1).astype(np.int64)


def _merfish_class_volume_and_names():
    """
    Load the processed MERFISH CLASS-level density volume and the exact class
    names corresponding to the returned volume channels.

    Important:
    `merfish.load_volume()` returns a label vector that is guaranteed to match
    the volume channels. This avoids the 339-vs-338 type mismatch that occurs
    when a dataframe is paired manually with a cached volume.

    Returns
    -------
    volume : np.ndarray, [n_classes, ml, dv, ap]
        Processed MERFISH class-composition volume.
    names : np.ndarray[str], [n_classes]
        Human-readable MERFISH class names.
    """
    # Official IBL API: labels match the returned volume channels exactly.
    volume, class_ids, _ = merfish.load_volume(
        level="class",
        label="processed",
    )
    volume = np.asarray(volume, dtype=np.float32)
    class_ids = np.asarray(class_ids)

    # The class table is the second dataframe returned by merfish.load().
    _, df_classes, *_ = merfish.load()

    # Find the actual human-readable class-name column.
    if "class" in df_classes.columns:
        name_column = "class"
    else:
        preferred = (
            "class_label",
            "class_name",
            "name",
            "label",
        )
        name_column = next(
            (
                column
                for column in preferred
                if column in df_classes.columns
            ),
            None,
        )

        if name_column is None:
            string_columns = []
            for column in df_classes.columns:
                series = df_classes[column]
                if (
                    series.dtype == object
                    or str(series.dtype).startswith("string")
                    or str(series.dtype).startswith("category")
                ):
                    lower = column.lower()
                    if not any(
                        token in lower
                        for token in ("color", "rgba", "hex")
                    ):
                        string_columns.append(column)

            if len(string_columns) == 1:
                name_column = string_columns[0]
            elif len(string_columns) > 1:
                ranked = sorted(
                    string_columns,
                    key=lambda column: (
                        0 if "class" in column.lower() else 1,
                        0 if "label" in column.lower() else 1,
                        0 if "name" in column.lower() else 1,
                        column,
                    ),
                )
                name_column = ranked[0]

    if name_column is None:
        raise RuntimeError(
            "Could not identify the MERFISH class-name column. "
            f"Available columns: {list(df_classes.columns)}"
        )

    # The load_volume() documentation states that class_ids match
    # df_classes.index. Reindex rather than assuming dataframe row order.
    class_rows = df_classes.reindex(class_ids)

    if class_rows[name_column].isna().any():
        missing = class_ids[
            class_rows[name_column].isna().to_numpy()
        ]
        raise RuntimeError(
            "Some MERFISH class IDs returned by load_volume() were not found "
            f"in df_classes.index. Missing IDs: {missing[:10]}"
        )

    names = (
        class_rows[name_column]
        .astype(str)
        .to_numpy()
    )

    if len(names) != volume.shape[0]:
        raise RuntimeError(
            "MERFISH class names and class-density channels do not match: "
            f"{len(names)} names vs {volume.shape[0]} channels."
        )

    print(
        "[figure 3 panel c] MERFISH CLASS level:",
        f"{volume.shape[0]} classes",
    )
    print(
        "[figure 3 panel c] class-name column:",
        name_column,
    )
    print(
        "[figure 3 panel c] classes:",
        list(names),
    )

    return volume, names


def _sample_merfish_class_density(
    ba,
    xyz_m: np.ndarray,
    volume: np.ndarray,
):
    """
    Sample MERFISH class densities at xyz locations.

    The context atlas is mirrored to the left hemisphere and downsampled by 8,
    matching the ContextAtlasManager convention used by this project.
    """
    xyz = np.asarray(xyz_m, dtype=np.float32).copy()
    xyz[:, 0] = -np.abs(xyz[:, 0])

    indices = ba.bc.xyz2i(
        xyz,
        mode="clip",
    )

    # volume is [cell_type, Xc, Zc, Yc]
    _, Xc, Zc, Yc = volume.shape
    xi = np.clip(
        np.round(indices[:, 0] / 8).astype(int),
        0,
        Xc - 1,
    )
    yi = np.clip(
        np.round(indices[:, 1] / 8).astype(int),
        0,
        Yc - 1,
    )
    zi = np.clip(
        np.round(indices[:, 2] / 8).astype(int),
        0,
        Zc - 1,
    )

    return volume[:, xi, zi, yi].T.astype(np.float32)


def _weighted_cooccurrence_matrix(
    component_fraction: np.ndarray,
    merfish_fraction: np.ndarray,
    weights: np.ndarray,
):
    """
    Weighted cosine co-occurrence across observed voxels.

    Each GMM component and MERFISH class is represented by its non-negative
    abundance vector across observed voxels. The weighted cosine similarity is:

        0 -> no spatial co-occurrence
        1 -> perfectly proportional spatial co-occurrence

    Returns
    -------
    score : [n_components, n_merfish_types]
        Values bounded to [0, 1].
    """
    Y = np.asarray(component_fraction, dtype=np.float64)
    X = np.asarray(merfish_fraction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    w = np.maximum(w, 0.0)
    if not np.any(w > 0):
        return np.zeros(
            (Y.shape[1], X.shape[1]),
            dtype=np.float64,
        )

    # sqrt(w) weighting lets the usual cosine similarity implement the desired
    # weighted inner product.
    sqrt_w = np.sqrt(w / w.sum())[:, None]

    Yw = Y * sqrt_w
    Xw = X * sqrt_w

    numerator = Yw.T @ Xw

    y_norm = np.sqrt(
        np.sum(Yw * Yw, axis=0)
    )
    x_norm = np.sqrt(
        np.sum(Xw * Xw, axis=0)
    )

    denominator = (
        y_norm[:, None]
        * x_norm[None, :]
    )

    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )

    return np.clip(score, 0.0, 1.0)


def _panel_c_cooccurrence(
    data,
    standardized_latents,
    model_gmm,
    ba,
    fig_cfg,
    negative_mask,
):
    """
    Build the observed-voxel MERFISH-class × GMM-component association map.
    """
    xyz = np.asarray(data.xyz_m, dtype=np.float32).copy()
    xyz[:, 0] = -np.abs(xyz[:, 0])

    valid = (
        np.asarray(negative_mask, dtype=bool)
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(standardized_latents).all(axis=1)
    )
    unit_ids = np.flatnonzero(valid)

    xyz_valid = xyz[unit_ids]
    z_valid = np.asarray(
        standardized_latents[unit_ids],
        dtype=np.float32,
    )
    component = _gmm_component_assignments(
        z_valid,
        model_gmm,
    )
    n_components = int(
        model_gmm.means.shape[0]
    )

    # Aggregate observed units into the same physical scale used by the atlas.
    bin_m = float(fig_cfg.panel_c_voxel_um) * 1e-6
    origin = np.floor(
        np.nanmin(xyz_valid, axis=0) / bin_m
    ) * bin_m
    keys = np.floor(
        (xyz_valid - origin[None]) / bin_m
    ).astype(np.int64)

    _, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True,
    )
    n_voxels = int(inverse.max()) + 1
    counts = np.bincount(
        inverse,
        minlength=n_voxels,
    )

    # Mean xyz of each occupied voxel.
    xyz_sum = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=xyz_valid[:, dim],
                minlength=n_voxels,
            )
            for dim in range(3)
        ]
    )
    voxel_xyz = xyz_sum / np.maximum(
        counts[:, None],
        1,
    )

    # Observed putative-component fractions per voxel.
    component_counts = np.zeros(
        (n_voxels, n_components),
        dtype=np.float32,
    )
    np.add.at(
        component_counts,
        (inverse, component),
        1.0,
    )
    component_fraction = component_counts / np.maximum(
        counts[:, None],
        1,
    )

    keep_voxel = (
        counts >= fig_cfg.panel_c_min_units_per_voxel
    )
    voxel_xyz = voxel_xyz[keep_voxel]
    counts = counts[keep_voxel]
    component_fraction = component_fraction[keep_voxel]

    merfish_volume, merfish_names = (
        _merfish_class_volume_and_names()
    )
    merfish_density = _sample_merfish_class_density(
        ba,
        voxel_xyz,
        merfish_volume,
    )

    # Compare composition rather than total cell density.
    density_sum = merfish_density.sum(
        axis=1,
        keepdims=True,
    )
    has_merfish = density_sum[:, 0] > 0

    merfish_density = merfish_density[has_merfish]
    component_fraction = component_fraction[has_merfish]
    counts = counts[has_merfish]

    merfish_fraction = merfish_density / np.maximum(
        merfish_density.sum(axis=1, keepdims=True),
        1e-12,
    )

    # Keep the most represented classes in the observed voxels so the
    # publication heatmap remains readable. Set max_cell_types=None for all.
    if fig_cfg.panel_c_max_cell_types is not None:
        weighted_abundance = np.average(
            merfish_fraction,
            axis=0,
            weights=counts,
        )
        n_keep = min(
            int(fig_cfg.panel_c_max_cell_types),
            merfish_fraction.shape[1],
        )
        cell_keep = np.argsort(
            weighted_abundance
        )[::-1][:n_keep]

        merfish_fraction = merfish_fraction[:, cell_keep]
        merfish_names = merfish_names[cell_keep]

    cooccurrence = _weighted_cooccurrence_matrix(
        component_fraction,
        merfish_fraction,
        counts,
    )

    return cooccurrence, merfish_names


def draw_panel_c(
    ax,
    data,
    standardized_latents,
    model_gmm,
    ba,
    fig_cfg,
    negative_mask,
):
    """
    Panel c: voxel-wise co-occurrence between transcriptomic cell types and
    putative electrophysiological GMM components, using observed voxels only.
    """
    cooccurrence, merfish_names = _panel_c_cooccurrence(
        data,
        standardized_latents,
        model_gmm,
        ba,
        fig_cfg,
        negative_mask,
    )

    image = ax.imshow(
        cooccurrence,
        aspect="auto",
        interpolation="nearest",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )

    ax.set_ylabel("Putative cell type\n(GMM component)")
    ax.set_xlabel("MERFISH class")
    ax.set_yticks(
        np.arange(cooccurrence.shape[0]),
        [
            f"GMM {i + 1}"
            for i in range(cooccurrence.shape[0])
        ],
    )
    ax.set_xticks(
        np.arange(len(merfish_names)),
        merfish_names,
        rotation=90,
        ha="center",
        fontsize=5,
    )

    cb = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.015,
    )
    cb.set_label(
        "Spatial co-occurrence\n(weighted cosine; 0–1)",
        fontsize=6,
    )
    cb.ax.tick_params(labelsize=5)

    ax.set_title(
        "Observed cell-type co-occurrence",
        pad=2,
    )
    _panel_label(ax, "c")


def draw_panel_d_gmm(ax):
    """
    Panel d: context-conditioned GMM.

    Global Gaussian component parameters:
        mu_k, sigma_k

    Context-conditioned mixture weights:
        gamma_k(x)

    The probe cartoon uses a Neuropixels 1.0-style staggered four-column site
    pattern.
    """
    ax.axis("off")

    # --------------------------------------------------------------
    # Neuropixels 1.0-style probe cartoon.
    # --------------------------------------------------------------
    body_x = 0.035
    body_y = 0.17
    body_w = 0.070
    body_h = 0.66

    ax.add_patch(
        Rectangle(
            (body_x, body_y),
            body_w,
            body_h,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            lw=1.0,
        )
    )

    ax.add_patch(
        Polygon(
            [
                (body_x, body_y),
                (body_x + body_w, body_y),
                (body_x + body_w / 2, 0.075),
            ],
            transform=ax.transAxes,
            closed=True,
            facecolor="white",
            edgecolor="black",
            lw=1.0,
        )
    )

    # NP1.0-like 4-column staggered recording-site pattern.
    x_cols = np.array([0.014, 0.028, 0.042, 0.056])
    row_ys = np.linspace(
        body_y + 0.035,
        body_y + body_h - 0.035,
        14,
    )

    query_y = row_ys[len(row_ys) // 2]

    for row_i, yy in enumerate(row_ys):
        # Alternate between the two interleaved site pairs:
        # [col0, col2] and [col1, col3].
        if row_i % 2 == 0:
            cols = (0, 2)
        else:
            cols = (1, 3)

        is_neighbor = abs(yy - query_y) < 0.095
        site_color = "green" if is_neighbor else "0.72"

        for col_i in cols:
            ax.add_patch(
                Rectangle(
                    (
                        body_x + x_cols[col_i] - 0.004,
                        yy - 0.006,
                    ),
                    0.008,
                    0.012,
                    transform=ax.transAxes,
                    facecolor=site_color,
                    edgecolor=site_color,
                    lw=0.4,
                )
            )

    # Query site in red at the center.
    ax.add_patch(
        Rectangle(
            (
                body_x + x_cols[2],
                query_y - 0.006,
            ),
            0.008,
            0.012,
            transform=ax.transAxes,
            facecolor="red",
            edgecolor="red",
            lw=0.4,
        )
    )

    ax.text(
        0.070,
        0.88,
        "query voxel +\nneighboring units",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )

    # --------------------------------------------------------------
    # Frozen encoder branch.
    # --------------------------------------------------------------
    _box(
        ax,
        (0.155, 0.58),
        0.12,
        0.16,
        "pre-trained\nencoders",
    )
    _arrow(
        ax,
        (0.11, query_y + 0.06),
        (0.145, 0.66),
        color="green",
    )

    ax.text(
        0.215,
        0.28,
        "anatomical\ncontext",
        transform=ax.transAxes,
        color="red",
        ha="center",
        va="center",
        fontsize=7,
    )
    _arrow(
        ax,
        (0.11, query_y),
        (0.165, 0.31),
        color="red",
    )

    # --------------------------------------------------------------
    # Point transformer: narrower than before.
    # --------------------------------------------------------------
    pt_x = 0.335
    pt_y = 0.405
    pt_w = 0.095
    pt_h = 0.19

    _box(
        ax,
        (pt_x, pt_y),
        pt_w,
        pt_h,
        "point\ntransformer",
    )

    _arrow(
        ax,
        (0.285, 0.66),
        (pt_x - 0.01, 0.54),
    )
    _arrow(
        ax,
        (0.265, 0.31),
        (pt_x - 0.01, 0.46),
    )

    # --------------------------------------------------------------
    # Global Gaussian parameter boxes.
    # --------------------------------------------------------------
    gaussian_center_x = 0.585

    ax.text(
        gaussian_center_x,
        0.80,
        "GLOBAL Gaussian parameters",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    component_y = [0.67, 0.55, 0.43]
    component_colors = [
        "tab:blue",
        "tab:orange",
        "tab:purple",
    ]

    box_x = 0.535
    box_w = 0.100

    for k, (yy, cc) in enumerate(
        zip(component_y, component_colors),
        start=1,
    ):
        ax.add_patch(
            FancyBboxPatch(
                (box_x, yy - 0.040),
                box_w,
                0.080,
                transform=ax.transAxes,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                facecolor="white",
                edgecolor=cc,
                lw=1.1,
            )
        )
        ax.text(
            box_x + box_w / 2,
            yy,
            rf"$\mu_{{{k}}},\ \sigma_{{{k}}}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=cc,
        )

    ax.text(
        gaussian_center_x,
        0.31,
        "shared across all brain locations",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
    )

    # Short PT output arrow; stop before the Gaussian boxes.
    _arrow(
        ax,
        (pt_x + pt_w + 0.01, 0.50),
        (0.505, 0.50),
    )

    # --------------------------------------------------------------
    # Context-conditioned gamma weights.
    # --------------------------------------------------------------
    gamma_center_x = 0.82

    ax.text(
        gamma_center_x,
        0.69,
        "CONTEXT-CONDITIONED\nmixture weights",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    gamma_labels = [
        r"$\gamma_1(x)$",
        r"$\gamma_2(x)$",
        r"$\gamma_3(x)$",
    ]
    gamma_widths = [
        0.105,
        0.068,
        0.042,
    ]

    y0 = 0.56

    for i, (label, width, cc) in enumerate(
        zip(
            gamma_labels,
            gamma_widths,
            component_colors,
        )
    ):
        yy = y0 - 0.10 * i

        ax.add_patch(
            Rectangle(
                (0.745, yy - 0.022),
                width,
                0.044,
                transform=ax.transAxes,
                facecolor=cc,
                edgecolor="black",
                lw=0.6,
                alpha=0.75,
            )
        )
        ax.text(
            0.875,
            yy,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7,
        )

    ax.text(
        gamma_center_x,
        0.19,
        r"$p(z\mid x)=\sum_k \gamma_k(x)\,\mathcal{N}(z;\mu_k,\sigma_k^2)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.text(
        gamma_center_x,
        0.08,
        r"global $\mu_k,\sigma_k$; context-conditioned $\gamma_k(x)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    _panel_label(ax, "d")


def _distribution_features(
    waveforms: np.ndarray,
    sampling_rate_hz: float,
):
    """
    Peak value, continuous trough-to-peak time, and repolarization slope.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    out = np.full((len(waveforms), 3), np.nan, dtype=np.float32)
    dt_ms = 1000.0 / float(sampling_rate_hz)

    for i, waveform in enumerate(waveforms):
        channel = int(
            np.unravel_index(
                np.argmax(np.abs(waveform)),
                waveform.shape,
            )[0]
        )
        trace = waveform[channel]

        trough = int(np.argmin(trace))
        post_peak = trough + int(np.argmax(trace[trough:]))

        trough_sub = trough + _parabolic_extremum_offset(trace, trough)
        peak_sub = post_peak + _parabolic_extremum_offset(trace, post_peak)
        duration_ms = max((peak_sub - trough_sub) * dt_ms, 1e-6)

        peak_value = float(trace[post_peak])
        trough_value = float(trace[trough])

        out[i] = (
            peak_value,
            duration_ms,
            (peak_value - trough_value) / duration_ms,
        )

    return out


def _feature_distributions(
    model_ae,
    model_gmm,
    scaler,
    loader,
    data,
    cfg,
    ba,
    fig_cfg,
    negative_mask,
):
    rng = np.random.default_rng(fig_cfg.seed + 303)

    means = model_gmm.means.detach().cpu().numpy()
    sigma = np.exp(
        0.5 * model_gmm.log_var.detach().cpu().numpy()
    )

    region_ids = {
        name: region_id(ba, name)
        for name in fig_cfg.panel_d_regions
    }
    obs = {
        region: [[], [], []]
        for region in fig_cfg.panel_d_regions
    }
    pred = {
        region: [[], [], []]
        for region in fig_cfg.panel_d_regions
    }

    cosmos = cosmos_ids_for_xyz(
        ba,
        data.xyz_m,
    )

    with torch.no_grad():
        for raw in loader:
            batch = move(raw, cfg.device)
            logits = model_gmm.logits(
                batch["neighbor_z"],
                batch["relative_position"],
                batch["context"],
                batch["neighbor_padding_mask"],
            )
            gamma = torch.softmax(
                logits,
                -1,
            ).cpu().numpy()

            mask = batch["target_mask"].cpu().numpy()
            target_idx = batch["target_indices"].cpu().numpy()

            for i in range(len(gamma)):
                ids = target_idx[i][mask[i]]
                ids = ids[
                    negative_mask[ids]
                ]

                if len(ids):
                    vals, counts = np.unique(
                        cosmos[ids],
                        return_counts=True,
                    )
                    rid = int(
                        vals[np.argmax(counts)]
                    )
                else:
                    rid = -1

                name = next(
                    (
                        region_name
                        for region_name, region_rid in region_ids.items()
                        if region_rid == rid
                    ),
                    None,
                )
                if name is None:
                    continue

                fobs = _distribution_features(
                    data.waveforms[ids],
                    cfg.waveform_sampling_rate_hz,
                )
                for j in range(3):
                    obs[name][j].append(
                        fobs[:, j]
                    )

                n = min(
                    fig_cfg.panel_d_samples_per_voxel,
                    max(len(ids), 1),
                )
                component = rng.choice(
                    len(means),
                    size=n,
                    p=gamma[i] / gamma[i].sum(),
                )
                zstd = (
                    means[component]
                    + sigma[component]
                    * rng.standard_normal(
                        (n, means.shape[1])
                    ).astype(np.float32)
                )
                zraw = scaler.inverse_transform(
                    zstd
                ).astype(np.float32)
                wt = model_ae.decode_waveform_from_shared(
                    torch.from_numpy(zraw).to(cfg.device)
                ).cpu().numpy()

                pred_negative = negative_dominant_mask(wt)
                wt = wt[pred_negative]
                if len(wt) == 0:
                    continue

                fp = _distribution_features(
                    wt,
                    cfg.waveform_sampling_rate_hz,
                )
                for j in range(3):
                    pred[name][j].append(
                        fp[:, j]
                    )

    return obs, pred


def draw_panel_e(
    fig,
    spec,
    model_ae,
    model_gmm,
    scaler,
    loader,
    data,
    cfg,
    ba,
    fig_cfg,
    negative_mask,
):
    obs, pred = _feature_distributions(
        model_ae,
        model_gmm,
        scaler,
        loader,
        data,
        cfg,
        ba,
        fig_cfg,
        negative_mask,
    )

    gs = GridSpecFromSubplotSpec(
        3,
        len(fig_cfg.panel_d_regions),
        subplot_spec=spec,
        hspace=0.35,
        wspace=0.25,
    )
    first = None
    names = (
        "Peak value",
        "Peak time",
        "Repolarization slope",
    )

    for row in range(3):
        pooled = []
        for region in fig_cfg.panel_d_regions:
            for source in (obs, pred):
                if source[region][row]:
                    pooled.append(
                        np.concatenate(source[region][row])
                    )

        pooled = np.concatenate(pooled)
        lo, hi = np.quantile(
            pooled[np.isfinite(pooled)],
            [0.005, 0.995],
        )
        bins = np.linspace(lo, hi, 32)

        for col, region in enumerate(fig_cfg.panel_d_regions):
            ax = fig.add_subplot(
                gs[row, col]
            )
            first = ax if first is None else first

            if obs[region][row]:
                ax.hist(
                    np.concatenate(obs[region][row]),
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.2,
                    label="Observed",
                )

            if pred[region][row]:
                ax.hist(
                    np.concatenate(pred[region][row]),
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.2,
                    ls="--",
                    label="Predicted",
                )

            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(region)
            if col == 0:
                ax.set_ylabel(names[row])
            if row == 2:
                ax.set_xlabel("Feature value")
            if row == 2 and col == 0:
                ax.legend(frameon=False)

    _panel_label(first, "e")


def make_figure3(fig_cfg=FigureConfig()):
    figure_style()
    ba = AllenAtlas()

    (
        cfg,
        data,
        model_ae,
        model_gmm,
        scaler,
        standardized,
        datasets,
        loaders,
    ) = unit_release(
        fig_cfg.repo_id,
        fig_cfg.vintage,
        cache_dir=fig_cfg.cache_dir,
    )

    # One shared inclusion criterion for all observed unit-level analyses.
    negative_mask = negative_dominant_mask(
        data.waveforms
    )
    print(
        "[figure 3] negative-dominant units: "
        f"{negative_mask.sum():,}/{len(negative_mask):,} "
        f"({100 * negative_mask.mean():.1f}%)"
    )

    fig = double_column_fig()
    fig.set_size_inches(
        fig.get_size_inches()[0] * 1.12,
        10.2,
    )

    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.85, 1.55, 1.45, 2.15],
        hspace=0.34,
    )

    draw_panel_a(
        fig,
        outer[0],
        data,
        cfg,
        fig_cfg,
        ba,
        negative_mask,
    )

    row_bc = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[1.0, 1.35],
        wspace=0.20,
    )

    draw_panel_b(
        fig,
        row_bc[0, 0],
        data,
        cfg,
        fig_cfg,
        ba,
        negative_mask,
    )

    ax_c = fig.add_subplot(
        row_bc[0, 1]
    )
    draw_panel_c(
        ax_c,
        data,
        standardized,
        model_gmm,
        ba,
        fig_cfg,
        negative_mask,
    )

    ax_d = fig.add_subplot(
        outer[2]
    )
    draw_panel_d_gmm(ax_d)

    draw_panel_e(
        fig,
        outer[3],
        model_ae,
        model_gmm,
        scaler,
        loaders[2],
        data,
        cfg,
        ba,
        fig_cfg,
        negative_mask,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.985,
        bottom=0.05,
    )

    fig_cfg.save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        fig_cfg.save_path,
        dpi=fig_cfg.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"saved: {fig_cfg.save_path}")


if __name__ == "__main__":
    make_figure3()
