from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

from ephysatlas.unit_level_encoder import Config, load_unit_model, prepare_unit_data
from ephysatlas.unit_level_encoder.data import load_prepared_data
from ephysatlas.unit_level_encoder.gmm_models import sample_conditional
from ephysatlas.unit_level_encoder.unit_level_vis import get_model_space_waveform_features


# Resolve paths from this file rather than from the current working directory.
# figure3.py lives in <repo>/examples/figures/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED_DATA_DIR = REPO_ROOT / "unit_level_model_data" / "prepared_data"
DEFAULT_CACHE_DIR = REPO_ROOT / "unit_level_model_results" / "figure_cache"


@dataclass
class FigureConfig:
    repo_id: str = "AlonSaguy/ephys-atlas-models"
    vintage: str = "2026_W26"
    revision: str = "main"
    token: Optional[str] = None
    prepared_data_dir: Path = DEFAULT_PREPARED_DATA_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    recompute_cache: bool = False
    save_path: Path = Path("figure3_unit_level_atlas.pdf")
    dpi: int = 600
    seed: int = 0

    panel_a_max_units: int = 30_000
    panel_a_mesh_stride: int = 4
    panel_a_clip_quantiles: tuple[float, float] = (0.01, 0.99)
    panel_a_elev: float = 16.0
    panel_a_azim: float = 135.0
    panel_a_zoom: float = 1.95
    panel_a_brain_alpha: float = 0.25

    panel_b_regions: tuple[str, ...] = ("Isocortex", "HB")
    panel_b_max_units_per_region: int = 6_000
    panel_b_axis_quantiles: tuple[float, float] = (0.005, 0.995)

    panel_d_regions: tuple[str, ...] = ("Isocortex", "TH", "MB", "HB", "CB")
    panel_d_samples_per_unit: int = 4


def cosmos_ids_for_xyz(ba, xyz_m):
    return np.asarray(ba.get_labels(np.asarray(xyz_m), mapping="Cosmos"), np.int64)


def region_id(ba, acronym):
    br = BrainRegions()
    matches = np.flatnonzero(np.asarray(br.acronym).astype(str) == str(acronym))
    if not len(matches):
        raise KeyError(f"Unknown region acronym: {acronym}")
    return int(br.id[matches[0]])


def region_color(ba, acronym):
    br = BrainRegions()
    matches = np.flatnonzero(np.asarray(br.acronym).astype(str) == str(acronym))
    if not len(matches):
        return "0.4"
    rgb = np.asarray(br.rgb[matches[0]], float)
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    return tuple(rgb[:3])

def _panel_label(ax, label):
    """Legacy axes-relative panel label helper."""
    kwargs = dict(
        transform=ax.transAxes,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    if hasattr(ax, "text2D"):
        ax.text2D(-0.08, 1.04, label, **kwargs)
    else:
        ax.text(-0.08, 1.04, label, **kwargs)


def _panel_label_left(fig, y, label, *, x=None):
    """Place a panel label at the left-most figure coordinate.

    The coordinates are clipped to remain inside the saved figure boundary, so
    the label stays visible even when using bbox_inches="tight".
    """
    if x is None:
        x = float(fig.subplotpars.left) + 0.002
    y = min(float(y), 0.985)
    x = max(0.008, float(x))
    fig.text(
        x,
        y,
        label,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=1000,
    )


def _panel_label_figure(fig, ax, label, *, x=None, dy=0.004):
    """Place a panel label slightly above a plot at the left edge of the figure."""
    bbox = ax.get_position()
    return _panel_label_left(
        fig,
        float(bbox.y1) + float(dy),
        label,
        x=x,
    )


def _panel_label_right(fig, y, label, *, x=None):
    """Compatibility alias: panel labels are now placed on the left."""
    return _panel_label_left(fig, y, label, x=x)


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
    Panel a: duration and peak value for individual negative-dominant units.

    The two 3-D axes are positioned manually so the pair can be shifted left,
    whitespace can be reduced, and the inter-plot spacing can be increased.
    """
    features, names = extract_three_waveform_features_continuous(
        data.waveforms,
        cfg.waveform_sampling_rate_hz,
    )

    feature_order = (2, 1)  # duration, peak value

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

    panel_bbox = spec.get_position(fig)
    panel_x0 = panel_bbox.x0
    panel_y0 = panel_bbox.y0
    panel_w = panel_bbox.width
    panel_h = panel_bbox.height

    # Move the full pair left and enlarge the gap between the two 3-D plots.
    panel_a_shift = -0.055 * panel_w
    panel_x0 += panel_a_shift

    axis_w = 0.405 * panel_w
    axis_h = 0.82 * panel_h
    axis_y = panel_y0 + 0.08 * panel_h

    cbar_w = 0.010 * panel_w
    cbar_h = 0.44 * panel_h
    cbar_gap = 0.004 * panel_w

    left_margin = 0.004 * panel_w
    gap = 0.050 * panel_w
    axis_x_positions = [
        panel_x0 + left_margin,
        panel_x0 + left_margin + axis_w + cbar_gap + cbar_w + gap,
    ]

    for axis_x, feature_index in zip(axis_x_positions, feature_order):
        name = names[feature_index]

        ax = fig.add_axes(
            [axis_x, axis_y, axis_w, axis_h],
            projection="3d",
        )

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

        xyz = np.asarray(
            data.xyz_m[ids],
            dtype=float,
        )
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
        ax.set_title(
            name,
            pad=1,
        )
        _clean_3d(ax)

        cbar_x = axis_x + axis_w + cbar_gap
        cbar_y = panel_y0 + 0.28 * panel_h
        cax = fig.add_axes(
            [
                cbar_x,
                cbar_y,
                cbar_w,
                cbar_h,
            ]
        )
        cb = fig.colorbar(
            sc,
            cax=cax,
        )
        cb.ax.tick_params(
            labelsize=5,
            length=2,
            pad=1,
        )

    _panel_label_left(fig, panel_bbox.y1 + 0.004, "a")

def _panel_b_raw_data(
    data,
    cfg,
    fig_cfg,
    ba,
    negative_mask,
):
    """
    Return duration and peak-value features for the selected brain regions,
    using only negative-dominant units.
    """
    features, names = extract_three_waveform_features_continuous(
        data.waveforms,
        cfg.waveform_sampling_rate_hz,
    )
    region_ids = np.asarray(
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
            & (region_ids == rid)
        )

        if len(ids) > fig_cfg.panel_b_max_units_per_region:
            ids = rng.choice(
                ids,
                fig_cfg.panel_b_max_units_per_region,
                replace=False,
            )

        points = np.column_stack(
            [
                features[ids, 2],  # duration
                features[ids, 1],  # peak value
            ]
        ).astype(np.float32)

        out[acronym] = {
            "ids": ids,
            "points": points,
            "color": region_color(ba, acronym),
        }

    axis_names = (
        names[2],
        names[1],
    )
    return out, axis_names


def draw_panel_b(fig, spec, data, cfg, fig_cfg, ba, negative_mask):
    """
    Panel b: Isocortex vs hindbrain in the two structured waveform features
    shown in panel a.

    The axes are positioned manually inside the GridSpec slot so panel b has
    the same visible height as panel a, and the panel letter is aligned in
    height with panel a.
    """
    panel_bbox = spec.get_position(fig)
    ax = fig.add_axes(
        [
            panel_bbox.x0,
            panel_bbox.y0 + 0.08 * panel_bbox.height,
            panel_bbox.width,
            0.82 * panel_bbox.height,
        ]
    )

    region_data, names = _panel_b_raw_data(
        data,
        cfg,
        fig_cfg,
        ba,
        negative_mask,
    )

    pooled = []
    display_region_names = {
        "Isocortex": "Isocortex",
        "HB": "Hindbrain",
    }

    for acronym in fig_cfg.panel_b_regions:
        pts = region_data[acronym]["points"]
        pooled.append(pts)
        color = region_data[acronym]["color"]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=2.0,
            alpha=0.22,
            color=color,
            label=display_region_names.get(acronym, acronym),
            rasterized=True,
        )

    pooled = np.concatenate(pooled, axis=0)

    ax.set_xlabel(names[0])
    ax.set_ylabel(
        names[1],
        labelpad=1,
    )

    qlo, qhi = fig_cfg.panel_b_axis_quantiles
    xlo, xhi = np.quantile(
        pooled[:, 0][np.isfinite(pooled[:, 0])],
        [qlo, qhi],
    )
    ylo, yhi = np.quantile(
        pooled[:, 1][np.isfinite(pooled[:, 1])],
        [qlo, qhi],
    )
    ax.set_xlim(float(xlo), float(xhi))
    ax.set_ylim(float(ylo), float(yhi))

    legend = ax.legend(
        frameon=False,
        loc="upper right",
    )

    for legend_text, acronym in zip(
        legend.get_texts(),
        fig_cfg.panel_b_regions,
    ):
        legend_text.set_color(
            region_data[acronym]["color"]
        )

    ax.spines[["top", "right"]].set_visible(False)
    _panel_label_left(fig, panel_bbox.y1 + 0.004, "b", x=max(0.008, panel_bbox.x0 - 0.014))


def _box(ax, xy, width, height, text, *, facecolor="white", edgecolor="black", textcolor="black", fontsize=7):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=1.0,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
    )


def _arrow(ax, start, end, color="black", lw=0.72):
    """Draw a narrow connector that does not visually merge with box borders."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=float(lw),
            color=color,
            mutation_scale=8,
            shrinkA=1.5,
            shrinkB=1.5,
        ),
    )





def draw_panel_c(fig, ax, seed=0):
    """Cartoon of the released multimodal K25/context/kNN20 model."""
    from matplotlib.lines import Line2D

    ax.axis("off")
    rng = np.random.default_rng(int(seed) + 100)

    modality_colors = ["tab:blue", "tab:orange", "tab:green"]
    names = ["waveform", "ACG", "stPC"]
    ys = [0.78, 0.52, 0.26]

    # ------------------------------------------------------------------
    # Modality inputs + encoders
    # ------------------------------------------------------------------
    encoder_x = 0.190
    encoder_w = 0.112
    encoder_h = 0.090

    for name, y, color in zip(names, ys, modality_colors):
        x0, w, h = 0.015, 0.125, 0.16
        inset = ax.inset_axes([x0, y - h / 2, w, h])

        if name == "waveform":
            t = np.linspace(-1.0, 1.0, 96)
            channels = np.arange(20)
            center = 9.5
            amp = np.exp(-0.5 * ((channels - center) / 3.2) ** 2)

            waveform = np.empty((20, len(t)), dtype=float)
            for ch in range(20):
                trough = -amp[ch] * np.exp(
                    -0.5 * ((t + 0.10 + 0.008 * (ch - center)) / 0.12) ** 2
                )
                rebound = 0.43 * amp[ch] * np.exp(
                    -0.5 * ((t - 0.24 - 0.004 * (ch - center)) / 0.18) ** 2
                )
                waveform[ch] = trough + rebound
            waveform += 0.012 * rng.normal(size=waveform.shape)

            lim = max(float(np.max(np.abs(waveform))), 1e-8)
            inset.imshow(
                waveform,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
            )

        elif name == "ACG":
            lag = np.linspace(-3.0, 3.0, 120)
            rows = []
            for r in range(10):
                width = 0.45 + 0.06 * r
                profile = np.exp(-0.5 * (lag / width) ** 2)
                refractory = 1.0 - 0.88 * np.exp(-0.5 * (lag / 0.18) ** 2)
                row = profile * refractory
                row += 0.025 * rng.normal(size=len(lag))
                rows.append(np.clip(row, 0.0, None))

            inset.imshow(
                np.asarray(rows),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
            )

        else:
            x = np.linspace(0, 1, 161)
            stpc = (
                0.48 * np.sin(4 * np.pi * x)
                + 0.24 * np.sin(9 * np.pi * x + 0.3)
                + 0.10 * np.cos(15 * np.pi * x)
            )
            stpc += 0.035 * rng.normal(size=len(x))
            inset.plot(x, stpc, color=color, lw=1.0)

        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(color)
            spine.set_linewidth(0.7)
        inset.set_title(name, fontsize=6, color=color, pad=1)

        _box(
            ax,
            (encoder_x, y - encoder_h / 2),
            encoder_w,
            encoder_h,
            f"{name}\nencoder",
            facecolor="white",
            edgecolor=color,
            textcolor=color,
            fontsize=6.1,
        )

        _arrow(ax, (0.143, y), (encoder_x - 0.010, y), color=color, lw=0.64)
        _arrow(
            ax,
            (encoder_x + encoder_w + 0.008, y),
            (0.388, 0.52),
            color=color,
            lw=0.68,
        )

    # ------------------------------------------------------------------
    # Joint latent -> context-conditioned GMM
    # ------------------------------------------------------------------
    latent_x = 0.398
    latent_w = 0.112
    _box(
        ax,
        (latent_x, 0.455),
        latent_w,
        0.13,
        "z\nlatent\nrepresentation",
        fontsize=6.8,
    )

    _arrow(ax, (latent_x + latent_w + 0.010, 0.52), (0.572, 0.52), lw=0.72)

    gmm_x = 0.578
    gmm_w = 0.118
    _box(ax, (gmm_x, 0.43), gmm_w, 0.18, "GMM\nK = 25", fontsize=7.8)

    ax.text(
        0.637,
        0.37,
        r"global $\mu_k,\Sigma_k$" + "\n" + r"context-conditioned $\gamma_k(x)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
    )

    _arrow(ax, (gmm_x + gmm_w + 0.014, 0.52), (0.795, 0.52), lw=0.72)

    # ------------------------------------------------------------------
    # Latent manifold + kNN projection
    # ------------------------------------------------------------------
    m_ax = ax.inset_axes([0.805, 0.12, 0.185, 0.72], projection="3d")

    u_grid = np.linspace(-1.7, 1.7, 34)
    v_grid = np.linspace(-1.4, 1.1, 32)
    uu, vv = np.meshgrid(u_grid, v_grid)

    def manifold_z(x, y):
        return (
            0.10 * x**2
            - 0.08 * y**2
            + 0.07 * np.sin(1.7 * x)
            + 0.03 * np.cos(2.0 * y)
        )

    zz = manifold_z(uu, vv)

    m_ax.plot_surface(
        uu,
        vv,
        zz,
        color="0.18",
        alpha=0.20,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=0,
    )

    # yellow, green, purple
    centers_uv = np.asarray(
        [
            [-1.00, -0.30],   # yellow
            [0.10, -1.00],    # green
            [1.08, -0.18],    # purple
        ],
        dtype=float,
    )
    cluster_colors = ["gold", "mediumseagreen", "purple"]

    points = []
    labels = []
    shown_points = {}

    for k, center_uv in enumerate(centers_uv):
        uv = center_uv[None, :] + rng.normal(scale=[0.22, 0.16], size=(48, 2))
        x = uv[:, 0]
        y = uv[:, 1]
        z = manifold_z(x, y)
        pts = np.column_stack([x, y, z])
        points.append(pts)
        labels.extend([k] * len(x))

        # Show 5-10 representative points per cluster, with a denser
        # visible purple cluster so the eventual kNN exemplars are selected
        # from a richer local cloud.
        n_show_target = 10 if k == 2 else 7
        n_show = min(n_show_target, len(pts))
        show_ids = rng.choice(len(pts), size=n_show, replace=False)
        pts_show = pts[show_ids]
        shown_points[k] = pts_show

        m_ax.scatter(
            pts_show[:, 0],
            pts_show[:, 1],
            pts_show[:, 2],
            color=cluster_colors[k],
            s=26,
            alpha=0.94,
            edgecolors="white",
            linewidths=0.40,
            depthshade=False,
            zorder=3,
        )

    points = np.concatenate(points, axis=0)
    labels = np.asarray(labels)

    # Red predicted latent: midpoint between the green and purple clusters.
    query_uv = 0.50 * centers_uv[1] + 0.50 * centers_uv[2]
    query = np.asarray(
        [
            query_uv[0],
            query_uv[1],
            manifold_z(query_uv[0], query_uv[1]),
        ]
    )

    # Use only the closest visible purple-cluster points as the kNN exemplars.
    purple_visible = shown_points[2]
    d_purple_visible = np.linalg.norm(purple_visible - query[None, :], axis=1)
    n_knn = min(4, len(purple_visible))
    knn_points = purple_visible[np.argsort(d_purple_visible)[:n_knn]]

    m_ax.scatter(
        *query,
        c="red",
        s=42,
        marker="o",
        edgecolors="black",
        linewidths=0.80,
        depthshade=False,
        zorder=12,
    )

    # kNN points: purple fill with red outline.
    m_ax.scatter(
        knn_points[:, 0],
        knn_points[:, 1],
        knn_points[:, 2],
        c="purple",
        s=22,
        marker="o",
        edgecolors="red",
        linewidths=0.90,
        depthshade=False,
        zorder=16,
    )

    for pt in knn_points:
        m_ax.plot(
            [query[0], pt[0]],
            [query[1], pt[1]],
            [query[2], pt[2]],
            color="black",
            lw=0.8,
            ls="--",
            zorder=11,
        )

    m_ax.set_xlabel("z1", fontsize=5, labelpad=-3)
    m_ax.set_ylabel("z2", fontsize=5, labelpad=-3)
    m_ax.set_zlabel("z3", fontsize=5, labelpad=-3)
    m_ax.tick_params(labelsize=4, pad=-2)
    m_ax.set_title("kNN based latent projection", fontsize=7, pad=2)
    m_ax.view_init(elev=34, azim=-121)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=5.5,
            label="predicted latent",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="purple",
            markeredgecolor="red",
            markeredgewidth=0.9,
            markersize=5.5,
            label="kNN",
        ),
    ]
    m_ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.02),
        frameon=False,
        fontsize=5.5,
        handletextpad=0.4,
        borderpad=0.2,
    )

    _panel_label_left(fig, ax.get_position().y1 + 0.040, "c")

def _feature_index(feature_names, name):
    """Return the index of a waveform feature, with optional alias support.

    Parameters
    ----------
    feature_names : sequence of str
    name : str | sequence[str]
        If a sequence is provided, the first matching alias is used.
    """
    names = list(feature_names)

    if isinstance(name, (tuple, list)):
        for alias in name:
            if alias in names:
                return names.index(alias)
        raise KeyError(
            f"None of the aliases {tuple(name)!r} are present. "
            f"Available={names}"
        )

    if name not in names:
        raise KeyError(f"Required feature {name!r} is absent. Available={names}")
    return names.index(name)


def _panel_d_distributions(bundle, fig_cfg):
    """Compute observed vs final-model feature distributions for panel d."""
    data = bundle.data
    rng = np.random.default_rng(int(fig_cfg.seed) + 303)

    features = get_model_space_waveform_features(data, bundle.cfg)

    requested_features = [
        (("trough_val", "trough_value"), "Trough value", 1.0),
        ("peak_time_secs", "Peak time (ms)", 1e3),
        (("tip_time_secs", "tip_time_sec"), "Tip time (ms)", 1e3),
        (("depolarisation_slope", "depolarization_slope"), "Depolarization slope", 1.0),
    ]
    selected = [_feature_index(data.waveform_feature_names, key) for key, _, _ in requested_features]

    test = np.flatnonzero(data.split == 2)

    # Resolve the broad Cosmos region IDs directly from the BrainRegions table.
    br = BrainRegions()
    # BrainRegions IDs can be signed by hemisphere, whereas the prepared
    # Cosmos labels are stored as canonical positive IDs. Compare absolute
    # IDs so anatomical identity is preserved independently of hemisphere sign.
    acronym_to_id = {
        str(acronym): abs(int(region_id_))
        for acronym, region_id_ in zip(br.acronym, br.id)
    }

    out = {
        "_feature_titles": [title for _, title, _ in requested_features],
        "_feature_scales": [scale for _, _, scale in requested_features],
    }
    print("[Figure 3 panel d] held-out units by requested Cosmos region:")

    cosmos_test = np.abs(np.asarray(data.cosmos_ids[test], dtype=np.int64))

    for region in fig_cfg.panel_d_regions:
        if region not in acronym_to_id:
            raise KeyError(
                f"Unknown BrainRegions acronym {region!r}. "
                f"Examples include: {list(acronym_to_id)[:20]}"
            )

        rid = acronym_to_id[region]
        ids = test[cosmos_test == rid]

        print(f"  {region}: n_test_units={len(ids):,}, canonical_region_id={rid}")

        if len(ids) == 0:
            out[region] = None
            continue

        draws = sample_conditional(
            ids,
            int(fig_cfg.panel_d_samples_per_unit),
            bundle.gmm,
            bundle.context_model,
            rng,
        )

        if not draws:
            out[region] = None
            continue

        pred_z = np.concatenate(draws, axis=0)
        pred_feat = bundle.knn_decoder.sample_features(pred_z, rng)

        obs = np.asarray(features[ids][:, selected], dtype=np.float32)
        pred = np.asarray(pred_feat[:, selected], dtype=np.float32)

        if obs.size == 0 or pred.size == 0:
            out[region] = None
            continue

        out[region] = {
            "observed": obs,
            "predicted": pred,
        }

    n_present = sum(
        (region in out) and (out[region] is not None)
        for region in fig_cfg.panel_d_regions
    )
    if n_present == 0:
        unique_ids, counts = np.unique(
            np.abs(np.asarray(data.cosmos_ids[test], dtype=np.int64)),
            return_counts=True,
        )
        top = sorted(
            zip(unique_ids.tolist(), counts.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        raise RuntimeError(
            "Panel d found zero held-out units in all requested Cosmos regions. "
            "This indicates a region-ID mismatch rather than a plotting problem. "
            f"Most common held-out Cosmos IDs are: {top}"
        )

    return out


PANEL_D_CACHE_VERSION = 6


def _cached_panel_d_distributions(bundle, fig_cfg):
    """Load or compute panel-d distributions with cache validation.

    Version 1 of the cache could preserve an all-None payload indefinitely,
    which made panel d appear completely blank.  Version 2 rejects those stale
    caches and recomputes automatically.
    """
    cache_dir = Path(fig_cfg.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        cache_dir
        / f"figure3_panel_d_v{PANEL_D_CACHE_VERSION}_{fig_cfg.vintage}.npz"
    )

    if cache_path.exists() and not bool(fig_cfg.recompute_cache):
        print(f"[Figure 3] loading cached panel-d data: {cache_path}")
        payload = np.load(cache_path, allow_pickle=False)

        cache_version = int(
            np.asarray(
                payload["cache_version"]
                if "cache_version" in payload.files
                else -1
            ).item()
        )

        valid_cache = cache_version == PANEL_D_CACHE_VERSION
        out = {}

        if valid_cache:
            for region in fig_cfg.panel_d_regions:
                present_key = f"{region}__present"
                obs_key = f"{region}__observed"
                pred_key = f"{region}__predicted"

                if present_key not in payload.files:
                    valid_cache = False
                    break

                present = bool(np.asarray(payload[present_key]).item())

                if not present:
                    out[region] = None
                    continue

                if obs_key not in payload.files or pred_key not in payload.files:
                    valid_cache = False
                    break

                observed = np.asarray(payload[obs_key], dtype=np.float32)
                predicted = np.asarray(payload[pred_key], dtype=np.float32)

                if observed.size == 0 or predicted.size == 0:
                    valid_cache = False
                    break

                out[region] = {
                    "observed": observed,
                    "predicted": predicted,
                }

        # A cache with no drawable region is invalid even if structurally valid.
        if valid_cache and any(block is not None for block in out.values()):
            if "feature_titles" in payload.files:
                out["_feature_titles"] = list(np.asarray(payload["feature_titles"]).astype(str))
            if "feature_scales" in payload.files:
                out["_feature_scales"] = list(np.asarray(payload["feature_scales"], dtype=float))
            return out

        print(
            "[Figure 3] panel-d cache is stale/empty/incompatible; "
            "recomputing it."
        )

    print("[Figure 3] computing panel-d distributions")
    out = _panel_d_distributions(bundle, fig_cfg)

    save_payload = {
        "cache_version": np.asarray(PANEL_D_CACHE_VERSION, dtype=np.int64),
        "vintage": np.asarray(str(fig_cfg.vintage)),
        "samples_per_unit": np.asarray(
            int(fig_cfg.panel_d_samples_per_unit),
            dtype=np.int64,
        ),
    }

    if "_feature_titles" in out:
        save_payload["feature_titles"] = np.asarray(
            out["_feature_titles"],
            dtype="U",
        )
    if "_feature_scales" in out:
        save_payload["feature_scales"] = np.asarray(
            out["_feature_scales"],
            dtype=np.float32,
        )

    for region in fig_cfg.panel_d_regions:
        block = out.get(region)

        save_payload[f"{region}__present"] = np.asarray(
            block is not None,
            dtype=np.bool_,
        )

        if block is not None:
            save_payload[f"{region}__observed"] = np.asarray(
                block["observed"],
                dtype=np.float32,
            )
            save_payload[f"{region}__predicted"] = np.asarray(
                block["predicted"],
                dtype=np.float32,
            )

    np.savez_compressed(cache_path, **save_payload)
    print(f"[Figure 3] saved panel-d cache: {cache_path}")

    return out


def draw_panel_d(fig, spec, bundle, fig_cfg):
    """Observed vs final conditional-GMM+kNN20 feature distributions.

    Each feature row shares one y-axis range across brain regions so density
    magnitudes are directly comparable between columns.
    """
    distributions = _cached_panel_d_distributions(bundle, fig_cfg)
    regions = list(fig_cfg.panel_d_regions)
    panel_bbox = spec.get_position(fig)

    feature_titles = list(
        distributions.get(
            "_feature_titles",
            ["Trough value", "Peak time (ms)", "Tip time (ms)", "Depolarization slope"],
        )
    )
    feature_scales = list(
        distributions.get("_feature_scales", [1.0, 1e3, 1e3, 1.0])
    )
    n_rows = len(feature_titles)

    # Precompute the line data. This lets us obtain a single y-limit per
    # feature row before creating any subplot.
    plot_data = {}
    row_ymax = np.zeros(n_rows, dtype=float)

    for col, region in enumerate(regions):
        block = distributions.get(region)
        if block is None:
            continue

        for row, scale in enumerate(feature_scales):
            obs = np.asarray(block["observed"][:, row], dtype=float) * float(scale)
            pred = np.asarray(block["predicted"][:, row], dtype=float) * float(scale)

            obs = obs[np.isfinite(obs)]
            pred = pred[np.isfinite(pred)]
            if len(obs) == 0 or len(pred) == 0:
                continue

            vals = np.concatenate([obs, pred])
            lo, hi = np.quantile(vals, [0.005, 0.995])
            if hi <= lo:
                hi = lo + 1e-8

            bins = np.linspace(lo, hi, 40)
            centers = 0.5 * (bins[:-1] + bins[1:])
            obs_density, _ = np.histogram(obs, bins=bins, density=True)
            pred_density, _ = np.histogram(pred, bins=bins, density=True)

            plot_data[(row, col)] = {
                "centers": centers,
                "observed": obs_density,
                "predicted": pred_density,
            }

            local_max = max(
                float(np.nanmax(obs_density)) if len(obs_density) else 0.0,
                float(np.nanmax(pred_density)) if len(pred_density) else 0.0,
            )
            row_ymax[row] = max(row_ymax[row], local_max)

    # A small headroom prevents the tallest curve from touching the top.
    row_ymax = np.where(row_ymax > 0, row_ymax * 1.08, 1.0)

    gs = GridSpecFromSubplotSpec(
        n_rows,
        len(regions),
        subplot_spec=spec,
        hspace=0.30,
        wspace=0.28,
    )

    first_drawable = None

    for col, region in enumerate(regions):
        block = distributions.get(region)

        for row, feature_title in enumerate(feature_titles):
            ax = fig.add_subplot(gs[row, col])

            if block is None or (row, col) not in plot_data:
                ax.axis("off")
                if row == 0 and block is None:
                    ax.text(
                        0.5,
                        0.5,
                        f"{region}\nno held-out units",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
                continue

            if first_drawable is None:
                first_drawable = ax

            payload = plot_data[(row, col)]
            ax.plot(
                payload["centers"],
                payload["observed"],
                lw=1.2,
                label="Observed",
            )
            ax.plot(
                payload["centers"],
                payload["predicted"],
                lw=1.2,
                ls="--",
                label="K25 + context + kNN20",
            )

            # Same density scale across all regions in this feature row.
            ax.set_ylim(0.0, float(row_ymax[row]))

            if row == 0:
                ax.set_title(region)

            if col == 0:
                ax.set_ylabel("Density", fontsize=6)

            ax.set_xlabel(feature_title, fontsize=6, labelpad=1)
            ax.tick_params(axis="x", labelsize=5, pad=1)
            ax.tick_params(axis="y", labelsize=5, pad=1)
            ax.spines[["top", "right"]].set_visible(False)

    if first_drawable is None:
        raise RuntimeError(
            "Panel d has no drawable region after distribution computation."
        )

    handles, labels = first_drawable.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, float(panel_bbox.y1) + 0.022),
        )

    _panel_label_left(fig, min(0.985, panel_bbox.y1 + 0.024), "d")



_REQUIRED_PREPARED_FILES = (
    "waveforms.npy",
    "acgs.npy",
    "stpc.npy",
    "ctx.npy",
    "xyz.npy",
    "pids.npy",
    "cosmos.npy",
    "allen.npy",
    "waveform_features.npy",
    "waveform_feature_names.json",
)


def _is_complete_prepared_dir(path: Path) -> bool:
    """Return True only for a complete prepared unit-level dataset."""
    path = Path(path)
    return path.is_dir() and all((path / name).exists() for name in _REQUIRED_PREPARED_FILES)


def _find_existing_prepared_data_dir(configured_path: Path) -> Path | None:
    """Find an existing prepared-data directory without rebuilding anything.

    Older runs used paths relative to the current working directory, whereas
    newer runs use a repository-root-relative path.  Figure generation accepts
    either layout and prefers the explicitly configured path when it is valid.
    """
    configured_path = Path(configured_path).expanduser()

    this_file = Path(__file__).resolve()
    figure_dir = this_file.parent
    repo_root = this_file.parents[2]
    cwd = Path.cwd().resolve()

    candidates = [
        configured_path,
        cwd / "unit_level_model_data" / "prepared_data",
        figure_dir / "unit_level_model_data" / "prepared_data",
        repo_root / "unit_level_model_data" / "prepared_data",
        repo_root / "examples" / "unit_level_model_data" / "prepared_data",
        repo_root / "examples" / "figures" / "unit_level_model_data" / "prepared_data",
    ]

    # The configured path may itself be relative to several historical roots.
    if not configured_path.is_absolute():
        candidates.extend(
            [
                cwd / configured_path,
                figure_dir / configured_path,
                repo_root / configured_path,
            ]
        )

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_complete_prepared_dir(candidate):
            return candidate

    # Last-resort bounded search.  This only inspects directory names/files;
    # it does not read the large arrays.  Searching the repo parent is useful
    # when an earlier run was made from a sibling ibleatools checkout.
    search_roots = [repo_root, cwd]
    try:
        search_roots.append(repo_root.parent)
    except Exception:
        pass

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            for waveforms_path in root.glob("**/unit_level_model_data/prepared_data/waveforms.npy"):
                candidate = waveforms_path.parent
                if _is_complete_prepared_dir(candidate):
                    return candidate.resolve()
        except (OSError, PermissionError):
            continue

    return None


def _load_or_prepare_unit_data(cfg, *, allow_prepare_if_missing: bool = True):
    """Reuse prepared unit arrays whenever possible.

    This is intentionally different from blindly calling ``prepare_unit_data``:
    first we search all historical/current cache locations.  Only when no
    complete prepared dataset exists do we optionally run preparation once.
    """
    existing = _find_existing_prepared_data_dir(cfg.prepared_data_dir)

    if existing is not None:
        cfg.prepared_data_dir = existing
        if hasattr(cfg, "prepare_data_if_missing"):
            cfg.prepare_data_if_missing = False
        if hasattr(cfg, "force_reprepare_data"):
            cfg.force_reprepare_data = False

        print(f"[unit figures] reusing prepared unit data: {existing}")
        return load_prepared_data(existing, cfg)

    if not allow_prepare_if_missing:
        raise FileNotFoundError(
            "Could not find a complete prepared unit-level dataset. "
            "Set FigureConfig.prepared_data_dir to the directory containing "
            "waveforms.npy, acgs.npy, stpc.npy, ctx.npy, xyz.npy, pids.npy, "
            "cosmos.npy, allen.npy, waveform_features.npy and "
            "waveform_feature_names.json."
        )

    # No usable cache exists.  Prepare once at the repository-root location so
    # subsequent figure runs are fast and independent of PyCharm's cwd.
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "unit_level_model_data" / "prepared_data"
    target.mkdir(parents=True, exist_ok=True)

    cfg.prepared_data_dir = target
    if hasattr(cfg, "force_reprepare_data"):
        cfg.force_reprepare_data = False
    if hasattr(cfg, "prepare_data_if_missing"):
        cfg.prepare_data_if_missing = True

    print(
        "[unit figures] no complete prepared-data cache was found.\n"
        f"[unit figures] preparing it once at: {target}\n"
        "[unit figures] later figure runs will reuse this cache."
    )
    return prepare_unit_data(cfg)


def make_figure3(fig_cfg=FigureConfig()):
    figure_style()
    cfg = Config(
        repo_id=fig_cfg.repo_id,
        vintage=fig_cfg.vintage,
        prepared_data_dir=Path(fig_cfg.prepared_data_dir).expanduser().resolve(),
        prepare_data_if_missing=False,
        force_reprepare_data=False,
    )
    data = _load_or_prepare_unit_data(cfg)
    bundle = load_unit_model(
        cfg,
        source="hub",
        data=data,
        token=fig_cfg.token,
        revision=fig_cfg.revision,
    )
    cfg = bundle.cfg
    data = bundle.data
    ba = AllenAtlas()
    negative_mask = negative_dominant_mask(data.waveforms)

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0] * 1.08, 10.2)
    outer = fig.add_gridspec(3, 1, height_ratios=[2.20, 2.15, 3.75], hspace=0.16)

    first = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.38, 0.62], wspace=0.07)
    draw_panel_a(fig, first[0], data, cfg, fig_cfg, ba, negative_mask)
    draw_panel_b(fig, first[1], data, cfg, fig_cfg, ba, negative_mask)

    ax_c = fig.add_subplot(outer[1])
    draw_panel_c(fig, ax_c, seed=fig_cfg.seed)
    draw_panel_d(fig, outer[2], bundle, fig_cfg)

    fig.subplots_adjust(left=0.062, right=0.988, top=0.989, bottom=0.055)
    fig_cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_cfg.save_path, dpi=fig_cfg.dpi, bbox_inches="tight", pad_inches=0.004)
    plt.close(fig)
    print(f"saved: {fig_cfg.save_path}")


if __name__ == "__main__":
    make_figure3()
