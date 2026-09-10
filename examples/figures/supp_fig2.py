from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
import numpy as np

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice

from ephysatlas.unit_level_encoder import Config, load_unit_model, prepare_unit_data
from ephysatlas.unit_level_encoder.data import load_prepared_data
from ephysatlas.unit_level_encoder.baselines import RegionalGaussianBaseline, SpatialKDEBaseline
from ephysatlas.unit_level_encoder.gmm_models import GlobalWeightModel
from ephysatlas.unit_level_encoder.unit_level_vis import (
    choose_feature_slice_indices,
    feature_nll_comparison,
    publication_feature_slice_data,
    reconstruction_examples_all_modalities,
)



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


@dataclass
class FigureConfig:
    repo_id: str = "AlonSaguy/ephys-atlas-models"
    vintage: str = "2026_W26"
    revision: str = "main"
    token: Optional[str] = None
    prepared_data_dir: Path = Path("unit_level_model_data/prepared_data")
    save_path: Path = Path("supp_figure2_unit_model_validation.pdf")
    dpi: int = 600
    seed: int = 0
    reconstruction_examples_per_modality: int = 2
    feature_count: int = 5
    nll_samples_per_test_unit: int = 4
    panel_b_color_quantiles: tuple[float, float] = (0.005, 0.995)
    panel_a_candidate_pool: int = 256


METHOD_ORDER = (
    "Cosmos Gaussian",
    "Beryl Gaussian",
    "KDE",
    "Unconditional GMM",
    "Conditional GMM",
    "Conditional + kNN",
)


def _panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def _panel_label_right(fig, y, label, *, x=0.992):
    """Place a panel label at a shared right-edge figure coordinate."""
    fig.text(
        float(x),
        float(y),
        label,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _panel_label_figure(fig, ax, label, *, x=0.992, dy=0.006):
    """Place a panel label slightly above an axis at the figure's right edge."""
    bbox = ax.get_position()
    _panel_label_right(fig, float(bbox.y1) + float(dy), label, x=x)


def _dominant_trace(waveform):
    waveform = np.asarray(waveform)
    return waveform[int(np.argmax(np.ptp(waveform, axis=1)))]


def _choose_good_reconstruction_examples(
    bundle,
    fig_cfg,
):
    """Choose held-out units with good reconstruction across all modalities.

    We score a reproducible candidate pool from the TEST split using normalized
    per-modality MSE and choose the units with the lowest mean normalized error.
    This avoids publication examples that look poor simply because they were
    selected randomly.
    """
    data = bundle.data
    cfg = bundle.cfg

    test_ids = np.flatnonzero(np.asarray(data.split) == 2)
    if len(test_ids) == 0:
        raise RuntimeError("No TEST units are available for reconstruction examples.")

    # Ask the existing helper for a reproducible pool of held-out examples,
    # then rank that pool ourselves.  This preserves compatibility with the
    # current helper API, which only requires n_examples + seed.
    n_pool = min(int(fig_cfg.panel_a_candidate_pool), len(test_ids))

    candidate_examples = reconstruction_examples_all_modalities(
        bundle.autoencoder,
        data,
        cfg,
        n_examples=n_pool,
        seed=int(fig_cfg.seed) + 811,
    )
    candidate_ids = np.asarray(candidate_examples["indices"])

    modality_pairs = (
        ("waveform", "waveform_reconstruction"),
        ("acg", "acg_reconstruction"),
        ("stpc", "stpc_reconstruction"),
    )

    errors = []
    for original_key, reconstruction_key in modality_pairs:
        original = np.asarray(candidate_examples[original_key], dtype=np.float32)
        reconstruction = np.asarray(
            candidate_examples[reconstruction_key],
            dtype=np.float32,
        )

        reduce_axes = tuple(range(1, original.ndim))
        mse = np.mean((original - reconstruction) ** 2, axis=reduce_axes)

        # Normalize by each sample's signal energy so modalities with different
        # numerical scales contribute comparably.
        energy = np.mean(original**2, axis=reduce_axes)
        nmse = mse / np.maximum(energy, 1e-8)
        errors.append(nmse)

    score = np.mean(np.column_stack(errors), axis=1)
    n_keep = min(
        int(fig_cfg.reconstruction_examples_per_modality),
        len(candidate_ids),
    )
    keep = np.argsort(score)[:n_keep]

    examples = {}
    for key, value in candidate_examples.items():
        if key == "indices":
            examples[key] = np.asarray(value)[keep]
            continue

        arr = np.asarray(value)
        if len(arr) == len(candidate_ids):
            examples[key] = arr[keep]
        else:
            examples[key] = value

    print(
        "[Supp Fig. 2 panel a] selected low reconstruction-error TEST units: "
        f"{examples['indices'].tolist()}"
    )
    return examples


def draw_panel_a(fig, spec, examples):
    """Two examples per modality; top row observed, bottom row reconstructed."""
    n = min(2, len(examples["indices"]))
    gs = GridSpecFromSubplotSpec(2, 6, subplot_spec=spec, hspace=0.22, wspace=0.28)
    modalities = ["Waveform", "ACG", "stPC"]
    first = None

    for modality_idx, modality in enumerate(modalities):
        for ex in range(n):
            col = 2 * modality_idx + ex
            for row, reconstructed in enumerate((False, True)):
                ax = fig.add_subplot(gs[row, col])
                if first is None:
                    first = ax
                if modality == "Waveform":
                    key = "waveform_reconstruction" if reconstructed else "waveform"
                    ax.plot(_dominant_trace(examples[key][ex]), lw=0.9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                elif modality == "ACG":
                    key = "acg_reconstruction" if reconstructed else "acg"
                    ax.imshow(examples[key][ex], aspect="auto", origin="lower", interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    key = "stpc_reconstruction" if reconstructed else "stpc"
                    ax.plot(np.asarray(examples[key][ex]).reshape(-1), lw=0.9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
                if row == 0:
                    ax.set_title(f"{modality} {ex + 1}", fontsize=7, pad=2)
                if col == 0:
                    ax.set_ylabel("Original" if row == 0 else "Recon.", fontsize=7)
    _panel_label_figure(fig, first, "a")


def _build_methods(bundle):
    data = bundle.data
    train = data.split == 0
    baselines = {
        "Cosmos Gaussian": RegionalGaussianBaseline(
            bundle.z_scaled, data.cosmos_ids, train, bundle.cfg.region_gaussian_variance_floor
        ),
        "Beryl Gaussian": RegionalGaussianBaseline(
            bundle.z_scaled, data.beryl_ids, train, bundle.cfg.region_gaussian_variance_floor
        ),
        "KDE": SpatialKDEBaseline(bundle.z_scaled, data.xyz_m, train, bundle.cfg),
    }
    unconditional = GlobalWeightModel(bundle.gmm.weights_, len(data.waveforms))
    return {
        "Cosmos Gaussian": {"method_kind": "cosmos_gaussian", "baseline": baselines["Cosmos Gaussian"]},
        "Beryl Gaussian": {"method_kind": "beryl_gaussian", "baseline": baselines["Beryl Gaussian"]},
        "KDE": {"method_kind": "kde", "baseline": baselines["KDE"]},
        "Unconditional GMM": {
            "method_kind": "experimental",
            "gmm": bundle.gmm,
            "conditional_model": unconditional,
        },
        "Conditional GMM": {
            "method_kind": "experimental",
            "gmm": bundle.gmm,
            "conditional_model": bundle.context_model,
        },
        "Conditional + kNN": {
            "method_kind": "experimental_knn",
            "gmm": bundle.gmm,
            "conditional_model": bundle.context_model,
            "empirical_decoder": bundle.knn_decoder,
        },
    }


def _central_sagittal_coord_um(data, step_um):
    test_x_um = np.asarray(data.xyz_m[data.split == 2, 0], float) * 1e6
    # Median sampled hemisphere is more informative than the empty midline.
    return float(np.round(np.median(test_x_um) / float(step_um)) * float(step_um))


def _empty_atlas(ax, ba, coord_um):
    plot_points_on_slice(
        np.empty((0, 3)),
        values=None,
        coord=float(coord_um),
        slice="sagittal",
        mapping="Cosmos",
        background="boundary",
        show_cbar=False,
        aggr="mean",
        fwhm=0,
        brain_atlas=ba,
        ax=ax,
    )


def draw_panel_b(fig, spec, bundle, methods, feature_indices, fig_cfg):
    data = bundle.data
    cfg = bundle.cfg
    ba = AllenAtlas()
    coord_um = _central_sagittal_coord_um(
        data,
        cfg.diagnostic_voxel_size_um,
    )

    predictions = {}
    for name in METHOD_ORDER:
        spec_m = methods[name]
        predictions[name] = publication_feature_slice_data(
            bundle.autoencoder,
            data,
            bundle.latent_scaler,
            cfg,
            feature_indices=feature_indices,
            method_kind=spec_m["method_kind"],
            sagittal_coord_um=coord_um,
            gmm=spec_m.get("gmm"),
            conditional_model=spec_m.get("conditional_model"),
            baseline=spec_m.get("baseline"),
            empirical_decoder=spec_m.get("empirical_decoder"),
        )

    gs = GridSpecFromSubplotSpec(
        len(feature_indices),
        len(METHOD_ORDER),
        subplot_spec=spec,
        hspace=0.08,
        wspace=0.05,
    )
    first = None

    # Each method/feature subplot gets its own robust color limits.  Use nearly
    # the full finite range so genuine values are not visually saturated by
    # isolated outliers.
    qlo, qhi = fig_cfg.panel_b_color_quantiles
    limits = {}

    for row, _ in enumerate(feature_indices):
        for name in METHOD_ORDER:
            values = np.asarray(
                predictions[name]["features"][:, row],
                dtype=float,
            )
            finite = values[np.isfinite(values)]

            if len(finite) == 0:
                lo, hi = -1.0, 1.0
            else:
                lo, hi = np.quantile(finite, [qlo, qhi])
                if hi <= lo:
                    center = float(np.nanmedian(finite))
                    eps = max(abs(center) * 1e-3, 1e-8)
                    lo, hi = center - eps, center + eps

            # 'seismic' is a diverging map. Keep zero as the neutral point
            # whenever the data span zero, without artificially forcing
            # symmetric ranges for one-sided quantities.
            if lo < 0.0 < hi:
                magnitude = max(abs(float(lo)), abs(float(hi)))
                lo, hi = -magnitude, magnitude

            limits[(row, name)] = (float(lo), float(hi))

    for row, findex in enumerate(feature_indices):
        for col, name in enumerate(METHOD_ORDER):
            ax = fig.add_subplot(gs[row, col])
            if first is None:
                first = ax

            payload = predictions[name]
            xyz_um = payload["xyz_m"] * 1e6
            values = payload["features"][:, row]
            vmin, vmax = limits[(row, name)]

            _empty_atlas(ax, ba, coord_um)
            ax.scatter(
                xyz_um[:, 1],
                xyz_um[:, 2],
                c=values,
                cmap="seismic",
                norm=Normalize(vmin=vmin, vmax=vmax),
                s=4,
                marker="s",
                linewidths=0,
                rasterized=True,
                zorder=3,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(name, fontsize=6.5, pad=2)
            if col == 0:
                ax.set_ylabel(
                    data.waveform_feature_names[int(findex)],
                    fontsize=6.5,
                )

    first.text(
        0.0,
        1.16,
        f"Sagittal slice: ML={coord_um:.0f} µm",
        transform=first.transAxes,
        fontsize=6.5,
        ha="left",
    )
    _panel_label_figure(fig, first, "b")



def draw_panel_c(fig, ax, nll, feature_names):
    """Held-out feature NLL with features on x and methods as grouped bars."""
    methods = list(METHOD_ORDER)
    feature_names = list(feature_names)

    n_features = len(feature_names)
    n_methods = len(methods)
    if n_features == 0:
        raise ValueError("No features were provided for panel c.")

    x = np.arange(n_features, dtype=float)
    group_width = 0.84
    bar_width = group_width / max(n_methods, 1)
    offsets = (
        np.arange(n_methods, dtype=float)
        - (n_methods - 1) / 2.0
    ) * bar_width

    colors = plt.cm.tab10(
        np.linspace(0, 1, max(n_methods, 3))
    )[:n_methods]

    for method_idx, method_name in enumerate(methods):
        values = np.asarray(nll[method_name], dtype=float)

        ax.bar(
            x + offsets[method_idx],
            values,
            width=0.92 * bar_width,
            label=method_name,
            color=colors[method_idx],
            edgecolor="none",
        )

    ax.set_xticks(
        x,
        feature_names,
        rotation=22,
        ha="right",
    )
    ax.set_ylabel("Held-out feature NLL")

    # Give the title and legend distinct vertical bands.
    ax.set_title(
        "Distribution fidelity of the five features (lower is better)",
        pad=36,
    )
    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    ax.spines[["top", "right"]].set_visible(False)
    _panel_label_figure(fig, ax, "c", dy=0.055)



def make_supp_figure2(fig_cfg=FigureConfig()):
    figure_style()
    cfg = Config(
        repo_id=fig_cfg.repo_id,
        vintage=fig_cfg.vintage,
        prepared_data_dir=fig_cfg.prepared_data_dir,
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
    cfg.feature_slice_count = int(fig_cfg.feature_count)
    cfg.feature_slice_seed = int(fig_cfg.seed) + 20260831

    examples = _choose_good_reconstruction_examples(
        bundle,
        fig_cfg,
    )
    feature_indices = choose_feature_slice_indices(data, cfg)
    feature_names = [data.waveform_feature_names[int(i)] for i in feature_indices]
    methods = _build_methods(bundle)
    nll = feature_nll_comparison(
        bundle.autoencoder,
        data,
        bundle.z_scaled,
        bundle.latent_scaler,
        cfg,
        feature_indices=feature_indices,
        methods=methods,
        samples_per_test_unit=fig_cfg.nll_samples_per_test_unit,
    )

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0] * 1.10, 11.4)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.8, 5.2, 2.05], hspace=0.42)
    draw_panel_a(fig, outer[0], examples)
    draw_panel_b(fig, outer[1], bundle, methods, feature_indices, fig_cfg)
    ax_c = fig.add_subplot(outer[2])
    draw_panel_c(fig, ax_c, nll, feature_names)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.09)
    fig_cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_cfg.save_path, dpi=fig_cfg.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    np.savez_compressed(
        fig_cfg.save_path.with_suffix(".npz"),
        feature_indices=np.asarray(feature_indices, int),
        feature_names=np.asarray(feature_names, dtype="U"),
        method_names=np.asarray(METHOD_ORDER, dtype="U"),
        nll=np.vstack([nll[name] for name in METHOD_ORDER]),
    )
    print(f"saved: {fig_cfg.save_path}")


if __name__ == "__main__":
    make_supp_figure2()
