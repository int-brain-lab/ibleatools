from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import gaussian_kde

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice

from ephysatlas.unit_level_encoder import Config, load_unit_model, prepare_unit_data
from ephysatlas.unit_level_encoder.data import load_prepared_data
from ephysatlas.unit_level_encoder.baselines import RegionalGaussianBaseline, SpatialKDEBaseline
from ephysatlas.unit_level_encoder.unit_level_vis import (
    choose_feature_slice_indices,
    feature_nll_comparison,
    publication_feature_slice_data,
    reconstruction_examples_all_modalities,
    get_model_space_waveform_features,
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
    panel_a_candidate_pool: int = 512
    # TEMPORARY diagnostics. Set False later to disable all extra outputs.
    run_diagnostics: bool = True
    diagnostics_dir: Path = Path("unit_level_model_results/supp_fig2_diagnostics")
    panel_a_quality_quantile: float = 0.35
    region_diagnostic_max_units: int = 120
    region_diagnostic_samples_per_unit: int = 2
    peak_diagnostic_samples_per_unit: int = 64
    peak_diagnostic_shuffle_repeats: int = 5
    peak_diagnostic_bootstrap_repeats: int = 2000
    # 0 means use every held-out probe.
    peak_diagnostic_max_probes: int = 0


METHOD_ORDER = (
    "Cosmos Gaussian",
    "Beryl Gaussian",
    "KDE",
    "Conditional GMM",
    "Conditional + kNN",
)

PANEL_B_COLUMN_ORDER = ("Observed TEST",) + METHOD_ORDER


def _panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def _panel_label_right(fig, y, label, *, x=0.012):
    """Place a panel label at the left side of the figure."""
    fig.text(
        float(x),
        float(y),
        label,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _panel_label_figure(fig, ax, label, *, x=None, dy=0.006):
    """Anchor a panel label above its row so layout adjustments preserve it."""
    ax.text(
        -0.075 if x is None else float(x),
        1.015 + float(dy),
        label,
        transform=ax.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _dominant_trace(waveform):
    waveform = np.asarray(waveform)
    return waveform[int(np.argmax(np.ptp(waveform, axis=1)))]


def _choose_good_reconstruction_examples(bundle, fig_cfg):
    """Choose TEST examples that are both well reconstructed and morphologically diverse.

    We first keep the best-reconstructed fraction of a reproducible candidate pool,
    then greedily maximize distance in a joint waveform/ACG/stPC morphology space.
    This avoids choosing two nearly identical "easy" examples.
    """
    data = bundle.data
    cfg = bundle.cfg
    test_ids = np.flatnonzero(np.asarray(data.split) == 2)
    if len(test_ids) == 0:
        raise RuntimeError("No TEST units are available for reconstruction examples.")

    n_pool = min(int(fig_cfg.panel_a_candidate_pool), len(test_ids))
    cand = reconstruction_examples_all_modalities(
        bundle.autoencoder, data, cfg, n_examples=n_pool, seed=int(fig_cfg.seed) + 811
    )
    candidate_ids = np.asarray(cand["indices"])

    pairs = (("waveform", "waveform_reconstruction"),
             ("acg", "acg_reconstruction"),
             ("stpc", "stpc_reconstruction"))
    nmse_cols = []
    morphology = []
    for original_key, reconstruction_key in pairs:
        original = np.asarray(cand[original_key], dtype=np.float32)
        reconstruction = np.asarray(cand[reconstruction_key], dtype=np.float32)
        reduce_axes = tuple(range(1, original.ndim))
        mse = np.mean((original - reconstruction) ** 2, axis=reduce_axes)
        energy = np.mean(original ** 2, axis=reduce_axes)
        nmse_cols.append(mse / np.maximum(energy, 1e-8))

        flat = original.reshape(len(original), -1).astype(np.float64)
        flat -= np.mean(flat, axis=1, keepdims=True)
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        # Random projection keeps diversity computation light while preserving shape differences.
        rng = np.random.default_rng(int(fig_cfg.seed) + 991 + len(morphology))
        n_proj = min(24, flat.shape[1])
        proj = rng.normal(size=(flat.shape[1], n_proj)) / np.sqrt(n_proj)
        morphology.append(flat @ proj)

    score = np.mean(np.column_stack(nmse_cols), axis=1)
    q = float(np.clip(fig_cfg.panel_a_quality_quantile, 0.05, 1.0))
    cutoff = np.quantile(score, q)
    eligible = np.flatnonzero(score <= cutoff)
    n_keep = min(int(fig_cfg.reconstruction_examples_per_modality), len(eligible))
    morph = np.column_stack(morphology)
    morph = (morph - np.mean(morph, axis=0, keepdims=True)) / np.maximum(
        np.std(morph, axis=0, keepdims=True), 1e-8
    )

    # Start from the best reconstruction, then choose the most different good example.
    chosen = [int(eligible[np.argmin(score[eligible])])]
    while len(chosen) < n_keep:
        remaining = np.asarray([i for i in eligible if i not in chosen], dtype=int)
        d = np.linalg.norm(morph[remaining, None, :] - morph[np.asarray(chosen)][None, :, :], axis=2)
        min_d = np.min(d, axis=1)
        chosen.append(int(remaining[np.argmax(min_d)]))
    keep = np.asarray(chosen, dtype=int)

    examples = {}
    for key, value in cand.items():
        if key == "indices":
            examples[key] = np.asarray(value)[keep]
        else:
            arr = np.asarray(value)
            examples[key] = arr[keep] if len(arr) == len(candidate_ids) else value

    print("[Supp Fig. 2 panel a] diverse, well-reconstructed TEST units:")
    for j, idx in enumerate(keep):
        print(f"  example {j+1}: unit={candidate_ids[idx]}, mean_NMSE={score[idx]:.4g}, "
              f"per_modality={[float(x[idx]) for x in nmse_cols]}")
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
    return {
        "Cosmos Gaussian": {"method_kind": "cosmos_gaussian", "baseline": baselines["Cosmos Gaussian"]},
        "Beryl Gaussian": {"method_kind": "beryl_gaussian", "baseline": baselines["Beryl Gaussian"]},
        "KDE": {"method_kind": "kde", "baseline": baselines["KDE"]},
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


def _remove_peak_value_feature(data, feature_indices):
    """Remove peak value while preserving all other selected feature rows."""
    excluded_names = {
        "peak_val",
        "peak_value",
        "peak value",
    }

    kept = []
    removed = []

    for index in feature_indices:
        name = str(data.waveform_feature_names[int(index)])
        if name.lower() in excluded_names:
            removed.append(name)
        else:
            kept.append(int(index))

    if removed:
        print(
            "[Supp Fig. 2] removed feature row(s): "
            + ", ".join(removed)
        )
    else:
        print(
            "[Supp Fig. 2] peak_val was not among the selected features; "
            "no feature row was removed."
        )

    if not kept:
        raise RuntimeError(
            "Removing peak_val left no features to plot."
        )

    return np.asarray(kept, dtype=int)


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


def _observed_test_feature_data(data, cfg, feature_indices, coord_um, slab_width_um):
    """Return held-out observations, both globally and within the shown slice.

    The global TEST values determine color scaling and distribution diagnostics.
    The slice subset contains TEST units whose mediolateral coordinate lies in
    the same voxel-width sagittal slab used for panel b.
    """
    test = np.asarray(data.split) == 2
    xyz_m = np.asarray(data.xyz_m, dtype=float)
    # This is the same canonical feature loader used by the unit-model
    # visualizations. It applies the model's feature ordering, sign conventions,
    # units and transformations before indices are selected.
    all_features = get_model_space_waveform_features(data, cfg)
    features = np.asarray(all_features, dtype=float)[:, feature_indices]
    finite_xyz = np.all(np.isfinite(xyz_m), axis=1)
    in_slab = (
        test
        & finite_xyz
        & (np.abs(xyz_m[:, 0] * 1e6 - float(coord_um)) <= float(slab_width_um) / 2.0)
    )

    if not np.any(in_slab):
        # Avoid a blank diagnostic column if no unit falls exactly in the voxel
        # slab; use the nearest held-out probe/unit plane reproducibly.
        test_ids = np.flatnonzero(test & finite_xyz)
        nearest_distance = np.min(np.abs(xyz_m[test_ids, 0] * 1e6 - float(coord_um)))
        in_slab[test_ids[np.isclose(
            np.abs(xyz_m[test_ids, 0] * 1e6 - float(coord_um)),
            nearest_distance,
        )]] = True

    print(
        "[Supp Fig. 2 panel b] observed TEST units in sagittal slab: "
        f"{int(np.sum(in_slab))} / {int(np.sum(test))}"
    )
    slab_pids = np.asarray(data.pids)[in_slab]
    n_probes = len(np.unique(slab_pids))
    print(f"[Supp Fig. 2 panel b] relevant TEST probes: {n_probes}")
    return {
        "xyz_m": xyz_m[in_slab],
        "features": features[in_slab],
        "all_test_features": features[test],
        "pids": slab_pids,
        "n_probes": n_probes,
    }


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

    observed = _observed_test_feature_data(
        data,
        cfg,
        feature_indices,
        coord_um,
        cfg.diagnostic_voxel_size_um,
    )
    predictions = {"Observed TEST": observed, **predictions}

    gs = GridSpecFromSubplotSpec(
        len(feature_indices),
        len(PANEL_B_COLUMN_ORDER),
        subplot_spec=spec,
        hspace=0.08,
        wspace=0.05,
    )
    first = None

    # Use the empirical held-out TEST distribution as the reference scale.
    # Every method and the observed slice therefore share exactly the same
    # original-unit color limits for a given feature.
    qlo, qhi = fig_cfg.panel_b_color_quantiles
    limits = {}
    for row, _ in enumerate(feature_indices):
        observed_test = np.asarray(observed["all_test_features"][:, row], dtype=float)
        finite = observed_test[np.isfinite(observed_test)]
        if len(finite) == 0:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = np.quantile(finite, [qlo, qhi])
            if hi <= lo:
                center = float(np.nanmedian(finite))
                eps = max(abs(center) * 1e-3, 1e-8)
                lo, hi = center - eps, center + eps
        if lo < 0.0 < hi:
            magnitude = max(abs(float(lo)), abs(float(hi)))
            lo, hi = -magnitude, magnitude
        for name in PANEL_B_COLUMN_ORDER:
            limits[(row, name)] = (float(lo), float(hi))

    for row, findex in enumerate(feature_indices):
        for col, name in enumerate(PANEL_B_COLUMN_ORDER):
            ax = fig.add_subplot(gs[row, col])
            if first is None:
                first = ax

            payload = predictions[name]
            xyz_um = payload["xyz_m"] * 1e6
            values = np.asarray(payload["features"][:, row], dtype=float)
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
                title = name
                if name == "Observed TEST":
                    title = f"Observed TEST probes\n(n={payload['n_probes']})"
                ax.set_title(title, fontsize=6.5, pad=2)
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
    _panel_label_figure(fig, first, "b", dy=0.025)
    return predictions


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

    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    ax.spines[["top", "right"]].set_visible(False)
    _panel_label_figure(fig, ax, "c", dy=0.025)



def _temporary_scale_diagnostics(predictions, feature_indices, feature_names, fig_cfg):
    """Compare every predicted marginal with the correctly loaded TEST marginal."""
    outdir = Path(fig_cfg.diagnostics_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row, (fidx, fname) in enumerate(zip(feature_indices, feature_names)):
        observed = np.asarray(predictions["Observed TEST"]["all_test_features"][:, row], float)
        observed = observed[np.isfinite(observed)]
        obs_q25, obs_median, obs_q75 = np.quantile(observed, [.25, .5, .75])
        obs_iqr = max(float(obs_q75 - obs_q25), 1e-12)
        obs_std = max(float(np.std(observed)), 1e-12)
        for method in PANEL_B_COLUMN_ORDER:
            source_key = "all_test_features" if method == "Observed TEST" else "features"
            x = np.asarray(predictions[method][source_key][:, row], float)
            x = x[np.isfinite(x)]
            if not len(x):
                continue
            q01, q05, q25, q50, q75, q95, q99 = np.quantile(x, [.01,.05,.25,.5,.75,.95,.99])
            iqr = q75-q25
            central90 = q95-q05
            tail_span = q99-q01
            rows.append([
                fname, method, len(x), np.mean(x), np.std(x), q01, q05, q25,
                q50, q75, q95, q99, iqr, central90, tail_span,
                tail_span / max(iqr, 1e-12), np.std(x) / obs_std,
                iqr / obs_iqr, (q50 - obs_median) / obs_iqr,
                (
                    "too_narrow" if iqr / obs_iqr < 0.5 else
                    "too_wide" if iqr / obs_iqr > 2.0 else
                    "shifted" if abs((q50 - obs_median) / obs_iqr) > 1.0 else
                    "approximately_calibrated"
                ),
            ])
    header = (
        "feature,method,n,mean,std,q01,q05,q25,median,q75,q95,q99,iqr,"
        "central90,tail_span_q99_q01,tail_to_iqr,std_ratio_to_test,"
        "iqr_ratio_to_test,median_shift_in_test_iqr,scale_flag"
    )
    path = outdir / "panel_b_scale_diagnostics.csv"
    with path.open("w", encoding="utf8") as f:
        f.write(header+"\n")
        for r in rows:
            f.write(",".join(map(str,r))+"\n")
    print(f"[TEMP diagnostics] panel-b scale statistics: {path}")
    for fname in feature_names:
        print(f"\n[Panel b scale] {fname}")
        for r in rows:
            if r[0] == fname:
                print(f"  {r[1]:20s} std={r[4]:.4g} IQR={r[12]:.4g} "
                      f"std/test={r[16]:.2f} IQR/test={r[17]:.2f} "
                      f"median shift={r[18]:+.2f} test-IQR")

    # Compact calibration view: 0 means the predicted and TEST widths match.
    model_rows = [r for r in rows if r[1] != "Observed TEST"]
    matrix_sd = np.full((len(METHOD_ORDER), len(feature_names)), np.nan)
    matrix_shift = np.full_like(matrix_sd, np.nan)
    for r in model_rows:
        i = METHOD_ORDER.index(r[1])
        j = list(feature_names).index(r[0])
        matrix_sd[i, j] = np.log2(max(float(r[16]), 1e-12))
        matrix_shift[i, j] = abs(float(r[18]))

    fig, axes = plt.subplots(2, 1, figsize=(max(7, 1.25 * len(feature_names)), 5.2))
    for ax, matrix, title, cmap, vmin, vmax in [
        (axes[0], matrix_sd, "Predicted width relative to TEST (log2 SD ratio)", "coolwarm", -3, 3),
        (axes[1], matrix_shift, "Absolute median error (TEST IQR units)", "magma", 0, None),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
        ax.set_xticks(np.arange(len(feature_names)), feature_names, rotation=25, ha="right")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "panel_b_scale_calibration.pdf", dpi=250)
    plt.close(fig)


def _temporary_nll_sign_diagnostics(nll, feature_names):
    """TEMP: explain negative differential NLL values in panel c."""
    print("\n[TEMP diagnostics] Panel-c NLL sign check")
    print("  Continuous-density NLL is -log p(x); it can be negative whenever density p(x) > 1.")
    for j, fname in enumerate(feature_names):
        vals = {m: float(np.asarray(nll[m])[j]) for m in METHOD_ORDER}
        print(f"  {fname}: " + ", ".join(f"{m}={v:.4f}" for m,v in vals.items()))
    if "peak_val" in feature_names:
        j = feature_names.index("peak_val")
        neg = [m for m in METHOD_ORDER if float(np.asarray(nll[m])[j]) < 0]
        print(f"  peak_val methods with negative mean NLL: {neg}")
        print("  This is not a probability >1: a probability DENSITY may exceed 1 when a continuous distribution is narrow.")


def _feature_index(names, aliases):
    """Resolve one feature index from a string or a sequence of aliases."""
    names = [str(name) for name in names]
    if isinstance(aliases, str):
        aliases = (aliases,)
    lowered = {name.lower(): i for i, name in enumerate(names)}
    for alias in aliases:
        if str(alias).lower() in lowered:
            return lowered[str(alias).lower()]
    raise KeyError(f"None of {tuple(aliases)!r} found in waveform features: {names}")


def _bootstrap_mean_ci(values, rng, repeats):
    """Probe-bootstrap mean and percentile confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(np.mean(values)), float(lo), float(hi)


def _peak_value_diagnostics(bundle, methods, fig_cfg):
    """Diagnose why peak-value NLL is low and whether spatial KDE adds information.

    The controls distinguish absolute differential NLL from useful spatial
    information:
      * Global Gaussian: no anatomical information.
      * Cosmos/Beryl Gaussian: region only.
      * Shuffled-latent KDE: preserves KDE geometry and the empirical latent
        marginal, but destroys the phenotype-to-location relationship.
      * Spatial KDE: intact continuous spatial information.
      * Conditional models: final learned alternatives.

    NLL is evaluated separately for each held-out probe, allowing paired
    probe-level deltas and probe-bootstrap confidence intervals.
    """
    data = bundle.data
    cfg = bundle.cfg
    outdir = Path(fig_cfg.diagnostics_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(fig_cfg.seed) + 1907)

    peak_index = _feature_index(
        data.waveform_feature_names,
        ("peak_val", "peak_value", "peak value"),
    )
    peak_indices = np.asarray([peak_index], dtype=int)
    original_split = np.asarray(data.split).copy()
    train = original_split == 0
    test = original_split == 2
    model_features = np.asarray(
        get_model_space_waveform_features(data, cfg), dtype=float
    )
    train_peak = model_features[train, peak_index]
    train_peak = train_peak[np.isfinite(train_peak)]
    global_feature_mean = float(np.mean(train_peak))
    global_feature_std = max(float(np.std(train_peak)), 1e-8)
    global_feature_kde = gaussian_kde(train_peak)

    # A single-region Gaussian is the constant/global parametric marginal.
    global_labels = np.zeros(len(original_split), dtype=np.int64)
    global_gaussian = RegionalGaussianBaseline(
        bundle.z_scaled,
        global_labels,
        train,
        cfg.region_gaussian_variance_floor,
    )

    control_methods = dict(methods)
    control_methods["Global Gaussian"] = {
        "method_kind": "cosmos_gaussian",
        "baseline": global_gaussian,
    }

    shuffled_names = []
    train_ids = np.flatnonzero(train)
    for repeat in range(int(fig_cfg.peak_diagnostic_shuffle_repeats)):
        # Keep the anatomical sampling geometry unchanged, but randomly assign
        # training latent phenotypes to locations. This is an empirical KDE
        # null with no true spatial phenotype relationship.
        shuffled_z = np.asarray(bundle.z_scaled).copy()
        shuffled_z[train_ids] = shuffled_z[rng.permutation(train_ids)]
        name = f"Shuffled-latent KDE {repeat + 1}"
        shuffled_names.append(name)
        control_methods[name] = {
            "method_kind": "kde",
            "baseline": SpatialKDEBaseline(shuffled_z, data.xyz_m, train, cfg),
        }

    test_pids = np.unique(np.asarray(data.pids)[test])
    max_probes = int(fig_cfg.peak_diagnostic_max_probes)
    if max_probes > 0 and len(test_pids) > max_probes:
        test_pids = np.sort(rng.choice(test_pids, size=max_probes, replace=False))

    method_names = list(control_methods)
    records = []
    try:
        for number, pid in enumerate(test_pids, start=1):
            probe_test = test & (np.asarray(data.pids) == pid)
            split = original_split.copy()
            split[test] = 1
            split[probe_test] = 2
            data.split = split
            result = feature_nll_comparison(
                bundle.autoencoder,
                data,
                bundle.z_scaled,
                bundle.latent_scaler,
                cfg,
                feature_indices=peak_indices,
                methods=control_methods,
                samples_per_test_unit=int(fig_cfg.peak_diagnostic_samples_per_unit),
            )
            record = {"pid": str(pid), "n_test_units": int(np.sum(probe_test))}
            for name in method_names:
                record[name] = float(np.asarray(result[name], dtype=float)[0])
            observed_peak = model_features[probe_test, peak_index]
            observed_peak = observed_peak[np.isfinite(observed_peak)]
            gaussian_log_density = (
                -0.5 * ((observed_peak - global_feature_mean) / global_feature_std) ** 2
                - np.log(global_feature_std)
                - 0.5 * np.log(2.0 * np.pi)
            )
            record["Global feature Gaussian"] = float(-np.mean(gaussian_log_density))
            record["Global feature KDE"] = float(
                -np.mean(global_feature_kde.logpdf(observed_peak))
            )
            record["Shuffled-latent KDE mean"] = float(np.mean([
                record[name] for name in shuffled_names
            ]))
            records.append(record)
            print(f"[peak_val diagnostics] probe {number}/{len(test_pids)}: {pid}")
    finally:
        data.split = original_split

    per_probe_path = outdir / "peak_value_nll_per_probe.csv"
    output_names = list(METHOD_ORDER) + [
        "Global Gaussian", "Global feature Gaussian", "Global feature KDE"
    ] + shuffled_names + [
        "Shuffled-latent KDE mean"
    ]
    with per_probe_path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pid", "n_test_units"] + output_names)
        writer.writeheader()
        writer.writerows(records)

    # Paired deltas are positive when the named method improves over its control.
    comparisons = {
        "Spatial KDE vs global feature KDE": ("Global feature KDE", "KDE"),
        "Spatial KDE vs global feature Gaussian": ("Global feature Gaussian", "KDE"),
        "Spatial KDE vs global latent Gaussian": ("Global Gaussian", "KDE"),
        "Spatial KDE vs shuffled KDE": ("Shuffled-latent KDE mean", "KDE"),
        "Conditional+kNN vs Global Gaussian": ("Global Gaussian", "Conditional + kNN"),
        "Conditional+kNN vs Conditional GMM": ("Conditional GMM", "Conditional + kNN"),
        "Conditional+kNN vs Spatial KDE": ("KDE", "Conditional + kNN"),
    }
    summary_rows = []
    bootstrap_rng = np.random.default_rng(int(fig_cfg.seed) + 1908)
    for name in list(METHOD_ORDER) + [
        "Global Gaussian", "Global feature Gaussian", "Global feature KDE",
        "Shuffled-latent KDE mean",
    ]:
        values = np.asarray([record[name] for record in records], dtype=float)
        mean, lo, hi = _bootstrap_mean_ci(
            values, bootstrap_rng, fig_cfg.peak_diagnostic_bootstrap_repeats
        )
        summary_rows.append({
            "quantity": "mean_nll",
            "comparison": name,
            "estimate": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "fraction_below_zero": float(np.mean(values < 0)),
            "fraction_method_wins": np.nan,
        })
    for label, (control, method) in comparisons.items():
        delta = np.asarray(
            [record[control] - record[method] for record in records], dtype=float
        )
        mean, lo, hi = _bootstrap_mean_ci(
            delta, bootstrap_rng, fig_cfg.peak_diagnostic_bootstrap_repeats
        )
        summary_rows.append({
            "quantity": "paired_delta_nll_control_minus_method",
            "comparison": label,
            "estimate": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "fraction_below_zero": np.nan,
            "fraction_method_wins": float(np.mean(delta > 0)),
        })

    summary_path = outdir / "peak_value_nll_summary.csv"
    with summary_path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Compare empirical TEST peak scale to the marginal slice predictions.
    coord_um = _central_sagittal_coord_um(data, cfg.diagnostic_voxel_size_um)
    peak_predictions = {}
    for name in METHOD_ORDER:
        method = methods[name]
        peak_predictions[name] = publication_feature_slice_data(
            bundle.autoencoder, data, bundle.latent_scaler, cfg,
            feature_indices=peak_indices,
            method_kind=method["method_kind"],
            sagittal_coord_um=coord_um,
            gmm=method.get("gmm"),
            conditional_model=method.get("conditional_model"),
            baseline=method.get("baseline"),
            empirical_decoder=method.get("empirical_decoder"),
        )
    observed = _observed_test_feature_data(
        data, cfg, peak_indices, coord_um, cfg.diagnostic_voxel_size_um
    )
    observed_peak = np.asarray(observed["all_test_features"][:, 0], dtype=float)
    observed_peak = observed_peak[np.isfinite(observed_peak)]
    obs_q25, obs_median, obs_q75 = np.quantile(observed_peak, [.25, .5, .75])
    obs_iqr = max(float(obs_q75 - obs_q25), 1e-12)
    obs_std = max(float(np.std(observed_peak)), 1e-12)

    scale_rows = []
    scale_sources = {"Observed TEST": observed_peak}
    scale_sources.update({
        name: np.asarray(peak_predictions[name]["features"][:, 0], dtype=float)
        for name in METHOD_ORDER
    })
    for name, values in scale_sources.items():
        values = values[np.isfinite(values)]
        q01, q05, q25, median, q75, q95, q99 = np.quantile(
            values, [.01, .05, .25, .5, .75, .95, .99]
        )
        scale_rows.append({
            "method": name,
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(median),
            "iqr": float(q75 - q25),
            "central90": float(q95 - q05),
            "q01": float(q01),
            "q99": float(q99),
            "std_ratio_to_test": float(np.std(values) / obs_std),
            "iqr_ratio_to_test": float((q75 - q25) / obs_iqr),
            "median_shift_in_test_iqr": float((median - obs_median) / obs_iqr),
            "scale_flag": (
                "too_narrow" if (q75 - q25) / obs_iqr < 0.5 else
                "too_wide" if (q75 - q25) / obs_iqr > 2.0 else
                "shifted" if abs((median - obs_median) / obs_iqr) > 1.0 else
                "approximately_calibrated"
            ),
        })
    with (outdir / "peak_value_scale_calibration.csv").open(
        "w", newline="", encoding="utf8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scale_rows[0]))
        writer.writeheader()
        writer.writerows(scale_rows)

    # One diagnostic PDF combines absolute NLL, spatial-information deltas and
    # peak-value scale calibration.
    plot_names = list(METHOD_ORDER) + [
        "Global feature Gaussian", "Global feature KDE", "Shuffled-latent KDE mean"
    ]
    nll_matrix = [np.asarray([record[name] for record in records]) for name in plot_names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].boxplot(nll_matrix, tick_labels=plot_names, showfliers=False)
    axes[0].axhline(0, color="0.4", lw=.8, ls="--")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set_ylabel("Held-out peak_val NLL per probe")
    axes[0].set_title("Absolute differential NLL")

    delta_labels = list(comparisons)
    delta_matrix = [
        np.asarray([record[c] - record[m] for record in records])
        for c, m in comparisons.values()
    ]
    axes[1].boxplot(delta_matrix, tick_labels=delta_labels, showfliers=False)
    axes[1].axhline(0, color="0.4", lw=.8, ls="--")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_ylabel("Control NLL − method NLL")
    axes[1].set_title("Positive means useful information")

    scale_method_rows = [row for row in scale_rows if row["method"] != "Observed TEST"]
    x = np.arange(len(scale_method_rows))
    axes[2].bar(x - .18, [np.log2(max(row["std_ratio_to_test"], 1e-12)) for row in scale_method_rows],
                width=.36, label="log2 SD ratio")
    axes[2].bar(x + .18, [row["median_shift_in_test_iqr"] for row in scale_method_rows],
                width=.36, label="median shift / TEST IQR")
    axes[2].axhline(0, color="0.4", lw=.8)
    axes[2].set_xticks(x, [row["method"] for row in scale_method_rows], rotation=55, ha="right")
    axes[2].set_title("Peak-value scale calibration")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "peak_value_diagnostics.pdf", dpi=250)
    plt.close(fig)

    print(f"[peak_val diagnostics] wrote {per_probe_path}")
    print(f"[peak_val diagnostics] wrote {summary_path}")
    print(f"[peak_val diagnostics] outputs completed in {outdir}")


def _temporary_region_method_diagnostics(bundle, methods, feature_indices, feature_names, fig_cfg):
    """TEMP: Cosmos-region method comparison using exactly the existing NLL helper.

    Produces (1) regional mean NLL heatmap and CSV, and (2) sampled unit-level
    winner counts per Cosmos region. The latter is intentionally capped because
    calling the existing NLL helper once per unit is expensive.
    """
    data = bundle.data
    outdir = Path(fig_cfg.diagnostics_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    original_split = np.asarray(data.split).copy()
    test_ids = np.flatnonzero(original_split == 2)
    cosmos = np.abs(np.asarray(data.cosmos_ids, dtype=np.int64))
    br = __import__("iblatlas.regions", fromlist=["BrainRegions"]).BrainRegions()
    id_to_acr = {abs(int(i)): str(a) for i,a in zip(br.id, br.acronym)}
    region_ids, counts = np.unique(cosmos[test_ids], return_counts=True)
    order = np.argsort(counts)[::-1]
    region_ids = region_ids[order]
    region_names = [id_to_acr.get(int(r), str(int(r))) for r in region_ids]

    regional = np.full((len(METHOD_ORDER), len(region_ids)), np.nan)
    winners = np.zeros((len(METHOD_ORDER), len(region_ids)), dtype=int)
    evaluated = np.zeros(len(region_ids), dtype=int)
    rng = np.random.default_rng(int(fig_cfg.seed)+404)

    try:
        for c, rid in enumerate(region_ids):
            ids = test_ids[cosmos[test_ids] == rid]
            # Regional aggregate: train unchanged, TEST restricted to this region.
            tmp = original_split.copy()
            tmp[(original_split == 2)] = 1
            tmp[ids] = 2
            data.split = tmp
            reg_nll = feature_nll_comparison(
                bundle.autoencoder, data, bundle.z_scaled, bundle.latent_scaler, bundle.cfg,
                feature_indices=feature_indices, methods=methods,
                samples_per_test_unit=int(fig_cfg.region_diagnostic_samples_per_unit),
            )
            for m, method in enumerate(METHOD_ORDER):
                regional[m,c] = float(np.nanmean(np.asarray(reg_nll[method], float)))

            # Unit-level winner counts, sampled for runtime.
            use = ids if len(ids) <= fig_cfg.region_diagnostic_max_units else rng.choice(
                ids, int(fig_cfg.region_diagnostic_max_units), replace=False)
            for uid in use:
                tmp = original_split.copy()
                tmp[(original_split == 2)] = 1
                tmp[int(uid)] = 2
                data.split = tmp
                one = feature_nll_comparison(
                    bundle.autoencoder, data, bundle.z_scaled, bundle.latent_scaler, bundle.cfg,
                    feature_indices=feature_indices, methods=methods,
                    samples_per_test_unit=int(fig_cfg.region_diagnostic_samples_per_unit),
                )
                scores = np.asarray([np.nanmean(np.asarray(one[m],float)) for m in METHOD_ORDER])
                if np.any(np.isfinite(scores)):
                    winners[int(np.nanargmin(scores)), c] += 1
                    evaluated[c] += 1
    finally:
        data.split = original_split

    # CSVs
    with (outdir/"cosmos_region_mean_nll.csv").open("w",encoding="utf8") as f:
        f.write("method,"+",".join(region_names)+"\n")
        for m, method in enumerate(METHOD_ORDER):
            f.write(method+","+",".join(map(str,regional[m]))+"\n")
    with (outdir/"cosmos_region_unit_winner_counts.csv").open("w",encoding="utf8") as f:
        f.write("method,"+",".join(region_names)+"\n")
        for m, method in enumerate(METHOD_ORDER):
            f.write(method+","+",".join(map(str,winners[m]))+"\n")

    # Heatmaps
    for matrix, title, fname, fmt in [
        (regional, "Mean held-out feature NLL by Cosmos region", "cosmos_region_mean_nll_heatmap.pdf", ".2f"),
        (winners, "Best-method counts by Cosmos region (sampled test units)", "cosmos_region_winner_counts_heatmap.pdf", "d")]:
        fig, ax = plt.subplots(figsize=(max(8, .65*len(region_names)), 3.4))
        im=ax.imshow(matrix, aspect="auto")
        ax.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
        ax.set_xticks(np.arange(len(region_names)), region_names, rotation=60, ha="right")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
        if len(region_names) <= 15:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val=matrix[i,j]
                    txt = (f"{int(val)}" if fmt=="d" else f"{val:.2f}") if np.isfinite(val) else ""
                    ax.text(j,i,txt,ha="center",va="center",fontsize=6)
        fig.tight_layout(); fig.savefig(outdir/fname, dpi=250); plt.close(fig)
    print(f"[TEMP diagnostics] Cosmos diagnostics written to {outdir}")
    print("[TEMP diagnostics] unit winner counts are based on at most "
          f"{fig_cfg.region_diagnostic_max_units} TEST units/region; see evaluated counts in console.")
    for name,n in zip(region_names,evaluated): print(f"  {name}: n_unit_winner_evaluated={n}")

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
    feature_indices = _remove_peak_value_feature(
        data,
        feature_indices,
    )
    feature_names = [
        data.waveform_feature_names[int(i)]
        for i in feature_indices
    ]
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
    fig.set_size_inches(fig.get_size_inches()[0] * 1.10, 9.8)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.8, 4.45, 1.85], hspace=0.26)
    draw_panel_a(fig, outer[0], examples)
    predictions = draw_panel_b(fig, outer[1], bundle, methods, feature_indices, fig_cfg)
    ax_c = fig.add_subplot(outer[2])
    draw_panel_c(fig, ax_c, nll, feature_names)

    if fig_cfg.run_diagnostics:
        _temporary_nll_sign_diagnostics(nll, feature_names)
        _temporary_scale_diagnostics(predictions, feature_indices, feature_names, fig_cfg)
        _peak_value_diagnostics(bundle, methods, fig_cfg)
        _temporary_region_method_diagnostics(bundle, methods, feature_indices, feature_names, fig_cfg)

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
