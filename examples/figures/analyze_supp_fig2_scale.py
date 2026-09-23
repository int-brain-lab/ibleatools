from __future__ import annotations

"""Standalone robust-scale diagnostic for Supplementary Figure 2 panel b.

Place this file beside the current Supp. Fig. 2 generation script and run it.
It imports that script so data loading, model loading, feature selection and
prediction generation are identical to the figure itself.
"""

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


# Set this explicitly if auto-discovery does not select the intended script.
SUPP_FIG2_SCRIPT: Path | None = None
OUTPUT_DIR = Path("unit_level_model_results/supp_fig2_scale_analysis")

# Candidate symmetric TEST-observation limits. For example, 5 means Q05-Q95.
CMAP_TAIL_PERCENTILES = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0)

# A shared scale is considered visually flat for a method if its central 90%
# occupies less than this fraction of the available color range.
FLAT_CENTRAL90_FRACTION = 0.15


def _progress(items: Iterable, *, desc: str, unit: str):
    """Use tqdm when available, otherwise print a compact progress bar."""
    items = list(items)
    if _tqdm is not None:
        yield from _tqdm(items, desc=desc, unit=unit)
        return
    total = len(items)
    for index, item in enumerate(items, start=1):
        fraction = index / max(total, 1)
        filled = int(round(24 * fraction))
        bar = "#" * filled + "-" * (24 - filled)
        print(f"\r{desc}: [{bar}] {index}/{total} {unit}", end="", flush=True)
        yield item
    print()


def _find_reference_script() -> Path:
    if SUPP_FIG2_SCRIPT is not None:
        path = Path(SUPP_FIG2_SCRIPT).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Supp. Fig. 2 script does not exist: {path}")
        return path

    this_file = Path(__file__).resolve()
    candidates = [
        path.resolve()
        for pattern in ("supp_fig2*.py", "*supp*fig*2*.py")
        for path in Path.cwd().glob(pattern)
        if path.resolve() != this_file
    ]
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the Supp. Fig. 2 script. Set SUPP_FIG2_SCRIPT at "
            "the top of this diagnostic file to its full path."
        )
    return candidates[0]


def _load_reference_module(path: Path):
    print(f"[1/7] Importing figure code from: {path}")
    spec = importlib.util.spec_from_file_location("supp_fig2_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _quantile(values: np.ndarray, percentile: float) -> float:
    return float(np.quantile(values, float(percentile) / 100.0))


def _percentile_in_reference(value: float, sorted_reference: np.ndarray) -> float:
    return 100.0 * float(np.searchsorted(sorted_reference, value, side="right")) / len(sorted_reference)


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = _finite(values)
    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    iqr = float(q75 - q25)
    lower_fence = float(q25 - 1.5 * iqr)
    upper_fence = float(q75 + 1.5 * iqr)
    inlier = values[(values >= lower_fence) & (values <= upper_fence)]
    full_std = float(np.std(values))
    inlier_std = float(np.std(inlier)) if len(inlier) else np.nan
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": full_std,
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "iqr": iqr,
        "central90": float(q95 - q05),
        "central98": float(q99 - q01),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "tukey_lower_fence": lower_fence,
        "tukey_upper_fence": upper_fence,
        "tukey_outlier_fraction": float(1.0 - len(inlier) / len(values)),
        "inlier_std": inlier_std,
        "std_inflation_from_outliers": (
            full_std / max(inlier_std, 1e-12) if np.isfinite(inlier_std) else np.nan
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_bundle(reference):
    print("[2/7] Loading prepared TEST observations and trained model...")
    fig_cfg = reference.FigureConfig()
    cfg = reference.Config(
        repo_id=fig_cfg.repo_id,
        vintage=fig_cfg.vintage,
        prepared_data_dir=fig_cfg.prepared_data_dir,
    )
    data = reference._load_or_prepare_unit_data(cfg)
    bundle = reference.load_unit_model(
        cfg,
        source="hub",
        data=data,
        token=fig_cfg.token,
        revision=fig_cfg.revision,
    )
    cfg = bundle.cfg
    cfg.feature_slice_count = int(fig_cfg.feature_count)
    cfg.feature_slice_seed = int(fig_cfg.seed) + 20260831
    return fig_cfg, bundle


def _selected_features(reference, bundle, fig_cfg):
    print("[3/7] Reproducing panel-b feature selection...")
    indices = reference.choose_feature_slice_indices(bundle.data, bundle.cfg)
    if hasattr(reference, "_remove_peak_value_feature"):
        indices = reference._remove_peak_value_feature(bundle.data, indices)
    indices = np.asarray(indices, dtype=int)
    names = [str(bundle.data.waveform_feature_names[int(index)]) for index in indices]
    print("      Selected features: " + ", ".join(names))
    return indices, names


def _generate_predictions(reference, bundle, feature_indices):
    data = bundle.data
    cfg = bundle.cfg
    methods = reference._build_methods(bundle)
    coord_um = reference._central_sagittal_coord_um(data, cfg.diagnostic_voxel_size_um)

    print(f"[4/7] Generating method predictions at sagittal ML={coord_um:.0f} um...")
    predictions = {}
    for name in _progress(reference.METHOD_ORDER, desc="Interpolation methods", unit="method"):
        method = methods[name]
        predictions[name] = reference.publication_feature_slice_data(
            bundle.autoencoder,
            data,
            bundle.latent_scaler,
            cfg,
            feature_indices=feature_indices,
            method_kind=method["method_kind"],
            sagittal_coord_um=coord_um,
            gmm=method.get("gmm"),
            conditional_model=method.get("conditional_model"),
            baseline=method.get("baseline"),
            empirical_decoder=method.get("empirical_decoder"),
        )

    print("      Loading TEST features through the canonical model-space loader...")
    all_features = np.asarray(
        reference.get_model_space_waveform_features(data, cfg), dtype=float
    )
    test = np.asarray(data.split) == 2
    xyz_m = np.asarray(data.xyz_m, dtype=float)
    finite_xyz = np.all(np.isfinite(xyz_m), axis=1)
    half_width = float(cfg.diagnostic_voxel_size_um) / 2.0
    in_slab = (
        test
        & finite_xyz
        & (np.abs(xyz_m[:, 0] * 1e6 - coord_um) <= half_width)
    )
    if not np.any(in_slab):
        test_ids = np.flatnonzero(test & finite_xyz)
        distance = np.abs(xyz_m[test_ids, 0] * 1e6 - coord_um)
        nearest = np.min(distance)
        in_slab[test_ids[np.isclose(distance, nearest)]] = True

    observed_all = all_features[test][:, feature_indices]
    observed_slab = all_features[in_slab][:, feature_indices]
    n_probes = len(np.unique(np.asarray(data.pids)[in_slab]))
    print(
        f"      TEST observations: {int(test.sum()):,} units total; "
        f"{int(in_slab.sum()):,} units from {n_probes} probes in the plotted slab."
    )
    return predictions, observed_all, observed_slab, coord_um


def _analyze_distributions(method_order, feature_names, predictions, observed_all, observed_slab):
    print("[5/7] Computing IQR, outlier, and robust distribution-alignment metrics...")
    summary_rows: list[dict] = []
    alignment_rows: list[dict] = []

    for feature_position, feature_name in enumerate(
        _progress(feature_names, desc="Feature distributions", unit="feature")
    ):
        observed = _finite(observed_all[:, feature_position])
        observed_slice = _finite(observed_slab[:, feature_position])
        obs_stats = _distribution_stats(observed)
        obs_iqr = max(obs_stats["iqr"], 1e-12)
        obs_sorted = np.sort(observed)

        for source, values in (
            ("Observed TEST all", observed),
            ("Observed TEST slab", observed_slice),
        ):
            stats = _distribution_stats(values)
            summary_rows.append({"feature": feature_name, "source": source, **stats})

        quantile_grid = np.linspace(0.05, 0.95, 19)
        observed_quantiles = np.quantile(observed, quantile_grid)

        for method in method_order:
            predicted = _finite(predictions[method]["features"][:, feature_position])
            stats = _distribution_stats(predicted)
            summary_rows.append({"feature": feature_name, "source": method, **stats})

            predicted_quantiles = np.quantile(predicted, quantile_grid)
            qrmse = float(np.sqrt(np.mean((predicted_quantiles - observed_quantiles) ** 2)))
            pred_q25, pred_q75 = np.quantile(predicted, [0.25, 0.75])
            pred_iqr = float(pred_q75 - pred_q25)
            pred_central90 = float(np.quantile(predicted, 0.95) - np.quantile(predicted, 0.05))
            obs_q25, obs_q75 = obs_stats["q25"], obs_stats["q75"]

            if obs_stats["iqr"] <= 1e-12:
                width_class = "TEST IQR is zero; use discrete-feature diagnostics"
            else:
                ratio = pred_iqr / obs_stats["iqr"]
                if ratio < 0.5:
                    width_class = "severely narrower than TEST"
                elif ratio < 0.8:
                    width_class = "moderately narrower than TEST"
                elif ratio <= 1.25:
                    width_class = "similar robust width"
                else:
                    width_class = "wider than TEST"

            alignment_rows.append({
                "feature": feature_name,
                "method": method,
                "test_q25": obs_q25,
                "test_q75": obs_q75,
                "test_iqr": obs_stats["iqr"],
                "pred_q25": float(pred_q25),
                "pred_q75": float(pred_q75),
                "pred_iqr": pred_iqr,
                "pred_iqr_over_test_iqr": pred_iqr / obs_iqr,
                "pred_central90_over_test_central90": (
                    pred_central90 / max(obs_stats["central90"], 1e-12)
                ),
                "pred_q25_as_test_percentile": _percentile_in_reference(pred_q25, obs_sorted),
                "pred_q75_as_test_percentile": _percentile_in_reference(pred_q75, obs_sorted),
                "fraction_predictions_inside_test_iqr": float(
                    np.mean((predicted >= obs_q25) & (predicted <= obs_q75))
                ),
                "wasserstein_over_test_iqr": float(
                    wasserstein_distance(observed, predicted) / obs_iqr
                ),
                "quantile_rmse_over_test_iqr": qrmse / obs_iqr,
                "median_shift_in_test_iqr": (
                    stats["median"] - obs_stats["median"]
                ) / obs_iqr,
                "width_assessment": width_class,
            })

    return summary_rows, alignment_rows


def _evaluate_cmap_limits(method_order, feature_names, predictions, observed_all):
    print("[6/7] Evaluating TEST-percentile color limits...")
    candidate_rows: list[dict] = []
    recommendation_rows: list[dict] = []

    for feature_position, feature_name in enumerate(
        _progress(feature_names, desc="Color-limit search", unit="feature")
    ):
        observed = _finite(observed_all[:, feature_position])
        feature_candidates = []

        for tail in CMAP_TAIL_PERCENTILES:
            lower = _quantile(observed, tail)
            upper = _quantile(observed, 100.0 - tail)
            width = upper - lower
            if not np.isfinite(width) or width <= 0:
                continue

            per_method = []
            for method in method_order:
                predicted = _finite(predictions[method]["features"][:, feature_position])
                p05, p25, p75, p95 = np.quantile(predicted, [.05, .25, .75, .95])
                saturation = float(np.mean((predicted < lower) | (predicted > upper)))
                central90_use = float((p95 - p05) / width)
                iqr_use = float((p75 - p25) / width)
                flat = central90_use < FLAT_CENTRAL90_FRACTION
                row = {
                    "feature": feature_name,
                    "test_lower_percentile": tail,
                    "test_upper_percentile": 100.0 - tail,
                    "vmin": lower,
                    "vmax": upper,
                    "test_fraction_clipped": 2.0 * tail / 100.0,
                    "method": method,
                    "method_fraction_saturated": saturation,
                    "method_iqr_color_utilization": iqr_use,
                    "method_central90_color_utilization": central90_use,
                    "method_would_look_flat": flat,
                }
                candidate_rows.append(row)
                per_method.append(row)

            max_saturation = max(row["method_fraction_saturated"] for row in per_method)
            min_use = min(row["method_central90_color_utilization"] for row in per_method)
            n_flat = sum(row["method_would_look_flat"] for row in per_method)
            # Prefer no more than 5% TEST clipping and 2.5% method saturation;
            # among valid scales, maximize visibility of the narrowest method.
            valid = tail <= 5.0 and max_saturation <= 0.025
            score = min_use - 2.0 * max_saturation - 0.25 * n_flat
            feature_candidates.append({
                "tail": tail,
                "lower": lower,
                "upper": upper,
                "valid": valid,
                "score": score,
                "max_method_saturation": max_saturation,
                "minimum_central90_utilization": min_use,
                "number_of_flat_methods": n_flat,
            })

        valid_candidates = [item for item in feature_candidates if item["valid"]]
        pool = valid_candidates if valid_candidates else feature_candidates
        best = max(pool, key=lambda item: item["score"])
        recommendation_rows.append({
            "feature": feature_name,
            "recommended_test_lower_percentile": best["tail"],
            "recommended_test_upper_percentile": 100.0 - best["tail"],
            "recommended_vmin": best["lower"],
            "recommended_vmax": best["upper"],
            "test_fraction_clipped": 2.0 * best["tail"] / 100.0,
            "max_method_fraction_saturated": best["max_method_saturation"],
            "minimum_method_central90_color_utilization": best["minimum_central90_utilization"],
            "number_of_methods_still_flat": best["number_of_flat_methods"],
            "all_constraints_satisfied": bool(best["valid"]),
        })

    return candidate_rows, recommendation_rows


def _print_report(summary_rows, alignment_rows, recommendation_rows, feature_names, method_order):
    print("\n" + "=" * 88)
    print("ROBUST SCALE DIAGNOSTIC SUMMARY")
    print("=" * 88)
    for feature in feature_names:
        observed = next(
            row for row in summary_rows
            if row["feature"] == feature and row["source"] == "Observed TEST all"
        )
        print(f"\n{feature}")
        print(
            f"  TEST Q25-Q75: [{observed['q25']:.6g}, {observed['q75']:.6g}]  "
            f"IQR={observed['iqr']:.6g}"
        )
        print(
            f"  Tukey outliers: {100 * observed['tukey_outlier_fraction']:.2f}%  |  "
            f"full-SD / inlier-SD={observed['std_inflation_from_outliers']:.2f}"
        )
        for method in method_order:
            row = next(
                item for item in alignment_rows
                if item["feature"] == feature and item["method"] == method
            )
            print(
                f"  {method:20s} IQR/TEST={row['pred_iqr_over_test_iqr']:.3f}  "
                f"Q25-Q75 map to TEST P{row['pred_q25_as_test_percentile']:.1f}-"
                f"P{row['pred_q75_as_test_percentile']:.1f}  "
                f"W1/IQR={row['wasserstein_over_test_iqr']:.3f}  "
                f"{row['width_assessment']}"
            )
        rec = next(row for row in recommendation_rows if row["feature"] == feature)
        print(
            f"  Suggested common cmap: TEST P{rec['recommended_test_lower_percentile']:g}-"
            f"P{rec['recommended_test_upper_percentile']:g} "
            f"[{rec['recommended_vmin']:.6g}, {rec['recommended_vmax']:.6g}]"
        )
        print(
            f"    clipped TEST={100 * rec['test_fraction_clipped']:.1f}%, "
            f"max method saturation={100 * rec['max_method_fraction_saturated']:.1f}%, "
            f"flat methods={rec['number_of_methods_still_flat']}"
        )


def _plot_iqr_alignment(summary_rows, feature_names, method_order, output_path):
    fig, axes = plt.subplots(
        len(feature_names), 1,
        figsize=(10, max(2.4 * len(feature_names), 4.0)),
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_order)))
    for row_index, feature in enumerate(feature_names):
        ax = axes[row_index, 0]
        observed = next(
            row for row in summary_rows
            if row["feature"] == feature and row["source"] == "Observed TEST all"
        )
        ax.axvspan(observed["q25"], observed["q75"], color="0.85", label="TEST IQR")
        ax.axvline(observed["median"], color="black", lw=1.2, label="TEST median")
        for y, (method, color) in enumerate(zip(method_order, colors), start=1):
            stats = next(
                item for item in summary_rows
                if item["feature"] == feature and item["source"] == method
            )
            ax.plot([stats["q25"], stats["q75"]], [y, y], color=color, lw=5, solid_capstyle="butt")
            ax.plot(stats["median"], y, marker="|", color="black", ms=9)
        ax.set_yticks(np.arange(1, len(method_order) + 1), method_order)
        ax.set_title(feature)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = _find_reference_script()
    reference = _load_reference_module(reference_path)
    fig_cfg, bundle = _build_bundle(reference)
    feature_indices, feature_names = _selected_features(reference, bundle, fig_cfg)
    predictions, observed_all, observed_slab, coord_um = _generate_predictions(
        reference, bundle, feature_indices
    )

    method_order = tuple(reference.METHOD_ORDER)
    summary_rows, alignment_rows = _analyze_distributions(
        method_order,
        feature_names,
        predictions,
        observed_all,
        observed_slab,
    )
    candidate_rows, recommendation_rows = _evaluate_cmap_limits(
        method_order,
        feature_names,
        predictions,
        observed_all,
    )

    print("[7/7] Writing reports...")
    _write_csv(OUTPUT_DIR / "distribution_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "model_iqr_alignment.csv", alignment_rows)
    _write_csv(OUTPUT_DIR / "cmap_candidate_evaluation.csv", candidate_rows)
    _write_csv(OUTPUT_DIR / "cmap_recommendations.csv", recommendation_rows)
    _plot_iqr_alignment(
        summary_rows,
        feature_names,
        method_order,
        OUTPUT_DIR / "iqr_alignment.pdf",
    )
    _print_report(
        summary_rows,
        alignment_rows,
        recommendation_rows,
        feature_names,
        method_order,
    )

    print("\nImportant interpretation note:")
    print(
        "  Panel-b predictions are values across an atlas slice, whereas TEST observations "
        "are individual units. A narrower predicted map can reflect either genuine model "
        "under-dispersion or the expected difference between a spatial summary field and "
        "within-location unit heterogeneity. Use the IQR alignment together with held-out "
        "NLL/distribution diagnostics before calling it model collapse."
    )
    print(f"\nCompleted. Results written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
