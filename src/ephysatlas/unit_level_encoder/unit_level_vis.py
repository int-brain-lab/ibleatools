from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice
from iblatlas.regions import BrainRegions
from sklearn.neighbors import KernelDensity

from .data import infer_training_hemisphere_sign, mirror_xyz_to_hemisphere
from .gmm_models import posterior_mean_for_context, sample_conditional, sample_conditional_for_context
from .waveform_features import extract_generated_waveform_features


def _region_names(ids):
    br = BrainRegions()
    lookup = {int(r): str(a) for r, a in zip(br.id, br.acronym)}
    return {int(r): lookup.get(int(r), f"rid_{int(r)}") for r in np.unique(ids)}


def _dominant_trace(w):
    ch = int(np.argmax(np.ptp(w, axis=1)))
    return w[ch]


def plot_reconstructions(ae, data, cfg, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = np.flatnonzero(data.split == 2)
    rng = np.random.default_rng(cfg.seed + 11)
    ids = rng.choice(test, size=min(cfg.diagnostic_examples, len(test)), replace=False)
    with torch.no_grad():
        w = torch.from_numpy(data.waveforms[ids]).to(cfg.device)
        a = torch.from_numpy(data.acgs[ids]).to(cfg.device)
        s = torch.from_numpy(data.stpc[ids]).to(cfg.device)
        lat = ae.encode(w, a, s)
        rec = ae.decode(lat)

    cols = 4
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.3 * rows), squeeze=False)
    for j, idx in enumerate(ids):
        ax = axes.flat[j]
        ax.plot(_dominant_trace(data.waveforms[idx]), label="observed", lw=1.2)
        ax.plot(_dominant_trace(rec["waveform"][j].cpu().numpy()), label="reconstruction", ls="--", lw=1)
        ax.set_title(f"test unit {idx}", fontsize=8)
        ax.set_xticks([])
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    if len(ids):
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Shared multimodal autoencoder: waveform reconstruction")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "reconstruction_waveforms.png", dpi=220)
    plt.close(fig)

    metrics = {
        "n_examples": int(len(ids)),
        "waveform_test_mse": float(np.mean((rec["waveform"].cpu().numpy() - data.waveforms[ids]) ** 2)),
        "acg_test_mse": float(np.mean((rec["acg"].cpu().numpy() - data.acgs[ids]) ** 2)),
        "stpc_test_mse": float(np.mean((rec["stpc"].cpu().numpy() - data.stpc[ids]) ** 2)),
    }
    return metrics


def plot_gmm_components(ae, gmm, scaler, cfg, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = scaler.inverse_transform(gmm.means_.astype(np.float32)).astype(np.float32)
    with torch.no_grad():
        joint = torch.from_numpy(raw).to(cfg.device)
        lat = ae.split_joint_latent(joint, cfg.modality_latent_dim)
        dec = ae.decode(lat)

    k = len(raw)
    cols = 5 if k >= 20 else 4
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.1 * rows), squeeze=False)
    wave = dec["waveform"].cpu().numpy()
    for j in range(k):
        axes.flat[j].plot(_dominant_trace(wave[j]))
        axes.flat[j].set_title(f"component {j} | p={gmm.weights_[j]:.3f}", fontsize=7)
        axes.flat[j].set_xticks([])
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle("Global GMM component centroids decoded to waveform")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "gmm_component_waveforms.png", dpi=220)
    plt.close(fig)

    arr = dec["acg"].cpu().numpy()
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.1 * rows), squeeze=False)
    for j in range(k):
        axes.flat[j].imshow(arr[j], aspect="auto", origin="lower")
        axes.flat[j].set_title(f"component {j}", fontsize=7)
        axes.flat[j].set_xticks([])
        axes.flat[j].set_yticks([])
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle("Global GMM component centroids decoded to ACG")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "gmm_component_acgs.png", dpi=220)
    plt.close(fig)

    arr = dec["stpc"].cpu().numpy()
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.1 * rows), squeeze=False)
    for j in range(k):
        axes.flat[j].plot(arr[j])
        axes.flat[j].set_title(f"component {j}", fontsize=7)
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.suptitle("Global GMM component centroids decoded to stPC")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "gmm_component_stpc.png", dpi=220)
    plt.close(fig)


def _decode_waveform_scaled_latents(ae, scaler, z_scaled, cfg):
    z_scaled = np.asarray(z_scaled, np.float32)
    raw = scaler.inverse_transform(z_scaled).astype(np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(raw), cfg.eval_batch_size):
            joint = torch.from_numpy(raw[start:start + cfg.eval_batch_size]).to(cfg.device)
            lat = ae.split_joint_latent(joint, cfg.modality_latent_dim)
            out.append(ae.decode(lat)["waveform"].cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.empty((0, *cfg.waveform_shape), np.float32)


def _feature_kde_nll(predicted, observed, min_bandwidth):
    predicted = np.asarray(predicted, np.float64)
    observed = np.asarray(observed, np.float64)
    predicted = predicted[np.isfinite(predicted)]
    observed = observed[np.isfinite(observed)]
    if len(predicted) < 2 or len(observed) == 0:
        return np.nan
    sd = float(np.std(predicted))
    silverman = 1.06 * max(sd, 1e-12) * (len(predicted) ** -0.2)
    bw = max(float(silverman), float(min_bandwidth), 1e-9)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(predicted[:, None])
    return float(-np.mean(kde.score_samples(observed[:, None])))


def _categorical_nll(predicted, observed, categories, alpha=1.0):
    categories = np.asarray(categories, np.float64)
    predicted = np.asarray(predicted, np.float64).reshape(-1)
    observed = np.asarray(observed, np.float64).reshape(-1)
    pred_idx = np.argmin(np.abs(predicted[:, None] - categories[None, :]), axis=1)
    obs_idx = np.argmin(np.abs(observed[:, None] - categories[None, :]), axis=1)
    counts = np.bincount(pred_idx, minlength=len(categories)).astype(np.float64) + float(alpha)
    prob = counts / counts.sum()
    return float(-np.mean(np.log(np.maximum(prob[obs_idx], 1e-12))))


def _model_space_feature_cache_path(cfg):
    return Path(cfg.prepared_data_dir) / "waveform_features_model_space.npy"


def _get_model_space_waveform_features(data, cfg):
    """Features extracted from the exact normalized waveforms seen by the AE.

    The prepared ``data.waveform_features`` array is sourced primarily from the
    IBL cluster table and is in the original waveform amplitude convention. The
    AE, however, is trained on per-unit max-abs normalized waveforms. Features
    extracted from decoded AE waveforms therefore live in the normalized-waveform
    convention. Mixing those two conventions caused both saturated slice maps
    and an inconsistent feature-NLL standardization.

    Cache this deterministic conversion because it is shared by every method.
    """
    path = _model_space_feature_cache_path(cfg)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.shape == (len(data.waveforms), len(data.waveform_feature_names)):
            return cached.astype(np.float32, copy=False)
        print(f"[feature cache] ignoring stale cache with shape={cached.shape}: {path}")

    print("[feature cache] extracting model-space waveform features once ...")
    features, names, report = extract_generated_waveform_features(
        data.waveforms,
        sampling_rate_hz=cfg.waveform_sampling_rate_hz,
        return_report=True,
    )
    if tuple(names) != tuple(data.waveform_feature_names):
        raise RuntimeError(
            f"Model-space feature order mismatch: {list(names)} != {data.waveform_feature_names}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.astype(np.float32), allow_pickle=False)
    (path.with_suffix(".json")).write_text(
        json.dumps({
            "definition": "features extracted from normalized waveforms.npy using the same extractor as decoded predictions",
            "shape": list(features.shape),
            "extractor_report": report,
        }, indent=2),
        encoding="utf-8",
    )
    return features.astype(np.float32, copy=False)


def _feature_eval_transform(data, cfg):
    """TRAIN-only transform in the same waveform convention as predictions."""
    train = data.split == 0
    feat = np.asarray(_get_model_space_waveform_features(data, cfg)[train], np.float64)
    mu = np.mean(feat[:, :-1], axis=0)
    sd = np.maximum(np.std(feat[:, :-1], axis=0), 1e-12)
    categories = np.unique(feat[:, -1])
    return mu, sd, categories


def _sample_method(method_kind, ids, n_each, rng, *, gmm=None, conditional_model=None, baseline=None, data=None):
    if method_kind in ("experimental", "experimental_knn"):
        return sample_conditional(ids, n_each, gmm, conditional_model, rng)
    if method_kind in ("cosmos_gaussian", "beryl_gaussian"):
        return baseline.sample(ids, n_each, rng)
    if method_kind == "kde":
        return baseline.sample(data.xyz_m[ids], n_each, rng)
    raise ValueError(method_kind)


def plot_region_distributions_single_method(
    ae, data, z_scaled, scaler, cfg, out_dir: Path, *, method_name: str,
    method_kind: str, gmm=None, conditional_model=None, baseline=None, empirical_decoder=None,
):
    """Observed-vs-predicted latent and waveform-feature distributions for one method."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = np.flatnonzero(data.split == 2)
    names = _region_names(data.cosmos_ids[test])
    region_ids, counts = np.unique(data.cosmos_ids[test], return_counts=True)
    region_ids = region_ids[np.argsort(counts)[::-1][:cfg.max_regions_in_distribution_plot]]
    rng = np.random.default_rng(cfg.seed + 77)

    all_observed_features = _get_model_space_waveform_features(data, cfg)
    observed_features = all_observed_features[test]
    feature_names = tuple(data.waveform_feature_names)
    report = {
        "source": "cached_model_space_features",
        "cache_path": str(_model_space_feature_cache_path(cfg)),
        "n_waveforms": int(len(test)),
    }
    lookup = {int(idx): row for row, idx in enumerate(test)}
    feature_mu, feature_sd, polarity_categories = _feature_eval_transform(data, cfg)
    weighted_sum = np.zeros(len(feature_names), float)
    scored_count = np.zeros(len(feature_names), int)

    fig_lat, axes_lat = plt.subplots(3, len(region_ids), figsize=(2.7 * len(region_ids), 7.5), squeeze=False)
    fig_feat, axes_feat = plt.subplots(len(feature_names), len(region_ids),
                                      figsize=(2.7 * len(region_ids), 2.0 * len(feature_names)), squeeze=False)
    regions_summary = {}

    for col, rid in enumerate(region_ids):
        ids = test[data.cosmos_ids[test] == rid]
        if len(ids) > cfg.diagnostic_samples_per_region:
            ids = rng.choice(ids, cfg.diagnostic_samples_per_region, replace=False)
        n_each = max(1, int(cfg.feature_nll_samples_per_test_unit))
        sampled = _sample_method(
            method_kind, ids, n_each, rng, gmm=gmm, conditional_model=conditional_model,
            baseline=baseline, data=data,
        )
        pred_z = np.concatenate(sampled, axis=0)
        obs_z = z_scaled[ids]

        for row in range(3):
            vals = np.concatenate([obs_z[:, row], pred_z[:, row]])
            lo, hi = np.quantile(vals[np.isfinite(vals)], [0.005, 0.995])
            hi = max(hi, lo + 1e-6)
            bins = np.linspace(lo, hi, cfg.diagnostic_hist_bins + 1)
            axes_lat[row, col].hist(obs_z[:, row], bins=bins, density=True, histtype="step", label="observed")
            axes_lat[row, col].hist(pred_z[:, row], bins=bins, density=True, histtype="step", ls="--", label="predicted")
            if row == 0:
                axes_lat[row, col].set_title(names[int(rid)], fontsize=8)
            if col == 0:
                axes_lat[row, col].set_ylabel(f"z{row}")

        if method_kind == "experimental_knn":
            if empirical_decoder is None:
                raise RuntimeError("experimental_knn requires an empirical_decoder")
            pred_feat = empirical_decoder.sample_features(pred_z, rng)
            pred_report = {
                "method": "distance-weighted empirical kNN retrieval",
                "k": int(empirical_decoder.k),
                "note": "features come from real TRAIN exemplars; no neural waveform decoder",
            }
        else:
            pred_wave = _decode_waveform_scaled_latents(ae, scaler, pred_z, cfg)
            pred_feat, _, pred_report = extract_generated_waveform_features(
                pred_wave, sampling_rate_hz=cfg.waveform_sampling_rate_hz, return_report=True
            )
        obs_feat = observed_features[[lookup[int(i)] for i in ids]]
        regions_summary[names[int(rid)]] = {
            "n_test_units": int(len(ids)),
            "generated_feature_report": pred_report,
        }

        pred_cont = (pred_feat[:, :-1] - feature_mu[None, :]) / feature_sd[None, :]
        obs_cont = (obs_feat[:, :-1] - feature_mu[None, :]) / feature_sd[None, :]
        region_per_feature_nll = np.full(len(feature_names), np.nan, dtype=float)
        for fidx in range(len(feature_names) - 1):
            value = _feature_kde_nll(pred_cont[:, fidx], obs_cont[:, fidx], cfg.feature_nll_min_bandwidth_fraction)
            region_per_feature_nll[fidx] = value
            if np.isfinite(value):
                weighted_sum[fidx] += value * len(obs_feat)
                scored_count[fidx] += len(obs_feat)
        pol_idx = len(feature_names) - 1
        value = _categorical_nll(pred_feat[:, pol_idx], obs_feat[:, pol_idx], polarity_categories, cfg.feature_categorical_alpha)
        region_per_feature_nll[pol_idx] = value
        weighted_sum[pol_idx] += value * len(obs_feat)
        scored_count[pol_idx] += len(obs_feat)
        regions_summary[names[int(rid)]]["per_feature_nll"] = region_per_feature_nll.tolist()
        regions_summary[names[int(rid)]]["mean_feature_nll"] = float(np.nanmean(region_per_feature_nll))

        for row, fname in enumerate(feature_names):
            if row == len(feature_names) - 1:
                cats = polarity_categories
                width = 0.35
                obs_counts = [np.mean(np.isclose(obs_feat[:, row], c)) for c in cats]
                pred_counts = [np.mean(np.isclose(pred_feat[:, row], c)) for c in cats]
                x = np.arange(len(cats))
                axes_feat[row, col].bar(x - width / 2, obs_counts, width=width, fill=False, label="observed")
                axes_feat[row, col].bar(x + width / 2, pred_counts, width=width, fill=False, ls="--", label="predicted")
                axes_feat[row, col].set_xticks(x, [str(float(v)) for v in cats], fontsize=6)
            else:
                vals = np.concatenate([obs_feat[:, row], pred_feat[:, row]])
                finite = vals[np.isfinite(vals)]
                lo, hi = np.quantile(finite, [0.005, 0.995])
                hi = max(hi, lo + 1e-9)
                bins = np.linspace(lo, hi, cfg.diagnostic_hist_bins + 1)
                axes_feat[row, col].hist(obs_feat[:, row], bins=bins, density=True, histtype="step", label="observed")
                axes_feat[row, col].hist(pred_feat[:, row], bins=bins, density=True, histtype="step", ls="--", label="predicted")
            if row == 0:
                axes_feat[row, col].set_title(names[int(rid)], fontsize=8)
            if col == 0:
                axes_feat[row, col].set_ylabel(fname, fontsize=7)

    if len(region_ids):
        h, l = axes_lat[0, 0].get_legend_handles_labels()
        fig_lat.legend(h, l, loc="upper center", ncol=2, frameon=False)
        h, l = axes_feat[0, 0].get_legend_handles_labels()
        fig_feat.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig_lat.suptitle(f"Observed vs predicted latent distributions: {method_name}")
    fig_lat.tight_layout(rect=(0, 0, 1, 0.97))
    fig_lat.savefig(out_dir / "region_latent_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig_lat)
    fig_feat.suptitle(f"Observed vs decoded waveform-feature distributions: {method_name}")
    fig_feat.tight_layout(rect=(0, 0, 1, 0.985))
    fig_feat.savefig(out_dir / "region_waveform_feature_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig_feat)

    per_feature = np.divide(weighted_sum, scored_count, out=np.full_like(weighted_sum, np.nan), where=scored_count > 0)
    feature_nll = {
        "feature_names": list(feature_names),
        "per_feature_nll": per_feature.tolist(),
        "mean_feature_nll": float(np.nanmean(per_feature)),
        "n_scored_observations_per_feature": scored_count.tolist(),
        "continuous_feature_preprocessing": "TRAIN-only z-score before KDE likelihood",
        "polarity_preprocessing": "categorical NLL with Laplace smoothing; no KDE",
        "polarity_categories_train": polarity_categories.tolist(),
    }
    summary = {
        "observed_feature_extraction": report,
        "regions": regions_summary,
        "feature_nll": feature_nll,
    }
    (out_dir / "distribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _axis_grid(limits_m, step_um):
    lo, hi = sorted(np.asarray(limits_m, dtype=float))
    step_m = float(step_um) * 1e-6
    start = np.ceil(lo / step_m) * step_m
    stop = np.floor(hi / step_m) * step_m
    return np.arange(start, stop + 0.5 * step_m, step_m, dtype=np.float32)


def _atlas_slice_voxels(brain_atlas, view, coord_um, voxel_size_um):
    coord_m = float(coord_um) * 1e-6
    if view == "coronal":
        xs = _axis_grid(brain_atlas.bc.xlim, voxel_size_um)
        zs = _axis_grid(brain_atlas.bc.zlim, voxel_size_um)
        xx, zz = np.meshgrid(xs, zs, indexing="xy")
        xyz = np.column_stack([xx.ravel(), np.full(xx.size, coord_m, np.float32), zz.ravel()])
    elif view == "sagittal":
        ys = _axis_grid(brain_atlas.bc.ylim, voxel_size_um)
        zs = _axis_grid(brain_atlas.bc.zlim, voxel_size_um)
        yy, zz = np.meshgrid(ys, zs, indexing="xy")
        xyz = np.column_stack([np.full(yy.size, coord_m, np.float32), yy.ravel(), zz.ravel()])
    else:
        raise ValueError(view)
    cosmos = np.asarray(brain_atlas.get_labels(xyz, mapping="Cosmos"), np.int64)
    keep = cosmos != 0
    xyz = xyz[keep].astype(np.float32)
    cosmos = cosmos[keep]
    beryl = np.asarray(brain_atlas.get_labels(xyz, mapping="Beryl"), np.int64)
    return xyz, cosmos, beryl


def _raw_context_for_atlas_voxels(xyz_m, data, cfg):
    from ephysatlas.spatial_encoder.utils import AtlasPCAConfig, ContextAtlasManager
    manager = ContextAtlasManager(
        AtlasPCAConfig(n_cell_pcs=int(cfg.n_cell_pcs), n_gene_pcs=int(cfg.n_gene_pcs)),
        regenerate_context=False,
        output_dir=Path(cfg.prepared_data_dir) / str(cfg.context_atlas_subdir),
    )
    pack = manager.sample_context_numpy_m(np.asarray(xyz_m, np.float32), mode="clip")
    cell = np.asarray(pack["cell_pc"], np.float32)[:, :cfg.n_cell_pcs]
    gene = np.asarray(pack["gene_pc"], np.float32)[:, :cfg.n_gene_pcs]
    context = np.concatenate([cell, gene], axis=1).astype(np.float32)
    if context.shape[1] != data.context.shape[1]:
        raise RuntimeError(f"Atlas context dim {context.shape[1]} != unit context dim {data.context.shape[1]}")
    return context


def _plot_empty_atlas_background(ax, brain_atlas, view, coord_um):
    plot_points_on_slice(
        np.empty((0, 3)), values=None, coord=float(coord_um), slice=view,
        mapping="Cosmos", background="boundary", show_cbar=False, aggr="mean",
        fwhm=0, brain_atlas=brain_atlas, ax=ax,
    )


def choose_feature_slice_indices(data, cfg):
    """Choose five reproducibly random continuous hand-picked features once per run."""
    n_cont = len(data.waveform_feature_names) - 1
    n = min(int(cfg.feature_slice_count), n_cont)
    rng = np.random.default_rng(int(cfg.feature_slice_seed))
    idx = np.sort(rng.choice(np.arange(n_cont), size=n, replace=False))
    return idx.astype(int)


def _canonical_model_xyz(xyz_m, data, cfg):
    """Return coordinates used by every spatial predictor.

    The plotted coordinates remain bilateral, but the model is queried only on
    the canonical hemisphere. Therefore predictions on the opposite hemisphere
    are exact mirrored copies rather than independent extrapolations.
    """
    xyz = np.asarray(xyz_m, np.float32)
    if not bool(getattr(cfg, "mirror_x_to_single_hemisphere", False)):
        return xyz
    sign = float(getattr(cfg, "mirror_x_sign", infer_training_hemisphere_sign(data.xyz_m, data.split)))
    return mirror_xyz_to_hemisphere(xyz, sign)


def _labels_for_model_xyz(brain_atlas, model_xyz):
    cosmos = np.asarray(brain_atlas.get_labels(model_xyz, mapping="Cosmos"), np.int64)
    beryl = np.asarray(brain_atlas.get_labels(model_xyz, mapping="Beryl"), np.int64)
    return cosmos, beryl


def _scaled_latent_mean_at_voxels(method_kind, xyz_m, cosmos, beryl, data, cfg, *,
                                  gmm=None, conditional_model=None, baseline=None):
    model_xyz = _canonical_model_xyz(xyz_m, data, cfg)
    if method_kind in ("experimental", "experimental_knn"):
        context = _raw_context_for_atlas_voxels(model_xyz, data, cfg)
        return posterior_mean_for_context(gmm, conditional_model, context)
    if method_kind == "cosmos_gaussian":
        return baseline.mean_for_regions(cosmos)
    if method_kind == "beryl_gaussian":
        return baseline.mean_for_regions(beryl)
    if method_kind == "kde":
        return baseline.mean_for_xyz(model_xyz)
    raise ValueError(method_kind)



def _decode_features_for_scaled_latents(ae, scaler, z_scaled, cfg):
    """Decode standardized latent vectors and extract waveform features."""
    wave = _decode_waveform_scaled_latents(ae, scaler, z_scaled, cfg)
    feat, _ = extract_generated_waveform_features(
        wave,
        sampling_rate_hz=cfg.waveform_sampling_rate_hz,
    )
    return feat.astype(np.float32)


def _gmm_component_feature_expectations(ae, scaler, gmm, cfg, seed):
    """Deterministic E[feature | component] approximation shared by all voxels.

    The previous implementation independently sampled only eight latent vectors
    at every voxel. For a mixture model, that introduces large categorical Monte
    Carlo noise: adjacent voxels with almost identical mixture weights can draw
    different components and therefore get visibly different colors. Here each
    component is sampled once with a large, fixed Monte Carlo bank, and spatial
    predictions only combine those stable component expectations with the local
    mixture weights.
    """
    n = max(1, int(getattr(cfg, "feature_slice_component_mc_samples", 128)))
    rng = np.random.default_rng(int(seed))
    draws = []
    for k in range(gmm.n_components):
        if gmm.covariance_type == "full":
            z = rng.multivariate_normal(gmm.means_[k], gmm.covariances_[k], size=n)
        elif gmm.covariance_type == "diag":
            eps = rng.normal(size=(n, gmm.means_.shape[1]))
            z = gmm.means_[k][None, :] + eps * np.sqrt(gmm.covariances_[k])[None, :]
        else:
            raise ValueError(gmm.covariance_type)
        draws.append(np.asarray(z, np.float32))
    z = np.concatenate(draws, axis=0)
    feat = _decode_features_for_scaled_latents(ae, scaler, z, cfg)
    return feat.reshape(gmm.n_components, n, -1).mean(axis=1).astype(np.float32)


def _gmm_component_knn_feature_expectations(gmm, empirical_decoder, cfg, seed):
    """Stable E[real TRAIN waveform feature | GMM component] for kNN decoding."""
    n = max(1, int(getattr(cfg, "feature_slice_component_mc_samples", 128)))
    rng = np.random.default_rng(int(seed))
    draws = []
    for k in range(gmm.n_components):
        if gmm.covariance_type == "full":
            z = rng.multivariate_normal(gmm.means_[k], gmm.covariances_[k], size=n)
        elif gmm.covariance_type == "diag":
            eps = rng.normal(size=(n, gmm.means_.shape[1]))
            z = gmm.means_[k][None, :] + eps * np.sqrt(gmm.covariances_[k])[None, :]
        else:
            raise ValueError(gmm.covariance_type)
        draws.append(np.asarray(z, np.float32))
    z = np.concatenate(draws, axis=0)
    feat = empirical_decoder.expected_features(z)
    return feat.reshape(gmm.n_components, n, -1).mean(axis=1).astype(np.float32)


def _regional_feature_expectations(ae, scaler, baseline, region_ids, cfg, seed):
    """Stable feature expectation for each requested regional Gaussian."""
    n = max(1, int(getattr(cfg, "feature_slice_regional_mc_samples", 128)))
    rng = np.random.default_rng(int(seed))
    unique = np.unique(np.asarray(region_ids, int))
    lookup = {}
    for rid in unique:
        mu, var = baseline._params(int(rid))
        z = mu[None, :] + rng.normal(size=(n, len(mu))) * np.sqrt(var)[None, :]
        lookup[int(rid)] = _decode_features_for_scaled_latents(
            ae, scaler, np.asarray(z, np.float32), cfg
        ).mean(axis=0)
    return lookup


def _kde_train_center_features(ae, scaler, baseline, cfg):
    """Features of KDE latent centers, computed once and cached.

    For the spatial KDE, the latent conditional mean is the spatially weighted
    mean of its TRAIN centers. A per-voxel random draw is unnecessary for a mean
    atlas and was a major source of grain. We therefore decode every stored TRAIN
    center once and spatially average its waveform features with the exact KDE
    neighbor weights. This is deterministic and retains the local training-unit
    structure that makes KDE useful.
    """
    path = Path(cfg.prepared_data_dir) / "kde_train_center_waveform_features.npy"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.shape[0] == len(baseline.z_train) and cached.shape[1] == 11:
            return cached.astype(np.float32, copy=False)
    print("[feature slices] decoding KDE TRAIN centers once ...")
    feat = _decode_features_for_scaled_latents(
        ae, scaler, np.asarray(baseline.z_train, np.float32), cfg
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feat.astype(np.float32), allow_pickle=False)
    return feat.astype(np.float32, copy=False)


def _deterministic_mean_waveform_features_at_voxels(
    ae, scaler, xyz_m, cosmos, beryl, data, cfg, *,
    method_kind, gmm=None, conditional_model=None, baseline=None,
    gmm_component_features=None, kde_center_features=None, empirical_decoder=None,
):
    """Return a smooth deterministic feature atlas at arbitrary voxels.

    Weight-only GMMs use local mixture weights times stable per-component feature
    expectations. KDE uses local spatial weights times decoded TRAIN-center
    features. Regional Gaussians use stable per-region feature expectations.
    The delta-mu model uses the feature of each context-shifted component centroid
    weighted by local mixture probabilities; this is deterministic and avoids
    independent Monte Carlo noise while preserving its context-dependent means.
    """
    model_xyz = _canonical_model_xyz(xyz_m, data, cfg)

    if method_kind in ("experimental", "experimental_knn"):
        # A truly unconditional GMM must be exactly constant over space.  Do not
        # even query context here: repeating identical weights and multiplying them
        # in different BLAS batch sizes can leave tiny (~1e-7 relative) floating
        # differences. Adaptive color scaling then magnifies those meaningless
        # round-off errors into visible stripes.
        if hasattr(conditional_model, "global_weights"):
            if gmm_component_features is None:
                raise RuntimeError("Missing precomputed GMM component feature expectations")
            mean_feature = (
                np.asarray(conditional_model.global_weights, np.float64)
                @ np.asarray(gmm_component_features, np.float64)
            ).astype(np.float32)
            return np.repeat(mean_feature[None, :], len(model_xyz), axis=0)

        context = _raw_context_for_atlas_voxels(model_xyz, data, cfg)
        w = conditional_model.weights_for_context(context)
        if gmm_component_features is None:
            raise RuntimeError("Missing precomputed GMM component feature expectations")
        return (w @ gmm_component_features).astype(np.float32)

    if method_kind in ("cosmos_gaussian", "beryl_gaussian"):
        labels = cosmos if method_kind == "cosmos_gaussian" else beryl
        lookup = _regional_feature_expectations(
            ae, scaler, baseline, labels, cfg,
            seed=int(cfg.feature_slice_seed) + (1701 if method_kind == "cosmos_gaussian" else 1702),
        )
        return np.stack([lookup[int(r)] for r in labels]).astype(np.float32)

    if method_kind == "kde":
        if kde_center_features is None:
            raise RuntimeError("Missing decoded KDE center features")
        ind, spatial_w = baseline._neighbors(model_xyz)
        return np.sum(
            kde_center_features[ind] * spatial_w[:, :, None], axis=1
        ).astype(np.float32)

    raise ValueError(method_kind)


def plot_feature_brain_slices(
    ae, data, scaler, cfg, out_dir: Path, *, method_name: str, method_kind: str,
    feature_indices, gmm=None, conditional_model=None, baseline=None, empirical_decoder=None,
):
    """5x3 coronal/sagittal maps of stable predicted mean waveform features.

    The previous version used eight independent Monte-Carlo samples per voxel.
    That estimates the mean correctly in expectation but produces a visibly
    grainy map because neighboring voxels randomly select different GMM
    components / KDE centers. The atlas is now deterministic: reusable
    component/region expectations are computed once and combined with smoothly
    varying local weights.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_atlas = AllenAtlas()
    test_xyz_um = np.asarray(data.xyz_m[data.split == 2], np.float32) * 1e6
    q = np.asarray(cfg.diagnostic_slice_quantiles, float)
    step = float(cfg.diagnostic_voxel_size_um)
    coronal = np.round(np.quantile(test_xyz_um[:, 1], q) / step) * step
    sagittal = np.round(np.quantile(test_xyz_um[:, 0], q) / step) * step

    feature_indices = np.asarray(feature_indices, int)
    feature_names = [data.waveform_feature_names[i] for i in feature_indices]

    gmm_component_features = None
    kde_center_features = None
    if method_kind == "experimental_knn":
        if empirical_decoder is None:
            raise RuntimeError("experimental_knn requires an empirical_decoder")
        gmm_component_features = _gmm_component_knn_feature_expectations(
            gmm, empirical_decoder, cfg,
            seed=int(cfg.feature_slice_seed) + 3200 + int(gmm.n_components),
        )
    elif method_kind == "experimental":
        gmm_component_features = _gmm_component_feature_expectations(
            ae, scaler, gmm, cfg,
            seed=int(cfg.feature_slice_seed) + 2200 + int(gmm.n_components),
        )
    elif method_kind == "kde":
        kde_center_features = _kde_train_center_features(ae, scaler, baseline, cfg)

    cache = {}

    def predict(view, coord):
        key = (view, float(coord))
        if key in cache:
            return cache[key]
        xyz, _, _ = _atlas_slice_voxels(brain_atlas, view, coord, step)
        model_xyz = _canonical_model_xyz(xyz, data, cfg)
        cosmos, beryl = _labels_for_model_xyz(brain_atlas, model_xyz)
        feat = _deterministic_mean_waveform_features_at_voxels(
            ae, scaler, xyz, cosmos, beryl, data, cfg,
            method_kind=method_kind, gmm=gmm,
            conditional_model=conditional_model, baseline=baseline,
            gmm_component_features=gmm_component_features,
            kde_center_features=kde_center_features,
            empirical_decoder=empirical_decoder,
        )
        payload = {"xyz_m": xyz, "features": feat}
        cache[key] = payload
        return payload

    for coord in coronal:
        predict("coronal", coord)
    for coord in sagittal:
        predict("sagittal", coord)

    qlo, qhi = tuple(getattr(cfg, "feature_slice_display_quantiles", (0.02, 0.98)))
    padding = float(getattr(cfg, "feature_slice_display_padding_fraction", 0.03))
    limits = {}
    centers = {}
    for findex in feature_indices:
        vals = np.concatenate([payload["features"][:, findex] for payload in cache.values()])
        vals = vals[np.isfinite(vals)]
        lo, hi = np.quantile(vals, [qlo, qhi])
        if hi <= lo:
            eps = max(abs(float(lo)) * 1e-3, 1e-9)
            lo, hi = float(lo) - eps, float(hi) + eps
        span = float(hi - lo)
        lo = float(lo - padding * span)
        hi = float(hi + padding * span)
        center_mode = str(getattr(cfg, "feature_slice_center", "median"))
        center = float(np.median(vals)) if center_mode == "median" else 0.5 * (lo + hi)
        if not (lo < center < hi):
            center = 0.5 * (lo + hi)
        limits[int(findex)] = (lo, hi)
        centers[int(findex)] = center

    def make(view, coords):
        fig, axes = plt.subplots(
            len(feature_indices), len(coords),
            figsize=(4.2 * len(coords), 3.2 * len(feature_indices)),
            squeeze=False,
        )
        counts = []
        for col, coord in enumerate(coords):
            payload = predict(view, coord)
            xyz_um = payload["xyz_m"] * 1e6
            feat = payload["features"]
            counts.append(int(len(xyz_um)))
            for row, findex in enumerate(feature_indices):
                ax = axes[row, col]
                _plot_empty_atlas_background(ax, brain_atlas, view, coord)
                if view == "coronal":
                    xx, yy = xyz_um[:, 0], xyz_um[:, 2]
                else:
                    xx, yy = xyz_um[:, 1], xyz_um[:, 2]
                lo, hi = limits[int(findex)]
                center = centers[int(findex)]
                norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
                sc = ax.scatter(
                    xx, yy, c=feat[:, findex],
                    cmap=cfg.diagnostic_feature_slice_cmap, norm=norm,
                    marker="s", s=9, linewidths=0, rasterized=True, zorder=3,
                )
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{view} {coord:.0f} µm")
                if col == 0:
                    ax.set_ylabel(feature_names[row], fontsize=8)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
        fig.suptitle(
            f"{method_name}: predicted mean hand-picked waveform features\n"
            f"deterministic distribution expectation; mirrored across ML axis; rows=5, columns=3"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(
            out_dir / f"handpicked_feature_slices_{view}.png",
            dpi=240, bbox_inches="tight",
        )
        plt.close(fig)
        return counts

    cor_counts = make("coronal", coronal)
    sag_counts = make("sagittal", sagittal)
    if method_kind == "experimental_knn":
        expectation_definition = (
            f"mixture weights times fixed per-component E[real TRAIN waveform feature | component] "
            f"from distance-weighted kNN decoder (k={int(empirical_decoder.k)})"
        )
    elif method_kind == "experimental":
        expectation_definition = (
            "mixture weights times fixed per-component Monte-Carlo E[decoded waveform feature|component]"
        )
    elif method_kind == "kde":
        expectation_definition = (
            "spatial KDE neighbor weights times waveform features decoded from TRAIN latent centers"
        )
    else:
        expectation_definition = (
            "fixed Monte-Carlo E[waveform feature|regional Gaussian] per anatomical region"
        )

    summary = {
        "feature_indices": feature_indices.tolist(),
        "feature_names": feature_names,
        "definition": expectation_definition,
        "grain_fix": (
            "Removed independent per-voxel Monte Carlo draws. Random component/neighbor selection "
            "was injecting sampling noise into adjacent voxels. Expectations are now reusable and deterministic."
        ),
        "mirrored_x": bool(getattr(cfg, "mirror_x_to_single_hemisphere", False)),
        "mirror_policy": (
            "query each bilateral atlas voxel after folding x onto the canonical "
            "training hemisphere; render the result at the original voxel coordinate"
        ),
        "coronal_slice_coordinates_um": coronal.tolist(),
        "sagittal_slice_coordinates_um": sagittal.tolist(),
        "coronal_voxel_counts": cor_counts,
        "sagittal_voxel_counts": sag_counts,
        "cmap": str(cfg.diagnostic_feature_slice_cmap),
        "display_limit_source": "robust quantiles of this method's six deterministic voxel-level prediction maps",
        "display_quantiles": [float(qlo), float(qhi)],
        "display_padding_fraction": float(padding),
        "display_center": str(getattr(cfg, "feature_slice_center", "median")),
        "display_limits": {
            data.waveform_feature_names[i]: list(limits[int(i)]) for i in feature_indices
        },
        "display_centers": {
            data.waveform_feature_names[i]: float(centers[int(i)]) for i in feature_indices
        },
    }
    (out_dir / "feature_slice_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary



def run_experimental_diagnostics(
    ae, data, z_scaled, gmm, scaler, conditional_model, cfg, out_dir, feature_indices,
    *, empirical_decoder=None,
):
    """Atlas diagnostics for a GMM density with either neural or empirical decoding."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_kind = "experimental_knn" if empirical_decoder is not None else "experimental"
    summary = {
        "distributions": plot_region_distributions_single_method(
            ae, data, z_scaled, scaler, cfg, out_dir,
            method_name=out_dir.name, method_kind=method_kind, gmm=gmm,
            conditional_model=conditional_model, empirical_decoder=empirical_decoder,
        ),
        "feature_slices": plot_feature_brain_slices(
            ae, data, scaler, cfg, out_dir, method_name=out_dir.name,
            method_kind=method_kind, feature_indices=feature_indices, gmm=gmm,
            conditional_model=conditional_model, empirical_decoder=empirical_decoder,
        ),
    }
    if empirical_decoder is not None:
        summary["decoder"] = {
            "kind": "distance_weighted_empirical_knn",
            "k": int(empirical_decoder.k),
            "space": "shared standardized 60-D joint latent",
            "training_exemplars_only": True,
        }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def run_baseline_diagnostics(ae, data, z_scaled, scaler, baseline, baseline_name, cfg, out_dir, feature_indices):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "distributions": plot_region_distributions_single_method(
            ae, data, z_scaled, scaler, cfg, out_dir,
            method_name=baseline_name, method_kind=baseline_name, baseline=baseline,
        ),
        "feature_slices": plot_feature_brain_slices(
            ae, data, scaler, cfg, out_dir, method_name=baseline_name, method_kind=baseline_name,
            feature_indices=feature_indices, baseline=baseline,
        ),
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

# Public alias used by the runner when building the empirical kNN decoder.
get_model_space_waveform_features = _get_model_space_waveform_features


def reconstruction_examples_all_modalities(ae, data, cfg, *, n_examples=2, seed=0):
    """Return held-out observation/reconstruction pairs for waveform, ACG and stPC."""
    test = np.flatnonzero(data.split == 2)
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(test, size=min(int(n_examples), len(test)), replace=False)
    with torch.no_grad():
        w = torch.from_numpy(data.waveforms[ids]).to(cfg.device)
        a = torch.from_numpy(data.acgs[ids]).to(cfg.device)
        s = torch.from_numpy(data.stpc[ids]).to(cfg.device)
        rec = ae.decode(ae.encode(w, a, s))
    return {
        "indices": ids,
        "waveform": data.waveforms[ids],
        "waveform_reconstruction": rec["waveform"].cpu().numpy(),
        "acg": data.acgs[ids],
        "acg_reconstruction": rec["acg"].cpu().numpy(),
        "stpc": data.stpc[ids],
        "stpc_reconstruction": rec["stpc"].cpu().numpy(),
    }


def publication_feature_slice_data(
    ae,
    data,
    scaler,
    cfg,
    *,
    feature_indices,
    method_kind,
    sagittal_coord_um,
    gmm=None,
    conditional_model=None,
    baseline=None,
    empirical_decoder=None,
):
    """Return deterministic mean waveform-feature predictions for one sagittal slice."""
    brain_atlas = AllenAtlas()
    step = float(cfg.diagnostic_voxel_size_um)
    xyz, _, _ = _atlas_slice_voxels(brain_atlas, "sagittal", sagittal_coord_um, step)
    model_xyz = _canonical_model_xyz(xyz, data, cfg)
    cosmos, beryl = _labels_for_model_xyz(brain_atlas, model_xyz)

    component_features = None
    kde_center_features = None
    if method_kind == "experimental_knn":
        component_features = _gmm_component_knn_feature_expectations(
            gmm,
            empirical_decoder,
            cfg,
            seed=int(cfg.feature_slice_seed) + 3200,
        )
    elif method_kind == "experimental":
        component_features = _gmm_component_feature_expectations(
            ae,
            scaler,
            gmm,
            cfg,
            seed=int(cfg.feature_slice_seed) + 2200,
        )
    elif method_kind == "kde":
        kde_center_features = _kde_train_center_features(ae, scaler, baseline, cfg)

    features = _deterministic_mean_waveform_features_at_voxels(
        ae,
        scaler,
        xyz,
        cosmos,
        beryl,
        data,
        cfg,
        method_kind=method_kind,
        gmm=gmm,
        conditional_model=conditional_model,
        baseline=baseline,
        gmm_component_features=component_features,
        kde_center_features=kde_center_features,
        empirical_decoder=empirical_decoder,
    )
    return {
        "xyz_m": xyz,
        "feature_indices": np.asarray(feature_indices, int),
        "features": features[:, np.asarray(feature_indices, int)],
    }


def feature_nll_comparison(
    ae,
    data,
    z_scaled,
    scaler,
    cfg,
    *,
    feature_indices,
    methods,
    samples_per_test_unit=None,
):
    """Pooled held-out feature NLL for publication-method comparison.

    Parameters
    ----------
    methods : dict
        Mapping display name -> dict with ``method_kind`` and optional ``gmm``,
        ``conditional_model``, ``baseline`` and ``empirical_decoder``.
    """
    rng = np.random.default_rng(int(cfg.seed) + 8871)
    test = np.flatnonzero(data.split == 2)
    n_each = int(samples_per_test_unit or cfg.feature_nll_samples_per_test_unit)
    all_observed = _get_model_space_waveform_features(data, cfg)
    observed = all_observed[test]
    feature_indices = np.asarray(feature_indices, int)
    mu, sd, polarity_categories = _feature_eval_transform(data, cfg)

    out = {}
    for name, spec in methods.items():
        kind = spec["method_kind"]
        sampled = _sample_method(
            kind,
            test,
            n_each,
            rng,
            gmm=spec.get("gmm"),
            conditional_model=spec.get("conditional_model"),
            baseline=spec.get("baseline"),
            data=data,
        )
        pred_z = np.concatenate(sampled, axis=0)
        if kind == "experimental_knn":
            pred_feat = spec["empirical_decoder"].sample_features(pred_z, rng)
        else:
            pred_wave = _decode_waveform_scaled_latents(ae, scaler, pred_z, cfg)
            pred_feat, _ = extract_generated_waveform_features(
                pred_wave,
                sampling_rate_hz=cfg.waveform_sampling_rate_hz,
            )

        per_feature = []
        for idx in feature_indices:
            if idx == len(data.waveform_feature_names) - 1:
                value = _categorical_nll(
                    pred_feat[:, idx],
                    observed[:, idx],
                    polarity_categories,
                    cfg.feature_categorical_alpha,
                )
            else:
                pred_std = (pred_feat[:, idx] - mu[idx]) / sd[idx]
                obs_std = (observed[:, idx] - mu[idx]) / sd[idx]
                value = _feature_kde_nll(
                    pred_std,
                    obs_std,
                    cfg.feature_nll_min_bandwidth_fraction,
                )
            per_feature.append(float(value))
        out[name] = np.asarray(per_feature, float)
    return out
