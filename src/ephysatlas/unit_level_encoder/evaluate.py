from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import pairwise_distances

from .baselines import RegionalGaussianBaseline, SpatialKDEBaseline
from .gmm_models import conditional_log_prob


def nll_from_log_prob(lp):
    lp = np.asarray(lp, float)
    return float(-np.mean(lp))


def fit_or_load_baselines(data, z_scaled, cfg, model_path: Path, *, fit: bool):
    """Fit the three requested baselines exactly once in the shared latent space."""
    model_path = Path(model_path)
    if fit:
        train = data.split == 0
        models = {
            "cosmos_gaussian": RegionalGaussianBaseline(
                z_scaled, data.cosmos_ids, train, cfg.region_gaussian_variance_floor
            ),
            "beryl_gaussian": RegionalGaussianBaseline(
                z_scaled, data.beryl_ids, train, cfg.region_gaussian_variance_floor
            ),
            "kde": SpatialKDEBaseline(z_scaled, data.xyz_m, train, cfg),
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, model_path)
        return models
    if not model_path.exists():
        raise FileNotFoundError(f"Missing shared baseline checkpoint: {model_path}")
    return joblib.load(model_path)


def evaluate_baselines(data, z_scaled, baselines, out_path: Path):
    test = np.flatnonzero(data.split == 2)
    metrics = {
        "kde": {"test_nll": nll_from_log_prob(baselines["kde"].log_prob(z_scaled[test], data.xyz_m[test]))},
        "cosmos_gaussian": {"test_nll": nll_from_log_prob(baselines["cosmos_gaussian"].log_prob(test))},
        "beryl_gaussian": {"test_nll": nll_from_log_prob(baselines["beryl_gaussian"].log_prob(test))},
    }
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_experiment(data, z_scaled, gmm, conditional_model, out_path: Path):
    test = np.flatnonzero(data.split == 2)
    nll = nll_from_log_prob(conditional_log_prob(z_scaled, test, gmm, conditional_model))
    dist = pairwise_distances(gmm.means_)
    np.fill_diagonal(dist, np.nan)
    metrics = {
        "test_nll": nll,
        "gmm_geometry": {
            "mean_pairwise_component_distance": float(np.nanmean(dist)),
            "min_pairwise_component_distance": float(np.nanmin(dist)),
            "max_pairwise_component_distance": float(np.nanmax(dist)),
            "component_weights_global": gmm.weights_.tolist(),
        },
    }
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_latent_fidelity(data, z_scaled, gmm, conditional_model, cfg, out_path: Path):
    """Compare generated latent distribution with held-out encoded TEST latents.

    This deliberately stays in the frozen 60-D standardized latent space, so it
    does not depend on the waveform decoder. Metrics diagnose density fidelity:
    held-out NLL, first/second moments, marginal and sliced Wasserstein distances,
    and empirical support distance to held-out TEST latents.
    """
    from scipy.stats import wasserstein_distance
    from sklearn.neighbors import NearestNeighbors
    from .gmm_models import sample_conditional

    rng = np.random.default_rng(int(cfg.seed) + 41021)
    test = np.flatnonzero(data.split == 2)
    n = min(len(test), int(cfg.latent_fidelity_test_units))
    ids = np.sort(rng.choice(test, size=n, replace=False)) if len(test) > n else test
    observed = np.asarray(z_scaled[ids], np.float64)
    generated = np.concatenate(
        sample_conditional(ids, 1, gmm, conditional_model, rng), axis=0
    ).astype(np.float64)

    lp = conditional_log_prob(z_scaled, ids, gmm, conditional_model)
    obs_mean = observed.mean(axis=0)
    gen_mean = generated.mean(axis=0)
    obs_cov = np.cov(observed, rowvar=False)
    gen_cov = np.cov(generated, rowvar=False)
    obs_var = np.maximum(np.diag(obs_cov), 1e-12)
    gen_var = np.maximum(np.diag(gen_cov), 1e-12)

    marginal_w = np.asarray([
        wasserstein_distance(observed[:, j], generated[:, j])
        for j in range(observed.shape[1])
    ], dtype=float)

    n_proj = int(cfg.latent_fidelity_sliced_wasserstein_projections)
    directions = rng.normal(size=(n_proj, observed.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    sliced = np.asarray([
        wasserstein_distance(observed @ v, generated @ v)
        for v in directions
    ], dtype=float)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(observed.astype(np.float32))
    support_dist, _ = nn.kneighbors(generated.astype(np.float32), return_distance=True)
    support_dist = support_dist[:, 0]

    metrics = {
        "n_test_units": int(len(ids)),
        "heldout_latent_nll": float(-np.mean(lp)),
        "mean_vector_rmse": float(np.sqrt(np.mean((gen_mean - obs_mean) ** 2))),
        "relative_covariance_frobenius_error": float(
            np.linalg.norm(gen_cov - obs_cov, ord="fro")
            / max(np.linalg.norm(obs_cov, ord="fro"), 1e-12)
        ),
        "variance_ratio_generated_over_test_median": float(np.median(gen_var / obs_var)),
        "variance_ratio_generated_over_test_p10": float(np.quantile(gen_var / obs_var, 0.10)),
        "variance_ratio_generated_over_test_p90": float(np.quantile(gen_var / obs_var, 0.90)),
        "mean_marginal_wasserstein": float(np.mean(marginal_w)),
        "median_marginal_wasserstein": float(np.median(marginal_w)),
        "mean_sliced_wasserstein": float(np.mean(sliced)),
        "median_sliced_wasserstein": float(np.median(sliced)),
        "generated_to_test_nn_distance_mean": float(np.mean(support_dist)),
        "generated_to_test_nn_distance_median": float(np.median(support_dist)),
        "interpretation": {
            "heldout_latent_nll": "lower is better",
            "mean_vector_rmse": "lower is better",
            "relative_covariance_frobenius_error": "lower is better",
            "variance_ratio": "closer to 1 is better",
            "wasserstein": "lower is better",
            "generated_to_test_nn_distance": "lower indicates generated samples lie nearer the empirical held-out latent cloud",
        },
    }
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
