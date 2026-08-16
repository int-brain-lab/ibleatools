from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance

from ephysatlas.unit_level_encoder.gmm_models import diag_log_prob, move


def _clone_batch(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _batch_observed_mean(batch):
    target, mask = batch["target_z"], batch["target_mask"]
    return torch.stack([target[i][mask[i]].mean(0) for i in range(len(target))])


def _prediction_and_nll(model, batch, logits):
    gamma = F.softmax(logits, -1)
    prediction = gamma @ model.means
    b, t, d = batch["target_z"].shape
    flat = batch["target_z"].reshape(-1, d)
    comp = diag_log_prob(flat, model.means, model.log_var).reshape(b, t, -1)
    lp = torch.logsumexp(F.log_softmax(logits, -1)[:, None] + comp, -1)
    valid_lp = lp[batch["target_mask"]]
    return prediction, -valid_lp.mean()


def _global_prior_logits(model, batch_size):
    if hasattr(model, "prior_logits"):
        return model.prior_logits[None].expand(batch_size, -1)
    return torch.zeros(batch_size, len(model.means), device=model.means.device)


@torch.no_grad()
def run_shortcut_diagnostics(model, loader, cfg, out_dir: Path) -> Dict[str, object]:
    """Audit what information drives held-out-probe voxel prediction.

    This is a post-hoc input intervention audit. It does not replace separately
    trained ablation models, but it directly tests whether the fitted network can
    retain performance when context, neighboring latents, or positions are removed
    or deliberately mismatched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rng = np.random.default_rng(cfg.seed + cfg.shortcut_seed_offset)
    names = [
        "full",
        "context_only",
        "neighbors_only",
        "positions_only",
        "neighbor_latents_only",
        "global_prior",
        "shuffled_context",
        "shuffled_neighbor_sets",
        "shuffled_positions",
        "neighbor_mean_baseline",
        "distance_weighted_neighbor_mean",
        "shuffled_target_sanity",
    ]
    observed = {name: [] for name in names}
    predicted = {name: [] for name in names}
    nll = {name: [] for name in names if name not in {
        "neighbor_mean_baseline", "distance_weighted_neighbor_mean", "shuffled_target_sanity"
    }}
    batch_counter = 0

    for raw in loader:
        batch = move(raw, cfg.device)
        if cfg.shortcut_max_batches and batch_counter >= cfg.shortcut_max_batches:
            break
        batch_counter += 1
        y = _batch_observed_mean(batch)
        base_logits = model.logits(batch["neighbor_z"], batch["relative_position"], batch["context"], batch["neighbor_padding_mask"])

        variants = {}
        variants["full"] = base_logits

        b = len(y)
        all_pad = torch.ones_like(batch["neighbor_padding_mask"])
        variants["context_only"] = model.logits(
            torch.zeros_like(batch["neighbor_z"]), torch.zeros_like(batch["relative_position"]),
            batch["context"], all_pad,
        )
        variants["neighbors_only"] = model.logits(
            batch["neighbor_z"], batch["relative_position"], torch.zeros_like(batch["context"]),
            batch["neighbor_padding_mask"],
        )
        variants["positions_only"] = model.logits(
            torch.zeros_like(batch["neighbor_z"]), batch["relative_position"], torch.zeros_like(batch["context"]),
            batch["neighbor_padding_mask"],
        )
        variants["neighbor_latents_only"] = model.logits(
            batch["neighbor_z"], torch.zeros_like(batch["relative_position"]), torch.zeros_like(batch["context"]),
            batch["neighbor_padding_mask"],
        )
        variants["global_prior"] = _global_prior_logits(model, b)

        permutation = torch.as_tensor(rng.permutation(b), device=cfg.device, dtype=torch.long)
        variants["shuffled_context"] = model.logits(
            batch["neighbor_z"], batch["relative_position"], batch["context"][permutation],
            batch["neighbor_padding_mask"],
        )
        variants["shuffled_neighbor_sets"] = model.logits(
            batch["neighbor_z"][permutation], batch["relative_position"][permutation], batch["context"],
            batch["neighbor_padding_mask"][permutation],
        )
        variants["shuffled_positions"] = model.logits(
            batch["neighbor_z"], batch["relative_position"][permutation], batch["context"],
            batch["neighbor_padding_mask"],
        )

        for name, logits in variants.items():
            pred, loss = _prediction_and_nll(model, batch, logits)
            observed[name].append(y.cpu().numpy())
            predicted[name].append(pred.cpu().numpy())
            nll[name].append(float(loss.cpu()))

        valid = ~batch["neighbor_padding_mask"]
        counts = valid.sum(1).clamp_min(1)
        mean_pred = (batch["neighbor_z"] * valid[..., None]).sum(1) / counts[:, None]
        distance = torch.linalg.norm(batch["relative_position"], dim=-1)
        weights = torch.exp(-2.0 * distance).masked_fill(~valid, 0.0)
        weighted_pred = (batch["neighbor_z"] * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        for name, pred in (("neighbor_mean_baseline", mean_pred), ("distance_weighted_neighbor_mean", weighted_pred)):
            observed[name].append(y.cpu().numpy())
            predicted[name].append(pred.cpu().numpy())

        # A deliberately wrong target must destroy R². A positive value here flags an evaluation bug.
        shuffled_y = y[permutation]
        observed["shuffled_target_sanity"].append(shuffled_y.cpu().numpy())
        predicted["shuffled_target_sanity"].append((F.softmax(base_logits, -1) @ model.means).cpu().numpy())

    metrics = {}
    for name in names:
        y = np.concatenate(observed[name])
        p = np.concatenate(predicted[name])
        metrics[name] = {
            "variance_weighted_r2": float(r2_score(y, p, multioutput="variance_weighted")),
            "mean_per_dim_r2": float(np.mean(r2_score(y, p, multioutput="raw_values"))),
            "n_examples": int(len(y)),
        }
        if name in nll:
            metrics[name]["mean_batch_nll"] = float(np.mean(nll[name]))

    full = metrics["full"]["variance_weighted_r2"]
    metrics["interpretation"] = {
        "context_increment_over_neighbors_only": full - metrics["neighbors_only"]["variance_weighted_r2"],
        "neighbors_increment_over_context_only": full - metrics["context_only"]["variance_weighted_r2"],
        "drop_when_context_shuffled": full - metrics["shuffled_context"]["variance_weighted_r2"],
        "drop_when_neighbor_sets_shuffled": full - metrics["shuffled_neighbor_sets"]["variance_weighted_r2"],
        "drop_when_positions_shuffled": full - metrics["shuffled_positions"]["variance_weighted_r2"],
    }

    plot_names = [name for name in names if name != "shuffled_target_sanity"]
    values = [metrics[name]["variance_weighted_r2"] for name in plot_names]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(plot_names)), values)
    ax.axhline(0, lw=1)
    ax.set_xticks(np.arange(len(plot_names)), plot_names, rotation=40, ha="right")
    ax.set_ylabel("held-out-probe voxel-mean R²")
    ax.set_title("Shortcut audit: input interventions and local-copy baselines")
    fig.tight_layout()
    fig.savefig(out_dir / "shortcut_ablation_r2.png", dpi=220)
    plt.close(fig)

    (out_dir / "shortcut_diagnostics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def extract_waveform_features(waveforms: np.ndarray, sampling_rate_hz: float):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    features = np.zeros((len(waveforms), 4), dtype=np.float32)
    dt_ms = 1000.0 / float(sampling_rate_hz)
    for i, waveform in enumerate(waveforms):
        channel = int(np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)[0])
        trace = waveform[channel]
        trough = int(np.argmin(trace))
        pre_peak = int(np.argmax(trace[:trough + 1]))
        post_peak = trough + int(np.argmax(trace[trough:]))
        trough_value = float(trace[trough])
        pre_value = float(trace[pre_peak])
        peak_value = float(trace[post_peak])
        dep_dt = max((trough - pre_peak) * dt_ms, dt_ms)
        rep_dt = max((post_peak - trough) * dt_ms, dt_ms)
        features[i] = (
            peak_value,
            (post_peak - trough) * dt_ms,
            (peak_value - trough_value) / rep_dt,
            (trough_value - pre_value) / dep_dt,
        )
    return features, ("peak value", "peak time (ms)", "repolarization slope (/ms)", "depolarization slope (/ms)")


@torch.no_grad()
def _decode_shared(model_ae, shared_raw: np.ndarray, cfg) -> np.ndarray:
    outputs = []
    for start in range(0, len(shared_raw), cfg.validation_batch_size):
        z = torch.from_numpy(shared_raw[start:start + cfg.validation_batch_size].astype(np.float32)).to(cfg.device)
        outputs.append(model_ae.decode_waveform_from_shared(z).cpu().numpy())
    return np.concatenate(outputs)


@torch.no_grad()
def run_waveform_feature_diagnostics(model_ae, model_gmm, scaler, gmm_pred, data, cfg, regions, names, out_dir: Path):
    """Determine whether feature failure is caused by reconstruction, latent averaging, or PT prediction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 2307)

    # Unit-level reconstruction ceilings on held-out probes.
    test_ids = np.flatnonzero(data.split == 2)
    if len(test_ids) > cfg.feature_diagnostic_max_units:
        test_ids = rng.choice(test_ids, cfg.feature_diagnostic_max_units, replace=False)
    full_rec, shared_rec = [], []
    for start in range(0, len(test_ids), cfg.validation_batch_size):
        ids = test_ids[start:start + cfg.validation_batch_size]
        w = torch.from_numpy(data.waveforms[ids]).to(cfg.device)
        a = torch.from_numpy(data.acgs[ids]).to(cfg.device)
        s = torch.from_numpy(data.stpc[ids]).to(cfg.device) if cfg.use_stpc else None
        e = model_ae.encode(w, a, s)
        full_rec.append(model_ae.waveform_decoder(e["z_wave_all"]).cpu().numpy())
        shared_rec.append(model_ae.decode_waveform_from_shared(e["z_wave_shared"]).cpu().numpy())
    original_features, feature_names = extract_waveform_features(data.waveforms[test_ids], cfg.waveform_sampling_rate_hz)
    full_features, _ = extract_waveform_features(np.concatenate(full_rec), cfg.waveform_sampling_rate_hz)
    shared_features, _ = extract_waveform_features(np.concatenate(shared_rec), cfg.waveform_sampling_rate_hz)

    ceilings = {
        "unit_full_autoencoder_feature_r2": r2_score(original_features, full_features, multioutput="raw_values").tolist(),
        "unit_shared_plus_predicted_private_feature_r2": r2_score(original_features, shared_features, multioutput="raw_values").tolist(),
    }

    # Observed voxel-average features.
    observed = []
    for indices in gmm_pred["target_indices"]:
        f, _ = extract_waveform_features(data.waveforms[indices], cfg.waveform_sampling_rate_hz)
        observed.append(f.mean(0))
    observed = np.stack(observed)

    # Oracle latent-mean decoding: diagnoses the nonlinearity/averaging problem separately from PT error.
    oracle_raw = scaler.inverse_transform(gmm_pred["observed"]).astype(np.float32)
    oracle_wave = _decode_shared(model_ae, oracle_raw, cfg)
    oracle_features, _ = extract_waveform_features(oracle_wave, cfg.waveform_sampling_rate_hz)

    # Posterior-mean decoding.
    posterior_raw = scaler.inverse_transform(gmm_pred["posterior_mean"]).astype(np.float32)
    posterior_wave = _decode_shared(model_ae, posterior_raw, cfg)
    posterior_features, _ = extract_waveform_features(posterior_wave, cfg.waveform_sampling_rate_hz)

    # Correct density-aware estimate: E[f(decode(z))], not f(decode(E[z])).
    means = model_gmm.means.detach().cpu().numpy()
    sigma = np.exp(0.5 * model_gmm.log_var.detach().cpu().numpy())
    gamma = gmm_pred["gamma"]
    mc_features = np.zeros_like(observed)
    sample_chunk = max(8, min(cfg.feature_mc_samples, 32))
    for start in range(0, len(gamma), 64):
        g = gamma[start:start + 64]
        all_features = []
        remaining = cfg.feature_mc_samples
        while remaining > 0:
            m = min(sample_chunk, remaining)
            component = np.stack([rng.choice(len(means), size=m, p=row / row.sum()) for row in g])
            eps = rng.standard_normal((len(g), m, means.shape[1])).astype(np.float32)
            z_scaled = means[component] + sigma[component] * eps
            z_raw = scaler.inverse_transform(z_scaled.reshape(-1, z_scaled.shape[-1])).astype(np.float32)
            wave = _decode_shared(model_ae, z_raw, cfg)
            feat, _ = extract_waveform_features(wave, cfg.waveform_sampling_rate_hz)
            all_features.append(feat.reshape(len(g), m, -1))
            remaining -= m
        mc_features[start:start + len(g)] = np.concatenate(all_features, axis=1).mean(1)

    diagnostics = {
        **ceilings,
        "voxel_oracle_mean_latent_decoding_feature_r2": r2_score(observed, oracle_features, multioutput="raw_values").tolist(),
        "voxel_pt_posterior_mean_decoding_feature_r2": r2_score(observed, posterior_features, multioutput="raw_values").tolist(),
        "voxel_pt_density_mc_feature_r2": r2_score(observed, mc_features, multioutput="raw_values").tolist(),
        "feature_names": list(feature_names),
        "note": "Density MC computes E[feature(decode(z))]. Posterior-mean decoding computes feature(decode(E[z])) and is generally biased for nonlinear features.",
    }

    # Region-level distribution comparison. Unlike posterior-mean decoding,
    # this retains multimodality by drawing individual latent samples from each
    # voxel's predicted GMM and decoding every sample before feature extraction.
    observed_by_region = [[[] for _ in range(4)] for _ in range(len(names))]
    predicted_by_region = [[[] for _ in range(4)] for _ in range(len(names))]
    distribution_metrics = {}

    means = model_gmm.means.detach().cpu().numpy()
    sigma = np.exp(0.5 * model_gmm.log_var.detach().cpu().numpy())

    for voxel_index, unit_indices in enumerate(gmm_pred["target_indices"]):
        region = int(regions[voxel_index])
        obs_features, _ = extract_waveform_features(
            data.waveforms[unit_indices], cfg.waveform_sampling_rate_hz
        )
        for feature_index in range(4):
            observed_by_region[region][feature_index].append(obs_features[:, feature_index])

        n_draw = min(
            int(len(unit_indices)),
            int(cfg.feature_distribution_max_predicted_per_voxel),
        )
        if n_draw <= 0:
            continue
        probabilities = gamma[voxel_index] / gamma[voxel_index].sum()
        component = rng.choice(len(means), size=n_draw, p=probabilities)
        eps = rng.standard_normal((n_draw, means.shape[1])).astype(np.float32)
        z_scaled = means[component] + sigma[component] * eps
        z_raw = scaler.inverse_transform(z_scaled).astype(np.float32)
        predicted_wave = _decode_shared(model_ae, z_raw, cfg)
        pred_features, _ = extract_waveform_features(
            predicted_wave, cfg.waveform_sampling_rate_hz
        )
        for feature_index in range(4):
            predicted_by_region[region][feature_index].append(pred_features[:, feature_index])

    fig, axes = plt.subplots(
        4, len(names), figsize=(2.55 * len(names), 9.5), squeeze=False
    )
    for row, feature_name in enumerate(feature_names):
        row_values = []
        for region in range(len(names)):
            if observed_by_region[region][row]:
                row_values.append(np.concatenate(observed_by_region[region][row]))
            if predicted_by_region[region][row]:
                row_values.append(np.concatenate(predicted_by_region[region][row]))
        finite = np.concatenate(row_values) if row_values else np.array([0.0, 1.0])
        finite = finite[np.isfinite(finite)]
        lo, hi = np.quantile(finite, [0.005, 0.995]) if len(finite) else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, int(cfg.feature_distribution_bins) + 1)

        for col, region_name in enumerate(names):
            ax = axes[row, col]
            observed_region = (
                np.concatenate(observed_by_region[col][row])
                if observed_by_region[col][row] else np.empty(0)
            )
            predicted_region = (
                np.concatenate(predicted_by_region[col][row])
                if predicted_by_region[col][row] else np.empty(0)
            )
            observed_region = observed_region[np.isfinite(observed_region)]
            predicted_region = predicted_region[np.isfinite(predicted_region)]

            if len(observed_region):
                ax.hist(
                    observed_region, bins=bins, density=True,
                    histtype="step", linewidth=1.5, label="observed",
                )
            if len(predicted_region):
                ax.hist(
                    predicted_region, bins=bins, density=True,
                    histtype="step", linewidth=1.5, linestyle="--",
                    label="predicted",
                )

            key = f"{region_name}/{feature_name}"
            if len(observed_region) and len(predicted_region):
                wd = float(wasserstein_distance(observed_region, predicted_region))
                distribution_metrics[key] = {
                    "wasserstein_distance": wd,
                    "n_observed_units": int(len(observed_region)),
                    "n_predicted_samples": int(len(predicted_region)),
                    "observed_mean": float(np.mean(observed_region)),
                    "predicted_mean": float(np.mean(predicted_region)),
                    "observed_std": float(np.std(observed_region)),
                    "predicted_std": float(np.std(predicted_region)),
                }
                ax.text(
                    0.04, 0.95, f"W={wd:.3g}\n"
                    f"n={len(observed_region)}/{len(predicted_region)}",
                    transform=ax.transAxes, va="top", fontsize=7,
                )

            ax.set_xlim(lo, hi)
            if row == 0:
                ax.set_title(str(region_name), fontsize=9)
            if col == 0:
                ax.set_ylabel(feature_name)
            if row == 3:
                ax.set_xlabel("feature value")
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Observed and PT-GMM-predicted waveform-feature distributions by Cosmos region",
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "observed_vs_predicted_feature_distributions_by_cosmos_region.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(fig)

    diagnostics["region_feature_distribution_metrics"] = distribution_metrics
    diagnostics["distribution_note"] = (
        "Observed curves pool target-unit features. Predicted curves sample each "
        "voxel's full GMM density and decode each sampled latent; no posterior-mean "
        "waveform is used for the regional distribution comparison."
    )

    (out_dir / "waveform_feature_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics
