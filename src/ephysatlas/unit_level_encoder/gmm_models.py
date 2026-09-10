from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


LOG2PI = float(np.log(2.0 * np.pi))


def component_log_prob(gmm, z: np.ndarray) -> np.ndarray:
    """Return log p(z | k) for a sklearn diagonal/full-covariance GMM."""
    z = np.asarray(z, np.float64)
    means = np.asarray(gmm.means_, np.float64)
    if gmm.covariance_type == "diag":
        var = np.asarray(gmm.covariances_, np.float64)
        return -0.5 * (
            LOG2PI
            + np.log(var)[None, :, :]
            + (z[:, None, :] - means[None, :, :]) ** 2 / var[None, :, :]
        ).sum(axis=2)
    if gmm.covariance_type == "full":
        n, d = z.shape
        out = np.empty((n, len(means)), np.float64)
        for k in range(len(means)):
            cov = np.asarray(gmm.covariances_[k], np.float64)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise RuntimeError(f"Non-positive-definite covariance for component {k}")
            delta = z - means[k]
            sol = np.linalg.solve(cov, delta.T).T
            out[:, k] = -0.5 * (d * LOG2PI + logdet + np.sum(delta * sol, axis=1))
        return out
    raise ValueError(f"Unsupported covariance_type={gmm.covariance_type!r}")


def responsibilities(gmm: GaussianMixture, z: np.ndarray) -> np.ndarray:
    return gmm.predict_proba(np.asarray(z, np.float64)).astype(np.float32)


def fit_latent_scaler(z_joint: np.ndarray, train_mask: np.ndarray, path: Path | None = None):
    scaler = StandardScaler().fit(np.asarray(z_joint)[train_mask])
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)
    return scaler


def fit_global_gmm(z_joint, train_mask, cfg, out_dir: Path, *, scaler=None):
    """Fit global GMM. A supplied scaler guarantees identical coordinates across experiments."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scaler is None:
        scaler = StandardScaler().fit(z_joint[train_mask])
    z = scaler.transform(z_joint).astype(np.float32)

    gmm = GaussianMixture(
        n_components=int(cfg.gmm_components),
        covariance_type=str(cfg.gmm_covariance_type),
        reg_covar=float(cfg.gmm_reg_covar),
        n_init=int(cfg.gmm_n_init),
        max_iter=int(cfg.gmm_max_iter),
        random_state=int(cfg.seed),
        init_params="kmeans",
    ).fit(z[train_mask].astype(np.float64))
    if not gmm.converged_:
        raise RuntimeError("Global GMM did not converge")

    train_resp = responsibilities(gmm, z[train_mask])
    mass = train_resp.mean(axis=0)
    rare = np.flatnonzero(mass < float(cfg.gmm_min_component_fraction))
    joblib.dump(scaler, out_dir / "latent_scaler.joblib")
    joblib.dump(gmm, out_dir / "global_gmm.joblib")
    info = {
        "n_components": int(gmm.n_components),
        "covariance_type": str(gmm.covariance_type),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "component_mass_train": mass.tolist(),
        "min_component_mass": float(mass.min()),
        "max_component_mass": float(mass.max()),
        "rare_component_indices": rare.tolist(),
        "rare_component_threshold": float(cfg.gmm_min_component_fraction),
    }
    (out_dir / "global_gmm_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return gmm, scaler, z, train_resp, info


class WeightModel:
    def weights(self, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GlobalWeightModel(WeightModel):
    def __init__(self, global_weights, n_units: int):
        self.global_weights = np.asarray(global_weights, np.float32)
        self.global_weights /= self.global_weights.sum()
        self.n_units = int(n_units)

    def weights(self, indices):
        return np.repeat(self.global_weights[None, :], len(indices), axis=0)

    def weights_for_context(self, context_raw):
        return np.repeat(self.global_weights[None, :], len(context_raw), axis=0)


class ContextWeightNet(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, n_components):
        super().__init__()
        seq = []
        d = input_dim
        for _ in range(max(1, int(layers) - 1)):
            seq.extend([nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
            d = hidden
        seq.append(nn.Linear(d, n_components))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class ContextWeightModel(WeightModel):
    def __init__(self, net, context_pc, device, transform=None):
        self.net = net
        self.context_pc = np.asarray(context_pc, np.float32)
        self.device = device
        self.transform = transform

    @torch.no_grad()
    def _weights_from_pc(self, context_pc):
        x = torch.from_numpy(np.asarray(context_pc, np.float32)).to(self.device)
        return torch.softmax(self.net(x), dim=1).cpu().numpy().astype(np.float32)

    def weights(self, indices):
        return self._weights_from_pc(self.context_pc[np.asarray(indices, int)])

    def weights_for_context(self, context_raw):
        if self.transform is None:
            raise RuntimeError("Context transform unavailable for arbitrary-voxel prediction")
        return self._weights_from_pc(self.transform.transform(np.asarray(context_raw, np.float32)))


def fit_context_weight_model(context_pc, train_mask, val_mask, resp_train, resp_val, component_mass, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train = torch.from_numpy(np.asarray(context_pc[train_mask], np.float32))
    y_train = torch.from_numpy(np.asarray(resp_train, np.float32))
    x_val = torch.from_numpy(np.asarray(context_pc[val_mask], np.float32))
    y_val = torch.from_numpy(np.asarray(resp_val, np.float32))

    mass = np.maximum(np.asarray(component_mass, np.float32), 1e-6)
    class_weight = mass ** (-float(cfg.rare_component_power))
    class_weight /= np.average(class_weight, weights=mass)
    class_weight = np.minimum(class_weight, float(cfg.rare_component_weight_cap)).astype(np.float32)
    cw = torch.from_numpy(class_weight).to(cfg.device)

    net = ContextWeightNet(x_train.shape[1], cfg.context_hidden_dim, cfg.context_layers,
                           cfg.context_dropout, y_train.shape[1]).to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.context_weight_lr, weight_decay=cfg.context_weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.context_weight_batch_size, shuffle=True)

    def soft_ce(logits, target):
        logp = F.log_softmax(logits, dim=1)
        wt = target * cw[None, :]
        wt = wt / wt.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return -(wt * logp).sum(dim=1).mean()

    best = np.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, cfg.context_weight_epochs + 1):
        net.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = soft_ce(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        net.eval()
        with torch.no_grad():
            val_loss = float(soft_ce(net(x_val.to(cfg.device)), y_val.to(cfg.device)).cpu())
        row = {"epoch": epoch, "train_balanced_ce": total / max(n, 1), "val_balanced_ce": val_loss}
        history.append(row)
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = copy.deepcopy(net.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg.context_weight_patience:
            break
    if best_state is None:
        raise RuntimeError("Context weight model produced no checkpoint")
    net.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "class_weight": class_weight, "history": history},
               out_dir / "context_weight_model.pt")
    return ContextWeightModel(net, context_pc, cfg.device), {
        "best_val_balanced_ce": float(best),
        "rare_component_class_weights": class_weight.tolist(),
        "history": history,
    }



def conditional_log_prob(z_scaled, indices, gmm, weight_model):
    """Held-out log p(z | conditioning) for fixed global GMM geometry."""
    indices = np.asarray(indices, int)
    w = weight_model.weights(indices)
    comp = component_log_prob(gmm, z_scaled[indices])
    return logsumexp(comp + np.log(np.maximum(w, 1e-12)), axis=1)


def sample_conditional(indices, n_per_index, gmm, weight_model, rng):
    """Sample standardized latents for observed unit indices."""
    indices = np.asarray(indices, int)
    w = weight_model.weights(indices)
    outputs = []
    for row in range(len(indices)):
        prob = w[row] / np.maximum(w[row].sum(), 1e-12)
        comp = rng.choice(gmm.n_components, size=int(n_per_index), p=prob)
        means = gmm.means_[comp]
        if gmm.covariance_type == "diag":
            draw = means + rng.normal(size=means.shape) * np.sqrt(gmm.covariances_[comp])
        elif gmm.covariance_type == "full":
            draw = np.vstack([
                rng.multivariate_normal(means[j], gmm.covariances_[k])
                for j, k in enumerate(comp)
            ])
        else:
            raise ValueError(gmm.covariance_type)
        outputs.append(draw.astype(np.float32))
    return outputs


def sample_conditional_for_context(context_raw, n_per_context, gmm, weight_model, rng):
    """Sample standardized latents at arbitrary atlas contexts."""
    context_raw = np.asarray(context_raw, np.float32)
    w = weight_model.weights_for_context(context_raw)
    outputs = []
    for row in range(len(context_raw)):
        prob = w[row] / np.maximum(w[row].sum(), 1e-12)
        comp = rng.choice(gmm.n_components, size=int(n_per_context), p=prob)
        means = gmm.means_[comp]
        if gmm.covariance_type == "diag":
            draw = means + rng.normal(size=means.shape) * np.sqrt(gmm.covariances_[comp])
        elif gmm.covariance_type == "full":
            draw = np.vstack([
                rng.multivariate_normal(means[j], gmm.covariances_[k])
                for j, k in enumerate(comp)
            ])
        else:
            raise ValueError(gmm.covariance_type)
        outputs.append(draw.astype(np.float32))
    return outputs


def posterior_mean(gmm, weight_model, indices):
    w = weight_model.weights(np.asarray(indices, int))
    return (w @ gmm.means_).astype(np.float32)


def posterior_mean_for_context(gmm, weight_model, context_raw):
    w = weight_model.weights_for_context(context_raw)
    return (w @ gmm.means_).astype(np.float32)


def save_context_weight_bundle(model, transform, out_dir, cfg):
    out_dir = Path(out_dir)
    joblib.dump(transform, out_dir / "context_transform.joblib")
    torch.save({
        "model_state_dict": model.net.state_dict(),
        "input_dim": int(model.context_pc.shape[1]),
        "hidden": int(cfg.context_hidden_dim),
        "layers": int(cfg.context_layers),
        "dropout": float(cfg.context_dropout),
        "n_components": int(model.net.net[-1].out_features),
    }, out_dir / "context_weight_model_bundle.pt")


def load_context_weight_bundle(context_raw, out_dir, cfg):
    out_dir = Path(out_dir)
    transform = joblib.load(out_dir / "context_transform.joblib")
    context_pc = transform.transform(context_raw)
    payload = torch.load(
        out_dir / "context_weight_model_bundle.pt",
        map_location=cfg.device,
        weights_only=False,
    )
    net = ContextWeightNet(
        payload["input_dim"], payload["hidden"], payload["layers"],
        payload["dropout"], payload["n_components"],
    ).to(cfg.device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return ContextWeightModel(net, context_pc, cfg.device, transform=transform), transform
