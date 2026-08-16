from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ephysatlas.unit_level_encoder.config import Config
from ephysatlas.unit_level_encoder.data import PreparedData, assert_strict_probe_split

LOG2PI = float(np.log(2 * np.pi))


def _release_config(cfg: Config) -> dict:
    payload = asdict(cfg)
    payload.pop("output_dir", None)
    payload["device"] = str(cfg.device)
    return payload


def diag_log_prob(z, means, log_var):
    return -0.5 * (
        LOG2PI + log_var[None]
        + (z[:, None] - means[None]).square() * torch.exp(-log_var[None])
    ).sum(-1)


class VoxelNeighborhoodDataset(Dataset):
    """One example per (probe, atlas voxel), with target voxel excluded from inputs."""

    def __init__(self, data: PreparedData, shared_z: np.ndarray, split_value: int, cfg: Config):
        self.examples = []
        self.cfg = cfg
        self.z = shared_z.astype(np.float32)
        self.data_xyz = data.xyz_m
        ids = np.flatnonzero(data.split == split_value)

        by_probe_voxel: Dict[Tuple[int, int], List[int]] = {}
        for i in ids:
            by_probe_voxel.setdefault((int(data.probe_index[i]), int(data.voxel_id[i])), []).append(int(i))

        probe_to_indices = {
            int(probe): ids[data.probe_index[ids] == probe]
            for probe in np.unique(data.probe_index[ids])
        }
        size_m = cfg.voxel_size_um * 1e-6

        for (probe, voxel), target_list in by_probe_voxel.items():
            if len(target_list) < cfg.min_target_units_per_voxel:
                continue
            target = np.asarray(target_list, dtype=np.int64)
            center = (data.voxel_key[voxel].astype(np.float64) + 0.5) * size_m
            candidates = probe_to_indices[probe]
            candidates = candidates[data.voxel_id[candidates] != voxel]
            if np.intersect1d(candidates, target).size:
                raise RuntimeError("FATAL target leakage")

            if len(candidates):
                distance_um = np.linalg.norm(data.xyz_m[candidates] - center[None], axis=1) * 1e6
                keep = distance_um <= cfg.max_neighbor_distance_um
                candidates = candidates[keep]
                distance_um = distance_um[keep]
                order = np.argsort(distance_um, kind="stable")[: cfg.max_neighbor_units]
                neighbors = candidates[order]
            else:
                neighbors = np.empty(0, dtype=np.int64)

            context = data.context[target].mean(0).astype(np.float32)
            self.examples.append((probe, voxel, target, neighbors, center.astype(np.float32), context))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        probe, voxel, target, neighbors, center, context = self.examples[item]
        rel = (self.data_xyz[neighbors] - center[None]) * 1e6 / self.cfg.max_neighbor_distance_um
        return {
            "probe_index": probe,
            "voxel_id": voxel,
            "target_indices": target,
            "neighbor_indices": neighbors,
            "neighbor_z": self.z[neighbors],
            "relative_position": rel.astype(np.float32),
            "target_z": self.z[target],
            "context": context,
            "neighbor_count": len(neighbors),
        }


def collate_voxels(batch):
    max_n = max(1, max(len(x["neighbor_z"]) for x in batch))
    max_t = max(len(x["target_z"]) for x in batch)
    d = batch[0]["neighbor_z"].shape[1]
    c = batch[0]["context"].shape[0]
    nz = torch.zeros(len(batch), max_n, d)
    pos = torch.zeros(len(batch), max_n, 3)
    nmask = torch.ones(len(batch), max_n, dtype=torch.bool)
    ni = torch.full((len(batch), max_n), -1, dtype=torch.long)
    tz = torch.zeros(len(batch), max_t, d)
    tmask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    ti = torch.full((len(batch), max_t), -1, dtype=torch.long)
    ctx = torch.zeros(len(batch), c)
    count = torch.zeros(len(batch), dtype=torch.long)
    voxel = torch.zeros(len(batch), dtype=torch.long)
    probe = torch.zeros(len(batch), dtype=torch.long)
    for b, x in enumerate(batch):
        n, t = len(x["neighbor_z"]), len(x["target_z"])
        nz[b, :n] = torch.from_numpy(x["neighbor_z"])
        pos[b, :n] = torch.from_numpy(x["relative_position"])
        nmask[b, :n] = False
        ni[b, :n] = torch.from_numpy(x["neighbor_indices"])
        tz[b, :t] = torch.from_numpy(x["target_z"])
        tmask[b, :t] = True
        ti[b, :t] = torch.from_numpy(x["target_indices"])
        ctx[b] = torch.from_numpy(x["context"])
        count[b] = n
        voxel[b] = x["voxel_id"]
        probe[b] = x["probe_index"]
    return {
        "neighbor_z": nz,
        "relative_position": pos,
        "neighbor_padding_mask": nmask,
        "neighbor_indices": ni,
        "target_z": tz,
        "target_mask": tmask,
        "target_indices": ti,
        "context": ctx,
        "neighbor_count": count,
        "voxel_id": voxel,
        "probe_index": probe,
    }


class PointTransformerGMM(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, n_components: int, cfg: Config):
        super().__init__()
        h = cfg.pt_hidden_dim
        if h % cfg.pt_heads != 0:
            raise ValueError("pt_hidden_dim must be divisible by pt_heads")
        self.means = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.raw_sigma = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.sigma_min = cfg.sigma_min
        self.register_buffer("prior_logits", torch.zeros(n_components))
        self.unit_embed = nn.Sequential(nn.Linear(latent_dim + 3, h), nn.GELU(), nn.Linear(h, h))
        self.query_embed = nn.Sequential(nn.Linear(context_dim, h), nn.GELU(), nn.Linear(h, h))
        layer = nn.TransformerEncoderLayer(
            h, cfg.pt_heads, 4 * h, cfg.pt_dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.pt_layers)
        self.gate = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_components))

    @property
    def log_var(self):
        return 2 * torch.log(self.sigma_min + F.softplus(self.raw_sigma))

    def logits(self, nz, pos, context, pad):
        tokens = self.unit_embed(torch.cat([nz, pos], -1))
        q = self.query_embed(context)[:, None]
        x = torch.cat([q, tokens], 1)
        mask = torch.cat([torch.zeros(len(pad), 1, dtype=torch.bool, device=pad.device), pad], 1)
        return self.gate(self.encoder(x, src_key_padding_mask=mask)[:, 0])

    def batch_log_prob(self, batch):
        logits = self.logits(batch["neighbor_z"], batch["relative_position"], batch["context"], batch["neighbor_padding_mask"])
        b, t, d = batch["target_z"].shape
        flat = batch["target_z"].reshape(-1, d)
        comp = diag_log_prob(flat, self.means, self.log_var).reshape(b, t, -1)
        lp = torch.logsumexp(F.log_softmax(logits, -1)[:, None] + comp, -1)
        return lp[batch["target_mask"]], logits

    def posterior_mean(self, batch):
        logits = self.logits(batch["neighbor_z"], batch["relative_position"], batch["context"], batch["neighbor_padding_mask"])
        return F.softmax(logits, -1) @ self.means


def apply_neighbor_dropout(batch, cfg: Config):
    pad = batch["neighbor_padding_mask"].clone()
    valid = ~pad
    if cfg.neighbor_token_dropout_probability > 0:
        pad |= (torch.rand(valid.shape, device=valid.device) < cfg.neighbor_token_dropout_probability) & valid
    if cfg.full_neighbor_dropout_probability > 0:
        pad[torch.rand(len(pad), device=pad.device) < cfg.full_neighbor_dropout_probability] = True
    dropped = dict(batch)
    dropped["neighbor_padding_mask"] = pad
    dropped["neighbor_z"] = batch["neighbor_z"].masked_fill(pad[..., None], 0.0)
    dropped["relative_position"] = batch["relative_position"].masked_fill(pad[..., None], 0.0)
    dropped["effective_neighbor_count"] = (~pad).sum(1)
    return dropped


def make_neighborhood_datasets(data, z, cfg):
    return tuple(VoxelNeighborhoodDataset(data, z, s, cfg) for s in (0, 1, 2))


def make_loaders(datasets, cfg):
    return tuple(
        DataLoader(
            ds,
            batch_size=cfg.pt_batch_size,
            shuffle=(i == 0),
            collate_fn=collate_voxels,
            num_workers=0,
            generator=torch.Generator().manual_seed(cfg.seed + i),
        )
        for i, ds in enumerate(datasets)
    )


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate_nll(model, loader, cfg):
    model.eval(); total = 0.0; n = 0
    with torch.no_grad():
        for b in loader:
            lp, _ = model.batch_log_prob(move(b, cfg.device))
            total += float(lp.sum().cpu()); n += len(lp)
    return -total / max(n, 1)


def fit_point_transformer_gmm(shared_z, data, cfg: Config, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    assert_strict_probe_split(data.pids, data.split)
    train_idx = np.flatnonzero(data.split == 0)

    scaler = StandardScaler().fit(shared_z[train_idx])
    z = scaler.transform(shared_z).astype(np.float32)
    joblib.dump(scaler, out / "shared_latent_scaler.joblib")

    datasets = make_neighborhood_datasets(data, z, cfg)
    if min(map(len, datasets)) == 0:
        raise RuntimeError("No valid neighborhood voxels")
    loaders = make_loaders(datasets, cfg)

    gmm = GaussianMixture(
        cfg.gmm_components,
        covariance_type="diag",
        reg_covar=cfg.gmm_reg_covar,
        max_iter=cfg.gmm_sklearn_max_iter,
        n_init=cfg.gmm_sklearn_n_init,
        random_state=cfg.seed,
    ).fit(z[train_idx])
    joblib.dump(gmm, out / "unconditional_gmm_train_only.joblib")

    model = PointTransformerGMM(z.shape[1], data.context.shape[1], cfg.gmm_components, cfg).to(cfg.device)
    with torch.no_grad():
        model.means.copy_(torch.tensor(gmm.means_, dtype=torch.float32, device=cfg.device))
        sigma = np.sqrt(gmm.covariances_)
        raw = np.log(np.expm1(np.maximum(sigma - cfg.sigma_min, 1e-5)))
        model.raw_sigma.copy_(torch.tensor(raw, dtype=torch.float32, device=cfg.device))
        prior_logits = torch.log(torch.tensor(gmm.weights_, dtype=torch.float32, device=cfg.device).clamp_min(1e-8))
        model.prior_logits.copy_(prior_logits)
        model.gate[-1].bias.copy_(prior_logits)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pt_learning_rate, weight_decay=cfg.pt_weight_decay)
    best = np.inf; state = None; bad = 0
    history = {"train_nll": [], "val_nll": [], "mean_effective_train_neighbors": []}
    for epoch in range(1, cfg.pt_epochs + 1):
        model.train(); total = 0.0; n = 0; effective_counts = []
        for b in tqdm(loaders[0], desc=f"PT-GMM {epoch:03d}", leave=False):
            b = move(b, cfg.device); b_train = apply_neighbor_dropout(b, cfg)
            effective_counts.append(float(b_train["effective_neighbor_count"].float().mean().cpu()))
            opt.zero_grad(set_to_none=True)
            lp, _ = model.batch_log_prob(b_train)
            loss = -lp.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PT-GMM loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            total += float(loss.detach().cpu()) * len(lp); n += len(lp)
        val = evaluate_nll(model, loaders[1], cfg)
        tr = total / max(n, 1)
        history["train_nll"].append(tr); history["val_nll"].append(val)
        history["mean_effective_train_neighbors"].append(float(np.mean(effective_counts)))
        print(f"PT-GMM epoch {epoch:03d}: train NLL={tr:.4f} val NLL={val:.4f}")
        if val < best - cfg.pt_min_delta:
            best = val; state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        if bad >= cfg.pt_patience:
            break

    if state is None:
        raise RuntimeError("PT-GMM produced no checkpoint")
    model.load_state_dict(state)
    torch.save(
        {
            "model_state_dict": state,
            "config": _release_config(cfg),
            "history": history,
            "latent_dim": int(z.shape[1]),
            "context_dim": int(data.context.shape[1]),
            "n_components": int(cfg.gmm_components),
        },
        out / cfg.pt_checkpoint_name,
    )
    return model, scaler, datasets, loaders, {
        "history": history,
        "best_val_nll": float(best),
        "test_nll": float(evaluate_nll(model, loaders[2], cfg)),
        "n_train_examples": len(datasets[0]),
        "n_validation_examples": len(datasets[1]),
        "n_test_examples": len(datasets[2]),
    }


def load_point_transformer_gmm(checkpoint_path: Path, data: PreparedData, standardized_shared: np.ndarray, cfg: Config):
    payload = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    latent_dim = int(payload.get("latent_dim", standardized_shared.shape[1]))
    context_dim = int(payload.get("context_dim", data.context.shape[1]))
    n_components = int(payload.get("n_components", cfg.gmm_components))
    model = PointTransformerGMM(latent_dim, context_dim, n_components, cfg).to(cfg.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    datasets = make_neighborhood_datasets(data, standardized_shared, cfg)
    loaders = make_loaders(datasets, cfg)
    return model, datasets, loaders, payload
