from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .data import split_indices
from .model import UnitAutoencoder, covariance_penalty, variance_penalty


class FeatureTargetTransform:
    """TRAIN-only feature transform used by the feature-fidelity AE.

    The first 10 waveform features are standardized. Polarity is encoded as a
    categorical variable using the exact unique TRAIN values.
    """

    def __init__(self, scaler: StandardScaler, polarity_values: np.ndarray):
        self.scaler = scaler
        self.polarity_values = np.asarray(polarity_values, np.float32)

    def continuous(self, features):
        return self.scaler.transform(np.asarray(features)[:, :-1]).astype(np.float32)

    def polarity_indices(self, polarity):
        x = np.asarray(polarity, np.float32).reshape(-1)
        dist = np.abs(x[:, None] - self.polarity_values[None, :])
        return np.argmin(dist, axis=1).astype(np.int64)


class UnitDataset(Dataset):
    def __init__(self, data, indices, feature_transform: FeatureTargetTransform | None = None):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.feature_transform = feature_transform

        self._feature_cont = None
        self._feature_pol = None
        if feature_transform is not None:
            feat = data.waveform_features[self.indices]
            self._feature_cont = feature_transform.continuous(feat)
            self._feature_pol = feature_transform.polarity_indices(feat[:, -1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        item = {
            "index": idx,
            "waveform": torch.from_numpy(self.data.waveforms[idx]),
        }
        if self.data.acgs is not None:
            item["acg"] = torch.from_numpy(self.data.acgs[idx])
        if self.data.stpc is not None:
            item["stpc"] = torch.from_numpy(self.data.stpc[idx])
        if self.feature_transform is not None:
            item["feature_continuous"] = torch.from_numpy(self._feature_cont[i])
            item["feature_polarity"] = torch.tensor(self._feature_pol[i], dtype=torch.long)
        return item


def fit_feature_target_transform(data) -> FeatureTargetTransform:
    train = data.split == 0
    train_features = np.asarray(data.waveform_features[train], np.float64)
    scaler = StandardScaler().fit(train_features[:, :-1])
    polarity_values = np.unique(train_features[:, -1].astype(np.float32))
    if len(polarity_values) < 2:
        raise RuntimeError(f"Polarity has only one TRAIN category: {polarity_values.tolist()}")
    return FeatureTargetTransform(scaler, polarity_values)


def _batch_loss(model, batch, cfg):
    waveform = batch["waveform"].to(cfg.device)
    acg = batch.get("acg")
    stpc = batch.get("stpc")
    if acg is not None:
        acg = acg.to(cfg.device)
    if stpc is not None:
        stpc = stpc.to(cfg.device)

    lat = model.encode(waveform, acg, stpc)
    rec = model.decode(lat)
    losses = {"waveform_reconstruction": F.mse_loss(rec["waveform"], waveform)}
    total = losses["waveform_reconstruction"]

    if cfg.use_acg:
        losses["acg_reconstruction"] = F.mse_loss(rec["acg"], acg)
        total = total + losses["acg_reconstruction"]
    if cfg.use_stpc:
        losses["stpc_reconstruction"] = F.mse_loss(rec["stpc"], stpc)
        total = total + losses["stpc_reconstruction"]

    var = torch.stack([variance_penalty(z, cfg.latent_std_target) for z in lat.values()]).mean()
    cov = torch.stack([covariance_penalty(z) for z in lat.values()]).mean()
    total = total + cfg.lambda_latent_variance * var + cfg.lambda_latent_covariance * cov
    losses["latent_variance_penalty"] = var
    losses["latent_covariance_penalty"] = cov

    if bool(getattr(cfg, "feature_fidelity", False)):
        target_cont = batch["feature_continuous"].to(cfg.device)
        target_pol = batch["feature_polarity"].to(cfg.device)
        pred_cont, pred_pol = model.predict_waveform_features(lat["waveform"])
        cont_loss = F.smooth_l1_loss(pred_cont, target_cont)
        pol_loss = F.cross_entropy(pred_pol, target_pol)
        losses["feature_continuous"] = cont_loss
        losses["feature_polarity"] = pol_loss
        total = total + float(cfg.lambda_feature_continuous) * cont_loss
        total = total + float(cfg.lambda_feature_polarity) * pol_loss

    losses["total"] = total
    return losses


def _run_epoch(model, loader, cfg, optimizer=None):
    train = optimizer is not None
    model.train(train)
    sums = {}
    n = 0
    for batch in tqdm(loader, desc="AE train" if train else "AE val", leave=False):
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            losses = _batch_loss(model, batch, cfg)
        if train:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        bs = len(batch["waveform"])
        n += bs
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * bs
    return {key: value / max(n, 1) for key, value in sums.items()}


def checkpoint_name(cfg) -> str:
    mods = "wave" + ("_acg" if cfg.use_acg else "") + ("_stpc" if cfg.use_stpc else "")
    suffix = "_feature_fidelity" if bool(getattr(cfg, "feature_fidelity", False)) else ""
    return f"ae_{mods}_d{cfg.modality_latent_dim}{suffix}.pt"


def train_autoencoder(data, cfg, checkpoint_dir: Path, initialize_from: Path | None = None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, _ = split_indices(data)

    feature_transform = fit_feature_target_transform(data) if cfg.feature_fidelity else None
    train_loader = DataLoader(
        UnitDataset(data, train_ids, feature_transform),
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        UnitDataset(data, val_ids, feature_transform),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    n_classes = len(feature_transform.polarity_values) if feature_transform is not None else 2
    model = UnitAutoencoder(cfg, n_continuous_features=10, n_polarity_classes=n_classes).to(cfg.device)

    if initialize_from is not None and Path(initialize_from).exists():
        source = torch.load(initialize_from, map_location=cfg.device, weights_only=False)
        source_state = source["model_state_dict"]
        compatible = {k: v for k, v in source_state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"[AE] initialized compatible weights from {initialize_from}; new parameters={len(missing)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ae_learning_rate, weight_decay=cfg.ae_weight_decay)
    best = np.inf
    best_state = None
    history = []
    bad = 0

    for epoch in range(1, cfg.ae_epochs + 1):
        tr = _run_epoch(model, train_loader, cfg, opt)
        va = _run_epoch(model, val_loader, cfg, None)
        history.append({"epoch": epoch, "train": tr, "validation": va})
        print(f"[AE] epoch={epoch:03d} train={tr['total']:.6f} val={va['total']:.6f}")
        if va["total"] < best - cfg.ae_min_delta:
            best = va["total"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg.ae_patience:
            break

    if best_state is None:
        raise RuntimeError("AE did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    payload = {
        "model_state_dict": best_state,
        "config": {**asdict(cfg), "device": str(cfg.device)},
        "history": history,
        "best_validation_loss": float(best),
        "active_modalities": list(cfg.active_modalities()),
        "feature_fidelity": bool(cfg.feature_fidelity),
        "polarity_values": feature_transform.polarity_values.tolist() if feature_transform else None,
    }
    path = checkpoint_dir / checkpoint_name(cfg)
    torch.save(payload, path)
    if feature_transform is not None:
        joblib.dump(feature_transform, checkpoint_dir / f"{path.stem}_feature_transform.joblib")
    (checkpoint_dir / f"{path.stem}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, payload, path


def load_autoencoder_file(path: Path | str, cfg):
    """Load an autoencoder from an explicit release checkpoint path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing AE checkpoint: {path}")
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    polarity_values = payload.get("polarity_values")
    n_classes = len(polarity_values) if polarity_values is not None else 2
    model = UnitAutoencoder(cfg, n_continuous_features=10, n_polarity_classes=n_classes).to(cfg.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload, path


def load_autoencoder(cfg, checkpoint_dir: Path):
    path = Path(checkpoint_dir) / checkpoint_name(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Missing AE checkpoint: {path}")
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    polarity_values = payload.get("polarity_values")
    n_classes = len(polarity_values) if polarity_values is not None else 2
    model = UnitAutoencoder(cfg, n_continuous_features=10, n_polarity_classes=n_classes).to(cfg.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload, path


@torch.no_grad()
def encode_all(model, data, cfg):
    ids = np.arange(len(data.waveforms))
    loader = DataLoader(UnitDataset(data, ids), batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    chunks = {name: [] for name in cfg.active_modalities()}
    model.eval()
    for batch in tqdm(loader, desc="encode units", leave=False):
        waveform = batch["waveform"].to(cfg.device)
        acg = batch.get("acg")
        stpc = batch.get("stpc")
        acg = acg.to(cfg.device) if acg is not None else None
        stpc = stpc.to(cfg.device) if stpc is not None else None
        lat = model.encode(waveform, acg, stpc)
        for name, value in lat.items():
            chunks[name].append(value.cpu().numpy())
    result = {name: np.concatenate(parts).astype(np.float32) for name, parts in chunks.items()}
    result["joint"] = np.concatenate([result[name] for name in cfg.active_modalities()], axis=1).astype(np.float32)
    return result
