from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ephysatlas.unit_level_encoder.config import Config
from ephysatlas.unit_level_encoder.data import PreparedData, UnitDataset, assert_strict_probe_split, split_indices
from ephysatlas.unit_level_encoder.gmm_models import fit_point_transformer_gmm, move
from ephysatlas.unit_level_encoder.model import (
    MultimodalAutoencoder,
    acg_reconstruction_loss,
    augment_acg,
    augment_waveform,
    latent_scale_loss,
    vicreg_loss,
    waveform_morphology_loss,
    waveform_reconstruction_loss,
)


def _release_config(cfg: Config) -> dict:
    payload = asdict(cfg)
    payload.pop("output_dir", None)
    payload["device"] = str(cfg.device)
    return payload


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_ae_epoch(model, loader, optimizer, scaler, cfg, train=True):
    model.train(train); sums = {}; n = 0
    for batch in tqdm(loader, desc="AE train" if train else "AE val", leave=False):
        batch = _to_device(batch, cfg.device)
        waveform, acg = batch["waveform"], batch["acg"]
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=cfg.amp and str(cfg.device).startswith("cuda"),
        ):
            clean = model.encode(waveform, acg)
            rec = model.reconstruct(clean)
            wave_view = model.encode(augment_waveform(waveform, cfg), acg)
            acg_view = model.encode(waveform, augment_acg(acg, cfg))
            vic = vicreg_loss(wave_view["p_wave"], acg_view["p_acg"], cfg)
            wave_loss = waveform_reconstruction_loss(rec["waveform_reconstruction"], waveform)
            morph = waveform_morphology_loss(rec["waveform_reconstruction"], waveform, cfg)
            acg_loss = acg_reconstruction_loss(rec["acg_reconstruction"], acg, cfg)
            raw_scale = latent_scale_loss(clean["z_wave_shared"], clean["z_acg_shared"], cfg)
            loss = (
                cfg.lambda_waveform_reconstruction * wave_loss
                + cfg.lambda_waveform_morphology * morph
                + cfg.lambda_acg_reconstruction * acg_loss
                + cfg.lambda_vicreg_invariance * vic["invariance"]
                + cfg.lambda_vicreg_variance * vic["variance"]
                + cfg.lambda_vicreg_covariance * vic["covariance"]
                + cfg.lambda_raw_latent_scale * raw_scale
            )
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        bs = len(waveform); n += bs
        values = {
            "total": loss, "wave_recon": wave_loss, "wave_morphology": morph,
            "acg_recon": acg_loss, "raw_latent_scale": raw_scale,
            "vic_invariance": vic["invariance"], "vic_variance": vic["variance"],
            "vic_covariance": vic["covariance"],
            "wave_shared_std": clean["z_wave_shared"].std(0, unbiased=False).mean(),
            "acg_shared_std": clean["z_acg_shared"].std(0, unbiased=False).mean(),
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * bs
    return {key: value / max(n, 1) for key, value in sums.items()}


def train_autoencoder(data: PreparedData, cfg: Config):
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert_strict_probe_split(data.pids, data.split)
    train_indices, validation_indices, _ = split_indices(data)
    train_loader = DataLoader(UnitDataset(data, train_indices), batch_size=cfg.ae_batch_size, shuffle=True,
                              num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed))
    validation_loader = DataLoader(UnitDataset(data, validation_indices), batch_size=cfg.validation_batch_size,
                                   shuffle=False, num_workers=cfg.num_workers)
    model = MultimodalAutoencoder(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ae_learning_rate, weight_decay=cfg.ae_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and str(cfg.device).startswith("cuda"))
    history = {}; best = math.inf; state = None; bad = 0
    for epoch in range(1, cfg.ae_epochs + 1):
        start = time.perf_counter()
        tr = run_ae_epoch(model, train_loader, optimizer, scaler, cfg, True)
        va = run_ae_epoch(model, validation_loader, optimizer, scaler, cfg, False)
        monitor = va["wave_recon"] + cfg.lambda_acg_reconstruction * va["acg_recon"] + 0.1 * cfg.lambda_waveform_morphology * va["wave_morphology"]
        for prefix, metrics in (("train", tr), ("val", va)):
            for key, value in metrics.items():
                history.setdefault(f"{prefix}_{key}", []).append(value)
        history.setdefault("epoch_seconds", []).append(time.perf_counter() - start)
        if monitor < best - cfg.min_delta:
            best = monitor; bad = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": state, "config": _release_config(cfg), "history": history, "epoch": epoch}, out / cfg.ae_checkpoint_name)
        else:
            bad += 1
        print(f"AE epoch {epoch:03d}: train={tr['total']:.4f} val wave={va['wave_recon']:.4f} acg={va['acg_recon']:.4f}")
        if bad >= cfg.patience:
            break
    if state is None:
        raise RuntimeError("Autoencoder produced no checkpoint")
    model.load_state_dict(state)
    return model, {"history": history, "best_monitor": float(best)}


@torch.no_grad()
def encode_all(model, data, cfg):
    loader = DataLoader(UnitDataset(data, np.arange(len(data.waveforms))), batch_size=cfg.validation_batch_size,
                        shuffle=False, num_workers=cfg.num_workers)
    model.eval(); unit_shared = []
    for batch in tqdm(loader, desc="encode units", leave=False):
        batch = _to_device(batch, cfg.device)
        unit_shared.append(model.encode(batch["waveform"], batch["acg"])["z_unit_shared"].cpu().numpy())
    return np.concatenate(unit_shared).astype(np.float32)


@torch.no_grad()
def collect_mean_predictions(model, loader, cfg):
    observed, predicted = [], []
    model.eval()
    for raw in loader:
        batch = move(raw, cfg.device)
        target, mask = batch["target_z"], batch["target_mask"]
        observed.append(torch.stack([target[i][mask[i]].mean(0) for i in range(len(target))]).cpu().numpy())
        predicted.append(model.posterior_mean(batch).cpu().numpy())
    return np.concatenate(observed), np.concatenate(predicted)


def fit_and_evaluate(model_ae, data: PreparedData, cfg: Config, training_outputs=None):
    """Minimal scientific evaluation; publication figures live in figure scripts."""
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    split_audit = assert_strict_probe_split(data.pids, data.split)
    shared = encode_all(model_ae, data, cfg)
    model_gmm, scaler, datasets, loaders, gmm_info = fit_point_transformer_gmm(shared, data, cfg, out)
    observed, predicted = collect_mean_predictions(model_gmm, loaders[2], cfg)
    summary = {
        "split_audit": split_audit,
        "autoencoder": training_outputs or {},
        "pt_gmm": gmm_info,
        "test_posterior_mean_r2_variance_weighted": float(r2_score(observed, predicted, multioutput="variance_weighted")),
        "config": _release_config(cfg),
    }
    (out / cfg.summary_name).write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return model_gmm, scaler, summary


def _json_default(value: Any):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)
