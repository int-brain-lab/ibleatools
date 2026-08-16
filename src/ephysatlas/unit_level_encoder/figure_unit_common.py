from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from ephysatlas.unit_level_encoder.config import Config
from ephysatlas.unit_level_encoder.data import prepare_data
from ephysatlas.unit_level_encoder.gmm_models import (
    PointTransformerGMM,
    collate_voxels,
    make_neighborhood_datasets,
)
from ephysatlas.unit_level_encoder.hf_io import download_json, download_unit_artifacts, download_unit_data
from ephysatlas.unit_level_encoder.model import MultimodalAutoencoder
from ephysatlas.unit_level_encoder.train import encode_all


def apply_saved_config(cfg: Config, payload: dict) -> Config:
    valid = {f.name for f in fields(Config)}
    tuple_fields = {"waveform_shape", "acg_shape", "cosmos_region_names"}
    for key, value in payload.items():
        if key in valid and key not in {"device", "output_dir"}:
            setattr(cfg, key, tuple(value) if key in tuple_fields else value)
    return cfg


def load_released_unit_model(
    repo_id: str,
    vintage: str,
    *,
    token: Optional[str] = None,
    device: Optional[str] = None,
):
    arrays = download_unit_data(repo_id, vintage, token)
    artifacts = download_unit_artifacts(repo_id, vintage, token)
    split_manifest = download_json(repo_id, "split.json", vintage, token)

    cfg = Config(device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved_cfg = json.loads(artifacts["config"].read_text(encoding="utf-8"))
    apply_saved_config(cfg, saved_cfg)
    cfg.waveform_shape = tuple(arrays["waveforms"].shape[1:])
    cfg.acg_shape = tuple(arrays["acgs"].shape[1:])

    data = prepare_data(
        arrays["waveforms"].astype(np.float32),
        arrays["acgs"].astype(np.float32),
        arrays["context"].astype(np.float32),
        arrays["xyz"].astype(np.float32),
        arrays["pids"],
        cfg,
        split_manifest=split_manifest,
    )

    ae_payload = torch.load(artifacts["autoencoder"], map_location=cfg.device, weights_only=False)
    model_ae = MultimodalAutoencoder(cfg).to(cfg.device)
    model_ae.load_state_dict(ae_payload["model_state_dict"], strict=True)
    model_ae.eval()

    shared = encode_all(model_ae, data, cfg)
    scaler = joblib.load(artifacts["scaler"])
    standardized = scaler.transform(shared).astype(np.float32)

    pt_payload = torch.load(artifacts["pt_gmm"], map_location=cfg.device, weights_only=False)
    model_gmm = PointTransformerGMM(
        standardized.shape[1], data.context.shape[1], cfg.gmm_components, cfg
    ).to(cfg.device)
    model_gmm.load_state_dict(pt_payload["model_state_dict"], strict=True)
    model_gmm.eval()

    datasets = make_neighborhood_datasets(data, standardized, cfg)
    loaders = tuple(
        DataLoader(ds, batch_size=cfg.pt_batch_size, shuffle=False, collate_fn=collate_voxels, num_workers=0)
        for ds in datasets
    )
    return cfg, data, model_ae, model_gmm, scaler, standardized, datasets, loaders, artifacts


def extract_three_waveform_features(waveforms: np.ndarray, sampling_rate_hz: float):
    """Pre-peak value, post-trough peak value, and trough-to-peak duration."""
    waveforms = np.asarray(waveforms, dtype=np.float32)
    out = np.zeros((len(waveforms), 3), dtype=np.float32)
    dt_ms = 1000.0 / float(sampling_rate_hz)
    for i, waveform in enumerate(waveforms):
        channel = int(np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)[0])
        trace = waveform[channel]
        trough = int(np.argmin(trace))
        pre_peak = int(np.argmax(trace[: trough + 1]))
        post_peak = trough + int(np.argmax(trace[trough:]))
        out[i] = (float(trace[pre_peak]), float(trace[post_peak]), float((post_peak - trough) * dt_ms))
    return out, ("Pre-peak value", "Peak value", "Duration (ms)")


def cosmos_ids_for_xyz(brain_atlas, xyz_m: np.ndarray):
    return np.asarray(brain_atlas.get_labels(np.asarray(xyz_m), mapping="Cosmos"), dtype=np.int64)


def region_color(brain_atlas, acronym: str):
    hit = np.flatnonzero(np.asarray(brain_atlas.regions.acronym, dtype=object) == acronym)
    if len(hit) == 0:
        raise RuntimeError(f"Atlas acronym not found: {acronym}")
    rgb = np.asarray(brain_atlas.regions.rgb[hit[0]], dtype=float)
    return rgb / 255.0 if rgb.max() > 1 else rgb


def region_id(brain_atlas, acronym: str) -> int:
    hit = np.flatnonzero(np.asarray(brain_atlas.regions.acronym, dtype=object) == acronym)
    if len(hit) == 0:
        raise RuntimeError(f"Atlas acronym not found: {acronym}")
    return int(brain_atlas.regions.id[hit[0]])
