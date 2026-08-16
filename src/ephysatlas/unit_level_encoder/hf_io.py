from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ephysatlas.unit_level_encoder.config import Config


UNIT_DATA_FILES = ("waveforms.npy", "acgs.npy", "ctx.npy", "xyz.npy", "pids.npy", "latest_cells_encoder_manifest.json")


def download_json(repo_id: str, filename: str, vintage: str, token: Optional[str] = None) -> dict:
    path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename, revision=vintage, token=token)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def download_unit_data(repo_id: str, vintage: str, token: Optional[str] = None) -> dict[str, np.ndarray]:
    names = {"waveforms": "waveforms.npy", "acgs": "acgs.npy", "context": "ctx.npy", "xyz": "xyz.npy", "pids": "pids.npy"}
    out = {}
    for key, basename in names.items():
        path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=f"data/unit/{basename}", revision=vintage, token=token)
        out[key] = np.load(path, allow_pickle=(key == "pids"))
    return out


def download_unit_artifacts(repo_id: str, vintage: str, token: Optional[str] = None) -> dict[str, Path]:
    files = {
        "config": "models/unit/config.json",
        "autoencoder": "models/unit/autoencoder.pt",
        "pt_gmm": "models/unit/point_transformer_gmm.pt",
        "scaler": "models/unit/shared_latent_scaler.joblib",
        "unconditional_gmm": "models/unit/unconditional_gmm_train_only.joblib",
        "unit_stats": "preprocessing/unit_stats.npz",
        "summary": "results/unit/summary.json",
    }
    return {
        key: Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename, revision=vintage, token=token))
        for key, filename in files.items()
    }


def _jsonable_config(cfg: Config) -> dict:
    payload = asdict(cfg); payload.pop("output_dir", None); payload["device"] = str(cfg.device)
    return payload


def stage_unit_release(stage_dir: Path, run_dir: Path, data_dir: Path, data, cfg: Config, summary: dict, include_data: bool, base_metadata: dict | None = None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir = stage_dir / "models" / "unit"; model_dir.mkdir(parents=True, exist_ok=True)
    result_dir = stage_dir / "results" / "unit"; result_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = stage_dir / "preprocessing"; prep_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(run_dir / cfg.ae_checkpoint_name, model_dir / "autoencoder.pt")
    shutil.copy2(run_dir / "pt_gmm" / cfg.pt_checkpoint_name, model_dir / "point_transformer_gmm.pt")
    shutil.copy2(run_dir / "pt_gmm" / "shared_latent_scaler.joblib", model_dir / "shared_latent_scaler.joblib")
    shutil.copy2(run_dir / "pt_gmm" / "unconditional_gmm_train_only.joblib", model_dir / "unconditional_gmm_train_only.joblib")
    (model_dir / "config.json").write_text(json.dumps(_jsonable_config(cfg), indent=2), encoding="utf-8")
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(prep_dir / "unit_stats.npz", context_mean=data.context_mean, context_std=data.context_std)

    if base_metadata is not None:
        metadata = dict(base_metadata)
        components = dict(metadata.get("components", {}))
        components["unit_level"] = {
            "available": True,
            "autoencoder": "models/unit/autoencoder.pt",
            "point_transformer_gmm": "models/unit/point_transformer_gmm.pt",
            "shared_latent_scaler": "models/unit/shared_latent_scaler.joblib",
            "config": "models/unit/config.json",
            "summary": "results/unit/summary.json",
            "data": "data/unit/",
        }
        metadata["components"] = components
        (stage_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if include_data:
        hf_data = stage_dir / "data" / "unit"; hf_data.mkdir(parents=True, exist_ok=True)
        for basename in UNIT_DATA_FILES:
            source = data_dir / basename
            if source.exists():
                shutil.copy2(source, hf_data / basename)


def publish_stage(repo_id: str, vintage: str, stage_dir: Path, token: Optional[str] = None, private: bool = False) -> str:
    """Upload only staged unit artifacts, preserving channel-level release files."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        path_in_repo=".",
        commit_message=f"Update unit-level Ephys Atlas model for {vintage}",
    )
    oid = getattr(commit, "oid", None) or "main"
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    if vintage in {x.name for x in refs.tags}:
        api.delete_tag(repo_id=repo_id, repo_type="model", tag=vintage)
    api.create_tag(repo_id=repo_id, repo_type="model", tag=vintage, revision=oid, tag_message=f"Data vintage {vintage}")
    return str(oid)
