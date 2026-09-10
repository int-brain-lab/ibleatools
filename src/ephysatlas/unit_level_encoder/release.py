from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np

from .config import Config
from .train import checkpoint_name


UNIT_MODEL_FILENAMES = (
    "README.md",
    "config.json",
    "autoencoder.pt",
    "shared_latent_scaler.joblib",
    "global_gmm.joblib",
    "global_gmm_summary.json",
    "context_transform.joblib",
    "context_weight_model_bundle.pt",
    "conditioning_summary.json",
    "knn_bank.npz",
    "split.json",
)


def _git_state(repo_dir: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_dir), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _package_versions() -> dict:
    names = ["torch", "numpy", "scipy", "scikit-learn", "iblatlas", "ONE-api", "ephys-atlas", "huggingface-hub"]
    out = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def _download_split(repo_id: str, vintage: str, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    # The channel split at the frozen vintage remains authoritative.  Unit-only
    # PIDs absent from it are assigned to TRAIN by data.build_split().
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="split.json",
            revision=vintage,
            token=token,
        )
    )


def _unit_readme() -> str:
    return """# Unit-level Ephys Atlas model

Released model: waveform + ACG + stPC modality autoencoders are concatenated
into a 60-D latent representation. A K=25 full-covariance GMM uses global
component means/covariances and molecular-context-conditioned mixture weights.
The final phenotype projection uses a distance-weighted k=20 nearest-neighbor
distribution over TRAIN exemplars only.

Artifacts:
- `autoencoder.pt`: multimodal autoencoder.
- `shared_latent_scaler.joblib`: TRAIN-only 60-D latent StandardScaler.
- `global_gmm.joblib`: K=25 full-covariance global GMM geometry.
- `context_transform.joblib`: TRAIN-only StandardScaler for 100-D molecular context.
- `context_weight_model_bundle.pt`: context -> GMM mixture weights.
- `knn_bank.npz`: standardized TRAIN latents and real TRAIN waveform-feature exemplars used by kNN=20.
- `split.json`: exact unit-level PID split used by this release.
- `config.json`: exact released configuration.

Preprocessing statistics are stored in `../../preprocessing/unit_stats.npz`; the
compact held-out evaluation is stored in `../../results/unit/summary.json`.
"""


def _write_unit_stats(path: Path, bundle) -> None:
    latent = bundle.latent_scaler
    transform = bundle.context_model.transform
    context_scaler = transform.scaler
    np.savez_compressed(
        path,
        latent_mean=np.asarray(latent.mean_, np.float32),
        latent_scale=np.asarray(latent.scale_, np.float32),
        context_mean=np.asarray(context_scaler.mean_, np.float32),
        context_scale=np.asarray(context_scaler.scale_, np.float32),
        waveform_shape=np.asarray(bundle.cfg.waveform_shape, np.int64),
        acg_shape=np.asarray(bundle.cfg.acg_shape, np.int64),
        stpc_shape=np.asarray(bundle.cfg.stpc_shape, np.int64),
        modality_latent_dim=np.asarray(bundle.cfg.modality_latent_dim, np.int64),
        joint_latent_dim=np.asarray(bundle.cfg.latent_dim(), np.int64),
        waveform_sampling_rate_hz=np.asarray(bundle.cfg.waveform_sampling_rate_hz, np.float64),
        gmm_components=np.asarray(bundle.cfg.gmm_components, np.int64),
        knn_k=np.asarray(bundle.cfg.knn_decoder_k, np.int64),
        mirror_x_sign=np.asarray(bundle.cfg.mirror_x_sign, np.float32),
        waveform_feature_names=np.asarray(bundle.data.waveform_feature_names, dtype="U"),
    )


def _load_existing_metadata(repo_id: str, token: str | None = None) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename="metadata.json", revision="main", token=token)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _updated_metadata(existing: dict, cfg: Config, summary: dict, code_repo_dir: Path) -> dict:
    out = dict(existing)
    out["format_version"] = max(int(out.get("format_version", 1)), 1)
    out["model_family"] = "ephys-atlas"
    out["release_tag"] = str(cfg.vintage)
    out["data_vintage"] = str(cfg.vintage)
    out["updated_utc"] = datetime.now(timezone.utc).isoformat()

    code = dict(out.get("code", {}))
    state = _git_state(code_repo_dir)
    code["unit_level_git_commit"] = state["git_commit"]
    code["unit_level_git_dirty"] = state["git_dirty"]
    out["code"] = code

    out["unit_level_environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }

    components = dict(out.get("components", {}))
    components["unit_level"] = {
        "available": True,
        "model_name": "context_weights_k25_knn20",
        "autoencoder": "models/unit/autoencoder.pt",
        "shared_latent_scaler": "models/unit/shared_latent_scaler.joblib",
        "global_gmm": "models/unit/global_gmm.joblib",
        "context_transform": "models/unit/context_transform.joblib",
        "context_weight_model": "models/unit/context_weight_model_bundle.pt",
        "knn_bank": "models/unit/knn_bank.npz",
        "config": "models/unit/config.json",
        "split": "models/unit/split.json",
        "preprocessing_stats": "preprocessing/unit_stats.npz",
        "summary": "results/unit/summary.json",
        "joint_latent_dim": int(cfg.latent_dim()),
        "gmm_components": int(cfg.gmm_components),
        "gmm_covariance_type": str(cfg.gmm_covariance_type),
        "knn_k": int(cfg.knn_decoder_k),
        "context": {
            "n_cell_pcs": int(cfg.n_cell_pcs),
            "n_gene_pcs": int(cfg.n_gene_pcs),
            "conditions": "mixture weights gamma only",
            "gmm_means_covariances": "global",
        },
    }
    out["components"] = components

    dataset = dict(out.get("dataset_summary", {}))
    dataset.update(summary.get("split_counts", {}))
    out["dataset_summary"] = dataset
    return out


def stage_unit_release(
    bundle,
    summary: dict,
    *,
    code_repo_dir: Path | str = Path("."),
    token: str | None = None,
    staging_dir: Path | str | None = None,
) -> Path:
    """Build a local tree mirroring only the Hub paths that should be replaced."""
    cfg = bundle.cfg
    staging = Path(staging_dir or cfg.release_staging_dir)
    if staging.exists():
        shutil.rmtree(staging)
    unit_dir = staging / "models" / "unit"
    prep_dir = staging / "preprocessing"
    result_dir = staging / "results" / "unit"
    unit_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    local_root = Path(bundle.model_root)
    # Local training layout from pipeline._model_paths().
    copies = {
        local_root / "autoencoder" / checkpoint_name(cfg): unit_dir / "autoencoder.pt",
        local_root / "shared_latent_scaler.joblib": unit_dir / "shared_latent_scaler.joblib",
        local_root / "gmm_k25_full" / "global_gmm.joblib": unit_dir / "global_gmm.joblib",
        local_root / "gmm_k25_full" / "global_gmm_summary.json": unit_dir / "global_gmm_summary.json",
        local_root / "context_weights_k25" / "context_transform.joblib": unit_dir / "context_transform.joblib",
        local_root / "context_weights_k25" / "context_weight_model_bundle.pt": unit_dir / "context_weight_model_bundle.pt",
        local_root / "context_weights_k25" / "conditioning_summary.json": unit_dir / "conditioning_summary.json",
        local_root / "knn_bank_k20.npz": unit_dir / "knn_bank.npz",
    }
    for src, dst in copies.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing release artifact: {src}")
        shutil.copy2(src, dst)

    (unit_dir / "config.json").write_text(json.dumps(cfg.to_json_dict(), indent=2), encoding="utf-8")
    (unit_dir / "README.md").write_text(_unit_readme(), encoding="utf-8")
    data = bundle.data
    split_payload = {
        "train": sorted(np.unique(data.pids[data.split == 0]).astype(str).tolist()),
        "validation": sorted(np.unique(data.pids[data.split == 1]).astype(str).tolist()),
        "test": sorted(np.unique(data.pids[data.split == 2]).astype(str).tolist()),
        "policy": (
            "Validation/test PIDs follow the frozen channel-level split; unit-only PIDs "
            "absent from that split are assigned to TRAIN. TEST is never modified."
        ),
    }
    (unit_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    _write_unit_stats(prep_dir / "unit_stats.npz", bundle)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    existing = _load_existing_metadata(cfg.repo_id, token=token)
    metadata = _updated_metadata(existing, cfg, summary, Path(code_repo_dir))
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return staging


def publish_unit_release(
    staging_dir: Path | str,
    *,
    repo_id: str,
    token: str | None = None,
    create_pr: bool = False,
    retag: str | None = None,
) -> None:
    """Replace the old Hub unit directory and update the three stable companion files."""
    from huggingface_hub import HfApi

    staging = Path(staging_dir)
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=staging / "models" / "unit",
        path_in_repo="models/unit",
        delete_patterns="*",
        commit_message="Replace unit-level model with K25 context + kNN20 release",
        create_pr=create_pr,
    )
    for local, remote in [
        (staging / "preprocessing" / "unit_stats.npz", "preprocessing/unit_stats.npz"),
        (staging / "results" / "unit" / "summary.json", "results/unit/summary.json"),
        (staging / "metadata.json", "metadata.json"),
    ]:
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=local,
            path_in_repo=remote,
            commit_message=f"Update {remote} for unit-level K25+kNN20 release",
            create_pr=create_pr,
        )

    if retag is not None:
        if create_pr:
            raise ValueError("Cannot retag main while create_pr=True; merge the PR first, then retag.")
        try:
            api.delete_tag(repo_id=repo_id, tag=retag)
        except Exception:
            pass
        api.create_tag(
            repo_id=repo_id,
            tag=retag,
            revision="main",
            tag_message=f"Ephys Atlas release {retag} with K25 context + kNN20 unit model",
        )


def download_unit_release(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """Download only the released unit-model files plus its stable metadata companions."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=[
            "models/unit/*",
            "preprocessing/unit_stats.npz",
            "results/unit/summary.json",
            "metadata.json",
            "split.json",
            "context/*",
        ],
    )
    return Path(root)
