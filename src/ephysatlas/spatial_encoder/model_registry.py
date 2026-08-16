from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch


REGISTRY_FORMAT_VERSION = 1
DEFAULT_REGISTRY_ROOT = Path(
    os.environ.get(
        "EPHYSATLAS_MODEL_REGISTRY",
        Path.home() / ".ephysatlas" / "model_registry",
    )
)


class RegistryError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=False)
        f.write("\n")


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_environment_metadata() -> dict:
    package_names = [
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "iblatlas",
        "ONE-api",
        "ephys-atlas",
        "huggingface-hub",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in package_names},
    }


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    repo_dir = Path(repo_dir or Path.cwd())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_git_is_dirty(repo_dir: Optional[Path] = None) -> Optional[bool]:
    repo_dir = Path(repo_dir or Path.cwd())
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


class EphysAtlasReleaseRegistry:
    """
    Local staging area for Hugging Face releases.

    Local layout:
        <root>/ephys-atlas-models/releases/<vintage>/
            README.md
            metadata.json
            config.json
            features.json
            split.json
            models/channel/spatial_encoder.pt
            models/channel/confidence_model.pt
            models/unit/                       # future unit-level artifacts
            context/agea_vol_pca.npy
            context/merfish_vol_pca.npy
            results/channel/test_r2.json
            results/channel/confidence_validation.json

    Each <vintage> folder is uploaded to the ROOT of the same Hugging Face model
    repository and tagged with exactly that vintage, e.g. revision="2026_W26".
    """

    def __init__(self, root: Path = DEFAULT_REGISTRY_ROOT):
        self.root = Path(root).expanduser().resolve()
        self.project_root = self.root / "ephys-atlas-models"
        self.releases_root = self.project_root / "releases"
        self.download_cache = self.project_root / "hf_cache"
        self.releases_root.mkdir(parents=True, exist_ok=True)
        self.download_cache.mkdir(parents=True, exist_ok=True)

    def release_dir(self, vintage: str) -> Path:
        return self.releases_root / str(vintage)

    def ensure_release_layout(self, vintage: str) -> Path:
        release = self.release_dir(vintage)
        for rel in [
            "models/channel",
            "models/unit",
            "context",
            "preprocessing",
            "results/channel",
            "results/unit",
        ]:
            (release / rel).mkdir(parents=True, exist_ok=True)
        unit_readme = release / "models" / "unit" / "README.md"
        if not unit_readme.exists():
            unit_readme.write_text(
                "# Unit-level model\n\nUnit-level waveform/ACG model artifacts are stored here when available.\n",
                encoding="utf-8",
            )
        return release

    def has_unit_level_model(self, vintage: str) -> bool:
        release = self.release_dir(vintage)
        required = [
            release / "models" / "unit" / "autoencoder.pt",
            release / "models" / "unit" / "point_transformer_gmm.pt",
            release / "models" / "unit" / "shared_latent_scaler.joblib",
            release / "models" / "unit" / "config.json",
            release / "models" / "unit" / "split.json",
            release / "preprocessing" / "unit_stats.npz",
        ]
        return all(path.exists() for path in required)

    def write_unit_level_artifacts(
        self,
        vintage: str,
        *,
        autoencoder_checkpoint: Path,
        pt_gmm_checkpoint: Path,
        shared_latent_scaler: Path,
        unit_config: Mapping[str, Any],
        unique_pids: Iterable[Any],
        probe_split: Iterable[int],
        context_mean: np.ndarray,
        context_std: np.ndarray,
        stpc_mean: Optional[np.ndarray] = None,
        stpc_std: Optional[np.ndarray] = None,
        preparation_manifest: Optional[Mapping[str, Any]] = None,
        summary_path: Optional[Path] = None,
        code_repo_dir: Optional[Path] = None,
    ) -> Path:
        """Stage a trained unit-level model inside an existing vintage release."""
        release = self.ensure_release_layout(vintage)
        unit_dir = release / "models" / "unit"
        unit_dir.mkdir(parents=True, exist_ok=True)

        sources = {
            "autoencoder.pt": Path(autoencoder_checkpoint),
            "point_transformer_gmm.pt": Path(pt_gmm_checkpoint),
            "shared_latent_scaler.joblib": Path(shared_latent_scaler),
        }
        for name, source in sources.items():
            if not source.exists():
                raise RegistryError(f"Missing unit-level artifact: {source}")
            shutil.copy2(source, unit_dir / name)

        save_json(
            unit_dir / "config.json",
            {
                "format_version": REGISTRY_FORMAT_VERSION,
                "model_family": "unit_level_multimodal_ptgmm",
                "config": unit_config,
            },
        )

        unique_pids = np.asarray(list(unique_pids), dtype=object)
        probe_split = np.asarray(list(probe_split), dtype=np.int8)
        if len(unique_pids) != len(probe_split):
            raise RegistryError(
                "unit unique_pids and probe_split lengths do not match"
            )
        split_payload = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "split_unit": "probe_insertion_pid",
            "train_pids": [str(x) for x in unique_pids[probe_split == 0]],
            "validation_pids": [str(x) for x in unique_pids[probe_split == 1]],
            "test_pids": [str(x) for x in unique_pids[probe_split == 2]],
        }
        split_payload["split_sha256"] = sha256_json(
            {
                "train_pids": split_payload["train_pids"],
                "validation_pids": split_payload["validation_pids"],
                "test_pids": split_payload["test_pids"],
            }
        )
        save_json(unit_dir / "split.json", split_payload)

        stats = {
            "context_mean": np.asarray(context_mean),
            "context_std": np.asarray(context_std),
        }
        if stpc_mean is not None:
            stats["stpc_mean"] = np.asarray(stpc_mean)
        if stpc_std is not None:
            stats["stpc_std"] = np.asarray(stpc_std)
        np.savez_compressed(
            release / "preprocessing" / "unit_stats.npz",
            **stats,
        )

        if preparation_manifest is not None:
            save_json(unit_dir / "data_manifest.json", preparation_manifest)

        if summary_path is not None:
            summary_path = Path(summary_path)
            if summary_path.exists():
                results_dir = release / "results" / "unit"
                results_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path, results_dir / "summary.json")

        unit_readme = unit_dir / "README.md"
        unit_readme.write_text(
            "# Unit-level model\\n\\n"
            "Artifacts:\\n"
            "- `autoencoder.pt`: multimodal unit autoencoder.\\n"
            "- `point_transformer_gmm.pt`: spatial PT-GMM over the shared unit latent.\\n"
            "- `shared_latent_scaler.joblib`: training-only StandardScaler for the shared latent.\\n"
            "- `config.json`: exact unit-level model/training configuration.\\n"
            "- `split.json`: authoritative unit-level PID split.\\n"
            "- `data_manifest.json`: prepared-array provenance when supplied.\\n"
            "\\nPreprocessing statistics are stored in `../../preprocessing/unit_stats.npz`.\\n",
            encoding="utf-8",
        )

        metadata_path = release / "metadata.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
        else:
            metadata = {
                "format_version": REGISTRY_FORMAT_VERSION,
                "model_family": "ephys-atlas",
                "release_tag": str(vintage),
                "data_vintage": str(vintage),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "code": {},
                "environment": get_environment_metadata(),
                "components": {},
            }
        metadata.setdefault("components", {})
        metadata["components"]["unit_level"] = {
            "available": True,
            "autoencoder": "models/unit/autoencoder.pt",
            "point_transformer_gmm": "models/unit/point_transformer_gmm.pt",
            "shared_latent_scaler": "models/unit/shared_latent_scaler.joblib",
            "config": "models/unit/config.json",
            "split": "models/unit/split.json",
            "preprocessing": "preprocessing/unit_stats.npz",
        }
        metadata["code"]["unit_level_git_commit"] = get_git_commit(code_repo_dir)
        metadata["code"]["unit_level_git_dirty"] = get_git_is_dirty(code_repo_dir)
        save_json(metadata_path, metadata)

        self.write_readme(vintage)
        self.write_checksums(vintage)
        return unit_dir

    def has_release(self, vintage: str, require_weights: bool = False) -> bool:
        release = self.release_dir(vintage)
        required = [
            release / "metadata.json",
            release / "config.json",
            release / "features.json",
            release / "split.json",
            release / "context" / "agea_vol_pca.npy",
            release / "context" / "merfish_vol_pca.npy",
            release / "preprocessing" / "channel_stats.npz",
        ]
        if require_weights:
            required.append(release / "models" / "channel" / "spatial_encoder.pt")
        return all(p.exists() for p in required)

    def load_split(self, vintage: str) -> dict:
        path = self.release_dir(vintage) / "split.json"
        if not path.exists():
            raise RegistryError(f"Missing split manifest: {path}")
        return load_json(path)

    def load_config(self, vintage: str) -> dict:
        path = self.release_dir(vintage) / "config.json"
        if not path.exists():
            raise RegistryError(f"Missing release config: {path}")
        return load_json(path)

    def load_features(self, vintage: str) -> list[str]:
        path = self.release_dir(vintage) / "features.json"
        if not path.exists():
            raise RegistryError(f"Missing feature manifest: {path}")
        payload = load_json(path)
        return [str(x) for x in payload["features"]]

    def validate_feature_order(self, vintage: str, current_features: Iterable[str]) -> None:
        saved = self.load_features(vintage)
        current = [str(x) for x in current_features]
        if saved != current:
            raise RegistryError(
                "FEATURE_LIST does not match the release feature order. "
                f"Release has {len(saved)} features; current code has {len(current)}."
            )


    def write_channel_preprocessing_stats(self, vintage: str, stats: Mapping[str, Any]) -> Path:
        release = self.ensure_release_layout(vintage)
        path = release / "preprocessing" / "channel_stats.npz"
        arrays = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                arrays[key] = value.detach().cpu().numpy()
            else:
                arrays[key] = np.asarray(value)
        np.savez_compressed(path, **arrays)
        return path

    def load_channel_preprocessing_stats(self, vintage: str) -> dict[str, np.ndarray]:
        path = self.release_dir(vintage) / "preprocessing" / "channel_stats.npz"
        if not path.exists():
            raise RegistryError(f"Missing channel preprocessing stats: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def write_split_manifest(
        self,
        vintage: str,
        split_info: Mapping[str, Any],
        *,
        seed: int,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.1,
        excluded_pids: Optional[Iterable[str]] = None,
    ) -> Path:
        release = self.ensure_release_layout(vintage)
        train = [str(x) for x in split_info["p_tr_names"]]
        val = [str(x) for x in split_info["p_va_names"]]
        test = [str(x) for x in split_info["p_te_names"]]
        payload = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "split_unit": "probe_insertion_pid",
            "seed_used_when_created": int(seed),
            "fractions_used_when_created": {
                "train": float(train_fraction),
                "validation": float(validation_fraction),
                "test": float(1.0 - train_fraction - validation_fraction),
            },
            "train_pids": train,
            "validation_pids": val,
            "test_pids": test,
            "excluded_pids": sorted({str(x) for x in (excluded_pids or [])}),
            "n_train": len(train),
            "n_validation": len(val),
            "n_test": len(test),
        }
        payload["split_sha256"] = sha256_json(
            {k: payload[k] for k in ["train_pids", "validation_pids", "test_pids"]}
        )
        path = release / "split.json"
        save_json(path, payload)
        return path

    def write_features(self, vintage: str, features: Iterable[str]) -> Path:
        release = self.ensure_release_layout(vintage)
        features = [str(x) for x in features]
        payload = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "features": features,
            "n_features": len(features),
            "feature_order_sha256": sha256_json(features),
        }
        path = release / "features.json"
        save_json(path, payload)
        return path

    def write_config(self, vintage: str, payload: Mapping[str, Any]) -> Path:
        release = self.ensure_release_layout(vintage)
        out = dict(payload)
        out.setdefault("format_version", REGISTRY_FORMAT_VERSION)
        path = release / "config.json"
        save_json(path, out)
        return path

    def write_metadata(
        self,
        vintage: str,
        *,
        code_repo_dir: Optional[Path] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        release = self.ensure_release_layout(vintage)
        payload = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "model_family": "ephys-atlas",
            "release_tag": str(vintage),
            "data_vintage": str(vintage),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code": {
                "git_commit": get_git_commit(code_repo_dir),
                "git_dirty": get_git_is_dirty(code_repo_dir),
            },
            "environment": get_environment_metadata(),
            "components": {
                "channel_level": {
                    "available": True,
                    "spatial_encoder": "models/channel/spatial_encoder.pt",
                    "confidence_model": "models/channel/confidence_model.pt",
                },
                "unit_level": {
                    "available": self.has_unit_level_model(vintage),
                    "directory": "models/unit",
                    "autoencoder": "models/unit/autoencoder.pt",
                    "point_transformer_gmm": "models/unit/point_transformer_gmm.pt",
                },
            },
        }
        if extra:
            payload.update(_jsonable(extra))
        path = release / "metadata.json"
        save_json(path, payload)
        return path

    def write_readme(self, vintage: str) -> Path:
        release = self.ensure_release_layout(vintage)
        text = f"""---
library_name: ephysatlas
tags:
- electrophysiology
- neuroscience
- neuropixels
---

# Ephys Atlas models — {vintage}

This directory is a complete release bundle for the Ephys Atlas models trained on
or associated with data vintage `{vintage}`. On Hugging Face this snapshot should
be tagged exactly `{vintage}`.

## Contents

- `models/channel/`: channel-level spatial encoder and probe-confidence model.
- `models/unit/`: unit-level multimodal autoencoder + PT-GMM artifacts when available.
- `context/`: frozen PCA context volumes used by the channel-level model.
- `split.json`: authoritative train/validation/test insertion PID strings.
- `features.json`: authoritative ordered channel feature list.
- `preprocessing/channel_stats.npz`: frozen clipping and normalization statistics.
- `config.json`: model, preprocessing, data, and training configuration.
- `metadata.json`: code/environment provenance and component availability.
- `results/`: release evaluation summaries.

Do not regenerate the split or PCA context when reproducing this release; load the
artifacts in this snapshot.
"""
        path = release / "README.md"
        path.write_text(text, encoding="utf-8")
        return path

    def verify_checksums(self, vintage: str) -> None:
        release = self.release_dir(vintage)
        path = release / "checksums.json"
        if not path.exists():
            return
        manifest = load_json(path)
        bad = []
        for item in manifest.get("files", []):
            file_path = release / item["path"]
            if not file_path.exists():
                bad.append(f"missing:{item['path']}")
                continue
            actual = sha256_file(file_path)
            if actual != item["sha256"]:
                bad.append(f"sha256:{item['path']}")
        if bad:
            raise RegistryError(
                "Release checksum verification failed: " + ", ".join(bad[:10])
            )

    def write_checksums(self, vintage: str) -> Path:
        release = self.release_dir(vintage)
        files = []
        for path in sorted(release.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "checksums.json":
                continue
            files.append(
                {
                    "path": str(path.relative_to(release)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        out = release / "checksums.json"
        save_json(out, {"format_version": REGISTRY_FORMAT_VERSION, "files": files})
        return out

    def upload_release_to_hf(
            self,
            vintage: str,
            *,
            repo_id: str,
            private: bool = False,
            token: Optional[str] = None,
            create_tag: bool = True,
            replace_existing_tag: bool = False,
    ) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RegistryError(
                "Install huggingface_hub first: "
                "pip install huggingface_hub"
            ) from exc

        release = self.release_dir(vintage)

        if not self.has_release(
                vintage,
                require_weights=True,
        ):
            raise RegistryError(
                f"Release {vintage} is incomplete and cannot be uploaded: "
                f"{release}"
            )

        # ------------------------------------------------------------
        # FINALIZE MANIFEST
        # ------------------------------------------------------------
        self.write_checksums(vintage)

        # Never upload a release that already fails locally.
        self.verify_checksums(vintage)

        api = HfApi(token=token)

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # Check tag BEFORE uploading
        # ------------------------------------------------------------
        refs = api.list_repo_refs(
            repo_id=repo_id,
            repo_type="model",
        )

        existing_tags = {
            ref.name
            for ref in refs.tags
        }

        tag_exists = str(vintage) in existing_tags

        if (
                create_tag
                and tag_exists
                and not replace_existing_tag
        ):
            raise RegistryError(
                f"Hugging Face tag {vintage!r} already exists. "
                "Refusing to replace a published release automatically. "
                "Set replace_existing_tag=True only when this is intentional."
            )

        # ------------------------------------------------------------
        # Upload exact release snapshot to main
        # ------------------------------------------------------------
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(release),
            path_in_repo=".",
            commit_message=f"Ephys Atlas release {vintage}",
            delete_patterns="*",
        )

        commit_oid = (
                getattr(commit, "oid", None)
                or "main"
        )

        # ------------------------------------------------------------
        # Update tag
        # ------------------------------------------------------------
        if create_tag:
            if tag_exists:
                api.delete_tag(
                    repo_id=repo_id,
                    repo_type="model",
                    tag=str(vintage),
                )

            api.create_tag(
                repo_id=repo_id,
                repo_type="model",
                tag=str(vintage),
                revision=commit_oid,
                tag_message=f"Data vintage {vintage}",
            )

    def download_release_from_hf(
        self,
        vintage: str,
        *,
        repo_id: str,
        token: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RegistryError(
                "Install huggingface_hub first: pip install huggingface_hub"
            ) from exc

        destination = self.release_dir(vintage)
        if self.has_release(vintage, require_weights=True) and not force:
            return destination

        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=str(vintage),
                cache_dir=str(self.download_cache),
                token=token,
            )
        )

        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(snapshot, destination)
        self.verify_checksums(vintage)
        return destination

    def resolve_release(
        self,
        vintage: str,
        *,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        require_weights: bool = True,
    ) -> Path:
        if self.has_release(vintage, require_weights=require_weights):
            return self.release_dir(vintage)
        if not repo_id:
            raise RegistryError(
                f"No complete local release found for {vintage} under {self.releases_root}, "
                "and no Hugging Face repo_id was provided."
            )
        return self.download_release_from_hf(
            vintage,
            repo_id=repo_id,
            token=token,
        )


def split_manifest_to_builder_format(split_manifest: Mapping[str, Any]) -> dict:
    """Normalize registry split.json to the keys expected by the data builder."""
    return {
        "train_pids": [str(x) for x in split_manifest.get("train_pids", [])],
        "validation_pids": [
            str(x) for x in split_manifest.get("validation_pids", [])
        ],
        "test_pids": [str(x) for x in split_manifest.get("test_pids", [])],
    }
