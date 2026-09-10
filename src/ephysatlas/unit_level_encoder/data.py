from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from iblatlas.regions import BrainRegions
from sklearn.preprocessing import StandardScaler


@dataclass
class UnitData:
    waveforms: np.ndarray
    acgs: Optional[np.ndarray]
    stpc: Optional[np.ndarray]
    context: np.ndarray
    xyz_m: np.ndarray
    pids: np.ndarray
    cosmos_ids: np.ndarray
    beryl_ids: np.ndarray
    allen_ids: np.ndarray
    waveform_features: np.ndarray
    waveform_feature_names: list[str]
    split: np.ndarray  # 0 train, 1 validation, 2 test


@dataclass
class ContextTransform:
    """Training-only standardization for the 100-D molecular atlas context.

    The input is already a PCA representation: 50 MERFISH PCs followed by
    50 AGEA PCs. Applying another PCA here would change the representation
    relative to the channel-level spatial encoder, so we only standardize it.
    """
    scaler: StandardScaler

    def transform(self, context: np.ndarray) -> np.ndarray:
        return self.scaler.transform(context).astype(np.float32)



def infer_training_hemisphere_sign(xyz_m: np.ndarray, split: np.ndarray | None = None) -> float:
    """Infer the canonical ML sign from recorded units, robust to a few midline points."""
    xyz = np.asarray(xyz_m, np.float64)
    if split is not None:
        xyz = xyz[np.asarray(split) == 0]
    x = xyz[:, 0]
    x = x[np.isfinite(x) & (np.abs(x) > 1e-9)]
    if len(x) == 0:
        return 1.0
    return 1.0 if float(np.median(x)) >= 0.0 else -1.0


def mirror_xyz_to_hemisphere(xyz_m: np.ndarray, hemisphere_sign: float) -> np.ndarray:
    """Fold ML coordinate x onto one hemisphere while preserving AP/DV."""
    out = np.asarray(xyz_m, np.float32).copy()
    sign = 1.0 if float(hemisphere_sign) >= 0 else -1.0
    out[:, 0] = sign * np.abs(out[:, 0])
    return out

def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_split(repo_id: str, vintage: str) -> dict:
    """Download the authoritative PID split from the release revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename="split.json",
        revision=vintage,
    )

    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def _split_pid_sets(manifest: dict) -> tuple[set[str], set[str], set[str]]:
    """Accept the split.json layouts used by the previous ephys-atlas runs."""
    aliases = {
        "train": ("train", "train_pids", "training"),
        "val": ("val", "validation", "val_pids", "validation_pids"),
        "test": ("test", "test_pids", "testing"),
    }

    def read_one(keys):
        for key in keys:
            if key in manifest:
                value = manifest[key]
                if isinstance(value, dict):
                    for sub in ("pids", "pid", "values"):
                        if sub in value:
                            value = value[sub]
                            break
                return set(map(str, value))
        return set()

    train = read_one(aliases["train"])
    val = read_one(aliases["val"])
    test = read_one(aliases["test"])
    if not test:
        raise ValueError("Could not identify a test PID list in split.json")
    return train, val, test


def build_split(pids: np.ndarray, manifest: dict) -> np.ndarray:
    """Preserve authoritative validation/test PIDs; unseen unit-only PIDs go to train."""
    train, val, test = _split_pid_sets(manifest)
    split = np.zeros(len(pids), dtype=np.int8)
    pid_str = np.asarray(pids).astype(str)
    split[np.isin(pid_str, list(val))] = 1
    split[np.isin(pid_str, list(test))] = 2

    # Critical invariant: never silently move a validation/test PID to train.
    for pid in np.unique(pid_str[split == 2]):
        if pid not in test:
            raise RuntimeError(f"test split corruption for PID {pid}")
    return split


def assert_probe_disjoint(data: UnitData) -> None:
    sets = [set(data.pids[data.split == s].astype(str)) for s in (0, 1, 2)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("PID leakage detected across train/validation/test splits")


def load_prepared_data(data_dir: Path, cfg, split_manifest: dict | None = None) -> UnitData:
    data_dir = Path(data_dir)
    required = [
        "waveforms.npy", "ctx.npy", "xyz.npy", "pids.npy", "cosmos.npy",
        "allen.npy", "waveform_features.npy", "waveform_feature_names.json",
    ]
    if cfg.use_acg:
        required.append("acgs.npy")
    if cfg.use_stpc:
        required.append("stpc.npy")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared arrays in {data_dir}: {missing}")

    waveforms = np.load(data_dir / "waveforms.npy").astype(np.float32)
    acgs = np.load(data_dir / "acgs.npy").astype(np.float32) if cfg.use_acg else None
    stpc = np.load(data_dir / "stpc.npy").astype(np.float32) if cfg.use_stpc else None
    context = np.load(data_dir / "ctx.npy").astype(np.float32)
    xyz = np.load(data_dir / "xyz.npy").astype(np.float32)
    pids = np.load(data_dir / "pids.npy", allow_pickle=True).astype(str)
    cosmos = np.load(data_dir / "cosmos.npy").astype(np.int64)
    allen = np.load(data_dir / "allen.npy").astype(np.int64)
    features = np.load(data_dir / "waveform_features.npy").astype(np.float32)
    feature_names = json.loads((data_dir / "waveform_feature_names.json").read_text(encoding="utf-8"))

    br = BrainRegions()
    beryl = br.remap(allen, source_map="Allen", target_map="Beryl").astype(np.int64)

    if split_manifest is None:
        split_manifest = _download_split(cfg.repo_id, cfg.vintage)
    split = build_split(pids, split_manifest)

    if bool(getattr(cfg, "mirror_x_to_single_hemisphere", False)):
        hemisphere_sign = float(getattr(cfg, "mirror_x_sign", infer_training_hemisphere_sign(xyz, split)))
        xyz = mirror_xyz_to_hemisphere(xyz, hemisphere_sign)
        print(
            f"[mirror-x] folded all unit xyz onto canonical training hemisphere "
            f"sign={hemisphere_sign:+.0f}; model/evaluation spatial coordinates are unilateral"
        )

    data = UnitData(
        waveforms=waveforms,
        acgs=acgs,
        stpc=stpc,
        context=context,
        xyz_m=xyz,
        pids=pids,
        cosmos_ids=cosmos,
        beryl_ids=beryl,
        allen_ids=allen,
        waveform_features=features,
        waveform_feature_names=feature_names,
        split=split,
    )
    assert_probe_disjoint(data)
    return data


def fit_context_transform(data: UnitData, cfg) -> ContextTransform:
    """Fit only a StandardScaler on TRAIN molecular-context vectors."""
    expected = int(cfg.n_cell_pcs) + int(cfg.n_gene_pcs)
    if data.context.shape[1] != expected:
        raise ValueError(
            f"Expected {expected}-D molecular context "
            f"({cfg.n_cell_pcs} MERFISH + {cfg.n_gene_pcs} AGEA PCs), "
            f"got shape={data.context.shape}. Re-run data preparation."
        )
    train = data.split == 0
    scaler = StandardScaler().fit(data.context[train])
    return ContextTransform(scaler=scaler)


def split_indices(data: UnitData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.flatnonzero(data.split == s) for s in (0, 1, 2))
