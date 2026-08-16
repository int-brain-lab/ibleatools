from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ephysatlas.unit_level_encoder.config import Config


@dataclass
class PreparedData:
    waveforms: np.ndarray
    acgs: np.ndarray
    context: np.ndarray
    xyz_m: np.ndarray
    pids: np.ndarray
    probe_index: np.ndarray
    unique_pids: np.ndarray
    probe_split: np.ndarray
    split: np.ndarray
    voxel_id: np.ndarray
    voxel_key: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_waveforms(waveforms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(waveforms, dtype=np.float32).copy()
    scale = np.max(np.abs(x), axis=(-2, -1), keepdims=True)
    x /= np.maximum(scale, eps)
    return np.clip(np.nan_to_num(x), -1.0, 1.0).astype(np.float32)


def normalize_acgs(acgs: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(acgs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)


def split_probes_from_manifest(
    pids: np.ndarray,
    split_manifest: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign unit-level PIDs using the authoritative channel/spatial split.

    Rules
    -----
    1. PIDs present in the spatial split keep their exact assignment.
    2. PIDs absent from the spatial split are assigned to TRAIN.
    3. The spatial TEST set is never modified or expanded.
    4. The spatial VALIDATION set is never expanded.

    Returns
    -------
    unit_split : np.ndarray
        Per-unit split labels:
            0 = train
            1 = validation
            2 = test

    unique_pids : np.ndarray
        Sorted unique unit-level PIDs.

    probe_split : np.ndarray
        Per-unique-PID split assignment.
    """

    pids = np.asarray(pids)
    unique_pids, probe_index = np.unique(
        pids,
        return_inverse=True,
    )

    train_pids = {
        str(pid)
        for pid in split_manifest.get("train_pids", [])
    }
    validation_pids = {
        str(pid)
        for pid in split_manifest.get("validation_pids", [])
    }
    test_pids = {
        str(pid)
        for pid in split_manifest.get("test_pids", [])
    }

    # ------------------------------------------------------------------
    # Validate the authoritative spatial split itself.
    # ------------------------------------------------------------------
    overlap_train_val = train_pids & validation_pids
    overlap_train_test = train_pids & test_pids
    overlap_val_test = validation_pids & test_pids

    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):
        raise RuntimeError(
            "The authoritative spatial split is invalid: "
            "some PIDs occur in more than one split.\n"
            f"train/validation overlap: "
            f"{sorted(overlap_train_val)[:10]}\n"
            f"train/test overlap: "
            f"{sorted(overlap_train_test)[:10]}\n"
            f"validation/test overlap: "
            f"{sorted(overlap_val_test)[:10]}"
        )

    if len(test_pids) == 0:
        raise RuntimeError(
            "Authoritative spatial split contains no test PIDs. "
            "Refusing to construct the unit-level split because "
            "the held-out test set must remain fixed."
        )

    # ------------------------------------------------------------------
    # Assign every unit-level PID.
    #
    # Unknown/new PIDs default to TRAIN only.
    # ------------------------------------------------------------------
    probe_split = np.zeros(
        len(unique_pids),
        dtype=np.int8,
    )

    extra_train_pids = []

    for i, pid in enumerate(unique_pids):
        pid_str = str(pid)

        if pid_str in test_pids:
            probe_split[i] = 2

        elif pid_str in validation_pids:
            probe_split[i] = 1

        elif pid_str in train_pids:
            probe_split[i] = 0

        else:
            # PID exists in unit-level data but not in the spatial split.
            # It may be used for training, but never validation/test.
            probe_split[i] = 0
            extra_train_pids.append(pid_str)

    unit_split = probe_split[probe_index]

    # ------------------------------------------------------------------
    # Hard safety checks.
    # ------------------------------------------------------------------
    unit_pid_to_split = {
        str(pid): int(split_value)
        for pid, split_value in zip(
            unique_pids,
            probe_split,
        )
    }

    # Every spatial test PID that exists in the unit dataset MUST be test.
    incorrectly_assigned_test = [
        pid
        for pid in test_pids
        if pid in unit_pid_to_split
        and unit_pid_to_split[pid] != 2
    ]
    if incorrectly_assigned_test:
        raise RuntimeError(
            "FATAL: a spatial-encoder test PID was assigned to a "
            "non-test unit split. First offending PIDs: "
            f"{incorrectly_assigned_test[:10]}"
        )

    # Unknown PIDs must never enter validation or test.
    known_pids = train_pids | validation_pids | test_pids
    wrongly_held_out_unknown = [
        str(pid)
        for pid, split_value in zip(
            unique_pids,
            probe_split,
        )
        if str(pid) not in known_pids
        and int(split_value) != 0
    ]
    if wrongly_held_out_unknown:
        raise RuntimeError(
            "FATAL: PIDs absent from the spatial split were assigned "
            "outside training. First offending PIDs: "
            f"{wrongly_held_out_unknown[:10]}"
        )

    print(
        "[unit split] using authoritative spatial PID split "
        "with unit-only PIDs added to training."
    )
    print(
        "[unit split] "
        f"train={int(np.sum(probe_split == 0))} PIDs, "
        f"validation={int(np.sum(probe_split == 1))} PIDs, "
        f"test={int(np.sum(probe_split == 2))} PIDs"
    )

    if extra_train_pids:
        print(
            "[unit split] added "
            f"{len(extra_train_pids)} unit-only PIDs to TRAIN "
            "(validation/test unchanged)."
        )
        print(
            "[unit split] first added training PIDs:",
            sorted(extra_train_pids)[:10],
        )

    return (
        unit_split.astype(np.int8),
        unique_pids,
        probe_split.astype(np.int8),
    )


def assert_strict_probe_split(
    pids: np.ndarray,
    split: np.ndarray,
    *,
    output_path: Path | None = None,
) -> Dict[str, object]:
    split_names = ("train", "validation", "test")
    pids = np.asarray(pids).astype(str)
    pid_sets = [set(pids[split == i].tolist()) for i in range(3)]
    overlaps = {
        "train_validation": sorted(pid_sets[0] & pid_sets[1]),
        "train_test": sorted(pid_sets[0] & pid_sets[2]),
        "validation_test": sorted(pid_sets[1] & pid_sets[2]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"FATAL split leakage: {overlaps}")

    audit: Dict[str, object] = {
        "strict_group_variable": "pid",
        "no_pid_overlap": True,
        "overlaps": overlaps,
        "splits": {
            split_names[i]: {
                "n_units": int((split == i).sum()),
                "n_probes": int(len(pid_sets[i])),
                "pids": sorted(pid_sets[i]),
            }
            for i in range(3)
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def make_voxel_ids(xyz_m: np.ndarray, voxel_size_um: float) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.floor(np.asarray(xyz_m, dtype=np.float64) * 1e6 / voxel_size_um).astype(np.int64)
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int64), unique


def prepare_data(
    waveforms,
    acgs,
    context,
    xyz,
    pids,
    cfg: Config,
    *,
    split_manifest: Mapping[str, object],
) -> PreparedData:
    n = len(waveforms)
    if not (len(acgs) == len(context) == len(xyz) == len(pids) == n):
        raise ValueError("All arrays must have the same first dimension")
    if tuple(waveforms.shape[1:]) != tuple(cfg.waveform_shape):
        raise ValueError(f"Expected waveform shape {cfg.waveform_shape}, got {waveforms.shape[1:]}")
    if tuple(acgs.shape[1:]) != tuple(cfg.acg_shape):
        raise ValueError(f"Expected ACG shape {cfg.acg_shape}, got {acgs.shape[1:]}")

    xyz_m = np.asarray(xyz, dtype=np.float32).copy()
    if not cfg.xyz_in_meters:
        xyz_m = xyz_m / 1e6
    if cfg.mirror_x_to_left_hemisphere:
        xyz_m[:, 0] = -np.abs(xyz_m[:, 0])

    pids = np.asarray(pids).astype(str)
    split, unique_pids, probe_split = split_probes_from_manifest(
        pids,
        split_manifest,
    )
    _, probe_index = np.unique(pids, return_inverse=True)
    assert_strict_probe_split(pids, split)
    voxel_id, voxel_key = make_voxel_ids(xyz_m, cfg.voxel_size_um)

    context = np.asarray(context, dtype=np.float32).copy()
    if cfg.mirror_x_to_left_hemisphere:
        if context.shape[1] < 3:
            raise ValueError("Expected context to begin with x, y, z coordinates")
        context[:, 0] = xyz_m[:, 0]

    train = split == 0
    mean = context[train].mean(0, keepdims=True)
    std = context[train].std(0, keepdims=True)
    unique_counts = np.array([len(np.unique(context[train, j])) for j in range(context.shape[1])])
    continuous = unique_counts > 4
    mean[:, ~continuous] = 0.0
    std[:, ~continuous] = 1.0
    std = np.maximum(std, 1e-6)
    context = (context - mean) / std

    return PreparedData(
        waveforms=normalize_waveforms(waveforms),
        acgs=normalize_acgs(acgs),
        context=context.astype(np.float32),
        xyz_m=xyz_m,
        pids=pids,
        probe_index=probe_index.astype(np.int64),
        unique_pids=unique_pids,
        probe_split=probe_split,
        split=split,
        voxel_id=voxel_id,
        voxel_key=voxel_key,
        context_mean=mean.astype(np.float32),
        context_std=std.astype(np.float32),
    )


class UnitDataset(Dataset):
    def __init__(self, data: PreparedData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        i = int(self.indices[item])
        return {
            "index": torch.tensor(i, dtype=torch.long),
            "waveform": torch.from_numpy(self.data.waveforms[i]),
            "acg": torch.from_numpy(self.data.acgs[i]),
        }


def split_indices(data: PreparedData):
    return tuple(np.flatnonzero(data.split == i) for i in range(3))
