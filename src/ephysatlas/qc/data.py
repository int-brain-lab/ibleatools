"""Inputs of the insertion quality control: every insertion of a feature release (its recording
and its human-aligned positions) and the session of each. The histology auto-alignment
(:func:`load_auto_alignment`) is no input of the QC: the example figures may draw it, as a
diagnostic only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS


def default_results_dir(vintage: str) -> Path:
    from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

    return Path(DEFAULT_RESULTS_DIR) / "qc" / vintage


def default_data_dir() -> Path:
    from ephysatlas.unit_level_encoder.config import DEFAULT_DATA_DIR

    return Path(DEFAULT_DATA_DIR)


@dataclass
class QCDataset:
    """Every insertion of a feature release, channel arrays in row order (top of the probe first).

    Attributes:
        pids: ``[N]`` insertion ids.
        features: ``[N, C, F]`` channel features (feature units), ``feature_names`` order.
        human_xyz, planned_xyz: ``[N, C, 3]`` human-aligned and planned channel positions (m).
        channel_labels: ``[N, C]`` bad-channel labels (0 good, 1 dead, 2 noisy, 3 outside).
        probe_model: ``[N]`` probe model (``3A``, ``PRB_1_4_0480_1``, ...).
    """

    pids: np.ndarray
    features: np.ndarray
    human_xyz: np.ndarray
    planned_xyz: np.ndarray
    channel_labels: np.ndarray
    probe_model: np.ndarray
    feature_names: list

    @property
    def probe_family(self) -> np.ndarray:
        """``3A`` or ``NP1`` per insertion."""
        return np.where(self.probe_model == "3A", "3A", "NP1")

    def component_index(self, component: str) -> np.ndarray:
        names = FEATURE_GROUPS[component]
        return np.array([self.feature_names.index(f) for f in names])

    def index(self, pid: str) -> int:
        return int(np.flatnonzero(self.pids == str(pid))[0])

    @classmethod
    def load(cls, vintage: str, data_dir: Optional[Path] = None) -> "QCDataset":
        """Load every insertion of ``vintage`` (``ea_active``, ``agg_full``), none excluded."""
        from ephysatlas.alignment.data import ChannelDataset
        from ephysatlas.spatial_encoder.utils import FEATURE_LIST

        data_dir = Path(data_dir or default_data_dir())
        ds = ChannelDataset.load(vintage, data_dir=data_dir, exclude_pids=[])
        channels = pd.read_parquet(
            data_dir / "ea_active" / vintage / "agg_full" / "channels.pqt"
        )
        features = pd.read_parquet(
            data_dir / "ea_active" / vintage / "agg_full" / "raw_ephys_features.pqt",
            columns=["channel_labels"],
        )
        channels = channels.join(features, how="left")
        labels = np.zeros(ds.human_xyz.shape[:2], dtype=np.int8)
        models = []
        for i, pid in enumerate(ds.pids):
            ch = channels.loc[pid].sort_index(ascending=False)  # row order: top first
            xyz = ch[["x", "y", "z"]].to_numpy(float)
            if not np.allclose(xyz, ds.human_xyz[i], atol=1e-7, equal_nan=True):
                raise ValueError(
                    f"{pid}: channel order differs from the feature arrays"
                )
            labels[i] = np.nan_to_num(ch["channel_labels"].to_numpy(float)).astype(
                np.int8
            )
            models.append(str(ch["probe_model"].iloc[0]))
        return cls(
            pids=ds.pids.astype(str),
            features=ds.features.astype(np.float32),
            human_xyz=ds.human_xyz.astype(np.float64),
            planned_xyz=ds.planned_xyz.astype(np.float64),
            channel_labels=labels,
            probe_model=np.asarray(models),
            feature_names=list(FEATURE_LIST),
        )


def insertion_sessions(
    pids, cache: Optional[Path] = None, one=None, chunk: int = 50
) -> dict:
    """``{pid: eid}``, the session of each insertion (for the probe swap of two-probe sessions),
    from Alyx, cached in ``cache`` (a CSV with columns ``pid`` and ``eid``; only the insertions
    it lacks are queried, ``chunk`` at a time)."""
    pids = [str(p) for p in pids]
    known = {}
    if cache is not None and Path(cache).exists():
        table = pd.read_csv(cache, dtype=str)
        known = dict(zip(table["pid"], table["eid"]))
    todo = [p for p in pids if p not in known]
    if todo:
        if one is None:
            from one.api import ONE

            one = ONE(base_url="https://alyx.internationalbrainlab.org", silent=True)
        for k in range(0, len(todo), chunk):
            part = todo[k : k + chunk]
            for rec in one.alyx.rest(
                "insertions", "list", django=f"id__in,{part}", no_cache=True
            ):
                known[str(rec["id"])] = str(rec["session"])
        if cache is not None:
            table = pd.DataFrame(sorted(known.items()), columns=["pid", "eid"])
            table.to_csv(cache, index=False)
    return {p: known[p] for p in pids if p in known}


def load_auto_alignment(results_dir: Path, pid: str):
    """The histology auto-alignment of ``pid`` (an :class:`AlignmentResult`), or None. Not used
    by any QC decision; the example figures may draw it as a diagnostic."""
    from ephysatlas.alignment.result import AlignmentResult

    path = Path(results_dir) / "results" / "histology" / f"{pid}.npz"
    return AlignmentResult.load(path) if path.exists() else None
