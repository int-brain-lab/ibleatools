"""Recording data for offline alignment runs: channel features, reference positions, histology.

Everything is cached under the data directory (``EPHYS_ATLAS_DATA_DIR``, default
``~/ephys-atlas/data``), outside any repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import (
    extend_trace_to_brain,
    infer_metres,
    order_top_to_bottom,
    resample_curve,
    valid_xyz_mask,
)
from .models import VINTAGE


def default_data_dir() -> Path:
    from ephysatlas.unit_level_encoder.config import DEFAULT_DATA_DIR

    return Path(DEFAULT_DATA_DIR)


@dataclass
class ChannelDataset:
    """The channel features of every insertion of a feature release, in row order (top first).

    Attributes:
        pids: ``[N]`` insertion ids.
        features: ``[N, C, F]`` recorded channel features (feature units).
        human_xyz: ``[N, C, 3]`` channel positions from the (human) histology alignment.
        planned_xyz: ``[N, C, 3]`` channel positions along the planned trajectory.
    """

    pids: np.ndarray
    features: np.ndarray
    human_xyz: np.ndarray
    planned_xyz: np.ndarray

    @classmethod
    def load(
        cls,
        vintage: str = VINTAGE,
        *,
        project: str = "ea_active",
        agg: str = "agg_full",
        data_dir: Optional[Path] = None,
    ) -> "ChannelDataset":
        """Load the release's features (downloaded from IBL S3 on first use; ONE credentials).

        Insertions on the misaligned list (``ephysatlas.fixtures.misaligned_pids``) are left
        out, exactly as for model training.
        """
        from ephysatlas.spatial_encoder.utils import LoadInsertionData

        pids, features, human, planned = LoadInsertionData(
            project=project, agg=agg, VINTAGE=vintage, path_data=data_dir or default_data_dir()
        )
        return cls(
            pids=np.asarray(pids).astype(str),
            features=np.asarray(features, dtype=np.float32),
            human_xyz=np.asarray(human, dtype=np.float32),
            planned_xyz=np.asarray(planned, dtype=np.float32),
        )

    def index(self, pid: str) -> int:
        idx = np.flatnonzero(self.pids == str(pid))
        if not len(idx):
            raise KeyError(f"{pid} is not in this dataset")
        return int(idx[0])

    def probe(self, pid: str) -> dict:
        """``features``, ``human_xyz`` and ``planned_xyz`` of one insertion."""
        i = self.index(pid)
        return dict(
            pid=str(pid),
            features=self.features[i],
            human_xyz=self.human_xyz[i],
            planned_xyz=self.planned_xyz[i],
        )


_TABLE_CACHE = {}


def load_probe(
    pid: str,
    vintage: str = VINTAGE,
    *,
    project: str = "ea_active",
    agg: str = "agg_full",
    data_dir: Optional[Path] = None,
) -> Optional[dict]:
    """One insertion's channels from the feature release, in row order (top of the probe first).

    Unlike :class:`ChannelDataset` this keeps insertions on the misaligned list -- those are the
    ones worth realigning. The release table is downloaded on first use (ONE credentials) and
    kept in memory.

    Returns:
        dict | None: ``features`` [C, F] (feature units, the release's feature order),
        ``human_xyz`` and ``planned_xyz`` [C, 3] (m), ``axial_um`` [C], ``channel`` [C];
        None when the insertion is not in the release.
    """
    from ephysatlas.data import download_tables, read_features_from_disk
    from ephysatlas.spatial_encoder.utils import FEATURE_LIST

    key = (vintage, project, agg)
    if key not in _TABLE_CACHE:
        root = Path(data_dir or default_data_dir())
        path = root / project / vintage / agg
        if not any(path.glob("raw_ephys_features*.pqt")):
            from one.api import ONE

            one = ONE(base_url="https://alyx.internationalbrainlab.org")
            path = download_tables(root, label=vintage, project=project, agg_level=agg, one=one)
        _TABLE_CACHE[key] = read_features_from_disk(path, strict=False)
    df = _TABLE_CACHE[key]
    if str(pid) not in df.index.get_level_values("pid"):
        return None
    d = df.loc[str(pid)].sort_index().iloc[::-1]  # highest channel (top of the probe) first
    features = d[list(FEATURE_LIST)].to_numpy(dtype=np.float32).copy()
    features[~np.isfinite(features)] = 0.0
    return dict(
        pid=str(pid),
        features=features,
        human_xyz=d[["x", "y", "z"]].to_numpy(dtype=np.float32),
        planned_xyz=d[["x_target", "y_target", "z_target"]].to_numpy(dtype=np.float32),
        axial_um=d["axial_um"].to_numpy(dtype=float),
        channel=d.index.to_numpy(),
    )


class HistologyTraces:
    """Reconstructed histology tracks (Alyx ``xyz_picks``), cached in a JSON file.

    Args:
        cache_path (Path, optional): Cache file; defaults to
            ``<data_dir>/alignment/histology_picks.json``.
        one: A ``one.api.ONE`` instance for fetching picks missing from the cache (created on
            first use).
    """

    def __init__(self, cache_path: Optional[Path] = None, one=None):
        self.cache_path = Path(
            cache_path or default_data_dir() / "alignment" / "histology_picks.json"
        )
        self._one = one
        self._picks = {}
        if self.cache_path.exists():
            self._picks = json.loads(self.cache_path.read_text(encoding="utf-8"))

    @property
    def one(self):
        if self._one is None:
            from one.api import ONE

            self._one = ONE(base_url="https://alyx.internationalbrainlab.org")
        return self._one

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._picks), encoding="utf-8")

    def picks(self, pid: str) -> Optional[np.ndarray]:
        """``[n, 3]`` histology picks (m), or None when the insertion has none."""
        pid = str(pid)
        if pid not in self._picks:
            recs = self.one.alyx.rest("insertions", "list", id=pid)
            picks = ((recs[0].get("json") or {}).get("xyz_picks")) if recs else None
            self._picks[pid] = picks
            self._save()
        picks = self._picks[pid]
        if picks is None:
            return None
        picks = infer_metres(np.asarray(picks, dtype=float))
        picks = picks[valid_xyz_mask(picks)]
        return picks if len(picks) >= 2 else None

    def trace(self, pid: str, brain_atlas, step_um: float = 10.0) -> Optional[np.ndarray]:
        """The histology trace of an insertion: picks ordered dorsal to ventral, resampled every
        ``step_um`` and extended to the brain boundary at both ends (None without picks)."""
        picks = self.picks(pid)
        if picks is None:
            return None
        trace = resample_curve(order_top_to_bottom(picks), step_um=step_um)
        return extend_trace_to_brain(trace, brain_atlas)


def split_pids(channel_model, split: str = "test") -> list[str]:
    """The insertions of one split of the channel model's release (``train``/``validation``/``test``)."""
    manifest = channel_model.split()
    key = {"train": "train_pids", "validation": "validation_pids", "test": "test_pids"}[split]
    return sorted(str(p) for p in manifest[key])
