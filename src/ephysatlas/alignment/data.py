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

from .geometry import histology_track, infer_metres, valid_xyz_mask
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
        exclude_pids=None,
    ) -> "ChannelDataset":
        """Load the release's features (downloaded from the public IBL S3 bucket on first use).

        Insertions the automatic QC marks bad (``ephysatlas.fixtures.pid_qc_filter``) are left
        out, exactly as for model training, unless ``exclude_pids`` lists others (an empty list
        keeps every insertion).
        """
        from ephysatlas.spatial_encoder.utils import LoadInsertionData

        pids, features, human, planned = LoadInsertionData(
            project=project,
            agg=agg,
            VINTAGE=vintage,
            path_data=data_dir or default_data_dir(),
            exclude_pids=exclude_pids,
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
    ones worth realigning. The release table is downloaded from the public S3 bucket on first
    use and kept in memory.

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
            path = download_tables(root, label=vintage, project=project, agg_level=agg)
        _TABLE_CACHE[key] = read_features_from_disk(path, strict=False)
    df = _TABLE_CACHE[key]
    if str(pid) not in df.index.get_level_values("pid"):
        return None
    d = (
        df.loc[str(pid)].sort_index().iloc[::-1]
    )  # highest channel (top of the probe) first
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
    """Reconstructed histology tracks (Alyx ``xyz_picks``) of the atlas insertions.

    The tracks of the published insertions are read from the public S3 bucket
    (:func:`ephysatlas.data.download_histology_picks`, downloaded on the first insertion missing
    from the local cache). Insertions that are not published are fetched from Alyx only when
    ``one`` is given, and kept in the local cache.

    Args:
        cache_path (Path, optional): Local cache of the tracks fetched from Alyx; defaults to
            ``<data_dir>/alignment/histology_picks.json``.
        one (one.api.ONE, optional): A logged-in ONE instance, to fetch the tracks of insertions
            that are not published. Without it, those insertions have no track.
        project (str, optional): Project of the published tracks.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        one=None,
        project: str = "ibl_neuropixel_brainwide_01",
    ):
        self.cache_path = Path(
            cache_path or default_data_dir() / "alignment" / "histology_picks.json"
        )
        self._one = one
        self._project = project
        self._published = None
        self._picks = {}
        if self.cache_path.exists():
            self._picks = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._picks), encoding="utf-8")

    def _published_picks(self) -> dict:
        """The published tracks ``{pid: picks}``, downloaded once."""
        if self._published is None:
            from ephysatlas.data import download_histology_picks

            path = download_histology_picks(default_data_dir(), project=self._project)
            self._published = json.loads(path.read_text(encoding="utf-8"))
        return self._published

    def picks(self, pid: str) -> Optional[np.ndarray]:
        """``[n, 3]`` histology picks (m), or None when the insertion has none."""
        pid = str(pid)
        if pid not in self._picks:
            published = self._published_picks()
            if pid in published:
                self._picks[pid] = published[pid]
            elif self._one is not None:
                recs = self._one.alyx.rest("insertions", "list", id=pid)
                self._picks[pid] = (
                    ((recs[0].get("json") or {}).get("xyz_picks")) if recs else None
                )
                self._save()
            else:
                return None
        picks = self._picks[pid]
        if picks is None:
            return None
        picks = infer_metres(np.asarray(picks, dtype=float))
        picks = picks[valid_xyz_mask(picks)]
        return picks if len(picks) >= 2 else None

    def trace(self, pid: str, brain_atlas) -> Optional[np.ndarray]:
        """The histology track of an insertion, top to bottom (None without picks); see
        :func:`~ephysatlas.alignment.geometry.histology_track`."""
        picks = self.picks(pid)
        return None if picks is None else histology_track(picks, brain_atlas)


def split_pids(channel_model, split: str = "test") -> list[str]:
    """The insertions of one split of the channel model's release (``train``/``validation``/``test``)."""
    manifest = channel_model.split()
    key = {"train": "train_pids", "validation": "validation_pids", "test": "test_pids"}[
        split
    ]
    return sorted(str(p) for p in manifest[key])
