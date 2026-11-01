"""The reference of the insertion QC: what the other insertions record in the same brain region.

The unit of comparison is the probe, not the channel (channels of one probe are not independent):

1. A probe's channels are reduced to 100 µm depth bins (:func:`bin_probe`, the median of the channels
   with a signal), each feature in robust units over all bins (:func:`bin_dataset`). Every bin gets
   its region from its position, in the finest parcellation first (:data:`PARCELLATIONS`: the Allen
   regions, with cortical layers and hippocampal strata, then Beryl and Cosmos).
2. For each probe and each region with at least ``MIN_BINS`` bins, the median of every feature over
   those bins is one *probe-region value*.
3. The reference of a region is the median and the MAD of the probe-region values of the *other*
   probes (the probe itself is always left out), with at least ``MIN_PROBES`` of them
   (:meth:`RegionReference.stats`). A region without a reference falls back to the next parcellation.
4. A bin's residual is its distance from the reference of its region, in MADs of that feature in
   that region (:meth:`RegionReference.residuals`), so a probe is judged by how much recordings of
   the same region differ from one another. Each feature's MAD is floored at ``MAD_FLOOR`` times
   its typical MAD over the regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

BAD_CHANNEL_LABELS = (1, 2)  # dead, noisy
BIN_CHANNELS = 10  # channels of a depth bin (100 µm)
PARCELLATIONS = ("Allen", "Beryl", "Cosmos")  # the finest first
MIN_BINS = 2  # bins of a probe in a region for a probe-region value
MIN_PROBES = 8  # other probes for the reference of a region
MAD_FLOOR = 0.5  # of a feature's typical MAD over the regions
MAD_TO_SD = 1.4826


def signal_rows(
    features: np.ndarray, labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """``[C]`` rows with finite, not all-zero features and not labelled dead or noisy."""
    x = np.asarray(features, dtype=float)
    ok = np.isfinite(x).all(axis=1) & ~np.all(np.nan_to_num(x) == 0.0, axis=1)
    if labels is not None:
        ok &= ~np.isin(np.asarray(labels), BAD_CHANNEL_LABELS)
    return ok


def region_labels(xyz: np.ndarray, brain_atlas, mapping: str) -> np.ndarray:
    """``[...]`` region id of each position in the parcellation ``mapping`` (0 outside the brain)."""
    xyz = np.asarray(xyz, dtype=float)
    ok = np.isfinite(xyz).all(axis=-1)
    out = np.zeros(xyz.shape[:-1], dtype=np.int64)
    if ok.any():
        out[ok] = brain_atlas.get_labels(xyz[ok], mapping=mapping, mode="clip")
    return out


def parcel_labels(xyz: np.ndarray, brain_atlas) -> dict:
    """``{parcellation: [...]}``, the region of each position in every :data:`PARCELLATIONS`."""
    return {m: region_labels(xyz, brain_atlas, m) for m in PARCELLATIONS}


def bin_probe(
    features: np.ndarray, labels: np.ndarray, xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``[B, F]`` median over the channels with a signal of each depth bin (NaN with fewer than half
    of them) and ``[B, 3]`` its mean position. Rows are in row order, top of the probe first."""
    features, xyz = np.asarray(features), np.asarray(xyz, dtype=float)
    rows = signal_rows(features, labels)
    n = len(features) // BIN_CHANNELS
    values = np.full((n, features.shape[1]), np.nan, dtype=np.float32)
    positions = np.full((n, 3), np.nan)
    for b in range(n):
        span = np.arange(b * BIN_CHANNELS, (b + 1) * BIN_CHANNELS)
        positions[b] = np.nanmean(xyz[span], axis=0)
        span = span[rows[span]]
        if len(span) >= BIN_CHANNELS // 2:
            values[b] = np.nanmedian(features[span], axis=0)
    return values, positions


@dataclass
class BinnedDataset:
    """Every insertion of a release in depth bins.

    Attributes:
        pids: ``[N]`` insertion ids.
        z: ``[N, B, F]`` features in robust units.
        xyz: ``[N, B, 3]`` positions of the bins (m).
        centre, scale: ``[F]`` median and robust SD of each feature over all bins.
    """

    pids: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    centre: np.ndarray
    scale: np.ndarray

    def standardize(self, values: np.ndarray) -> np.ndarray:
        """Binned features (:func:`bin_probe`) in the units of ``z``."""
        return (values - self.centre) / self.scale


def bin_dataset(ds) -> BinnedDataset:
    """The insertions of a :class:`~.data.QCDataset` in depth bins."""
    binned = [
        bin_probe(ds.features[i], ds.channel_labels[i], ds.human_xyz[i])
        for i in range(len(ds.pids))
    ]
    values = np.stack([b[0] for b in binned])
    xyz = np.stack([b[1] for b in binned])
    centre = np.nanmedian(values, axis=(0, 1))
    scale = MAD_TO_SD * np.nanmedian(np.abs(values - centre), axis=(0, 1))
    scale = np.where(scale > 0, scale, 1.0)
    return BinnedDataset(ds.pids, (values - centre) / scale, xyz, centre, scale)


class RegionReference:
    """What the probes record in each region.

    Args:
        z: ``[N, B, F]`` binned features in robust units.
        labels: ``{parcellation: [N, B]}`` region of every bin at the recorded positions.
    """

    def __init__(self, z: np.ndarray, labels: dict):
        self.tables = {}
        for name in PARCELLATIONS:
            table = {}
            for i in range(len(z)):
                good = np.isfinite(z[i]).all(axis=1)
                for region in np.unique(labels[name][i]):
                    bins = good & (labels[name][i] == region)
                    if region != 0 and bins.sum() >= MIN_BINS:
                        probes, values = table.setdefault(int(region), ([], []))
                        probes.append(i)
                        values.append(np.median(z[i][bins], axis=0))
            self.tables[name] = {
                r: (np.array(p), np.array(v, dtype=np.float32))
                for r, (p, v) in table.items()
            }
        self._stats = {}
        mads = [
            s[1]
            for r in self.tables[PARCELLATIONS[0]]
            if (s := self.stats(PARCELLATIONS[0], r)) is not None
        ]
        self.floor = (
            MAD_FLOOR * np.median(mads, axis=0) if mads else np.ones(z.shape[2])
        )

    def stats(self, parcellation: str, region: int, exclude=()) -> Optional[np.ndarray]:
        """``[2, F]`` median and robust SD of the probe-region values of ``region`` over the probes
        not in ``exclude``, or None with fewer than ``MIN_PROBES`` of them."""
        table = self.tables[parcellation].get(region)
        if table is None:
            return None
        probes, values = table
        inside = tuple(int(p) for p in exclude if p in probes)
        key = (parcellation, region, inside)
        if key not in self._stats:
            keep = values[~np.isin(probes, inside)]
            self._stats[key] = None
            if len(keep) >= MIN_PROBES:
                med = np.median(keep, axis=0)
                mad = MAD_TO_SD * np.median(np.abs(keep - med), axis=0)
                self._stats[key] = np.stack([med, mad])
        return self._stats[key]

    def _reference(self, regions, exclude) -> Optional[np.ndarray]:
        """The stats of the first parcellation with a reference for a bin's ``regions``."""
        for name, region in zip(PARCELLATIONS, regions):
            s = self.stats(name, int(region), exclude) if region != 0 else None
            if s is not None:
                return s
        return None

    def expected(self, labels: dict, exclude=()) -> tuple[np.ndarray, np.ndarray]:
        """``[B, F]`` what is expected of bins in the regions ``labels`` (``{parcellation: [B]}``):
        the median of their region and its robust SD, floored; NaN where no region has a reference.
        ``exclude``: the probes left out of the reference."""
        keys = np.stack([labels[name] for name in PARCELLATIONS], axis=1)
        median = np.full((len(keys), len(self.floor)), np.nan, dtype=np.float32)
        spread = median.copy()
        for regions in np.unique(keys, axis=0):
            s = self._reference(regions, exclude)
            if s is not None:
                bins = (keys == regions).all(axis=1)
                median[bins], spread[bins] = s[0], np.maximum(s[1], self.floor)
        return median, spread

    def residuals(self, z: np.ndarray, labels: dict, exclude=()) -> np.ndarray:
        """``[B, F]`` residuals of binned features ``z`` whose bins are in the regions ``labels``,
        in MADs of each feature in each region (:meth:`expected`)."""
        median, spread = self.expected(labels, exclude)
        return (z - median) / spread
