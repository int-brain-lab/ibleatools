"""Region outliers: is a probe's recording unusual for the brain regions it is said to be in?

The unit of comparison is the probe, not the channel (channels of one probe are not independent):

1. Each channel row gets its Beryl and Cosmos region from its position. Only *core* rows are used:
   rows with a signal, not labelled dead or noisy, and more than ``core_margin_um`` (axial) from a
   region border, because the alignment can be a few hundred µm off and rows near a border may be
   in the next region.
2. For each probe and each region with at least ``min_core_channels`` core rows, the median of
   every feature over those rows is one *probe-region value*.
3. The reference of a region is the probe-region values of the *other* reference probes (the probe
   itself is always left out): their median and MAD. A Beryl region with fewer than
   ``min_reference_probes`` other probes falls back to the reference of its Cosmos region.
4. ``z = (value - reference median) / (1.4826 MAD)``, with the scale floored at a fraction of the
   feature's spread over all probe-region values and ``|z|`` capped at ``z_clip``.
5. A component score (LF, AP, waveform) is the mean ``|z|`` over the component's features,
   averaged over the probe's regions weighted by their core rows (:func:`component_scores`), or,
   robust to one noisy feature or one odd region, the median over the features and the weighted
   median over the regions (:func:`robust_component_scores`). Two scores are kept: ``beryl``
   (Beryl with the Cosmos fallback, sensitive) and ``cosmos`` (Cosmos only, stable).

The QC decisions use a finer reference, local to each position (:class:`~.modelfree.LocalReference`):
brain regions are too coarse to see a probe a few hundred µm off. The region reference gives the
``h_rb_*`` band scores and the region medians drawn in the example figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ephysatlas.alignment.geometry import np1_axial_um_in_row_order, region_ids
from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS

LEVELS = ("Beryl", "Cosmos")
BAD_CHANNEL_LABELS = (1, 2)  # dead, noisy


@dataclass
class RegionSettings:
    core_margin_um: float = 100.0
    min_core_channels: int = 10
    min_reference_probes: int = 10
    z_clip: float = 10.0
    scale_floor_fraction: float = 0.1


def signal_rows(
    features: np.ndarray, labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """``[C]`` rows with finite, not all-zero features and not labelled dead or noisy."""
    x = np.asarray(features, dtype=float)
    ok = np.isfinite(x).all(axis=1) & ~np.all(np.nan_to_num(x) == 0.0, axis=1)
    if labels is not None:
        ok &= ~np.isin(np.asarray(labels), BAD_CHANNEL_LABELS)
    return ok


def distance_to_border_um(region: np.ndarray, axial_um: np.ndarray) -> np.ndarray:
    """``[C]`` axial distance of each row to the nearest change of ``region`` along the probe
    (rows in probe order); inf when the probe has a single region."""
    region = np.asarray(region)
    change = np.flatnonzero(region[1:] != region[:-1])
    if not len(change):
        return np.full(len(region), np.inf)
    borders = 0.5 * (axial_um[change] + axial_um[change + 1])
    return np.min(np.abs(axial_um[:, None] - borders[None, :]), axis=1)


def probe_region_values(
    features: np.ndarray,
    xyz: np.ndarray,
    labels: Optional[np.ndarray],
    brain_atlas,
    settings: RegionSettings,
    *,
    probe: int = 0,
) -> pd.DataFrame:
    """Probe-region values of one probe at positions ``xyz`` ([C, 3] m, row order).

    Returns:
        pd.DataFrame: one row per (level, region) with ``probe``, ``level``, ``region`` (atlas
        region index), ``cosmos`` (the Cosmos region of most of its core rows), ``n_core`` and
        ``f0..f{F-1}`` (feature medians).
    """
    features = np.asarray(features, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    n_rows, n_feat = features.shape
    finite = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0.0, axis=1)
    usable = signal_rows(features, labels) & finite
    axial = np1_axial_um_in_row_order(n_rows)
    labels_by_level = {}
    for level in LEVELS:
        lab = np.zeros(n_rows, dtype=np.int64)
        if finite.any():
            lab[finite] = region_ids(brain_atlas, xyz[finite], level)
        labels_by_level[level] = lab
    out = []
    for level in LEVELS:
        lab = labels_by_level[level]
        core = (
            usable
            & (lab > 0)
            & (distance_to_border_um(lab, axial) > settings.core_margin_um)
        )
        for region in np.unique(lab[core]):
            rows = np.flatnonzero(core & (lab == region))
            if len(rows) < settings.min_core_channels:
                continue
            cosmos = np.bincount(labels_by_level["Cosmos"][rows]).argmax()
            out.append(
                [probe, level, int(region), int(cosmos), len(rows)]
                + list(np.nanmedian(features[rows], axis=0))
            )
    columns = ["probe", "level", "region", "cosmos", "n_core"] + [
        f"f{i}" for i in range(n_feat)
    ]
    return pd.DataFrame(out, columns=columns)


def row_regions(xyz: np.ndarray, brain_atlas) -> dict:
    """``{"Beryl": [C], "Cosmos": [C]}`` region index of each row (0 outside the brain)."""
    xyz = np.asarray(xyz, dtype=float)
    finite = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0.0, axis=1)
    out = {}
    for level in LEVELS:
        lab = np.zeros(len(xyz), dtype=np.int64)
        if finite.any():
            lab[finite] = region_ids(brain_atlas, xyz[finite], level)
        out[level] = lab
    return out


class RegionReference:
    """Region references built from the probe-region values of the reference probes.

    Args:
        values: :func:`probe_region_values` rows of many probes (usually at their human positions).
        reference_probes: Probe numbers allowed in the references (e.g. not flagged in round 1).
        feature_names: The feature names of ``f0..``.
        settings: :class:`RegionSettings`.
        family: Optional ``{probe: family}``; when given, a probe is compared with probes of its
            own family where the region has enough of them (else with all probes).
    """

    def __init__(
        self,
        values: pd.DataFrame,
        reference_probes,
        feature_names: list,
        settings: RegionSettings,
        family: Optional[dict] = None,
    ):
        self.settings = settings
        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.family = family
        ref = values[values["probe"].isin(set(reference_probes))]
        self.fcols = [f"f{i}" for i in range(self.n_features)]
        self.groups = {}
        for (level, region), g in ref.groupby(["level", "region"]):
            self.groups[(level, int(region))] = (
                g["probe"].to_numpy(),
                g[self.fcols].to_numpy(float),
            )
        # Scale floor: a fraction of each feature's robust spread over all probe-region values.
        self.floor = {}
        for level in LEVELS:
            v = ref.loc[ref["level"] == level, self.fcols].to_numpy(float)
            spread = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=0)), axis=0)
            self.floor[level] = settings.scale_floor_fraction * np.where(
                np.isfinite(spread) & (spread > 0), spread, 1.0
            )

    def stats(self, level: str, region: int, probe: int):
        """``(median [F], scale [F], n)`` of a region's reference without ``probe``, or None
        when fewer than ``min_reference_probes`` remain."""
        key = (level, int(region))
        if key not in self.groups:
            return None
        probes, v = self.groups[key]
        keep = probes != probe
        if self.family is not None:
            fam = self.family.get(probe)
            same = keep & np.array([self.family.get(p) == fam for p in probes])
            if same.sum() >= self.settings.min_reference_probes:
                keep = same
        n = int(keep.sum())
        if n < self.settings.min_reference_probes:
            return None
        v = v[keep]
        med = np.nanmedian(v, axis=0)
        scale = np.maximum(
            1.4826 * np.nanmedian(np.abs(v - med), axis=0), self.floor[level]
        )
        return med, scale, n

    def zscores(self, values: pd.DataFrame) -> pd.DataFrame:
        """Signed z of every probe-region value, each probe left out of its own reference.

        Returns:
            pd.DataFrame: the rows of ``values`` that could be scored, with ``source``
            (``Beryl``, ``Cosmos`` or ``Cosmos fallback``), ``n_ref`` and ``z0..`` columns.
        """
        out = []
        for row in values.itertuples(index=False):
            d = row._asdict()
            level, probe = d["level"], d["probe"]
            st = self.stats(level, d["region"], probe)
            source = level
            if st is None and level == "Beryl":
                st = self.stats("Cosmos", d["cosmos"], probe)
                source = "Cosmos fallback"
            if st is None:
                continue
            med, scale, n = st
            x = np.array([d[c] for c in self.fcols], dtype=float)
            z = np.clip((x - med) / scale, -self.settings.z_clip, self.settings.z_clip)
            out.append(
                dict(
                    probe=probe,
                    level=level,
                    region=d["region"],
                    cosmos=d["cosmos"],
                    n_core=d["n_core"],
                    source=source,
                    n_ref=n,
                    **{f"z{i}": z[i] for i in range(self.n_features)},
                )
            )
        return pd.DataFrame(out)


def component_scores(
    z: pd.DataFrame, feature_names: list, prefix: str = ""
) -> pd.DataFrame:
    """Per probe: ``<prefix><level>_<component>`` = weighted (by core rows) mean over its regions
    of the mean ``|z|`` over the component's features; ``<prefix><level>_n_regions`` and the
    fraction of core rows scored on a Beryl reference (``<prefix>beryl_coverage``)."""
    if z.empty:
        return pd.DataFrame()
    idx = {
        c: [feature_names.index(f) for f in FEATURE_GROUPS[c]] for c in FEATURE_GROUPS
    }
    rows = []
    for (probe, level), g in z.groupby(["probe", "level"]):
        w = g["n_core"].to_numpy(float)
        zabs = np.abs(g[[f"z{i}" for i in range(len(feature_names))]].to_numpy(float))
        tag = level.lower()
        rec = {"probe": probe, f"{prefix}{tag}_n_regions": len(g)}
        for comp, cols in idx.items():
            per_region = zabs[:, cols].mean(axis=1)
            rec[f"{prefix}{tag}_{comp}"] = float(np.sum(w * per_region) / np.sum(w))
        if level == "Beryl":
            rec[f"{prefix}beryl_coverage"] = float(
                w[g["source"].to_numpy() == "Beryl"].sum() / w.sum()
            )
        rows.append(rec)
    return pd.DataFrame(rows).groupby("probe").first()


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cum, 0.5 * cum[-1])])


def robust_component_scores(
    z: pd.DataFrame, feature_names: list, prefix: str = ""
) -> pd.DataFrame:
    """Like :func:`component_scores`, with medians instead of means: per region the median ``|z|``
    over the component's features (one noisy feature does not make a region unusual), and per
    probe the median over its regions weighted by their core rows (a few unusual channels or one
    unusual region do not make the probe unusual: at least half of its core rows must be)."""
    if z.empty:
        return pd.DataFrame()
    idx = {
        c: [feature_names.index(f) for f in FEATURE_GROUPS[c]] for c in FEATURE_GROUPS
    }
    rows = []
    for (probe, level), g in z.groupby(["probe", "level"]):
        w = g["n_core"].to_numpy(float)
        zabs = np.abs(g[[f"z{i}" for i in range(len(feature_names))]].to_numpy(float))
        tag = level.lower()
        rec = {"probe": probe, f"{prefix}{tag}_n_regions": len(g)}
        for comp, cols in idx.items():
            rec[f"{prefix}{tag}_{comp}"] = _weighted_median(
                np.median(zabs[:, cols], axis=1), w
            )
        rows.append(rec)
    return pd.DataFrame(rows).groupby("probe").first()
