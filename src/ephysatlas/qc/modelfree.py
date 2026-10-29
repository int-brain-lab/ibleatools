"""Model-free evidence for the insertion QC: nothing here uses the channel- or unit-level model.

A probe's recording is compared with what the *other* probes record around the same positions --
their recorded features, not a model's prediction -- so excluding the probes this evidence flags does
not favour the model that is later trained without them.

- :class:`LocalReference` -- at any position, the median of the recordings of the nearby probes
  (one channel per probe, its nearest, within ``radius_um``); a probe's band score is how far its
  recording is from that reference, with medians over features and over channels so that one
  noisy feature or a few odd channels do not make a band unusual, and every feature scaled by how
  much it typically disagrees with its neighbours (the AP and waveform features are noisier);
- :func:`region_band_scores` -- the same question at the resolution of brain regions
  (:class:`~.regions.RegionReference`), kept for the figures;
- :func:`depth_scan` -- would the recording be more typical if the whole probe sat deeper or
  shallower along its axis? Each shift is compared with the human positions on the same channels;
- :func:`swap_gain` -- would two probes of one session be more typical on each other's tracks?
- :func:`band_harm`, :func:`position_harm` -- how large a planted problem is.

Positions are in metres, ``[C, 3]`` per probe in row order (top of the probe first).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .regions import probe_region_values, robust_component_scores, signal_rows

COMPONENTS = ("lf", "ap", "waveform")
SHIFTS_UM = np.array([-1000.0, -750.0, -500.0, -250.0, 250.0, 500.0, 750.0, 1000.0])
MIN_KEEP = (
    0.5  # a shift is scored only when half of the in-brain channels stay in the brain
)
MIN_CHANNELS = 20  # channels with a reference needed for a band score


def probe_axis(xyz: np.ndarray) -> np.ndarray:
    """Unit vector along the probe (first principal axis of the channel positions), top to tip."""
    ok = np.isfinite(xyz).all(axis=1)
    p = xyz[ok]
    c = p.mean(axis=0)
    axis = np.linalg.svd(p - c, full_matrices=False)[2][0]
    if np.dot(axis, p[-1] - p[0]) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def in_brain(brain_atlas, xyz: np.ndarray) -> np.ndarray:
    """``[C]`` whether each position is inside the brain (finite and not in the void)."""
    ok = np.isfinite(xyz).all(axis=1)
    out = np.zeros(len(xyz), dtype=bool)
    if ok.any():
        labels = np.asarray(
            brain_atlas.get_labels(xyz[ok], mode="clip"), dtype=np.int64
        )
        out[ok] = labels != 0
    return out


class LocalReference:
    """What the other probes record around a position.

    Args:
        ds: :class:`~.data.QCDataset`; its signal channels at their human positions form the bank.
        sd: ``[F]`` robust SD of every feature (:func:`~.pipeline.feature_sd`), the unit of the
            standardized features.
        radius_um: Neighbourhood radius.
        min_probes: Fewest other probes with a channel in the neighbourhood for a reference.
        max_probes: The nearest this many probes are used (one channel each).

    ``noise`` (``[F]``, ones until :meth:`fit_noise` sets it) is each feature's typical
    disagreement with its neighbours, the unit of :meth:`deltas`.
    """

    def __init__(
        self,
        ds,
        sd: np.ndarray,
        radius_um: float = 300.0,
        min_probes: int = 3,
        max_probes: int = 20,
    ):
        xs, fs, ps = [], [], []
        for i in range(len(ds.pids)):
            rows = signal_rows(ds.features[i], ds.channel_labels[i])
            rows &= np.isfinite(ds.human_xyz[i]).all(axis=1)
            xs.append(ds.human_xyz[i][rows] * 1e6)
            fs.append(ds.features[i][rows])
            ps.append(np.full(rows.sum(), i))
        self.X, F, self.P = np.concatenate(xs), np.concatenate(fs), np.concatenate(ps)
        self.med, self.sd = np.nanmedian(F, axis=0), sd
        self.Fz = (F - self.med) / sd
        self.tree = cKDTree(self.X)
        self.radius, self.min_probes, self.max_probes = (
            radius_um,
            min_probes,
            max_probes,
        )
        self.noise = np.ones(F.shape[1])
        self.bands = {c: ds.component_index(c) for c in COMPONENTS}

    def reference(self, xyz, exclude, rows) -> np.ndarray:
        """``[C, F]`` standardized reference at ``xyz`` for ``rows`` (NaN elsewhere): the
        median over the nearest ``max_probes`` other probes of their nearest channel."""
        xyz = np.asarray(xyz, dtype=float) * 1e6
        out = np.full((len(xyz), self.Fz.shape[1]), np.nan)
        idxs = np.flatnonzero(np.asarray(rows, bool) & np.isfinite(xyz).all(axis=1))
        if not len(idxs):
            return out
        excl = np.asarray(list(exclude))
        for c, idx in zip(idxs, self.tree.query_ball_point(xyz[idxs], r=self.radius)):
            idx = np.asarray(idx, dtype=int)
            idx = idx[~np.isin(self.P[idx], excl)]
            if len(idx) < self.min_probes:
                continue
            d = np.linalg.norm(self.X[idx] - xyz[c], axis=1)
            o = np.lexsort((d, self.P[idx]))
            idx, d, pr = idx[o], d[o], self.P[idx][o]
            first = np.r_[True, pr[1:] != pr[:-1]]
            idx, d = idx[first], d[first]
            if len(idx) < self.min_probes:
                continue
            idx = idx[np.argsort(d)[: self.max_probes]]
            out[c] = np.median(self.Fz[idx], axis=0)
        return out

    def deltas(self, features, xyz, exclude, rows) -> np.ndarray:
        """``[C, F]`` |recorded - reference| in units of each feature's typical disagreement."""
        fz = (np.asarray(features, dtype=float) - self.med) / self.sd
        return np.abs(fz - self.reference(xyz, exclude, rows)) / self.noise

    def fit_noise(self, ds) -> np.ndarray:
        """Each feature's typical disagreement with its neighbours: the median over every
        channel of every probe of ``|recorded - reference|`` (standardized units)."""
        self.noise = np.ones(self.Fz.shape[1])
        d = [
            self.deltas(
                ds.features[i],
                ds.human_xyz[i],
                [i],
                signal_rows(ds.features[i], ds.channel_labels[i]),
            )
            for i in range(len(ds.pids))
        ]
        noise = np.nanmedian(np.concatenate(d), axis=0)
        self.noise = np.where(np.isfinite(noise) & (noise > 0), noise, 1.0)
        return self.noise

    def band_scores(self, features, xyz, labels, exclude, rows=None) -> dict:
        """``{band: score}``: the median over channels of the median over the band's features of
        the scaled disagreement with the reference (NaN with fewer than ``MIN_CHANNELS``)."""
        sig = signal_rows(features, labels)
        rows = sig if rows is None else sig & np.asarray(rows, bool)
        d = self.deltas(features, xyz, exclude, rows)
        out = {}
        for c, cols in self.bands.items():
            per = (
                np.nanmedian(d[:, cols], axis=1)
                if len(cols)
                else np.full(len(d), np.nan)
            )
            per = per[np.isfinite(per)]
            out[c] = float(np.median(per)) if len(per) >= MIN_CHANNELS else np.nan
        return out


def typicality(scores: dict) -> float:
    """Mean of the band scores (lower is more typical)."""
    v = np.array([scores.get(c, np.nan) for c in COMPONENTS], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def depth_scan(
    local: LocalReference,
    brain_atlas,
    features,
    xyz,
    labels,
    probe: int,
    shifts_um: np.ndarray = SHIFTS_UM,
) -> tuple[dict, dict]:
    """Model-free depth scan.

    Returns:
        tuple: ``summary`` -- ``depth_gain`` (how much more typical the best shift is than the
        human positions, on the same channels; positive when another depth fits better),
        ``depth_best`` (that shift, µm, positive deeper) and ``depth_gain_<band>`` at that shift;
        ``profile`` -- ``{"all" | band: [S] gains}``.
    """
    xyz = np.asarray(xyz, dtype=float)
    axis = probe_axis(xyz)
    inside0 = in_brain(brain_atlas, xyz) & signal_rows(features, labels)
    shifts = np.asarray(shifts_um, dtype=float)
    profile = {k: np.full(len(shifts), np.nan) for k in ("all",) + COMPONENTS}
    for k, d in enumerate(shifts):
        moved = xyz + d * 1e-6 * axis
        rows = inside0 & in_brain(brain_atlas, moved)
        if rows.sum() < MIN_KEEP * max(inside0.sum(), 1):
            continue
        s0 = local.band_scores(features, xyz, labels, [probe], rows)
        sd = local.band_scores(features, moved, labels, [probe], rows)
        profile["all"][k] = typicality(s0) - typicality(sd)
        for c in COMPONENTS:
            profile[c][k] = s0[c] - sd[c]
    summary = {"depth_gain": np.nan, "depth_best": np.nan}
    summary.update({f"depth_gain_{c}": np.nan for c in COMPONENTS})
    g = profile["all"]
    if np.isfinite(g).any():
        k = int(np.nanargmax(g))
        summary.update(depth_gain=float(g[k]), depth_best=float(shifts[k]))
        summary.update({f"depth_gain_{c}": float(profile[c][k]) for c in COMPONENTS})
    return summary, profile


def swap_gain(local: LocalReference, a: dict, b: dict) -> float:
    """How much more typical two probes of one session are on each other's tracks (positive when
    swapped fits better). ``a``, ``b``: ``features``, ``xyz``, ``labels``, ``probe``. Neither
    probe is part of the reference."""
    excl = [a["probe"], b["probe"]]

    def t(src, pos):
        return typicality(
            local.band_scores(src["features"], pos["xyz"], src["labels"], excl)
        )

    return (t(a, a) + t(b, b)) - (t(a, b) + t(b, a))


def region_band_scores(
    features: np.ndarray,
    xyz: np.ndarray,
    labels: Optional[np.ndarray],
    brain_atlas,
    reference,
    settings,
    probe: int,
) -> dict:
    """``{"<beryl|cosmos>_<band>": score}``: how unusual each band is for its brain regions."""
    out = {f"{lvl}_{c}": np.nan for lvl in ("beryl", "cosmos") for c in COMPONENTS}
    values = probe_region_values(
        features, xyz, labels, brain_atlas, settings, probe=probe
    )
    if values.empty:
        return out
    z = reference.zscores(values)
    if z.empty:
        return out
    rec = robust_component_scores(z, reference.feature_names).iloc[0].to_dict()
    return {k: float(rec.get(k, np.nan)) for k in out}


def band_harm(f_true, f_fake, cols, sd, rows) -> float:
    """Median over ``rows`` of the mean over ``cols`` of ``|f_fake - f_true| / sd``."""
    d = (
        np.abs(np.asarray(f_fake, float) - np.asarray(f_true, float))[:, cols]
        / sd[cols]
    )
    d = d[np.asarray(rows, dtype=bool)]
    d = d[np.isfinite(d).all(axis=1)]
    return float(np.median(d.mean(axis=1))) if len(d) else np.nan


def position_harm(local: LocalReference, xyz_true, xyz_fake, probe, rows) -> float:
    """Median over channels of the mean over features of the change of the local reference
    between the true and the planted positions, in robust SDs: how different what other probes
    record at the planted positions is from what they record at the true ones."""
    r_t = local.reference(xyz_true, [probe], rows)
    r_f = local.reference(xyz_fake, [probe], rows)
    d = np.abs(r_t - r_f)
    d = d[np.isfinite(d).all(axis=1)]
    return float(np.median(d.mean(axis=1))) if len(d) else np.nan
