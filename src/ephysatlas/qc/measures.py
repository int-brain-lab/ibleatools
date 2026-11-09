"""What the insertion QC measures on a probe, from its recording and the region references.

- :func:`band_scores` -- is a band unusual for its regions? ``level``: how far the band is from what
  the other probes record in the same regions; ``shape``: the same after removing each feature's
  offset over the whole probe, so that it sees what changes along the probe (a band broken on part of
  the probe, noise) and not a probe-wide offset. Medians over the band's features and over the bins,
  so that one noisy feature or a few odd bins do not flag a band;
- :func:`fit` -- how well does the recording follow, along the probe, the profile expected from the
  regions it crosses? One measure for all the tests of position, which ask whether the recording
  would follow better somewhere else: :func:`scan`, deeper or shallower along the probe, or sideways;
  :class:`TrackProfiles`, on the tracks of the other insertions; :func:`swap_gains`, on another
  track of the same mouse.

Positions are in metres, ``[B, 3]``, in row order (top of the probe first).
"""

from __future__ import annotations

import numpy as np

from .regions import BAD_CHANNEL_LABELS, parcel_labels

SHIFTS_UM = tuple(float(d) for d in np.arange(-2000, 2001, 250) if d != 0)
SIDEWAYS_UM = (
    -1000.0,
    -500.0,
    500.0,
    1000.0,
)  # along the left-right and the front-back axes
MIN_KEEP = 0.5  # of the probe's bins that must stay scored for a move to count
MIN_SCAN_BINS = 5


def dead_fraction(labels: np.ndarray) -> float:
    """Fraction of a probe's channels labelled dead or noisy."""
    return float(np.isin(labels, BAD_CHANNEL_LABELS).mean())


def band_scores(residuals: np.ndarray, bands: dict) -> dict:
    """``{"<band>_level": score, "<band>_shape": score}`` of the residuals ``[B, F]`` of a probe;
    ``bands``: ``{band: feature columns}``."""
    out = {}
    with np.errstate(all="ignore"):
        for band, cols in bands.items():
            e = residuals[:, cols]
            ok = np.isfinite(e).any(axis=1)
            if not ok.any():
                out[f"{band}_level"] = out[f"{band}_shape"] = np.nan
                continue
            e = e[ok]
            out[f"{band}_level"] = float(np.median(np.nanmedian(np.abs(e), axis=1)))
            shaped = e - np.nanmedian(e, axis=0, keepdims=True)
            out[f"{band}_shape"] = float(
                np.median(np.nanmedian(np.abs(shaped), axis=1))
            )
    return out


def fit(z: np.ndarray, expected: np.ndarray, bands: dict) -> tuple[float, int]:
    """How well a recording ``z`` follows the ``expected`` profile (both ``[B, F]``): the correlation
    over the bins of each feature, averaged over each band's features and then over the bands, so that
    a band with many features counts once. Offsets do not matter, only the shape along the probe.

    Returns:
        tuple: the fit, NaN below ``MIN_SCAN_BINS`` scored bins, and the number of those bins.
    """
    scored = np.isfinite(expected).all(axis=1) & np.isfinite(z).all(axis=1)
    if scored.sum() < MIN_SCAN_BINS:
        return np.nan, int(scored.sum())
    a = z[scored] - z[scored].mean(axis=0)
    b = expected[scored] - expected[scored].mean(axis=0)
    r = (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0) + 1e-12)
    return float(np.mean([r[cols].mean() for cols in bands.values()])), int(
        scored.sum()
    )


def gain(here: tuple, there: tuple) -> float:
    """How much better the fit is ``there`` than ``here`` (:func:`fit`); NaN when too few of the bins
    scored ``here`` stay scored ``there``."""
    if not there[1] >= max(MIN_KEEP * here[1], MIN_SCAN_BINS):
        return np.nan
    return there[0] - here[0]


def probe_axis(xyz: np.ndarray) -> np.ndarray:
    """Unit vector along the probe (the line fitted to the positions), top to tip."""
    from iblatlas.atlas import Trajectory

    p = xyz[np.isfinite(xyz).all(axis=1)]
    axis = Trajectory.fit(p).vector
    if np.dot(axis, p[-1] - p[0]) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def moved_positions(xyz: np.ndarray) -> dict:
    """``{"depth": [(µm, positions)], "sideways": [(µm, positions)]}``, the places a probe is tried.
    A positive depth is deeper."""
    axis = probe_axis(xyz)
    depth = [(d, xyz + d * 1e-6 * axis) for d in SHIFTS_UM]
    sideways = []
    for direction in (np.eye(3)[0], np.eye(3)[1]):
        sideways += [(d, xyz + d * 1e-6 * direction) for d in SIDEWAYS_UM]
    return dict(depth=depth, sideways=sideways)


def scan(
    reference, brain_atlas, z: np.ndarray, xyz: np.ndarray, probe: int, bands: dict
) -> tuple[dict, dict]:
    """How much better a probe follows its regions elsewhere.

    Returns:
        tuple: ``summary`` -- ``depth`` and ``sideways``, the largest gain over the moves of each
        scan (:func:`gain`; positive when somewhere else fits better), and ``depth_best``,
        ``sideways_best``, that move (µm); ``profile`` -- ``{scan: [(µm, gain)]}``.
    """

    def at(positions):
        labels = parcel_labels(positions, brain_atlas)
        return fit(z, reference.expected(labels, [probe])[0], bands)

    here = at(xyz)
    summary, profile = {}, {}
    for name, moves in moved_positions(xyz).items():
        gains = [gain(here, at(pos)) for _, pos in moves]
        profile[name] = [(d, g) for (d, _), g in zip(moves, gains)]
        ok = np.isfinite(gains)
        k = int(np.nanargmax(gains)) if ok.any() else None
        summary[name] = gains[k] if ok.any() else np.nan
        summary[f"{name}_best"] = moves[k][0] if ok.any() else np.nan
    return summary, profile


class TrackProfiles:
    """What is expected along every recorded track, to tell whether a recording follows its own
    track better than the tracks of the other insertions. Nothing to calibrate: a track that changes
    little along the probe fits no better than the others, and the rank says so.

    Args:
        expected: ``[N, B, F]`` expected profile of each track (NaN where a bin is not scored).
        bands: ``{band: feature columns}``.
    """

    def __init__(self, expected: np.ndarray, bands: dict):
        self.scored = np.isfinite(expected).all(axis=2)
        centred = expected - np.nanmean(expected, axis=1, keepdims=True)
        self.centred = np.nan_to_num(centred).astype(np.float32)
        self.norm = (self.centred**2).sum(axis=1)
        self.bands = list(bands.values())

    def fits(self, z: np.ndarray, exclude=()) -> np.ndarray:
        """``[N]`` :func:`fit` of the recording ``z`` on every track (NaN on the tracks ``exclude``
        and on those with too few bins in common with it)."""
        ok = np.isfinite(z).all(axis=1)
        out = np.full(len(self.centred), np.nan)
        if ok.sum() < MIN_SCAN_BINS:
            return out
        a = z - z[ok].mean(axis=0)
        a[~ok] = 0.0
        r = np.einsum("bf,nbf->nf", a, self.centred)
        r = r / np.sqrt((a * a).sum(axis=0)[None] * self.norm + 1e-9)
        means = np.mean([r[:, cols].mean(axis=1) for cols in self.bands], axis=0)
        common = self.scored[:, ok].sum(axis=1) >= MIN_SCAN_BINS
        common[list(exclude)] = False
        out[common] = means[common]
        return out

    def rank(self, z: np.ndarray, own: float, exclude=()) -> float:
        """The share of the other tracks the recording ``z`` follows at least as well as its own,
        whose fit is ``own``: small for a recording that belongs where it is. ``exclude``: the probes
        whose track is not compared."""
        fits = self.fits(z, exclude)
        fits = fits[np.isfinite(fits)]
        return float((fits >= own).mean()) if len(fits) and np.isfinite(own) else np.nan


def swap_gains(
    reference, brain_atlas, a: tuple, b: tuple, exclude, bands: dict
) -> tuple[float, float]:
    """How much better each of two probes follows the track of the other than its own (positive when
    it does). ``a``, ``b``: ``(z, xyz)``; ``exclude``: the probes left out of the reference."""
    labels = {id(p): parcel_labels(p[1], brain_atlas) for p in (a, b)}

    def on(src, track):
        return fit(src[0], reference.expected(labels[id(track)], exclude)[0], bands)[0]

    return on(a, b) - on(a, a), on(b, a) - on(b, b)


def measure_probe(
    reference, brain_atlas, z, xyz, probe: int, bands: dict, tracks: TrackProfiles
) -> tuple[dict, dict]:
    """Everything measured on one probe, real or planted: :func:`band_scores`, :func:`fit`,
    :func:`scan` and its rank among the tracks (:meth:`TrackProfiles.rank`)."""
    labels = parcel_labels(xyz, brain_atlas)
    median, spread = reference.expected(labels, [probe])
    own = fit(z, median, bands)[0]
    summary, profile = scan(reference, brain_atlas, z, xyz, probe, bands)
    rec = {**band_scores((z - median) / spread, bands), "fit": own, **summary}
    rec["track_rank"] = tracks.rank(z, own, [probe])
    return rec, profile
