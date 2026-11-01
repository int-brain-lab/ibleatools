"""What the insertion QC measures on a probe, from its residuals against the region references.

- :func:`band_scores` -- is a band unusual for its regions? ``level``: how far the band is from what
  the other probes record in the same regions; ``shape``: the same after removing each feature's
  offset over the whole probe, so that it sees what changes along the probe (a band broken on part of
  the probe, noise) and not a probe-wide offset. Medians over the band's features and over the bins,
  so that one noisy feature or a few odd bins do not flag a band;
- :func:`scan` -- would the recording be more typical elsewhere: deeper or shallower along the probe,
  or sideways? A move changes the bins near region borders only, so the scan uses means over the
  bins and the features (of residuals clipped at ``COST_CLIP``), not medians;
- :func:`swap_gain` -- would the two probes of a session fit better on each other's tracks?

Positions are in metres, ``[B, 3]``, in row order (top of the probe first).
"""

from __future__ import annotations

import numpy as np

from .regions import BAD_CHANNEL_LABELS, parcel_labels

COST_CLIP = 3.0  # MADs: a bin's cost
SHIFTS_UM = (-1000.0, -750.0, -500.0, -250.0, 250.0, 500.0, 750.0, 1000.0)
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


def bin_cost(residuals: np.ndarray) -> np.ndarray:
    """``[B]`` mean over the features of the clipped ``|residual|`` (NaN without a reference)."""
    a = np.minimum(np.abs(residuals), COST_CLIP)
    with np.errstate(all="ignore"):
        n = np.isfinite(a).sum(axis=1)
        return np.where(n > 0, np.nansum(a, axis=1) / np.maximum(n, 1), np.nan)


def gain(here: np.ndarray, there: np.ndarray) -> float:
    """How much lower the cost ``there`` is than ``here`` (``[B]`` per bin, mean over the bins scored
    at both); NaN when too few of the bins scored ``here`` stay scored ``there``."""
    valid = np.isfinite(here) & np.isfinite(there)
    if valid.sum() < max(MIN_KEEP * np.isfinite(here).sum(), MIN_SCAN_BINS):
        return np.nan
    return float(np.mean(here[valid] - there[valid]))


def probe_axis(xyz: np.ndarray) -> np.ndarray:
    """Unit vector along the probe (first principal axis of the positions), top to tip."""
    ok = np.isfinite(xyz).all(axis=1)
    p = xyz[ok]
    axis = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)[2][0]
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
    reference, brain_atlas, z: np.ndarray, xyz: np.ndarray, probe: int
) -> tuple[dict, dict]:
    """How much more typical a probe is elsewhere.

    Returns:
        tuple: ``summary`` -- ``depth`` and ``sideways``, the largest gain over the moves of each
        scan (:func:`gain`; positive when somewhere else fits better), and ``depth_best``,
        ``sideways_best``, that move (µm); ``profile`` -- ``{scan: [(µm, gain)]}``.
    """
    here = bin_cost(reference.residuals(z, parcel_labels(xyz, brain_atlas), [probe]))
    summary, profile = {}, {}
    for name, moves in moved_positions(xyz).items():
        gains = [
            gain(
                here,
                bin_cost(
                    reference.residuals(z, parcel_labels(pos, brain_atlas), [probe])
                ),
            )
            for _, pos in moves
        ]
        profile[name] = [(d, g) for (d, _), g in zip(moves, gains)]
        ok = np.isfinite(gains)
        k = int(np.nanargmax(gains)) if ok.any() else None
        summary[name] = gains[k] if ok.any() else np.nan
        summary[f"{name}_best"] = moves[k][0] if ok.any() else np.nan
    return summary, profile


def swap_gain(
    reference,
    brain_atlas,
    a: tuple,
    b: tuple,
    exclude,
) -> float:
    """How much better two probes fit on each other's tracks than on their own (positive when the
    swap fits better). ``a``, ``b``: ``(z, xyz)``; ``exclude``: the probes left out of the
    reference."""
    labels = {id(p): parcel_labels(p[1], brain_atlas) for p in (a, b)}

    def cost(src, track):
        c = bin_cost(reference.residuals(src[0], labels[id(track)], exclude))
        return np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan

    return float(cost(a, a) + cost(b, b) - cost(a, b) - cost(b, a))


def measure_probe(
    reference, brain_atlas, z, xyz, probe: int, bands: dict
) -> tuple[dict, dict]:
    """Everything measured on one probe, real or planted: :func:`band_scores` and :func:`scan`."""
    e = reference.residuals(z, parcel_labels(xyz, brain_atlas), [probe])
    summary, profile = scan(reference, brain_atlas, z, xyz, probe)
    return {**band_scores(e, bands), **summary}, profile
