"""Signal checks: problems visible in a probe's recording without any atlas or position.

- ``sig_bad_channel_frac`` -- fraction of channels the bad-channel detector labelled dead or noisy
  (an AP-band detector);
- ``sig_lf_fit_r2`` -- median R^2 of the LF aperiodic (decay) fit over the channels with a signal;
- ``sig_fp_<component>`` -- the probe's *fingerprint* distance: each feature's 10th, 50th and 90th
  percentile over the probe's channels, as robust z against all probes, mean ``|z|`` (capped) over
  the component's features. It is large when a whole component is off (scale, flat profile,
  saturation) wherever the probe is;
- ``sig_fpr_<component>`` -- the same, robust: the median over the component's features
  (:func:`robust_fingerprint_scores`), so one noisy feature cannot flag a component. The QC
  decisions use this one (:data:`~.levels.TESTS`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS

from .regions import BAD_CHANNEL_LABELS, signal_rows

QUANTILES = (10, 50, 90)
Z_CLIP = 10.0


def probe_fingerprints(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``[N, F, 3]`` per-probe feature percentiles over the signal channels."""
    n, _, f = features.shape
    out = np.full((n, f, len(QUANTILES)), np.nan)
    for i in range(n):
        rows = signal_rows(features[i], labels[i])
        if rows.sum() >= 10:
            out[i] = np.nanpercentile(features[i][rows], QUANTILES, axis=0).T
    return out


def fingerprint_stats(fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population median and robust scale ``[F, 3]`` of the fingerprints of all probes."""
    med = np.nanmedian(fp, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(fp - med), axis=0)
    return med, np.where(scale > 0, scale, np.nanstd(fp, axis=0) + 1e-9)


def robust_fingerprint_scores(
    fp: np.ndarray, stats: tuple, feature_names: list
) -> dict:
    """``{component: [N]}``: the median over the component's features of the mean ``|z|`` over
    the three percentiles (``sig_fpr_<component>``). A component is unusual only when most of its
    features are, so one noisy feature (AP and waveform features are the noisiest) cannot flag it;
    percentiles over the channels already ignore a few odd channels."""
    med, scale = stats
    z = np.clip(np.abs(fp - med) / scale, 0, Z_CLIP).mean(axis=-1)
    out = {}
    for comp, feats in FEATURE_GROUPS.items():
        cols = [feature_names.index(f) for f in feats]
        out[comp] = np.nanmedian(z[:, cols], axis=1)
    return out


def signal_table(ds) -> pd.DataFrame:
    """Signal statistics of every probe of a :class:`~.data.QCDataset`."""
    names = ds.feature_names
    fp = probe_fingerprints(ds.features, ds.channel_labels)
    med, scale = fingerprint_stats(fp)
    z = np.clip(np.abs(fp - med) / scale, 0, Z_CLIP)
    table = pd.DataFrame(index=pd.Index(ds.pids, name="pid"))
    table["sig_bad_channel_frac"] = np.isin(ds.channel_labels, BAD_CHANNEL_LABELS).mean(
        axis=1
    )
    r2 = names.index("decay_fit_r_squared")
    table["sig_lf_fit_r2"] = fp[:, r2, 1]
    for comp, feats in FEATURE_GROUPS.items():
        cols = [names.index(f) for f in feats]
        table[f"sig_fp_{comp}"] = np.nanmean(z[:, cols, :], axis=(1, 2))
    for comp, v in robust_fingerprint_scores(fp, (med, scale), names).items():
        table[f"sig_fpr_{comp}"] = v
    return table
