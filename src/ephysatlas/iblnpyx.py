"""
Vendored numerics from npyx (SteinmetzLab/NeuroPyxels), used by
``ephysatlas.cells.compute_3d_acgs`` to log-time-resample linear autocorrelograms
into Han Yu's NEMO/ICLR 3D-ACG format.

Why vendored instead of depended on
------------------------------------
``npyx`` pulls in a large, fragile dependency chain for the ~40 lines actually
used here (``npyx.corr.convert_acg_log`` / ``npyx.utils.smooth``):

- Its ``scikit-learn<1.6.0`` pin conflicts with ``ibleatools``'s own validated
  ``scikit-learn`` range (1.6-1.8; see ``packages/ibleatools/pyproject.toml``).
- Its unpinned ``ipython`` dependency resolved to IPython>=9, which removed the
  ``IPython.core.display.display`` re-export that ``npyx.plot_utils`` imports
  at module load time — breaking ``import npyx`` outright.
- ``npyx/circuitProphyler.py`` (eagerly imported by ``npyx/__init__.py``, and
  otherwise unrelated to anything used here) does ``import imp``, a stdlib
  module removed in Python 3.12.

This module is a short-term fix. The plan is to reach a consensus with the
npyx maintainer (a lighter-weight ``npyx``-core package, or fixes to the above)
and remove this file in favour of a normal dependency.

License
-------
npyx is GPL-3.0-licensed (see https://github.com/m-beau/NeuroPyxels). The
functions below are copied near-verbatim from ``npyx/corr.py`` and
``npyx/utils.py``. Unlike the rest of ``ephysatlas`` (MIT), THIS FILE is
therefore under the terms of the GPL-3.0 license.
"""

import numpy as np
import scipy.stats


def smooth_gaussian_axis0(arr, sd):
    """
    Symmetric gaussian smoothing along axis 0.

    Axis/method-specialised port of ``npyx.utils.smooth(arr, method='gaussian',
    sd=sd, axis=0)``. Kept numerically identical, including the kernel
    zero-padding step below (a quirk of the original, general-purpose
    implementation): npyx re-centres an even-length kernel by zero-padding one
    side, based on the (for a symmetric gaussian, always slightly off-centre
    for an odd-length ``arange``) argmax position.

    Parameters
    ----------
    arr : numpy.ndarray
        1-D or 2-D array to smooth along axis 0.
    sd : int
        Gaussian kernel standard deviation, in samples along axis 0.

    Returns
    -------
    numpy.ndarray
        Smoothed array, same shape as `arr`.
    """
    c = arr.shape[0] // 2
    pad_width = [(c, c)] + [(0, 0)] * (arr.ndim - 1)
    padded = np.pad(arr, pad_width, "symmetric")

    x = np.arange(-4 * sd, 4 * sd + 1)
    kernel = scipy.stats.norm.pdf(x, 0, sd)
    mx = np.argmax(kernel)
    if mx < len(kernel) / 2:
        kernel = np.append(np.zeros(len(kernel) - 2 * mx), kernel)
    elif mx > len(kernel) / 2:
        kernel = np.append(kernel, np.zeros(mx - (len(kernel) - mx)))
    kernel = kernel / kernel.sum()

    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=padded)
    return smoothed[c + 1: smoothed.shape[0] - c + 1]


def convert_acg_log(lin_acg, cbin, cwin, n_log_bins=100, start_log_ms=0.8, smooth_sd=1):
    """
    Resample a linear-time ACG onto a log-time axis.

    Non-plotting port of ``npyx.corr.convert_acg_log``. Interpolates the linear
    ACG's positive lags onto `n_log_bins` log-spaced points, smooths, then
    mirrors to cover negative lags too.

    Parameters
    ----------
    lin_acg : numpy.ndarray
        (n_bins,) or (n_freqs, n_bins) linear-time ACG.
    cbin, cwin : float
        Linear bin size / full window used to compute `lin_acg`, in ms.
    n_log_bins : int
        Number of log-spaced bins per side (mirrored -> 2*n_log_bins+1 total).
    start_log_ms : float or None
        Discard log bins below this lag, in ms; None to keep all.
    smooth_sd : int or None
        Gaussian smoothing std, in log-bin samples; None to disable.

    Returns
    -------
    log_acg : numpy.ndarray
        (2*n_log_bins+1,) or (n_freqs, 2*n_log_bins+1) log-time ACG.
    t_log : numpy.ndarray
        (2*n_log_bins+1,) log-time bin centres, in ms (negative, zero, positive).
    """
    assert cbin <= cwin
    assert n_log_bins > 1
    if start_log_ms is not None:
        assert start_log_ms > 0

    original_bins = np.arange(-cwin // 2, cwin // 2 + cbin, cbin)
    lin_acg = lin_acg.T  # (n_freqs, n_bins) -> (n_bins, n_freqs) if 2D
    assert original_bins.shape[0] == lin_acg.shape[0]
    half_i = len(original_bins) // 2 + 1
    lin_bins = original_bins[half_i:]
    log_bins = np.logspace(np.log10(lin_bins[0]), np.log10(lin_bins[-1]), n_log_bins)

    if lin_acg.ndim == 2:
        log_acg = np.zeros((len(log_bins), lin_acg.shape[1]))
        for acg_i, lin_a in enumerate(lin_acg[half_i:, :].T):
            log_acg[:, acg_i] = np.interp(log_bins, lin_bins, lin_a)
    else:
        log_acg = np.interp(log_bins, lin_bins, lin_acg[half_i:])

    if start_log_ms is not None:
        keep = log_bins >= start_log_ms
        log_bins = log_bins[keep]
        log_acg = log_acg[keep]

    if smooth_sd is not None:
        log_acg = smooth_gaussian_axis0(log_acg, smooth_sd)

    t_log = np.concatenate((-log_bins[::-1], [0], log_bins))
    zeros = [0] if lin_acg.ndim == 1 else np.zeros((1, log_acg.shape[1]))
    log_acg = np.concatenate((log_acg[::-1], zeros, log_acg), axis=0)

    return log_acg.T, t_log