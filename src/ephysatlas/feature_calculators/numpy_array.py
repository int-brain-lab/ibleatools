"""Generic in-memory numpy-array feature calculator.

A source-agnostic ``BaseFeatureCalculator`` for LF already held as a plain
numpy array in memory (no file-format knowledge of any kind). Intended for a
pre-processed LF checkpoint borrowed from wherever it was produced (e.g. a
Cadzow-denoised, resampled reference array) -- that file-format-specific
loading glue belongs in the caller, not in this package.

Classes
-------
NumpyArrayFeatureCalculator
    Compute OOP features from an in-memory LF array.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseFeatureCalculator
from .types import RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


class NumpyArrayFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for an LF array already held in memory.

    Args:
        lf (np.ndarray): LF voltage shaped ``(n_channels, n_samples)`` in
            volts.
        fs_lf (float): LF sampling frequency in Hz.
        geometry (dict): Channel geometry with at least ``"x"`` and ``"y"``
            arrays; missing derived keys (``sample_shift``, ``shank``,
            ``col``, ``row``) are filled with defaults.
        name (str): Recording identifier used as the OOP ``pid`` in outputs.
        channel_metadata (pd.DataFrame, optional): Channel metadata (e.g.
            ``axial_um``/``lateral_um``/``x``/``y``/``z``/``atlas_id``/
            ``acronym``/``labels``/``shank``). ``None`` falls back to a
            minimal frame built from ``geometry`` alone.
        t0 (float): Session-clock time (seconds) of ``lf``'s first sample,
            for a caller whose array doesn't start at session time 0.
        neuropixel_version (int): Neuropixels version passed to
            ``compute_features_from_raw`` (unused for LF-only feature
            families, kept for interface parity).
    """

    def __init__(
        self,
        lf: np.ndarray,
        fs_lf: float,
        geometry: dict,
        name: str,
        channel_metadata: pd.DataFrame | None = None,
        t0: float = 0.0,
        neuropixel_version: int = 1,
    ) -> None:
        self.lf = lf
        self.fs_lf = float(fs_lf)
        self._geometry = geometry
        self._channel_metadata = channel_metadata
        self.t0 = float(t0)
        super().__init__(name=name, neuropixel_version=neuropixel_version)

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Slice one LF snippet directly out of the in-memory array.

        Args:
            window (SnippetWindow): Snippet time window to read
                (``duration_ap`` is ignored -- this source is LF-only).

        Returns:
            RawSnippet: ``raw_ap=None``; ``raw_lf`` shaped
            ``(channels, samples)`` in volts.
        """
        n0 = int(round((window.t_start - self.t0) * self.fs_lf))
        ns = int(round(window.duration_lf * self.fs_lf))
        raw_lf = np.asarray(self.lf[:, n0 : n0 + ns], dtype=np.float32)
        return RawSnippet(raw_ap=None, raw_lf=raw_lf, fs_ap=None, fs_lf=self.fs_lf)

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Return the supplied geometry, filling derived-default keys.

        Returns:
            dict[str, np.ndarray]: ``x``/``y`` as supplied, plus
            ``sample_shift``/``shank``/``col``/``row`` defaults when absent.
        """
        geometry = dict(self._geometry)
        n_channels = len(np.asarray(geometry["x"]))
        derived_defaults = {
            "sample_shift": lambda: np.zeros(n_channels),
            "shank": lambda: np.zeros(n_channels),
            "col": lambda: np.unique(np.asarray(geometry["x"]), return_inverse=True)[1],
            "row": lambda: np.unique(np.asarray(geometry["y"]), return_inverse=True)[1],
        }
        for key, make_default in derived_defaults.items():
            if key not in geometry:
                geometry[key] = make_default()
        return {key: np.asarray(value) for key, value in geometry.items()}

    def load_channel_metadata(self) -> pd.DataFrame:
        """Return the supplied channel metadata, or a minimal geometry-only frame.

        Returns:
            pd.DataFrame: ``channel_metadata`` as supplied, or (when omitted)
            a frame with ``channel``/``axial_um``/``lateral_um``/``shank``
            built from ``load_geometry()``.
        """
        if self._channel_metadata is not None:
            return self._channel_metadata
        geometry = self.load_geometry()
        n_channels = len(np.asarray(geometry["x"]))
        return pd.DataFrame(
            {
                "channel": np.arange(n_channels),
                "axial_um": np.asarray(geometry["y"], dtype=float),
                "lateral_um": np.asarray(geometry["x"], dtype=float),
                "shank": np.asarray(geometry["shank"], dtype=float),
            }
        )

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return ``(None, duration_lf)`` -- this source has no AP stream."""
        return None, self.lf.shape[1] / self.fs_lf
