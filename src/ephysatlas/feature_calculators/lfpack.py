"""lfpack-backed feature calculator.

Reads LF snippets from an ``lfpack``-compressed HDF5 archive via
:class:`lfpack.LFPackReader`. This does **not** subclass
:class:`ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`:
that intermediate class bakes in an AP/LF phase-alignment offset
(``LF_LATENCY_SAMPLES``) meaningful only for a real dual-band spikeglx-style
reader, and this source is LF-only and already resampled/denoised upstream (by
the ``lfpack`` compression pipeline), not on that clock.

The archive's LF has already been highpass-filtered, common-average
referenced, bad-channel-interpolated, decimated (typically to ~250 Hz), and
Cadzow-denoised by ``ibldsp.voltage.resample_denoise_lfp_cbin`` *before*
compression -- callers must pass ``skip_lf_destripe=True`` (and, for the
``csd`` family, ``feature_params=FeatureParams(csd=CsdParams(decimate=1,
denoise=False))``) so :func:`ephysatlas.feature_computation.compute_features_from_raw`
does not redo (or crash trying to redo) that preprocessing at the wrong
sample rate.

Classes
-------
LFPackFeatureCalculator
    Compute OOP features from an lfpack-compressed archive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseFeatureCalculator
from .types import RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)

# Channel-metadata fields lfpack may attach to an archive (see
# lfpack.LFPackReader.channels_full and 2026-06-lfpack/attach_ibl_metadata.py).
_CHANNEL_METADATA_KEYS = (
    "lateral_um",
    "axial_um",
    "x",
    "y",
    "z",
    "atlas_id",
    "acronym",
    "labels",
)


class LFPackFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for an lfpack-compressed LF archive.

    Args:
        h5_file (str | Path): lfpack HDF5 archive (single-recording legacy
            layout, or multi-recording keyed by ``recording``).
        recording (str, optional): Recording key within a multi-recording
            archive. Ignored (and not required) for a legacy single-recording
            file; required when ``h5_file`` holds more than one recording.
        scale (int): lfpack reconstruction scale (``0`` = full quality).
        bin_channels (int): Adjacent electrodes summed per output channel.
            Defaults to ``1`` (full electrode resolution) -- unlike
            ``2026-07_lfp-encoders/lfpack_io.py``'s ``bin_channels=4``, which
            is a choice specific to that regression project; region
            classification here is per-physical-channel.
        name (str, optional): Recording identifier used as the OOP ``pid`` in
            outputs. Defaults to ``recording`` or the file stem.
        neuropixel_version (int): Neuropixels version passed to
            ``compute_features_from_raw`` (unused for LF-only feature
            families, kept for interface parity).

    Note:
        The reader is opened lazily on first access, not at construction.
    """

    def __init__(
        self,
        h5_file: str | Path,
        recording: str | None = None,
        scale: int = 0,
        bin_channels: int = 1,
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> None:
        self.h5_file = Path(h5_file)
        self.recording = recording
        self.scale = scale
        self.bin_channels = bin_channels
        super().__init__(
            name=name or recording or self.h5_file.stem,
            neuropixel_version=neuropixel_version,
        )
        self._reader = None

    @property
    def reader(self):
        """Return the lazily opened, cached ``lfpack.LFPackReader``."""
        if self._reader is None:
            from lfpack import LFPackReader

            self._reader = LFPackReader(
                self.h5_file,
                recording=self.recording,
                scale=self.scale,
                bin_channels=self.bin_channels,
            )
        return self._reader

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read one LF snippet from the archive.

        Args:
            window (SnippetWindow): Snippet time window to read
                (``duration_ap`` is ignored -- this source is LF-only).

        Returns:
            RawSnippet: ``raw_ap=None``; ``raw_lf`` shaped
            ``(channels, samples)`` in volts.
        """
        reader = self.reader
        fs_lf = float(reader.fs)
        n0 = int(round(window.t_start * fs_lf))
        ns = int(round(window.duration_lf * fs_lf))
        # Not read_samples(): despite its docstring, it forwards to read()'s
        # sync=True default and returns a (data, None) tuple instead of the
        # bare array. sync=False here returns the array directly.
        raw_lf = reader.read(slice(n0, n0 + ns), slice(None), sync=False)
        raw_lf = np.asarray(raw_lf, dtype=np.float32).T
        return RawSnippet(raw_ap=None, raw_lf=raw_lf, fs_ap=None, fs_lf=fs_lf)

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Load ibldsp geometry from the reader, filling derived defaults.

        Returns:
            dict[str, np.ndarray]: ``x``/``y`` from the reader, plus
            ``sample_shift``/``shank``/``col``/``row`` defaults (lfpack
            archives carry only on-probe ``x``/``y``, matching
            ``SpikeGlxLikeFeatureCalculator.load_geometry``'s fallback for a
            reader without those derived keys).
        """
        geometry = dict(self.reader.geometry)
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
        """Build channel metadata from the archive's annotated channel info.

        Returns:
            pd.DataFrame: ``channel`` plus whichever of
            ``lateral_um``/``axial_um``/``x``/``y``/``z``/``atlas_id``/
            ``acronym``/``labels`` the archive carries (written by
            ``2026-06-lfpack/attach_ibl_metadata.py`` for annotated archives).
            ``shank`` is filled with zeros (lfpack is single-shank today).

        Raises:
            KeyError: If the reader exposes no channel geometry at all.
        """
        channels = self.reader.channels
        n_channels = len(np.asarray(channels["lateral_um"]))
        data = {"channel": np.arange(n_channels)}
        for key in _CHANNEL_METADATA_KEYS:
            if key in channels:
                data[key] = np.asarray(channels[key])
        data["shank"] = np.zeros(n_channels)
        return pd.DataFrame(data)

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return ``(None, duration_lf)`` -- this source has no AP stream."""
        reader = self.reader
        return None, reader.ns / reader.fs

    def saturation_times(self) -> pd.DataFrame:
        """Return the archive's saturation intervals on the session clock.

        Returns:
            pd.DataFrame: ``start_sample``/``stop_sample``/``start_time``/
            ``stop_time`` (see ``lfpack.LFPackReader.saturation_times``).
            Verified scale-independent, so reading it from any tier of a PID
            covers every tier.
        """
        return self.reader.saturation_times()

    @property
    def saturation_mask(self) -> np.ndarray:
        """Boolean saturation mask at this reader's sampling rate, shape ``(ns,)``."""
        return self.reader.saturation_mask
