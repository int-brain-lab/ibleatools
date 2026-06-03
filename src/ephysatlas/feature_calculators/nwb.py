"""SpikeInterface Zarr feature calculator.

This module implements a calculator for SpikeInterface-readable AP/LFP Zarr
recordings, including the NWB-style Zarr layout used by Allen/AIND examples.

Classes
-------
NWBZarrFeatureCalculator
    Compute OOP features from AP/LFP Zarr recordings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft

from .base import BaseFeatureCalculator
from .spikeglx import LF_LATENCY_SAMPLES
from .types import RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


class NWBZarrFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for SpikeInterface-readable AP/LFP Zarr recordings.

    Args:
        ap_zarr_path (str | Path, optional): Path to an AP Zarr recording.
        lf_zarr_path (str | Path, optional): Path to an LFP Zarr recording.
        name (str, optional): Output identifier. Defaults to the AP path stem,
            LF path stem, or ``"probe"``.
        neuropixel_version (int): Neuropixels version passed to destriping.

    Raises:
        ValueError: If neither AP nor LFP Zarr path is supplied.

    Note:
        SpikeInterface returns traces as ``(samples, channels)`` and, with
        ``return_in_uV=True``, in microvolts. This calculator transposes them to
        ``(channels, samples)`` and converts to volts before feature computation.
    """

    def __init__(
        self,
        ap_zarr_path: str | Path | None = None,
        lf_zarr_path: str | Path | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> None:
        if ap_zarr_path is None and lf_zarr_path is None:
            raise ValueError("At least one AP or LFP Zarr path must be provided")
        self.ap_zarr_path = Path(ap_zarr_path) if ap_zarr_path is not None else None
        self.lf_zarr_path = Path(lf_zarr_path) if lf_zarr_path is not None else None
        default_name = (
            self.ap_zarr_path.stem
            if self.ap_zarr_path is not None
            else self.lf_zarr_path.stem
            if self.lf_zarr_path is not None
            else "probe"
        )
        super().__init__(name=name or default_name, neuropixel_version=neuropixel_version)
        self._rec_ap = None
        self._rec_lf = None
        self._geometry: dict[str, np.ndarray] | None = None

    @property
    def rec_ap(self):
        """Return the lazily opened AP SpikeInterface recording."""
        if self.ap_zarr_path is None:
            return None
        if self._rec_ap is None:
            from spikeinterface.core import read_zarr

            self._rec_ap = read_zarr(str(self.ap_zarr_path))
        return self._rec_ap

    @property
    def rec_lf(self):
        """Return the lazily opened LFP SpikeInterface recording."""
        if self.lf_zarr_path is None:
            return None
        if self._rec_lf is None:
            from spikeinterface.core import read_zarr

            self._rec_lf = read_zarr(str(self.lf_zarr_path))
        return self._rec_lf

    @property
    def _reference_recording(self):
        """Return whichever recording is available, preferring AP."""
        return self.rec_ap if self.rec_ap is not None else self.rec_lf

    def _read_traces_volts(
        self, recording, start_frame: int, n_samples: int
    ) -> np.ndarray:
        """Read traces from SpikeInterface and convert microvolts to volts."""
        traces_uv = recording.get_traces(
            start_frame=int(start_frame),
            end_frame=int(start_frame + n_samples),
            return_in_uV=True,
        )
        return traces_uv.T.astype(np.float32) * 1e-6

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read AP/LFP snippets from Zarr recordings.

        Args:
            window (SnippetWindow): Snippet time window to read.

        Returns:
            RawSnippet: Raw AP/LF arrays shaped ``(channels, samples)`` in volts.
        """
        raw_ap = raw_lf = fs_ap = fs_lf = None

        if self.rec_ap is not None:
            fs_ap = float(self.rec_ap.get_sampling_frequency())
            ns_ap = scipy.fft.next_fast_len(int(fs_ap * window.duration_ap), real=True)
            n0_ap = int(fs_ap * window.t_start)
            raw_ap = self._read_traces_volts(self.rec_ap, n0_ap, ns_ap)

        if self.rec_lf is not None:
            fs_lf = float(self.rec_lf.get_sampling_frequency())
            ns_lf = scipy.fft.next_fast_len(int(fs_lf * window.duration_lf), real=True)
            n0_lf = int(fs_lf * window.t_start) + LF_LATENCY_SAMPLES
            raw_lf = self._read_traces_volts(self.rec_lf, n0_lf, ns_lf)

        return RawSnippet(raw_ap=raw_ap, raw_lf=raw_lf, fs_ap=fs_ap, fs_lf=fs_lf)

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Build ibldsp geometry from SpikeInterface channel properties.

        Returns:
            dict[str, np.ndarray]: Geometry dictionary compatible with ibldsp.
        """
        if self._geometry is not None:
            return self._geometry

        rec = self._reference_recording
        locations = np.asarray(rec.get_channel_locations(), dtype=float)
        n_channels = locations.shape[0]
        sample_shift = rec.get_property("inter_sample_shift")
        if sample_shift is None:
            LOGGER.warning("No inter_sample_shift property on %s; using zeros", self.name)
            sample_shift = np.zeros(n_channels, dtype=float)

        x = locations[:, 0].astype(float)
        y = locations[:, 1].astype(float)
        self._geometry = {
            "x": x,
            "y": y,
            "col": np.unique(x, return_inverse=True)[1].astype(float),
            "row": np.unique(y, return_inverse=True)[1].astype(float),
            "sample_shift": np.asarray(sample_shift, dtype=float),
            "shank": np.zeros(n_channels, dtype=float),
        }
        return self._geometry

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from Zarr geometry.

        Returns:
            pd.DataFrame: Columns ``channel``, ``rawInd``, ``axial_um``, and
            ``lateral_um``.
        """
        geometry = self.load_geometry()
        n_channels = len(geometry["x"])
        return pd.DataFrame(
            {
                "channel": np.arange(n_channels),
                "rawInd": np.arange(n_channels),
                "axial_um": np.asarray(geometry["y"], dtype=float),
                "lateral_um": np.asarray(geometry["x"], dtype=float),
            }
        )

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return AP and LF durations from SpikeInterface recordings."""
        max_ap = None
        if self.rec_ap is not None:
            max_ap = self.rec_ap.get_num_frames() / self.rec_ap.get_sampling_frequency()
        max_lf = None
        if self.rec_lf is not None:
            max_lf = self.rec_lf.get_num_frames() / self.rec_lf.get_sampling_frequency()
        return max_ap, max_lf
