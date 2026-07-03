"""SpikeGLX file-backed feature calculator.

This module implements a concrete calculator for local Neuropixels AP/LF
``.bin`` or ``.cbin`` files readable by ``spikeglx.Reader``.

Classes
-------
SpikeGLXFileFeatureCalculator
    Compute OOP features from local AP/LF SpikeGLX files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import ibldsp.voltage
import numpy as np
import pandas as pd
import scipy.fft

from ephysatlas.feature_computation import add_target_coordinates

from .base import BaseFeatureCalculator
from .types import FeatureComputationOptions, RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)
LF_LATENCY_SAMPLES = 3


class SpikeGLXFileFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for local SpikeGLX AP/LF files.

    Args:
        ap_file (str | Path, optional): AP-band ``.bin`` or ``.cbin`` file.
        lf_file (str | Path, optional): LF-band ``.bin`` or ``.cbin`` file.
        name (str, optional): Recording identifier used as the OOP ``pid`` in
            outputs. Defaults to the AP file stem, LF file stem, or ``"probe"``.
        neuropixel_version (int): Neuropixels version passed to destriping.
        traj_dict (dict, optional): Optional trajectory dictionary used to add
            target coordinates to channel metadata.

    Raises:
        ValueError: If neither AP nor LF file is supplied.

    Note:
        Readers are opened lazily so importing and constructing this class does
        not touch the filesystem until data are requested.
    """

    def __init__(
        self,
        ap_file: str | Path | None = None,
        lf_file: str | Path | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
        traj_dict: dict | None = None,
    ) -> None:
        if ap_file is None and lf_file is None:
            raise ValueError("At least one of ap_file or lf_file must be provided")
        self.ap_file = Path(ap_file) if ap_file is not None else None
        self.lf_file = Path(lf_file) if lf_file is not None else None
        default_name = (
            self.ap_file.stem
            if self.ap_file is not None
            else self.lf_file.stem
            if self.lf_file is not None
            else "probe"
        )
        super().__init__(
            name=name or default_name, neuropixel_version=neuropixel_version
        )
        self.traj_dict = traj_dict
        self._sr_ap = None
        self._sr_lf = None

    @property
    def sr_ap(self):
        """Return the lazily opened AP ``spikeglx.Reader``."""
        if self.ap_file is None:
            return None
        if self._sr_ap is None:
            from spikeglx import Reader

            self._sr_ap = Reader(self.ap_file)
        return self._sr_ap

    @property
    def sr_lf(self):
        """Return the lazily opened LF ``spikeglx.Reader``."""
        if self.lf_file is None:
            return None
        if self._sr_lf is None:
            from spikeglx import Reader

            self._sr_lf = Reader(self.lf_file)
        return self._sr_lf

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read AP/LF snippets from SpikeGLX files.

        Args:
            window (SnippetWindow): Snippet time window to read.

        Returns:
            RawSnippet: Raw AP/LF arrays shaped ``(channels, samples)`` in volts.

        Raises:
            IndexError: If the underlying reader cannot access the requested
                samples.
        """
        raw_ap = raw_lf = fs_ap = fs_lf = None

        if self.sr_ap is not None:
            fs_ap = float(self.sr_ap.fs)
            ns_ap = scipy.fft.next_fast_len(int(fs_ap * window.duration_ap), real=True)
            n0_ap = int(fs_ap * window.t_start)
            n_channels_ap = self.sr_ap.nc - self.sr_ap.nsync
            raw_ap = self.sr_ap[slice(n0_ap, n0_ap + ns_ap), :n_channels_ap].T

        if self.sr_lf is not None:
            fs_lf = float(self.sr_lf.fs)
            ns_lf = scipy.fft.next_fast_len(int(fs_lf * window.duration_lf), real=True)
            n0_lf = int(fs_lf * window.t_start) + LF_LATENCY_SAMPLES
            n_channels_lf = self.sr_lf.nc - self.sr_lf.nsync
            raw_lf = self.sr_lf[slice(n0_lf, n0_lf + ns_lf), :n_channels_lf].T

        return RawSnippet(raw_ap=raw_ap, raw_lf=raw_lf, fs_ap=fs_ap, fs_lf=fs_lf)

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Load geometry from the AP reader, falling back to LF.

        Returns:
            dict[str, np.ndarray]: Geometry dictionary compatible with ibldsp.
        """
        reader = self.sr_ap if self.sr_ap is not None else self.sr_lf
        geometry = dict(reader.geometry)
        n_channels = len(geometry["x"])
        geometry.setdefault("sample_shift", np.zeros(n_channels))
        geometry.setdefault("shank", np.zeros(n_channels))
        geometry.setdefault(
            "col", np.unique(np.asarray(geometry["x"]), return_inverse=True)[1]
        )
        geometry.setdefault(
            "row", np.unique(np.asarray(geometry["y"]), return_inverse=True)[1]
        )
        return {key: np.asarray(value) for key, value in geometry.items()}

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from SpikeGLX geometry.

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
        """Return AP and LF durations from the SpikeGLX readers."""
        max_ap = self.sr_ap.ns / self.sr_ap.fs if self.sr_ap is not None else None
        max_lf = self.sr_lf.ns / self.sr_lf.fs if self.sr_lf is not None else None
        return max_ap, max_lf

    def _resolve_channel_labels(
        self,
        raw: RawSnippet,
        channels: pd.DataFrame,
        channel_labels: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return bad-channel labels using the legacy SpikeGLX file behavior.

        Args:
            raw (RawSnippet): Loaded AP/LF snippet used as a fallback when the
                full AP file is unavailable.
            channels (pd.DataFrame): Channel metadata that may already include
                channel labels.
            channel_labels (np.ndarray, optional): Explicit labels supplied by
                the caller.

        Returns:
            np.ndarray | None: Channel labels for AP/LF destriping.
        """
        if channel_labels is not None:
            return channel_labels
        if "labels" in channels.columns or "channel_labels" in channels.columns:
            return super()._resolve_channel_labels(raw, channels, channel_labels)
        if self.sr_ap is not None and self.sr_ap.file_bin is not None:
            return ibldsp.voltage.detect_bad_channels_cbin(self.sr_ap.file_bin)
        return super()._resolve_channel_labels(raw, channels, channel_labels)

    def enrich_channel_metadata(
        self, channels: pd.DataFrame, options: FeatureComputationOptions
    ) -> pd.DataFrame:
        """Add optional trajectory target coordinates for local file sources.

        Args:
            channels (pd.DataFrame): Channel metadata.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: Channel metadata with optional ``x_target``,
            ``y_target``, and ``z_target`` columns.

        Raises:
            ValueError: If trajectory metadata are required but missing.
        """
        if not options.include_trajectory:
            return channels
        if self.traj_dict is None:
            if options.require_trajectory:
                raise ValueError(f"No trajectory dictionary available for {self.name}")
            LOGGER.info("No trajectory dictionary available for %s", self.name)
            return channels

        channel_dict = {column: channels[column].to_numpy() for column in channels}
        enriched = add_target_coordinates(
            channels=channel_dict, traj_dict=self.traj_dict
        )
        return pd.DataFrame(enriched)
