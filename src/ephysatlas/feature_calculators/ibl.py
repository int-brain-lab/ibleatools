"""IBL PID-backed feature calculator.

This module implements a concrete calculator for IBL recordings loaded through
``brainbox.io.one.SpikeSortingLoader`` and a ONE client.

Classes
-------
IBLPIDFeatureCalculator
    Compute OOP features from IBL probe insertion identifiers.
"""

from __future__ import annotations

import logging

import ibldsp.voltage
import numpy as np
import pandas as pd
import scipy.fft

from ephysatlas.feature_computation import add_target_coordinates

from .base import BaseFeatureCalculator
from .spikeglx import LF_LATENCY_SAMPLES
from .types import FeatureComputationOptions, RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


class IBLPIDFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for an IBL probe insertion loaded through ONE.

    Args:
        pid (str): Probe insertion ID.
        one: ONE or OneSdsc client.
        eid (str, optional): Session ID. Required for OneSdsc clients.
        probe_name (str, optional): Probe name. Required for OneSdsc clients.
        name (str, optional): Output identifier. Defaults to ``pid``.
        neuropixel_version (int): Neuropixels version passed to destriping.

    Raises:
        AssertionError: If an OneSdsc client is used without ``eid`` or
            ``probe_name``.

    Note:
        AP/LF readers and channels are loaded lazily and cached on the instance.
    """

    def __init__(
        self,
        pid: str,
        one,
        eid: str | None = None,
        probe_name: str | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> None:
        super().__init__(name=name or pid, neuropixel_version=neuropixel_version)
        self.pid = pid
        self.one = one
        self.eid = eid
        self.probe_name = probe_name
        self._ssl = None
        self._sr_ap = None
        self._sr_lf = None
        self._channels: dict | None = None

    @property
    def ssl(self):
        """Return the lazily constructed ``SpikeSortingLoader``."""
        if self._ssl is None:
            from brainbox.io.one import SpikeSortingLoader

            if self.one.__class__.__name__ == "OneSdsc":
                assert self.eid is not None, "eid is required for OneSdsc"
                assert self.probe_name is not None, "probe_name is required for OneSdsc"
                self._ssl = SpikeSortingLoader(
                    pid=self.pid, eid=self.eid, pname=self.probe_name, one=self.one
                )
            else:
                self._ssl = SpikeSortingLoader(pid=self.pid, one=self.one)
        return self._ssl

    @property
    def stream(self) -> bool:
        """Return whether raw electrophysiology should be streamed."""
        return self.one.__class__.__name__ != "OneSdsc"

    @property
    def sr_ap(self):
        """Return the lazily loaded AP raw electrophysiology reader."""
        if self._sr_ap is None:
            self._sr_ap = self.ssl.raw_electrophysiology(band="ap", stream=self.stream)
        return self._sr_ap

    @property
    def sr_lf(self):
        """Return the lazily loaded LF raw electrophysiology reader."""
        if self._sr_lf is None:
            self._sr_lf = self.ssl.raw_electrophysiology(band="lf", stream=self.stream)
        return self._sr_lf

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read AP/LF snippets from IBL raw electrophysiology readers.

        Args:
            window (SnippetWindow): Snippet time window to read.

        Returns:
            RawSnippet: Raw AP/LF arrays shaped ``(channels, samples)`` in volts.
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
        """Load ibldsp geometry from AP, falling back to LF.

        Returns:
            dict[str, np.ndarray]: Geometry dictionary compatible with ibldsp.
        """
        reader = self.sr_ap if self.sr_ap is not None else self.sr_lf
        geometry = dict(reader.geometry)
        n_channels = len(geometry["x"])
        # Fill missing geometry keys with derived defaults, but warn first: a
        # missing sample_shift/shank silently changes destriping, so a real reader
        # lacking these usually signals a problem worth surfacing.
        derived_defaults = {
            "sample_shift": lambda: np.zeros(n_channels),
            "shank": lambda: np.zeros(n_channels),
            "col": lambda: np.unique(np.asarray(geometry["x"]), return_inverse=True)[1],
            "row": lambda: np.unique(np.asarray(geometry["y"]), return_inverse=True)[1],
        }
        for key, make_default in derived_defaults.items():
            if key not in geometry:
                LOGGER.warning(
                    "Geometry missing '%s' for %s; using a derived default",
                    key,
                    self.name,
                )
                geometry[key] = make_default()
        return {key: np.asarray(value) for key, value in geometry.items()}

    def _load_channels_dict(self) -> dict:
        """Load channels once from ``SpikeSortingLoader`` with geometry fallback."""
        if self._channels is not None:
            return self._channels
        try:
            channels = dict(self.ssl.load_channels())
        except KeyError as exc:
            LOGGER.info("Channels key was not found for %s: %s", self.pid, exc)
            channels = {}
        except Exception:
            LOGGER.error("Failed to load channels for %s", self.pid, exc_info=True)
            channels = {}

        geometry = self.load_geometry()
        if "axial_um" not in channels and "y" in geometry:
            channels["axial_um"] = geometry["y"]
        if "lateral_um" not in channels and "x" in geometry:
            channels["lateral_um"] = geometry["x"]
        if "rawInd" not in channels and "channel" not in channels:
            channels["rawInd"] = np.arange(len(geometry["x"]))
        self._channels = channels
        return self._channels

    def load_channel_metadata(self) -> pd.DataFrame:
        """Load channel metadata from ONE/ALF with geometry fallback.

        Returns:
            pd.DataFrame: Channel metadata keyed by ``channel``.
        """
        channels = self._load_channels_dict()
        df_channels = pd.DataFrame(channels)
        if "channel" not in df_channels.columns and "rawInd" in df_channels.columns:
            df_channels["channel"] = df_channels["rawInd"]
        return df_channels

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return AP and LF durations from the IBL raw readers."""
        max_ap = self.sr_ap.ns / self.sr_ap.fs if self.sr_ap is not None else None
        max_lf = self.sr_lf.ns / self.sr_lf.fs if self.sr_lf is not None else None
        return max_ap, max_lf

    def enrich_channel_metadata(
        self, channels: pd.DataFrame, options: FeatureComputationOptions
    ) -> pd.DataFrame:
        """Add optional target coordinates from Alyx trajectories.

        Args:
            channels (pd.DataFrame): Channel metadata.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: Channel metadata with optional ``x_target``,
            ``y_target``, and ``z_target`` columns.

        Raises:
            ValueError: If trajectory metadata are required but cannot be loaded.
        """
        if not options.include_trajectory:
            return channels
        required = {"x_target", "y_target", "z_target"}
        if required.issubset(channels.columns):
            return channels

        channel_dict = {column: channels[column].to_numpy() for column in channels}
        try:
            enriched = add_target_coordinates(
                pid=self.pid,
                one=self.one,
                channels=channel_dict,
            )
        except Exception as exc:
            if options.require_trajectory:
                raise ValueError(
                    f"Failed to load trajectory metadata for {self.pid}"
                ) from exc
            LOGGER.warning(
                "No trajectory information available for %s; continuing without it",
                self.pid,
            )
            return channels
        return pd.DataFrame(enriched)

    def _resolve_channel_labels(
        self,
        raw: RawSnippet,
        channels: pd.DataFrame,
        channel_labels: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Resolve bad-channel labels, preferring whole-recording cbin detection.

        Mirrors ``online_feature_computation``: explicit/stored labels win, then
        ``detect_bad_channels_cbin`` when a local ``.cbin`` is available (e.g.
        pre-downloaded or SDSC readers whose ``file_bin`` is set), otherwise fall
        back to the base-class snippet-level detection. Streamed readers have
        ``file_bin=None`` and therefore keep using the snippet fallback.
        """
        has_stored = (
            channel_labels is not None
            or "labels" in channels.columns
            or "channel_labels" in channels.columns
        )
        if (
            not has_stored
            and self.sr_ap is not None
            and self.sr_ap.file_bin is not None
        ):
            return ibldsp.voltage.detect_bad_channels_cbin(self.sr_ap.file_bin)
        return super()._resolve_channel_labels(raw, channels, channel_labels)
