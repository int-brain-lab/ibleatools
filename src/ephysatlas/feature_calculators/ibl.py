"""IBL PID-backed feature calculator.

This module implements a concrete calculator for IBL recordings loaded through
``brainbox.io.one.SpikeSortingLoader`` and a ONE client. The reader-contract
logic (raw snippets, geometry, durations, channel labels) is inherited from
:class:`ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`.

Classes
-------
IBLPIDFeatureCalculator
    Compute OOP features from IBL probe insertion identifiers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ephysatlas.feature_computation import add_target_coordinates

from .spikeglx_like import SpikeGlxLikeFeatureCalculator
from .types import FeatureComputationOptions

LOGGER = logging.getLogger(__name__)


class IBLPIDFeatureCalculator(SpikeGlxLikeFeatureCalculator):
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

    def _open_reader(self, band: str):
        """Open the IBL raw-electrophysiology reader for a band."""
        return self.ssl.raw_electrophysiology(band=band, stream=self.stream)

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
        # shank feeds the physical-site channel merge in the base class.
        if "shank" not in channels and "shank" in geometry:
            channels["shank"] = geometry["shank"]
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
