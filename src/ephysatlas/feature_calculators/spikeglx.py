"""SpikeGLX file-backed feature calculator.

This module implements a concrete calculator for local Neuropixels AP/LF
``.bin`` or ``.cbin`` files readable by ``spikeglx.Reader``. The reader-contract
logic (raw snippets, geometry, durations, channel labels) is inherited from
:class:`ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`.

Classes
-------
SpikeGLXFileFeatureCalculator
    Compute OOP features from local AP/LF SpikeGLX files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.feature_computation import add_target_coordinates

from .spikeglx_like import SpikeGlxLikeFeatureCalculator
from .types import FeatureComputationOptions

LOGGER = logging.getLogger(__name__)


class SpikeGLXFileFeatureCalculator(SpikeGlxLikeFeatureCalculator):
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

    def _open_reader(self, band: str):
        """Open the SpikeGLX ``Reader`` for a band (``None`` if the file is absent)."""
        file = self.ap_file if band == "ap" else self.lf_file
        if file is None:
            return None
        from spikeglx import Reader

        return Reader(file)

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from SpikeGLX geometry.

        Returns:
            pd.DataFrame: Columns ``channel``, ``rawInd``, ``axial_um``,
            ``lateral_um``, and ``shank`` (``shank`` feeds the physical-site
            channel merge in the base class).
        """
        geometry = self.load_geometry()
        n_channels = len(geometry["x"])
        return pd.DataFrame(
            {
                "channel": np.arange(n_channels),
                "rawInd": np.arange(n_channels),
                "axial_um": np.asarray(geometry["y"], dtype=float),
                "lateral_um": np.asarray(geometry["x"], dtype=float),
                "shank": np.asarray(geometry["shank"], dtype=float),
            }
        )

    def enrich_channel_metadata(
        self, channels: pd.DataFrame, options: FeatureComputationOptions
    ) -> pd.DataFrame:
        """Add probe metadata and optional trajectory target coordinates.

        Args:
            channels (pd.DataFrame): Channel metadata.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: Channel metadata with ``probe_model`` and
            ``referencing_scheme`` columns, plus optional ``x_target``,
            ``y_target``, and ``z_target`` columns.

        Raises:
            ValueError: If trajectory metadata are required but missing.
        """
        channels = self._join_probe_metadata(channels)
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
