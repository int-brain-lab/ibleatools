"""SpikeInterface-native Zarr feature calculator.

Reads recordings saved in the SpikeInterface Zarr layout (``read_zarr``) -- e.g. the
AIND / open-ephys WavPack-compressed ``ecephys_compressed/*.zarr`` stores -- for local
paths and remote object stores (S3/GCS) alike. This is the SI-native-Zarr sibling of
:class:`ephysatlas.feature_calculators.nwb.NwbFeatureCalculator`: both subclass
:class:`ephysatlas.feature_calculators.spikeinterface_like.SpikeInterfaceFeatureCalculator`
and differ only in how they open a recording.

Note: SI-native Zarr (a ``ZarrRecordingExtractor`` store with ``traces_seg0`` etc.) is
NOT NWB-Zarr; NWB files -- including NWB with a Zarr backend -- are read by
:class:`~ephysatlas.feature_calculators.nwb.NwbFeatureCalculator` instead.

Classes
-------
SpikeInterfaceZarrFeatureCalculator
    Compute OOP features from local/remote SpikeInterface Zarr recordings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .spikeinterface_like import SpikeInterfaceFeatureCalculator

LOGGER = logging.getLogger(__name__)


class SpikeInterfaceZarrFeatureCalculator(SpikeInterfaceFeatureCalculator):
    """Feature calculator for SpikeInterface-native Zarr recordings.

    Args:
        ap_zarr (str | Path, optional): Path/URL of the AP-band Zarr store.
        lf_zarr (str | Path, optional): Path/URL of the LF-band Zarr store.
        name (str, optional): Output identifier. Defaults to a zarr-store-derived
            stem.
        neuropixel_version (int): Neuropixels version passed to destriping.
        storage_options (Mapping, optional): Forwarded to ``read_zarr`` for remote
            stores, e.g. ``{"anon": True}`` for a public S3 bucket. ``None`` for
            local stores.

    Raises:
        ValueError: If neither an AP nor an LF store is supplied.

    Note:
        Reading remote/compressed stores may need extra packages (``s3fs`` for S3,
        ``wavpack_numcodecs`` registered for AIND WavPack); import those in the
        caller. Stores are opened lazily via ``read_zarr`` on first access.
    """

    def __init__(
        self,
        ap_zarr: str | Path | None = None,
        lf_zarr: str | Path | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
        storage_options: Mapping[str, Any] | None = None,
    ) -> None:
        if ap_zarr is None and lf_zarr is None:
            raise ValueError("At least one of ap_zarr or lf_zarr must be provided")
        self.ap_zarr = str(ap_zarr) if ap_zarr is not None else None
        self.lf_zarr = str(lf_zarr) if lf_zarr is not None else None
        self.storage_options = storage_options
        super().__init__(
            name=name or self._default_name(), neuropixel_version=neuropixel_version
        )

    def _default_name(self) -> str:
        """Derive a recording name from the first available zarr-store stem."""
        for path in (self.ap_zarr, self.lf_zarr):
            if path is not None:
                return Path(path.rstrip("/")).stem
        return "probe"

    def _open_recording(self, band: str):
        """Open the band's Zarr store as a SpikeInterface recording via read_zarr."""
        path = self.ap_zarr if band == "ap" else self.lf_zarr
        if path is None:
            return None
        from spikeinterface.core import read_zarr

        kwargs: dict = {}
        if self.storage_options is not None:
            kwargs["storage_options"] = dict(self.storage_options)
        LOGGER.info("Opening SpikeInterface zarr (%s): %s", band, path)
        return read_zarr(path, **kwargs)
