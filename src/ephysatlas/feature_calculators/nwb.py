"""NWB feature calculator and its acquisition-source resolver.

An NWB recording can be read from several backends (HDF5 ``.nwb`` / Zarr
``.nwb.zarr``) and several locations (local file, direct S3/HTTPS stream, or a
DANDI asset addressed by dandiset id + version + filepath). Rather than a class
per (backend x location) cell, those two axes are handled by **composition**: a
:class:`NwbSource` value object describes *how to obtain one recording*, and
:class:`NwbFeatureCalculator` holds one source per band. The reader-contract logic
is inherited from
:class:`ephysatlas.feature_calculators.spikeinterface_like.SpikeInterfaceFeatureCalculator`.

All heavy, optional dependencies (SpikeInterface, pynwb, dandi, remfile) are
imported lazily inside :meth:`NwbSource.to_recording`, so importing this module
stays cheap and does not require the ``[full]`` install.

Classes
-------
NwbSource
    Describes how to open one NWB-backed SpikeInterface recording.
NwbFeatureCalculator
    Compute OOP features from NWB recordings (local, streamed, or DANDI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .spikeinterface_like import SpikeInterfaceFeatureCalculator

LOGGER = logging.getLogger(__name__)

# Shown when an optional dependency is missing; the NWB stack ships in [full].
_NWB_IMPORT_HINT = (
    "NWB feature computation needs the optional NWB/streaming dependencies "
    "(spikeinterface, pynwb, hdmf-zarr, remfile, dandi). "
    "Install them with `pip install ibleatools[full]`."
)


@dataclass(frozen=True)
class NwbSource:
    """Describes how to obtain one SpikeInterface recording from an NWB asset.

    This is the composition object for the acquisition axis: backend (HDF5/Zarr)
    and location (local/remote/DANDI) are fields here instead of subclasses. Build
    one with the named constructors :meth:`local`, :meth:`remote`, or
    :meth:`dandi`, then call :meth:`to_recording`.

    Attributes:
        kind (str): One of ``"local"``, ``"remote"``, or ``"dandi"``.
        path (str | None): Local path (``local``) or S3/HTTPS URL (``remote``).
        dandiset_id (str | None): DANDI dataset id (``dandi``).
        filepath (str | None): Asset path within the dandiset (``dandi``).
        version (str): DANDI version, e.g. ``"draft"`` or a published version.
        backend (str): ``"auto"`` (detect from the extension), ``"hdf5"``, or
            ``"zarr"``.
        electrical_series (str | None): Which ``ElectricalSeries`` to read. One NWB
            file often holds both an AP and an LFP series, so this selects the band.
        stream_mode (str | None): SpikeInterface stream mode override. When ``None``
            a sensible default is chosen per kind/backend (``None`` for local,
            ``"remfile"`` for streamed HDF5, ``"zarr"`` for streamed Zarr).
    """

    kind: str
    path: str | None = None
    dandiset_id: str | None = None
    filepath: str | None = None
    version: str = "draft"
    backend: str = "auto"
    electrical_series: str | None = None
    stream_mode: str | None = None

    @classmethod
    def local(
        cls,
        path: str | Path,
        backend: str = "auto",
        electrical_series: str | None = None,
    ) -> "NwbSource":
        """Build a source for a local NWB file (HDF5 or Zarr)."""
        return cls(
            kind="local",
            path=str(path),
            backend=backend,
            electrical_series=electrical_series,
        )

    @classmethod
    def remote(
        cls,
        url: str,
        backend: str = "auto",
        electrical_series: str | None = None,
        stream_mode: str | None = None,
    ) -> "NwbSource":
        """Build a source that streams a remote NWB file from an S3/HTTPS URL.

        ``stream_mode=None`` auto-selects ``"remfile"`` (HDF5) or ``"zarr"`` from
        the backend; pass an explicit mode to override.
        """
        return cls(
            kind="remote",
            path=str(url),
            backend=backend,
            electrical_series=electrical_series,
            stream_mode=stream_mode,
        )

    @classmethod
    def dandi(
        cls,
        dandiset_id: str,
        filepath: str,
        version: str = "draft",
        backend: str = "auto",
        electrical_series: str | None = None,
        stream_mode: str | None = None,
    ) -> "NwbSource":
        """Build a source that streams a DANDI asset (resolved to its S3 URL).

        ``stream_mode=None`` auto-selects ``"remfile"`` (HDF5) or ``"zarr"`` from
        the backend; pass an explicit mode to override.
        """
        return cls(
            kind="dandi",
            dandiset_id=dandiset_id,
            filepath=filepath,
            version=version,
            backend=backend,
            electrical_series=electrical_series,
            stream_mode=stream_mode,
        )

    def _resolve_url(self) -> str:
        """Return the file path/URL to hand SpikeInterface.

        For ``local``/``remote`` this is ``path`` verbatim. For ``dandi`` the asset
        is resolved to a streamable S3 URL via the DANDI API.
        """
        if self.kind in ("local", "remote"):
            return self.path
        try:
            from dandi.dandiapi import DandiAPIClient
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_NWB_IMPORT_HINT) from exc
        with DandiAPIClient() as client:
            asset = client.get_dandiset(
                self.dandiset_id, self.version
            ).get_asset_by_path(self.filepath)
            return asset.get_content_url(follow_redirects=1, strip_query=True)

    def _resolved_backend(self) -> str:
        """Return the effective backend (``"hdf5"``/``"zarr"``).

        Auto-detection uses the user-facing asset name (``filepath`` for DANDI,
        else ``path``) -- a resolved DANDI S3 content URL is keyed by blob/UUID and
        has no meaningful extension, so it must not be used for detection.
        """
        if self.backend != "auto":
            return self.backend
        name = self.filepath if self.kind == "dandi" else (self.path or "")
        return "zarr" if str(name).rstrip("/").endswith(".zarr") else "hdf5"

    def to_recording(self):
        """Open this source as a SpikeInterface ``BaseRecording`` (lazy imports).

        Returns:
            A SpikeInterface ``NwbRecordingExtractor``.

        Raises:
            ImportError: If the optional NWB/streaming dependencies are missing.
        """
        try:
            from spikeinterface.extractors import NwbRecordingExtractor
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_NWB_IMPORT_HINT) from exc

        url = self._resolve_url()
        backend = self._resolved_backend()

        # There is no separate backend kwarg on NwbRecordingExtractor: the backend
        # is expressed through stream_mode. Zarr is read via stream_mode="zarr"
        # (works for both local and remote zarr stores); remote HDF5 streams via
        # "remfile"; a local HDF5 file needs no stream_mode. An explicit
        # self.stream_mode always wins.
        stream_mode = self.stream_mode
        if stream_mode is None:
            if backend == "zarr":
                stream_mode = "zarr"
            elif self.kind != "local":
                stream_mode = "remfile"

        kwargs: dict = {}
        if self.electrical_series is not None:
            kwargs["electrical_series_path"] = self.electrical_series
        if stream_mode is not None:
            kwargs["stream_mode"] = stream_mode

        LOGGER.info(
            "Opening NWB recording (kind=%s, backend=%s, stream_mode=%s)",
            self.kind,
            backend,
            stream_mode,
        )
        return NwbRecordingExtractor(file_path=url, **kwargs)


class NwbFeatureCalculator(SpikeInterfaceFeatureCalculator):
    """Feature calculator for NWB recordings (local, streamed, or DANDI).

    Holds one :class:`NwbSource` per band and opens each lazily through the shared
    SpikeInterface reader-contract logic. Use the constructor with explicit
    :class:`NwbSource` objects, or the :meth:`from_local` / :meth:`from_url` /
    :meth:`from_dandi` convenience constructors.

    Args:
        ap_source (NwbSource, optional): Source for the AP band.
        lf_source (NwbSource, optional): Source for the LF band.
        name (str, optional): Output identifier. Defaults to a source-derived stem.
        neuropixel_version (int): Neuropixels version passed to destriping.

    Raises:
        ValueError: If neither an AP nor an LF source is provided.
    """

    def __init__(
        self,
        ap_source: NwbSource | None = None,
        lf_source: NwbSource | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> None:
        if ap_source is None and lf_source is None:
            raise ValueError("At least one of ap_source or lf_source must be provided")
        self.ap_source = ap_source
        self.lf_source = lf_source
        super().__init__(
            name=name or self._default_name(), neuropixel_version=neuropixel_version
        )

    def _default_name(self) -> str:
        """Derive a recording name from the first available source."""
        for source in (self.ap_source, self.lf_source):
            if source is None:
                continue
            if source.kind == "dandi" and source.filepath is not None:
                return Path(source.filepath).stem
            if source.path is not None:
                return Path(source.path).stem
        return "probe"

    def _open_recording(self, band: str):
        """Resolve the band's :class:`NwbSource` into a SpikeInterface recording."""
        source = self.ap_source if band == "ap" else self.lf_source
        if source is None:
            return None
        return source.to_recording()

    @classmethod
    def from_local(
        cls,
        ap_path: str | Path | None = None,
        lf_path: str | Path | None = None,
        backend: str = "auto",
        ap_electrical_series: str | None = None,
        lf_electrical_series: str | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> "NwbFeatureCalculator":
        """Build a calculator from local NWB file paths."""
        ap_source = (
            NwbSource.local(
                ap_path, backend=backend, electrical_series=ap_electrical_series
            )
            if ap_path is not None
            else None
        )
        lf_source = (
            NwbSource.local(
                lf_path, backend=backend, electrical_series=lf_electrical_series
            )
            if lf_path is not None
            else None
        )
        return cls(
            ap_source=ap_source,
            lf_source=lf_source,
            name=name,
            neuropixel_version=neuropixel_version,
        )

    @classmethod
    def from_url(
        cls,
        ap_url: str | None = None,
        lf_url: str | None = None,
        backend: str = "auto",
        ap_electrical_series: str | None = None,
        lf_electrical_series: str | None = None,
        stream_mode: str | None = "remfile",
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> "NwbFeatureCalculator":
        """Build a calculator that streams NWB files from S3/HTTPS URLs."""
        ap_source = (
            NwbSource.remote(
                ap_url,
                backend=backend,
                electrical_series=ap_electrical_series,
                stream_mode=stream_mode,
            )
            if ap_url is not None
            else None
        )
        lf_source = (
            NwbSource.remote(
                lf_url,
                backend=backend,
                electrical_series=lf_electrical_series,
                stream_mode=stream_mode,
            )
            if lf_url is not None
            else None
        )
        return cls(
            ap_source=ap_source,
            lf_source=lf_source,
            name=name,
            neuropixel_version=neuropixel_version,
        )

    @classmethod
    def from_dandi(
        cls,
        dandiset_id: str,
        ap_filepath: str | None = None,
        lf_filepath: str | None = None,
        version: str = "draft",
        backend: str = "auto",
        ap_electrical_series: str | None = None,
        lf_electrical_series: str | None = None,
        stream_mode: str | None = "remfile",
        name: str | None = None,
        neuropixel_version: int = 1,
    ) -> "NwbFeatureCalculator":
        """Build a calculator that streams NWB assets from a DANDI dataset.

        A single NWB file holding both bands is expressed by passing the same
        ``filepath`` for AP and LF with different ``*_electrical_series`` selectors.
        """
        ap_source = (
            NwbSource.dandi(
                dandiset_id,
                ap_filepath,
                version=version,
                backend=backend,
                electrical_series=ap_electrical_series,
                stream_mode=stream_mode,
            )
            if ap_filepath is not None
            else None
        )
        lf_source = (
            NwbSource.dandi(
                dandiset_id,
                lf_filepath,
                version=version,
                backend=backend,
                electrical_series=lf_electrical_series,
                stream_mode=stream_mode,
            )
            if lf_filepath is not None
            else None
        )
        return cls(
            ap_source=ap_source,
            lf_source=lf_source,
            name=name,
            neuropixel_version=neuropixel_version,
        )
