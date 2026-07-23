"""Shared base for SpikeInterface ``BaseRecording``-backed feature calculators.

This is the SpikeInterface twin of
:class:`ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`.
Where that class targets the ``spikeglx.Reader`` contract, this one targets the
SpikeInterface ``BaseRecording`` contract (``get_traces`` / ``get_sampling_frequency``
/ ``get_num_frames`` / ``get_channel_locations`` / ``get_property``). Every step that
only needs that contract -- raw-snippet slicing, geometry, durations, channel
metadata -- is implemented here once, so concrete calculators differ only in how
they open the underlying recording (see :meth:`_open_recording`).

Classes
-------
SpikeInterfaceFeatureCalculator
    Reader-contract logic shared by SpikeInterface-backed calculators.
"""

from __future__ import annotations

import abc
import logging

import numpy as np
import pandas as pd
import scipy.fft

from .base import BaseFeatureCalculator
from .spikeglx_like import LF_LATENCY_SAMPLES
from .types import RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


class SpikeInterfaceFeatureCalculator(BaseFeatureCalculator):
    """Base for calculators backed by a SpikeInterface ``BaseRecording``.

    Subclasses implement :meth:`_open_recording`; everything that only needs the
    recording's ``get_traces`` / ``get_sampling_frequency`` / ``get_num_frames`` /
    ``get_channel_locations`` / ``get_property`` API is implemented here once.

    Args:
        name (str): Recording identifier, used as the OOP ``pid`` in outputs.
        neuropixel_version (int): Neuropixels version passed to destriping.

    Note:
        Recordings are opened lazily and cached on first access. SpikeInterface
        returns traces as ``(samples, channels)`` and, with ``return_in_uV=True``,
        in microvolts; this class transposes to ``(channels, samples)`` and
        converts to volts before feature computation.
    """

    def __init__(self, name: str, neuropixel_version: int = 1) -> None:
        super().__init__(name=name, neuropixel_version=neuropixel_version)
        self._rec_ap = None
        self._rec_lf = None
        self._geometry: dict[str, np.ndarray] | None = None

    @abc.abstractmethod
    def _open_recording(self, band: str):
        """Open and return the SpikeInterface ``BaseRecording`` for a band.

        Args:
            band (str): Either ``"ap"`` or ``"lf"``.

        Returns:
            A SpikeInterface ``BaseRecording`` exposing the extractor interface, or
            ``None`` when the band is not available for this source.
        """
        raise NotImplementedError

    @property
    def rec_ap(self):
        """Return the lazily opened AP recording (``None`` if unavailable)."""
        if self._rec_ap is None:
            self._rec_ap = self._open_recording("ap")
        return self._rec_ap

    @property
    def rec_lf(self):
        """Return the lazily opened LF recording (``None`` if unavailable)."""
        if self._rec_lf is None:
            self._rec_lf = self._open_recording("lf")
        return self._rec_lf

    @property
    def _reference_recording(self):
        """Return whichever recording is available, preferring AP."""
        return self.rec_ap if self.rec_ap is not None else self.rec_lf

    def _read_traces_volts(
        self, recording, start_frame: int, n_samples: int
    ) -> np.ndarray:
        """Read traces from SpikeInterface and convert microvolts to volts.

        Args:
            recording: SpikeInterface ``BaseRecording``.
            start_frame (int): First sample to read.
            n_samples (int): Number of samples to read.

        Returns:
            np.ndarray: ``(channels, samples)`` array in volts.

        Note:
            ``next_fast_len`` (and the LF latency offset) can push the requested
            window past the last sample even when the duration validated against
            ``available_duration``. Unlike numpy slicing (which the spikeglx path
            uses and which silently clips), SpikeInterface ``get_traces`` raises on
            an out-of-range ``end_frame``. We therefore shift the window back to
            keep the requested sample count while staying in bounds.
        """
        n_frames = recording.get_num_frames()
        start = int(start_frame)
        n = int(n_samples)
        if start + n > n_frames:  # window runs past the end -> shift it back
            start = max(0, n_frames - n)
        end = min(start + n, n_frames)
        traces_uv = recording.get_traces(
            start_frame=start, end_frame=end, return_in_uV=True
        )
        return traces_uv.T.astype(np.float32) * 1e-6

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read AP/LF snippets from the SpikeInterface recordings.

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
            # LF is read a few samples late to compensate for the LF filter latency
            # relative to the AP band (shared constant with the spikeglx path).
            n0_lf = int(fs_lf * window.t_start) + LF_LATENCY_SAMPLES
            raw_lf = self._read_traces_volts(self.rec_lf, n0_lf, ns_lf)

        return RawSnippet(raw_ap=raw_ap, raw_lf=raw_lf, fs_ap=fs_ap, fs_lf=fs_lf)

    # (horizontal, vertical) electrode-property name pairs used as a geometry
    # fallback when SpikeInterface has no channel locations. Ordered most-standard
    # first: ``rel_x``/``rel_y`` is the NWB extracellular-channels convention;
    # ``probe_horizontal_position``/``probe_vertical_position`` is Allen's.
    _POSITION_PROPERTY_PAIRS = (
        ("rel_x", "rel_y"),
        ("probe_horizontal_position", "probe_vertical_position"),
    )

    def _channel_locations(self, recording) -> np.ndarray:
        """Return ``(n_channels, 2)`` ``[x=horizontal, y=vertical]`` sites in um.

        Prefers SpikeInterface channel locations; when those are unset (common for
        real DANDI NWB, where SpikeInterface raises), falls back to electrode-table
        position properties whose names vary by NWB convention.

        Raises:
            ValueError: If neither channel locations nor any known position
                property pair is available.
        """
        try:
            return np.asarray(recording.get_channel_locations(), dtype=float)
        except Exception as exc:
            # SpikeInterface raises a bare Exception when locations are not set; it
            # can also raise for other reasons, so log the actual message
            # before falling back to electrode-table properties.
            LOGGER.info(
                "get_channel_locations() unavailable for %s (%s); falling back to "
                "electrode position properties",
                self.name,
                exc,
            )

        available = set(recording.get_property_keys())
        for x_key, y_key in self._POSITION_PROPERTY_PAIRS:
            if x_key in available and y_key in available:
                x = np.asarray(recording.get_property(x_key), dtype=float)
                y = np.asarray(recording.get_property(y_key), dtype=float)
                LOGGER.info(
                    "Using electrode properties %r/%r as probe geometry for %s",
                    x_key,
                    y_key,
                    self.name,
                )
                return np.column_stack([x, y])

        known = [key for pair in self._POSITION_PROPERTY_PAIRS for key in pair]
        raise ValueError(
            f"Cannot determine probe geometry for {self.name}: SpikeInterface has "
            f"no channel locations and none of the known electrode position "
            f"properties {known} are present. Available properties: "
            f"{sorted(available)}."
        )

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Build ibldsp geometry from SpikeInterface channel properties.

        ``x``/``y`` come from ``get_channel_locations`` (with an electrode-property
        fallback, see :meth:`_channel_locations`); ``sample_shift`` from the
        ``inter_sample_shift`` property (zeros with a warning when absent, since a
        missing shift silently changes destriping); ``col``/``row`` are derived
        from the site coordinates and ``shank`` defaults to a single shank.

        Returns:
            dict[str, np.ndarray]: Geometry dictionary compatible with ibldsp.
        """
        if self._geometry is not None:
            return self._geometry

        rec = self._reference_recording
        # TODO(per-shank): multi-shank probes (e.g. Neuropixels 2.0) are not
        # supported yet. Destriping, CSD, and the physical-site channel merge all
        # assume a single shank, and features should be computed one shank at a
        # time. Until per-shank splitting is implemented, refuse a recording that
        # spans more than one shank rather than silently labelling every channel
        # shank=0. Detection uses SpikeInterface channel groups, which normally
        # encode shank; when the recording exposes none, we assume a single shank.
        try:
            shanks = np.unique(rec.get_channel_groups())
        except Exception:
            shanks = np.array([0])
        if shanks.size > 1:
            raise NotImplementedError(
                f"{self.name}: recording spans {shanks.size} shanks/groups "
                f"({shanks.tolist()}); per-shank feature computation is not "
                "implemented yet. Pass a single shank's channels per calculator "
                "(one ElectricalSeries per shank, or a per-shank channel subset)."
            )

        locations = self._channel_locations(rec)
        n_channels = locations.shape[0]
        sample_shift = rec.get_property("inter_sample_shift")
        if sample_shift is None:
            LOGGER.warning(
                "No inter_sample_shift property on %s; using zeros", self.name
            )
            sample_shift = np.zeros(n_channels, dtype=float)

        x = locations[:, 0].astype(float)
        y = locations[:, 1].astype(float)
        self._geometry = {
            "x": x,
            "y": y,
            "col": np.unique(x, return_inverse=True)[1].astype(float),
            "row": np.unique(y, return_inverse=True)[1].astype(float),
            "sample_shift": np.asarray(sample_shift, dtype=float),
            # single shank only for now (multi-shank guarded above; see TODO)
            "shank": np.zeros(n_channels, dtype=float),
        }
        return self._geometry

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from the recording geometry.

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

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return AP and LF durations (seconds) from the recordings."""
        max_ap = None
        if self.rec_ap is not None:
            max_ap = self.rec_ap.get_num_frames() / self.rec_ap.get_sampling_frequency()
        max_lf = None
        if self.rec_lf is not None:
            max_lf = self.rec_lf.get_num_frames() / self.rec_lf.get_sampling_frequency()
        return max_ap, max_lf
