"""Shared base for ``spikeglx.Reader``-backed feature calculators.

Both IBL recordings (streamed through ``brainbox.io.one.SpikeSortingLoader``) and
local SpikeGLX files are read through ``spikeglx.Reader``-like objects that expose
``.fs``, ``.ns``, ``.nc``, ``.nsync``, ``.geometry``, ``.file_bin`` and
``[slice, cols]`` indexing. This intermediate class implements every step that
depends only on that reader contract, so the concrete calculators differ only in
how they open the reader, load channel metadata, and source trajectory metadata.

Classes
-------
SpikeGlxLikeFeatureCalculator
    Reader-contract logic shared by the IBL and SpikeGLX-file calculators.
"""

from __future__ import annotations

import abc
import logging

import ibldsp.voltage
import numpy as np
import pandas as pd
import scipy.fft

from .base import BaseFeatureCalculator
from .types import RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)

# LF snippets are read a few samples late to compensate for the LF filter latency
# relative to the AP band.
LF_LATENCY_SAMPLES = 3


class SpikeGlxLikeFeatureCalculator(BaseFeatureCalculator):
    """Base for calculators backed by a ``spikeglx.Reader``-like AP/LF reader.

    Subclasses implement :meth:`_open_reader` (and the source-specific
    :meth:`load_channel_metadata` / :meth:`enrich_channel_metadata`); everything
    that only needs the reader's ``.fs``/``.ns``/``.nc``/``.nsync``/``.geometry``/
    ``.file_bin`` attributes and ``[slice, cols]`` indexing is implemented here
    once.

    Args:
        name (str): Recording identifier, used as the OOP ``pid`` in outputs.
        neuropixel_version (int): Neuropixels version passed to destriping.

    Note:
        Readers are opened lazily and cached on first access.
    """

    def __init__(self, name: str, neuropixel_version: int = 1) -> None:
        super().__init__(name=name, neuropixel_version=neuropixel_version)
        self._sr_ap = None
        self._sr_lf = None

    @abc.abstractmethod
    def _open_reader(self, band: str):
        """Open and return the ``spikeglx.Reader``-like object for a band.

        Args:
            band (str): Either ``"ap"`` or ``"lf"``.

        Returns:
            A reader exposing the spikeglx-like interface, or ``None`` when the
            band is not available for this source.
        """
        raise NotImplementedError

    @property
    def sr_ap(self):
        """Return the lazily opened AP reader (``None`` if unavailable)."""
        if self._sr_ap is None:
            self._sr_ap = self._open_reader("ap")
        return self._sr_ap

    @property
    def sr_lf(self):
        """Return the lazily opened LF reader (``None`` if unavailable)."""
        if self._sr_lf is None:
            self._sr_lf = self._open_reader("lf")
        return self._sr_lf

    def _join_probe_metadata(self, channels: pd.DataFrame) -> pd.DataFrame:
        """Broadcast probe-level SpikeGLX metadata onto every channel.

        Args:
            channels (pd.DataFrame): Channel metadata.

        Returns:
            pd.DataFrame: A copy of ``channels`` with ``probe_model`` and
            ``referencing_scheme`` columns, taken from the AP reader's SpikeGLX
            meta-data, falling back to the already-open LF reader. Both are
            ``pd.NA`` when neither carries meta-data (e.g. a reader opened
            without a companion .meta file).
        """
        import spikeglx

        # An AP reader can exist but carry no meta-data (no companion .meta file),
        # in which case LF can still supply it. The fallback reads the *already
        # opened* LF reader rather than the ``sr_lf`` property: opening a reader
        # costs an Alyx round-trip for streamed sources, and load_raw_snippet()
        # has opened both bands by the time this runs in the compute pipeline.
        meta = getattr(self.sr_ap, "meta", None)
        if meta is None:
            meta = getattr(self._sr_lf, "meta", None)

        probe_model = spikeglx.get_probe_model(meta) if meta is not None else pd.NA
        referencing = (
            spikeglx.get_referencing_scheme(meta) if meta is not None else pd.NA
        )
        # The explicit "string" dtype keeps an all-NA
        # column concat-compatible with a populated one when channel tables from
        # several probes are aggregated.
        return channels.assign(
            probe_model=pd.Series(probe_model, index=channels.index, dtype="string"),
            referencing_scheme=pd.Series(
                referencing, index=channels.index, dtype="string"
            ),
        )

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read AP/LF snippets from the spikeglx-like readers.

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
        """Load ibldsp geometry from the AP reader, falling back to LF.

        Missing derived keys (``sample_shift``, ``shank``, ``col``, ``row``) are
        filled with defaults, but a warning is logged first: a missing
        ``sample_shift``/``shank`` silently changes destriping, so a real reader
        lacking these usually signals a problem worth surfacing.

        Returns:
            dict[str, np.ndarray]: Geometry dictionary compatible with ibldsp.
        """
        reader = self.sr_ap if self.sr_ap is not None else self.sr_lf
        geometry = dict(reader.geometry)
        n_channels = len(geometry["x"])
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

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return AP and LF durations (seconds) from the readers."""
        max_ap = self.sr_ap.ns / self.sr_ap.fs if self.sr_ap is not None else None
        max_lf = self.sr_lf.ns / self.sr_lf.fs if self.sr_lf is not None else None
        return max_ap, max_lf

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
