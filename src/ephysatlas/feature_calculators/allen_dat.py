"""Allen Brain Observatory raw ``.dat`` feature calculator.

The Allen *Visual Coding - Neuropixels* release publishes its raw bands as
headerless flat int16 files on public S3::

    s3://allen-brain-observatory/visual-coding-neuropixels/raw-data/
        <session_id>/<probe_id>/spike_band.dat   # 384ch int16 @ 30 kHz
        <session_id>/<probe_id>/lfp_band.dat     # 384ch int16 @ 2.5 kHz

There is no ``settings.xml`` / ``structure.oebin`` / ``.meta`` companion, so the
sampling rate, channel count, scale and channel order all have to be supplied
externally. :class:`AllenDatReader` wraps such a file in the small
``spikeglx.Reader``-like interface that
:class:`~ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`
consumes, so every downstream feature, aggregation and denoise step is reused
unchanged.

Two properties of these files are easy to get wrong and fail silently:

* **Channel order.** Samples are stored in ADC order, not geometric order. The
  reader applies :data:`ALLEN_CHANNEL_ORDER` on every read so that column ``k``
  is always geometric channel ``k``.
* **Scale.** ``0.195`` uV/bit, for *both* bands (see
  :data:`MICROVOLTS_PER_BIT`).

Classes
-------
AllenDatReader
    Minimal spikeglx-like reader over one headerless int16 ``.dat`` on S3.
AllenDatFeatureCalculator
    Compute OOP features from an Allen raw AP/LF ``.dat`` pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .spikeglx_like import SpikeGlxLikeFeatureCalculator
from .types import FeatureComputationOptions

LOGGER = logging.getLogger(__name__)

N_CHANNELS = 384
DAT_DTYPE = np.dtype("<i2")

# int16 -> microvolts, for BOTH bands. From Allen's own pipeline,
# AllenInstitute/ecephys_spike_sorting, modules/extract_from_npx/
# create_settings_json.py, which sets bit_volts = 0.195 explicitly on the
# 'AP band' and 'LFP band' subprocessors, and scripts/create_input_json.py, which
# applies one bit_volts to both ap_band_file and lfp_band_file. Neuropixels does
# allow the two bands to use different gains, so this had to be checked rather
# than assumed.
#
# Deliberately a single constant, not a per-probe measurement. Measuring it
# against the DANDI LFP copy gives 0.17-0.20 across probes, but that spread is
# measurement noise (a real gain difference would be a factor of two or more, as
# gain is selected from 50/125/250/500/...). Amplitude features are absolute dB
# and cross-validation is grouped by probe, so a per-probe scale would inject
# ~1 dB of noise into exactly the axis the across-probe score measures.
MICROVOLTS_PER_BIT = 0.195

# Channels arrive on disk in ADC order. This remap puts geometric (``local_index``)
# channel k in column k, and is required for BOTH bands on BOTH probe generations
# (Phase 3A and PXI).
#
# Pattern from AllenInstitute/ecephys_spike_sorting, common/OEFileInfo.py::
# get_lfp_channel_order(). Allen's pipeline gates the equivalent fix on
# ``reorder_lfp_channels = probe_type == '3A'`` and applies it to the LFP band
# only, but that flag describes their internal intermediate files; on the
# published raw-data tree PXI needs it as much as Phase 3A, and AP as much as LFP.
# Verified in a 300-600 Hz band against the DANDI copy, which is stored in
# geometric order: remapped scores ~0.83 vs ~0.21 raw on PXI LFP, and 0.50 vs
# 0.17 on AP. A low-frequency check cannot see the difference -- LFP correlates
# 0.97 across four channels, which is finer than this remap displaces them.
#
# Unremapped, a channel sits up to 12 positions (240 um) from where it is
# believed to be, because disk columns 0-11 are the x-column-A sites of rows 0-11
# and disk 12-23 are the x-column-B sites of those same rows. Nothing raises; the
# spatial features are simply wrong.
_REMAP_BLOCK = np.array([
    0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19,
    8, 20, 9, 21, 10, 22, 11, 23, 24, 36, 25, 37, 26, 38,
    27, 39, 28, 40, 29, 41, 30, 42, 31, 43, 32, 44, 33, 45, 34, 46, 35, 47,
])
ALLEN_CHANNEL_ORDER = np.concatenate([_REMAP_BLOCK + 48 * i for i in range(8)])


class AllenDatReader:
    """Minimal ``spikeglx.Reader``-like view of one Allen raw ``.dat`` file.

    Implements only what
    :class:`~ephysatlas.feature_calculators.spikeglx_like.SpikeGlxLikeFeatureCalculator`
    needs: ``fs``, ``ns``, ``nc``, ``nsync``, ``geometry``, ``file_bin`` and
    ``[rows, cols]`` indexing returning volts.

    Args:
        url (str): ``s3://`` URL of the ``.dat`` file.
        fs (float): Sampling rate in Hz. Must be the per-probe value from
            ``probes.csv``: rates vary between probes (29999.92-30000.31 Hz), and
            a 0.4 Hz error accumulates to ~100 ms over a 3 h recording.
        storage_options (dict, optional): Passed to ``s3fs.S3FileSystem``.
            Defaults to anonymous access.
        remap_channels (bool): Apply :data:`ALLEN_CHANNEL_ORDER`. Leave ``True``
            for anything that uses channel positions.

    Raises:
        ValueError: If the file size is not an exact multiple of the frame size,
            which would mean the channel count or dtype assumption is wrong.

    Note:
        ``file_bin`` is ``None`` because the data are remote, so the base class
        falls back to snippet-level bad-channel detection rather than
        whole-recording ``detect_bad_channels_cbin``.
    """

    def __init__(
        self,
        url: str,
        fs: float,
        storage_options: dict | None = None,
        remap_channels: bool = True,
    ) -> None:
        import s3fs

        self.url = url
        self.fs = float(fs)
        self.nc = N_CHANNELS
        self.nsync = 0
        self.file_bin = None
        self._key = url.replace("s3://", "")
        self._fs = s3fs.S3FileSystem(**(storage_options or {"anon": True}))

        frame_bytes = self.nc * DAT_DTYPE.itemsize
        size = self._fs.size(self._key)
        if size % frame_bytes:
            raise ValueError(
                f"{url}: size {size} is not a multiple of {frame_bytes} bytes, so "
                f"the {self.nc}-channel int16 layout assumption is wrong"
            )
        self.ns = size // frame_bytes
        self._frame_bytes = frame_bytes
        self._order = (
            ALLEN_CHANNEL_ORDER if remap_channels else np.arange(self.nc)
        )
        self._handle = None

        # Allen's published channel positions match IBL's NP1 trace header
        # exactly (checked: max |dx| = max |dy| = 0.00 um), so use the trace
        # header, which additionally carries the sample_shift and shank that
        # destriping needs and channels.csv does not provide.
        import neuropixel

        self.geometry = neuropixel.trace_header(version=1)

    def _open(self):
        """Return a cached remote file handle, opening it on first use."""
        if self._handle is None:
            self._handle = self._fs.open(self._key, "rb")
        return self._handle

    def __getitem__(self, item) -> np.ndarray:
        """Read a window and return volts, shaped ``(samples, channels)``.

        Args:
            item: ``(rows, cols)`` where ``rows`` is a slice of samples.

        Returns:
            np.ndarray: float32 volts, channels in geometric order.
        """
        rows, cols = item if isinstance(item, tuple) else (item, slice(None))
        start = 0 if rows.start is None else max(0, int(rows.start))
        stop = self.ns if rows.stop is None else min(int(rows.stop), self.ns)
        n_samples = max(0, stop - start)

        handle = self._open()
        handle.seek(start * self._frame_bytes)
        raw = handle.read(n_samples * self._frame_bytes)
        window = np.frombuffer(raw, dtype=DAT_DTYPE).reshape(-1, self.nc)

        # Remap to geometric order first, then apply the caller's column slice,
        # so callers always index by geometric channel.
        window = window[:, self._order]
        return window[:, cols].astype(np.float32) * (MICROVOLTS_PER_BIT * 1e-6)

    def close(self) -> None:
        """Close the cached remote handle, if one is open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class AllenDatFeatureCalculator(SpikeGlxLikeFeatureCalculator):
    """Feature calculator for an Allen Visual Coding raw AP/LF ``.dat`` pair.

    Args:
        ap_url (str, optional): ``s3://`` URL of ``spike_band.dat``.
        lf_url (str, optional): ``s3://`` URL of ``lfp_band.dat``.
        ap_fs (float, optional): AP sampling rate from ``probes.csv``.
        lf_fs (float, optional): LF sampling rate from ``probes.csv``.
        name (str, optional): Recording identifier used as the OOP ``pid``.
        neuropixel_version (int): Neuropixels version passed to destriping.
        storage_options (dict, optional): Passed to ``s3fs.S3FileSystem``.
        remap_channels (bool): Apply :data:`ALLEN_CHANNEL_ORDER` on read.

    Raises:
        ValueError: If neither band is supplied, or a supplied band has no
            sampling rate (there is no metadata file to infer it from).

    Note:
        Readers are opened lazily, so constructing this class performs no I/O.
    """

    def __init__(
        self,
        ap_url: str | None = None,
        lf_url: str | None = None,
        ap_fs: float | None = None,
        lf_fs: float | None = None,
        name: str | None = None,
        neuropixel_version: int = 1,
        storage_options: dict | None = None,
        remap_channels: bool = True,
    ) -> None:
        if ap_url is None and lf_url is None:
            raise ValueError("At least one of ap_url or lf_url must be provided")
        # There is no .meta companion to fall back on, so a missing rate is fatal
        # rather than something to guess at.
        if ap_url is not None and ap_fs is None:
            raise ValueError("ap_fs is required when ap_url is given")
        if lf_url is not None and lf_fs is None:
            raise ValueError("lf_fs is required when lf_url is given")

        self.ap_url = ap_url
        self.lf_url = lf_url
        self.ap_fs = ap_fs
        self.lf_fs = lf_fs
        self.storage_options = storage_options or {"anon": True}
        self.remap_channels = remap_channels
        super().__init__(
            name=name or "allen_probe", neuropixel_version=neuropixel_version
        )

    def _open_reader(self, band: str):
        """Open the :class:`AllenDatReader` for a band (``None`` if absent)."""
        url = self.ap_url if band == "ap" else self.lf_url
        fs = self.ap_fs if band == "ap" else self.lf_fs
        if url is None:
            return None
        LOGGER.info("opening Allen %s band for %s: %s", band, self.name, url)
        return AllenDatReader(
            url,
            fs=fs,
            storage_options=self.storage_options,
            remap_channels=self.remap_channels,
        )

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from the NP1 trace-header geometry.

        Returns:
            pd.DataFrame: Columns ``channel``, ``rawInd``, ``axial_um``,
            ``lateral_um`` and ``shank``. ``channel`` is the geometric
            ``local_index``, which is what the reader's remap guarantees and what
            the Allen ``channels.csv`` region labels are keyed on.
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
        """Add probe metadata columns.

        These files carry no SpikeGLX meta-data, so ``probe_model`` and
        ``referencing_scheme`` come back as NA from the base implementation.

        Args:
            channels (pd.DataFrame): Channel metadata.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: Channel metadata with probe-level columns added.
        """
        return self._join_probe_metadata(channels)
