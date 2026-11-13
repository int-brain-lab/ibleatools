"""The recorded probe as the localizer sees it: which channels it uses and how.

The localizer was fitted on IBL Neuropixels 1.0 insertions: one shank, 384 channels two per
20 µm row (bank 0), read as rows 10 µm apart from the top of the probe
(:attr:`ScoreConfig.channel_spacing_um`), with an LF band. Such a probe is used as it is.
Any other probe -- Neuropixels 2.0 single- or four-shank, a quad-base probe, another bank or
channel map of a 1.0, an NHP or UHD probe -- is localized one shank at a time:

- the shank's channels are resampled by height onto rows 10 µm apart over the span they
  record (two channels at the same height are averaged, heights between channel rows are
  interpolated, rows in a gap of more than a few channel rows are left out), and the
  localizer's nominal probe length is scaled from the 1.0's 3830 µm span to that span;
- a probe without an LF band (Neuropixels 2.0 records one wideband signal) gets its LF from
  the wideband signal, low-passed at 1 kHz and decimated to 2.5 kHz, as IBL derives it for its
  2.0 recordings (``neuropixel.NP2Converter``);
- the destriping uses the probe's ADC sampling order (``neuropixel.adc_shifts``).

The model behind the localizer has only seen Neuropixels 1.0 features: how well it localizes
other probes has not been measured.

:class:`ProbeLayout` holds the result, built from SpikeGLX's site map (:meth:`from_geom_map`,
the command server's ``getGeomMap``) or a ``.meta`` file (:meth:`from_meta`).
"""

from __future__ import annotations

import copy
import dataclasses
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Rows of the localizer (ScoreConfig.channel_spacing_um) and the span of a full 1.0 probe's
# 384 channels on them (the probe the annealer's nominal length is for).
ROW_UM = 10.0
NP1_ROW_SPAN_UM = 383 * ROW_UM
# IBL convention: the tip-most site row sits 20 µm above the tip (neuropixel.CHANNEL_GRID Y0).
TIP_ROW_HEIGHT_UM = 20.0
# LF from a wideband signal (neuropixel.NP2Converter: 1 kHz low-pass, 30 kHz -> 2.5 kHz).
LF_DECIMATION = 12
LF_CUTOFF_HZ = 1000.0
# A shank is only localized if its channels span this much (two 100 µm depth bins).
MIN_SPAN_UM = 200.0


def np1_heights_um() -> np.ndarray:
    """``[384]`` heights above the tip of a Neuropixels 1.0's bank-0 channels (channel order)."""
    return TIP_ROW_HEIGHT_UM + 20.0 * (np.arange(384) // 2)


def lf_from_wideband(x: np.ndarray, fs: float, q: int = LF_DECIMATION):
    """``(lf [C, ceil(ns / q)], fs / q)``: the LF band of a wideband ``[C, ns]`` signal (volts),
    zero-phase low-passed at :data:`LF_CUTOFF_HZ` and decimated by ``q``."""
    import scipy.signal

    x = np.asarray(x)
    sos = scipy.signal.butter(4, LF_CUTOFF_HZ, btype="lowpass", fs=float(fs), output="sos")
    out = np.empty((x.shape[0], -(-x.shape[1] // q)), dtype=np.float32)
    for c0 in range(0, x.shape[0], 64):  # in blocks: float64 copies of 64 channels at a time
        block = x[c0 : c0 + 64].astype(np.float64)
        block -= np.median(block, axis=1, keepdims=True)
        out[c0 : c0 + 64] = scipy.signal.sosfiltfilt(sos, block, axis=1)[:, ::q]
    return out, float(fs) / q


def _adc_version(part_number: str, has_lf: bool, n_stream: int):
    """``neuropixel.adc_shifts`` version of a probe: 1.0 family (32 ADCs of 12 channels) or 2.0
    (24 of 16; a quad base as four 2.0 shanks)."""
    if str(part_number).upper().startswith("NP2") or not has_lf:
        return "NP2QB" if n_stream > 384 else 2
    return 1


@dataclass
class ProbeLayout:
    """The channels of one probe the localizer uses, and its rows.

    Attributes:
        part_number: The probe's part number ('' when unknown).
        has_lf: The probe records an LF band (else it is derived from the wideband signal).
        n_shanks: Shanks of the probe.
        shank: The shank localized.
        channels: ``[n]`` stream indices of the channels used (that shank's), by height then x.
        x_um: ``[n]`` µm from the shank's left edge.
        z_um: ``[n]`` µm above the centre of the tip-most electrode row.
        n_stream: AP channels in the stream (the ADC shifts are those of the stream index).
        identity: A standard Neuropixels 1.0 (bank 0 of one shank): its channels are the
            localizer's rows as they are, top first.
        row_height_um: ``[R]`` heights above the tip of the localizer's rows, top first.
        geometry: ibldsp geometry of the used channels (``x``, ``y`` above the tip, ``shank``,
            ``sample_shift``, ...), in their order.
    """

    part_number: str
    has_lf: bool
    n_shanks: int
    shank: int
    channels: np.ndarray
    x_um: np.ndarray
    z_um: np.ndarray
    n_stream: int
    identity: bool
    row_height_um: np.ndarray
    geometry: dict
    shank_channels: dict = dataclasses.field(default_factory=dict)

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def from_sites(
        cls,
        shank_ids,
        x_um,
        z_um,
        *,
        part_number: str = "",
        has_lf: bool = True,
        n_shanks: Optional[int] = None,
        shank: Optional[int] = None,
        np1_geometry: Optional[dict] = None,
    ) -> "ProbeLayout":
        """The layout of a probe from its sites, one per stream channel (stream order).

        Args:
            shank_ids, x_um, z_um: Per stream channel: shank, µm from the shank's left edge, µm
                above the centre of the tip-most electrode row (SpikeGLX's ``snsGeomMap``).
            part_number: E.g. ``"NP2014"``.
            has_lf: The stream has an LF band.
            n_shanks: Shanks of the probe (default: those with a channel).
            shank: The shank to localize (default: the one with the most channels).
            np1_geometry: The geometry to use for a standard Neuropixels 1.0 (the source's own,
                so that such a probe is processed exactly as before), or a function returning it.
        """
        s = np.asarray(shank_ids, dtype=float).astype(int)
        x = np.asarray(x_um, dtype=float)
        z = np.asarray(z_um, dtype=float)
        n_stream = len(s)
        counts = {int(k): int((s == k).sum()) for k in np.unique(s)}
        if shank is None:
            shank = max(counts, key=lambda k: (counts[k], np.ptp(z[s == k]), -k))
        elif int(shank) not in counts:
            raise ValueError(
                f"shank {shank} records no channel (channels per shank: {counts})"
            )
        shank = int(shank)
        on = np.flatnonzero(s == shank)
        channels = on[np.lexsort((x[on], z[on]))]
        span = float(np.ptp(z[channels])) if len(channels) else 0.0
        if span < MIN_SPAN_UM:
            raise ValueError(
                f"shank {shank} has {len(channels)} channels over {span:.0f} µm: the localizer "
                f"needs at least {MIN_SPAN_UM:.0f} µm of recorded shank"
            )
        identity = bool(
            has_lf
            and n_stream == 384
            and len(counts) == 1
            and np.allclose(z, np1_heights_um() - TIP_ROW_HEIGHT_UM, atol=0.5)
        )
        version = _adc_version(part_number, has_lf, n_stream)
        if identity:
            channels = np.arange(384)
            rows = np1_heights_um()[::-1].astype(float)
            if np1_geometry is not None:
                if callable(np1_geometry):
                    np1_geometry = np1_geometry()
                geometry = {k: np.asarray(v) for k, v in np1_geometry.items()}
            else:
                geometry = _geometry(channels, x, z, version, n_stream)
        else:
            h = z[channels] + TIP_ROW_HEIGHT_UM
            n_rows = int(np.floor((h.max() - h.min()) / ROW_UM + 1e-6)) + 1
            rows = h.max() - ROW_UM * np.arange(n_rows)
            geometry = _geometry(channels, x, z, version, n_stream)
        return cls(
            part_number=str(part_number or ""),
            has_lf=bool(has_lf),
            n_shanks=int(n_shanks or len(counts)),
            shank=shank,
            channels=np.asarray(channels, dtype=int),
            x_um=x[channels],
            z_um=z[channels],
            n_stream=n_stream,
            identity=identity,
            row_height_um=np.asarray(rows, dtype=float),
            geometry=geometry,
            shank_channels=counts,
        )

    @classmethod
    def from_geom_map(
        cls,
        kv: dict,
        n_stream: int,
        *,
        has_lf: bool,
        shank: Optional[int] = None,
        np1_geometry: Optional[dict] = None,
    ) -> "ProbeLayout":
        """From SpikeGLX's ``getGeomMap`` key/values (``head_partNumber``, ``head_numShanks``,
        ``ch<c>_s``, ``ch<c>_x``, ``ch<c>_z``); the default site map of a 1.0 (with an LF band)
        or a single-shank 2.0 (without) when it lacks the channels."""
        part = str(kv.get("head_partNumber", ""))
        try:
            s = [float(kv[f"ch{c}_s"]) for c in range(n_stream)]
            x = [float(kv[f"ch{c}_x"]) for c in range(n_stream)]
            z = [float(kv[f"ch{c}_z"]) for c in range(n_stream)]
            n_shanks = int(float(kv.get("head_numShanks", 0))) or None
        except (KeyError, ValueError):
            s, x, z = _default_sites(has_lf, n_stream)
            n_shanks = None
        return cls.from_sites(
            s,
            x,
            z,
            part_number=part,
            has_lf=has_lf,
            n_shanks=n_shanks,
            shank=shank,
            np1_geometry=np1_geometry,
        )

    @classmethod
    def from_meta(
        cls,
        meta: dict,
        n_stream: int,
        *,
        has_lf: bool,
        shank: Optional[int] = None,
        np1_geometry: Optional[dict] = None,
    ) -> "ProbeLayout":
        """From a SpikeGLX ``.meta`` (``spikeglx.read_meta_data``): ``snsGeomMap``, else
        ``snsShankMap``, else the default site map."""
        import spikeglx

        part = str(meta.get("imDatPrb_pn", ""))
        cm = spikeglx._map_channels_from_meta(meta) or {}
        if cm.get("x") is not None:
            s, x, z = cm["shank"], cm["x"], cm["y"]
        elif cm.get("row") is not None:
            th = spikeglx.geometry_from_meta(meta, sort=False)
            s, x, z = th["shank"], th["x"], np.asarray(th["y"]) - TIP_ROW_HEIGHT_UM
        else:
            s, x, z = _default_sites(has_lf, n_stream)
        s, x, z = (np.asarray(v, dtype=float)[:n_stream] for v in (s, x, z))
        # shanks of the probe: the snsGeomMap header "(part,shanks,pitch,width)"
        header = re.match(r"\(([^,()]*),(\d+),", str(meta.get("snsGeomMap", "")))
        n_shanks = int(header.group(2)) if header else None
        return cls.from_sites(
            s,
            x,
            z,
            part_number=part,
            has_lf=has_lf,
            n_shanks=n_shanks,
            shank=shank,
            np1_geometry=np1_geometry,
        )

    # -- the localizer's view ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        """Channels used (read from the stream)."""
        return len(self.channels)

    @property
    def n_rows(self) -> int:
        """Rows of the localizer's feature matrix and positions."""
        return len(self.row_height_um)

    @property
    def neuropixel_version(self) -> int:
        """The version the destriping is told: 1 or 2."""
        return 1 if _adc_version(self.part_number, self.has_lf, self.n_stream) == 1 else 2

    @property
    def length_factor(self) -> float:
        """The localizer's probe length relative to a full Neuropixels 1.0."""
        if self.identity:
            return 1.0
        return (self.n_rows - 1) * ROW_UM / NP1_ROW_SPAN_UM

    def describe(self) -> str:
        part = self.part_number or "probe"
        span = float(np.ptp(self.z_um))
        if self.identity:
            text = f"{part}: Neuropixels 1.0 layout, {self.n_channels} channels"
        else:
            text = (
                f"{part}: {self.n_shanks} shank{'s' if self.n_shanks > 1 else ''}"
                + (
                    f", localizing shank {self.shank} "
                    f"(channels per shank {self.shank_channels})"
                    if self.n_shanks > 1 or len(self.shank_channels) > 1
                    else ""
                )
                + f": {self.n_channels} channels, {self.z_um.min() + TIP_ROW_HEIGHT_UM:.0f}-"
                f"{self.z_um.max() + TIP_ROW_HEIGHT_UM:.0f} µm above the tip ({span:.0f} µm), "
                f"resampled to {self.n_rows} rows of {ROW_UM:.0f} µm"
            )
        if not self.has_lf:
            text += "; no LF band: LF derived from the wideband signal"
        return text

    def feature_rows(self, table) -> np.ndarray:
        """``[R, F]`` features of the localizer's rows (top first) from a per-channel table
        (:func:`~ephysatlas.realtime.features.aggregate_snippets` of the used channels)."""
        from .features import feature_matrix

        per_channel = feature_matrix(table, n_channels=self.n_channels)[::-1]
        return self.rows(per_channel)

    def rows(self, per_channel: np.ndarray) -> np.ndarray:
        """``[R, F]`` from ``[n, F]`` values of the used channels (in :attr:`channels` order):
        the channels reversed for a standard 1.0; otherwise averaged per height and
        interpolated linearly in height onto the rows (NaN in gaps wider than a few rows)."""
        v = np.asarray(per_channel, dtype=np.float32)
        if self.identity:
            return v[::-1].copy()
        h = np.round(self.z_um + TIP_ROW_HEIGHT_UM, 3)
        u = np.unique(h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN heights
            means = np.stack([np.nanmean(v[h == k], axis=0) for k in u])
        out = np.full((self.n_rows, v.shape[1]), np.nan, dtype=np.float32)
        max_gap = max(40.0, 2.5 * float(np.median(np.diff(u)))) if len(u) > 1 else 0.0
        for r, hr in enumerate(self.row_height_um):
            i = int(np.searchsorted(u, hr))
            if i < len(u) and abs(u[i] - hr) < 0.5:
                out[r] = means[i]
            elif i > 0 and abs(u[i - 1] - hr) < 0.5:
                out[r] = means[i - 1]
            elif 0 < i < len(u) and u[i] - u[i - 1] <= max_gap:
                w = (hr - u[i - 1]) / (u[i] - u[i - 1])
                out[r] = (1.0 - w) * means[i - 1] + w * means[i]
        return out

    def plan_rows(self, planned_xyz_m, heights_um=None) -> np.ndarray:
        """``[R, 3]`` planned positions (m) of the rows, from a plan's ``[N, 3]`` positions.

        ``heights_um``: ``[N]`` heights above the tip of the plan's positions (IBL convention);
        default the 384 channels of a Neuropixels 1.0, top first, which a standard 1.0 uses as
        they are. Otherwise the straight line through the plan's positions is evaluated at the
        rows' heights (a plan made for a 1.0 is the line of the shank localized).
        """
        xyz = np.asarray(planned_xyz_m, dtype=np.float64)
        if heights_um is None:
            if len(xyz) != 384:
                raise ValueError(
                    f"the plan has {len(xyz)} positions and no heights: expected the 384 "
                    "channels of a Neuropixels 1.0"
                )
            if self.identity:
                return xyz.copy()
            heights_um = np1_heights_um()[::-1]
        h = np.asarray(heights_um, dtype=np.float64)
        ok = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0.0, axis=1) & np.isfinite(h)
        if ok.sum() < 2 or np.ptp(h[ok]) <= 0:
            raise ValueError("the plan has fewer than two distinct valid positions")
        design = np.column_stack([np.ones(int(ok.sum())), h[ok]])
        coef = np.linalg.lstsq(design, xyz[ok], rcond=None)[0]
        return coef[0][None, :] + self.row_height_um[:, None] * coef[1][None, :]

    def channel_xyz(self, row_xyz: np.ndarray) -> np.ndarray:
        """``[n, 3]`` positions of the used channels (:attr:`channels` order) from the rows'
        ``[R, 3]`` positions: linear in height, extrapolated along the end rows."""
        p = np.asarray(row_xyz, dtype=np.float64)
        if self.identity:
            return p[::-1].copy()
        hr = self.row_height_um[::-1]  # ascending
        pr = p[::-1]
        h = self.z_um + TIP_ROW_HEIGHT_UM
        out = np.column_stack([np.interp(h, hr, pr[:, k]) for k in range(3)])
        if len(hr) > 1:
            lo, hi = h < hr[0], h > hr[-1]
            out[lo] = pr[0] + (h[lo] - hr[0])[:, None] * (pr[1] - pr[0]) / (hr[1] - hr[0])
            out[hi] = pr[-1] + (h[hi] - hr[-1])[:, None] * (pr[-1] - pr[-2]) / (
                hr[-1] - hr[-2]
            )
        return out

    def localizer(self, localizer):
        """The localizer for this probe: the same one for a standard 1.0, else a shallow copy
        whose nominal probe length (and its bounds and proposal widths) is scaled by
        :attr:`length_factor`."""
        f = self.length_factor
        if self.identity or abs(f - 1.0) < 1e-12:
            return localizer
        a = localizer.anneal_cfg
        clone = copy.copy(localizer)
        clone.anneal_cfg = dataclasses.replace(
            a,
            probe_length_um=a.probe_length_um * f,
            min_probe_length_um=a.min_probe_length_um * f,
            max_probe_length_um=a.max_probe_length_um * f,
            init_length_sigma_um=a.init_length_sigma_um * f,
            final_length_sigma_um=a.final_length_sigma_um * f,
        )
        return clone


def _default_sites(has_lf: bool, n_stream: int):
    """``(shank, x, z)`` of the default site map: a 1.0 (bank 0) with an LF band, else a
    single-shank 2.0 (bank 0)."""
    import neuropixel

    th = neuropixel.trace_header(version=1 if has_lf else 2)
    n = min(n_stream, len(th["x"]))
    s = np.zeros(n_stream)
    x = np.resize(np.asarray(th["x"], dtype=float)[:n], n_stream)
    z = np.resize(np.asarray(th["y"], dtype=float)[:n] - TIP_ROW_HEIGHT_UM, n_stream)
    return s, x, z


def _geometry(channels, x, z, version, n_stream: int) -> dict:
    """ibldsp geometry of ``channels`` (one shank): positions, the stream's ADC shifts."""
    import neuropixel

    shift, adc = neuropixel.adc_shifts(version=version)
    idx = np.asarray(channels) % len(shift)
    xs, zs = np.asarray(x)[channels], np.asarray(z)[channels]
    return dict(
        x=xs.astype(float),
        y=(zs + TIP_ROW_HEIGHT_UM).astype(float),
        shank=np.zeros(len(channels)),
        sample_shift=np.asarray(shift)[idx],
        adc=np.asarray(adc)[idx],
        ind=np.asarray(channels),
        row=np.unique(zs, return_inverse=True)[1].astype(float),
        col=np.unique(xs, return_inverse=True)[1].astype(float),
    )
