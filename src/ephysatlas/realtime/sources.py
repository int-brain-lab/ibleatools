"""Where the online pipeline gets its samples: a recording being acquired, or one replayed.

A source exposes one probe's AP and LF bands through a small interface the pipeline polls:

- ``fs_ap``, ``fs_lf``, ``n_channels`` (ephys channels, sync excluded), ``geometry`` (ibldsp),
  ``s2v_ap``, ``s2v_lf`` (``[n_channels]`` volts per int16 count);
- ``start()`` / ``stop()``;
- ``acquired_s()``: seconds of data acquired so far (on the AP clock);
- ``read(band, first_sample, n_samples)``: ``[n_samples, n_channels]`` int16 of a band.

:class:`FileReplaySource` replays SpikeGLX files (e.g. the test recordings of
:mod:`ephysatlas.realtime.data`) as if they were being acquired, at real-time speed, faster, or
with everything available at once (``speed=None``). :class:`SpikeGLXSource` reads a probe that a
running SpikeGLX is acquiring (or playing back as a simulated probe), through SpikeGLX's remote
command server and the SpikeGLX-CPP-SDK.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Where the SpikeGLX-CPP-SDK's Python wrapper lives (the folder holding ``sglx_pkg``, with
# ``SglxApi.dll`` copied into ``sglx_pkg``); override with the SGLX_SDK_DIR environment variable.
DEFAULT_SGLX_SDK_DIR = (
    Path.home() / "SpikeGLX" / "SpikeGLX-CPP-SDK-main" / "Windows" / "Python"
)


class FileReplaySource:
    """A SpikeGLX recording replayed as an acquisition.

    Args:
        ap_file: The ``.ap.bin`` file (with its ``.meta``); the ``.lf.bin`` next to it is used
            unless ``lf_file`` is given.
        lf_file: The ``.lf.bin`` file.
        speed: Acquisition speed relative to real time (1: real time); None makes the whole
            recording available at ``start()`` (for runs that should not wait).
    """

    def __init__(self, ap_file, lf_file=None, speed: Optional[float] = 1.0):
        import spikeglx

        ap_file = Path(ap_file)
        lf_file = (
            Path(lf_file)
            if lf_file is not None
            else Path(str(ap_file).replace(".ap.", ".lf."))
        )
        self.sr_ap = spikeglx.Reader(ap_file, ignore_warnings=True)
        self.sr_lf = spikeglx.Reader(lf_file, ignore_warnings=True)
        self.name = ap_file.name.split("_g")[0]
        self.fs_ap, self.fs_lf = float(self.sr_ap.fs), float(self.sr_lf.fs)
        self.n_channels = int(self.sr_ap.nc - self.sr_ap.nsync)
        self.geometry = {k: np.asarray(v) for k, v in self.sr_ap.geometry.items()}
        self.s2v_ap = np.asarray(
            self.sr_ap.channel_conversion_sample2v["ap"][: self.n_channels]
        )
        self.s2v_lf = np.asarray(
            self.sr_lf.channel_conversion_sample2v["lf"][: self.n_channels]
        )
        self.duration_s = float(
            min(self.sr_ap.ns / self.fs_ap, self.sr_lf.ns / self.fs_lf)
        )
        self.speed = speed
        self._t0 = None

    @classmethod
    def from_recording_dir(
        cls, rec_dir, speed: Optional[float] = 1.0
    ) -> "FileReplaySource":
        """The SpikeGLX files of a test recording directory (``<dir>/sglx/*.ap.bin``)."""
        ap = sorted(Path(rec_dir).joinpath("sglx").glob("*.ap.bin"))
        if len(ap) != 1:
            raise FileNotFoundError(
                f"expected one .ap.bin in {Path(rec_dir) / 'sglx'}, found {len(ap)}"
            )
        return cls(ap[0], speed=speed)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._t0 = None

    def acquired_s(self) -> float:
        if self._t0 is None:
            return 0.0
        if self.speed is None:
            return self.duration_s
        return float(
            min(self.duration_s, (time.perf_counter() - self._t0) * self.speed)
        )

    def read(self, band: str, first_sample: int, n_samples: int) -> np.ndarray:
        sr = self.sr_ap if band == "ap" else self.sr_lf
        fs = self.fs_ap if band == "ap" else self.fs_lf
        stop = int(first_sample) + int(n_samples)
        if stop > int(round(self.acquired_s() * fs)) + 1:
            raise ValueError(f"{band} samples up to {stop} are not acquired yet")
        return np.array(
            sr._raw[int(first_sample) : stop, : self.n_channels], dtype=np.int16
        )


def load_sglx(sdk_dir=None):
    """Import the SpikeGLX-CPP-SDK Python wrapper (``sglx_pkg.sglx``) from ``sdk_dir``."""
    sdk = Path(sdk_dir or os.environ.get("SGLX_SDK_DIR") or DEFAULT_SGLX_SDK_DIR)
    dll_dir = sdk / "sglx_pkg"
    if not (dll_dir / "SglxApi.dll").exists():
        raise FileNotFoundError(
            f"SglxApi.dll not found in {dll_dir}: download the SpikeGLX-CPP-SDK and copy "
            "Windows/API/SglxApi.dll into Windows/Python/sglx_pkg (or set SGLX_SDK_DIR)"
        )
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
    if str(sdk) not in sys.path:
        sys.path.insert(0, str(sdk))
    from sglx_pkg import sglx

    return sglx


class SpikeGLXSource:
    """An imec probe acquired by a running SpikeGLX, read through its remote command server.

    SpikeGLX streams a Neuropixels 1.0 probe as one stream of 16-bit words per sample: the AP
    channels, then the LF channels (upsampled to the AP rate in memory), then the sync word.
    AP samples are fetched as they are; LF samples are the LF channels at every 12th sample. The
    acquisition clock is the stream's sample count since :meth:`start`. Enable the command
    server in SpikeGLX (Options / Command Server Settings) and start a run before :meth:`start`.

    Args:
        probe: Logical imec probe index (``ip``).
        address, port: The SpikeGLX command server.
        sdk_dir: Folder holding the SDK's ``sglx_pkg`` (see :func:`load_sglx`).
        lf_downsample: AP samples per LF sample (12 for Neuropixels 1.0).
    """

    JS_IMEC = 2

    def __init__(
        self,
        probe: int = 0,
        address: str = "localhost",
        port: int = 4142,
        sdk_dir=None,
        lf_downsample: int = 12,
    ):
        from ctypes import byref, c_double, c_int

        self._sglx = s = load_sglx(sdk_dir)
        self.ip, self.lf_downsample = int(probe), int(lf_downsample)
        self.h = s.c_sglx_createHandle()
        if not s.c_sglx_connect(self.h, address.encode(), int(port)):
            raise ConnectionError(
                f"cannot connect to SpikeGLX at {address}:{port} ({self._error()}). Is SpikeGLX "
                "running, with Options / Command Server Settings enabled on this port? If that "
                "dialog shows a network address rather than 127.0.0.1, pass it with --sglx ADDRESS."
            )
        nval = c_int()
        if not s.c_sglx_getStreamAcqChans(byref(nval), self.h, self.JS_IMEC, self.ip):
            raise RuntimeError(
                f"cannot read the channels of imec probe {self.ip}: {self._error()}"
            )
        n_ap, n_lf, n_sy = (s.c_sglx_getint(self.h, i) for i in range(nval.value))
        if n_lf == 0:
            raise ValueError(
                f"imec probe {self.ip} has no LF band (the localizer needs LF features)"
            )
        self.n_channels = int(n_ap)
        self.fs_ap = float(s.c_sglx_getStreamSampleRate(self.h, self.JS_IMEC, self.ip))
        self.fs_lf = self.fs_ap / self.lf_downsample
        mult = c_double()

        def i16_to_volts(chan: int) -> float:
            if not s.c_sglx_getStreamI16ToVolts(
                byref(mult), self.h, self.JS_IMEC, self.ip, chan
            ):
                raise RuntimeError(
                    f"cannot read the gain of channel {chan}: {self._error()}"
                )
            return mult.value

        self.s2v_ap = np.array([i16_to_volts(c) for c in range(n_ap)])
        self.s2v_lf = np.array([i16_to_volts(n_ap + c) for c in range(n_lf)])
        self.geometry = self._geometry()
        self.name = f"spikeglx_imec{self.ip}"
        self.speed = 1.0  # live: data arrive in real time
        self.duration_s = np.inf
        self._count0 = None

    def _error(self) -> str:
        err = self._sglx.c_sglx_getError(self.h)
        return err.decode(errors="replace") if isinstance(err, bytes) else str(err)

    def _geometry(self) -> dict:
        """ibldsp geometry of a Neuropixels 1.0 probe, with SpikeGLX's site positions when
        available (``getGeomMap``: x from the shank's left edge, z from the tip row)."""
        from ctypes import byref, c_int

        import neuropixel

        geometry = {
            k: np.asarray(v) for k, v in neuropixel.trace_header(version=1).items()
        }
        geometry = {k: v[: self.n_channels] for k, v in geometry.items()}
        nval, length = c_int(), c_int()
        if self._sglx.c_sglx_getGeomMap(byref(nval), self.h, self.ip):
            kv = {}
            for i in range(nval.value):
                item = self._sglx.c_sglx_getstr(byref(length), self.h, i).decode()
                key, _, val = item.partition("=")
                kv[key.strip()] = val.strip()
            try:
                x = np.array([float(kv[f"ch{c}_x"]) for c in range(self.n_channels)])
                z = np.array([float(kv[f"ch{c}_z"]) for c in range(self.n_channels)])
                # ibldsp convention: y from the tip site (20 µm for NP1), x as in the meta.
                geometry["x"], geometry["y"] = x, z - z.min() + geometry["y"].min()
            except (KeyError, ValueError):
                pass
        return geometry

    def start(self) -> None:
        count = int(
            self._sglx.c_sglx_getStreamSampleCount(self.h, self.JS_IMEC, self.ip)
        )
        if count == 0:
            raise RuntimeError("SpikeGLX is not running (start a run first)")
        self._count0 = count

    def stop(self) -> None:
        pass  # the SpikeGLX run is the experimenter's

    def close(self) -> None:
        self._sglx.c_sglx_close(self.h)
        self._sglx.c_sglx_destroyHandle(self.h)

    def acquired_s(self) -> float:
        if self._count0 is None:
            return 0.0
        count = int(
            self._sglx.c_sglx_getStreamSampleCount(self.h, self.JS_IMEC, self.ip)
        )
        return max(0.0, (count - self._count0) / self.fs_ap)

    def read(self, band: str, first_sample: int, n_samples: int) -> np.ndarray:
        """``[n_samples, n_channels]`` int16 of a band, ``first_sample`` counted from :meth:`start`.

        SpikeGLX receives one LF sample per 12-sample packet and interpolates it linearly into
        the AP-rate stream, starting from the previous LF sample: LF sample ``m`` is exact at
        stream sample ``12 (m + 1)``. LF is therefore fetched at full rate and taken at those
        samples (SpikeGLX's own ``downsample`` does not land on them).
        """
        if band == "ap":
            chans = np.arange(self.n_channels)
            return self._fetch(chans, self._count0 + int(first_sample), int(n_samples))
        k = self.lf_downsample
        chans = self.n_channels + np.arange(len(self.s2v_lf))
        m = self._count0 // k + int(first_sample)  # LF sample index on SpikeGLX's clock
        full = self._fetch(chans, (m + 1) * k, int(n_samples) * k)
        return np.ascontiguousarray(full[::k])

    def _fetch(self, chans: np.ndarray, start: int, n_samples: int) -> np.ndarray:
        """``n_samples`` stream samples of ``chans`` from absolute sample ``start``."""
        from ctypes import POINTER, byref, c_int, c_short

        step = 1
        subset = (c_int * len(chans))(*chans.tolist())
        out, got = np.empty((int(n_samples), len(chans)), dtype=np.int16), 0
        data, ndata = POINTER(c_short)(), c_int()
        while got < n_samples:
            head = self._sglx.c_sglx_fetch(
                byref(data),
                byref(ndata),
                self.h,
                self.JS_IMEC,
                self.ip,
                start,
                int(n_samples - got),
                subset,
                len(chans),
                step,
            )
            if head == 0:
                raise RuntimeError(f"SpikeGLX fetch failed: {self._error()}")
            if head != start:
                raise RuntimeError(
                    f"SpikeGLX returned samples from {head}, not {start}: the requested data have "
                    "left SpikeGLX's buffer (the pipeline fell behind)"
                )
            nt = ndata.value // len(chans)
            if nt == 0:
                time.sleep(0.01)
                continue
            block = np.ctypeslib.as_array(data, shape=(nt * len(chans),)).reshape(
                nt, len(chans)
            )
            take = min(nt, int(n_samples) - got)
            out[got : got + take] = block[:take]
            got += take
            start = head + take * step
        return out
