"""Test recordings for the real-time tools: a time window of an IBL insertion.

:func:`download_window` streams a window of an insertion's raw AP and LF data from the IBL
servers (only the compressed chunks it needs, deleted once decompressed) and writes it once, as
SpikeGLX files with the original channel count (384 + sync), gains and meta-data -- what
SpikeGLX's simulated probes play back. :func:`write_open_ephys_view` then describes the same
files as an Open Ephys binary recording (``structure.oebin``, hard-linked ``continuous.dat``) for
the Open Ephys File Reader, without copying the samples. Layout of a recording directory::

    <dir>/sglx/<name>_g0_t0.imec0.ap.bin|.meta, <name>_g0_t0.imec0.lf.bin|.meta
    <dir>/oe/structure.oebin
    <dir>/oe/continuous/IBL-AP|IBL-LF/continuous.dat (hard link), sample_numbers.npy, timestamps.npy
    <dir>/recording.json  (pid, window, files)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

BANDS = ("ap", "lf")
# Streaming block (s): the Streamer downloads and decompresses this much at a time.
CHUNK_S = 10.0


def recording_name(pid: str) -> str:
    """SpikeGLX run name of a test recording (``<first 8 characters of the pid>``)."""
    return str(pid)[:8]


def sglx_files(rec_dir, pid: str, band: str) -> tuple[Path, Path]:
    """``(bin, meta)`` paths of one band of a test recording."""
    stem = Path(rec_dir) / "sglx" / f"{recording_name(pid)}_g0_t0.imec0.{band}"
    return stem.parent / f"{stem.name}.bin", stem.parent / f"{stem.name}.meta"


def _edit_meta(
    text: str, n_samples: int, n_channels: int, fs: float, first_sample: int
) -> str:
    """The original meta with the file size, duration and first sample of the window."""

    def put(key, value, s):
        pattern = rf"^{re.escape(key)}=.*$"
        line = f"{key}={value}"
        return (
            re.sub(pattern, line, s, flags=re.M)
            if re.search(pattern, s, flags=re.M)
            else s + f"\n{line}"
        )

    text = put("fileSizeBytes", n_samples * n_channels * 2, text)
    text = put("fileTimeSecs", repr(n_samples / fs), text)
    text = put("firstSample", first_sample, text)
    return text


def download_window(
    pid: str,
    rec_dir,
    t_start: float = 300.0,
    duration: float = 300.0,
    one=None,
    cache_folder: Optional[Path] = None,
) -> dict:
    """Write ``duration`` s of an insertion's AP and LF bands from ``t_start`` as SpikeGLX files.

    Args:
        pid: Insertion id.
        rec_dir: The recording directory (``sglx/`` is written inside).
        t_start, duration: The window, s from the start of the recording.
        one: ``one.api.ONE`` instance (created when None).
        cache_folder: Where the compressed chunks are staged (deleted after each block).

    Returns:
        dict: Per band ``bin``, ``meta``, ``fs``, ``n_channels``, ``first_sample``, ``n_samples``.
    """
    from brainbox.io.spikeglx import Streamer

    if one is None:
        from one.api import ONE

        one = ONE(base_url="https://alyx.internationalbrainlab.org")
    rec_dir = Path(rec_dir)
    (rec_dir / "sglx").mkdir(parents=True, exist_ok=True)
    out = {}
    for band in BANDS:
        sr = Streamer(
            pid=pid, one=one, typ=band, cache_folder=cache_folder, remove_cached=True
        )
        fs, nc = float(sr.fs), int(sr.nc)
        n0 = int(round(t_start * fs))
        n_total = int(round(duration * fs))
        if n0 + n_total > sr.ns:
            raise ValueError(
                f"{pid} {band}: the window {t_start}+{duration} s ends after the recording "
                f"({sr.ns / fs:.0f} s)"
            )
        bin_path, meta_path = sglx_files(rec_dir, pid, band)
        step = int(round(CHUNK_S * fs))
        t0 = time.time()
        with open(bin_path, "wb") as f:
            for a in range(n0, n0 + n_total, step):
                b = min(a + step, n0 + n_total)
                block = sr.read(nsel=slice(a, b), csel=slice(None), volts=False)
                np.ascontiguousarray(block, dtype=np.int16).tofile(f)
        meta_text = Path(sr.file_meta_data).read_text()
        meta_path.write_text(_edit_meta(meta_text, n_total, nc, fs, n0))
        out[band] = dict(
            bin=str(bin_path),
            meta=str(meta_path),
            fs=fs,
            n_channels=nc,
            first_sample=n0,
            n_samples=n_total,
            seconds=time.time() - t0,
        )
    return out


def write_open_ephys_view(rec_dir, pid: str) -> Path:
    """Describe the SpikeGLX files of a test recording as an Open Ephys binary recording.

    The ``continuous.dat`` of each stream is a hard link to the SpikeGLX ``.bin`` (int16,
    interleaved, all channels including sync), so no sample is copied; ``bit_volts`` comes from
    the meta's gains (the sync channel has 1). The stream layout follows the IBL exports the
    Probe-Viewer plugin was developed with.

    Returns:
        Path: The ``structure.oebin``.
    """
    import spikeglx

    rec_dir = Path(rec_dir)
    oe = rec_dir / "oe"
    streams = []
    for band in BANDS:
        bin_path, _ = sglx_files(rec_dir, pid, band)
        sr = spikeglx.Reader(bin_path, ignore_warnings=True)
        folder = f"IBL-{band.upper()}"
        sdir = oe / "continuous" / folder
        sdir.mkdir(parents=True, exist_ok=True)
        dat = sdir / "continuous.dat"
        if dat.exists():
            dat.unlink()
        os.link(bin_path, dat)
        n = int(sr.ns)
        samples = np.arange(n, dtype=np.int64)
        np.save(sdir / "sample_numbers.npy", samples)
        np.save(sdir / "timestamps.npy", samples / float(sr.fs))
        s2v = np.asarray(sr.channel_conversion_sample2v[band], dtype=float)
        n_data = int(sr.nc - sr.nsync)
        channels = [
            dict(
                channel_name=f"{band.upper()}_CH{c + 1}",
                description=f"IBL {band.upper()} ephys channel {c}",
                identifier="genericdata.continuous",
                history="IBL SpikeGLX window",
                bit_volts=round(
                    float(s2v[c]) * 1e6, 6
                ),  # float32 gains: 2.34375 µV, not ...1113
                units="uV",
                source_processor_index=c,
                recorded_processor_index=c,
            )
            for c in range(n_data)
        ] + [
            dict(
                channel_name=f"SYNC{k}",
                description="SpikeGLX sync word (not an ephys channel)",
                identifier="genericdata.continuous",
                history="IBL SpikeGLX window",
                bit_volts=1.0,
                units="",
                source_processor_index=n_data + k,
                recorded_processor_index=n_data + k,
            )
            for k in range(int(sr.nsync))
        ]
        streams.append(
            dict(
                folder_name=folder,
                sample_rate=float(sr.fs),
                source_processor_name="IBL File Export",
                source_processor_id=100,
                source_processor_sub_idx=0,
                recorded_processor="Demo source",
                recorded_processor_id=100,
                num_channels=int(sr.nc),
                stream_name=f"IBL {band.upper()} {pid}",
                channels=channels,
            )
        )
    oebin = oe / "structure.oebin"
    oebin.write_text(
        json.dumps(
            {"GUI version": "0.4.5", "continuous": streams, "events": [], "spikes": []},
            indent=1,
        ),
        encoding="utf-8",
    )
    return oebin


def prepare_test_recording(
    pid: str, root, t_start: float = 300.0, duration: float = 300.0, one=None
) -> Path:
    """:func:`download_window` + :func:`write_open_ephys_view` into ``root/IBL_<pid>_<t0>s_<t1>s``,
    with a ``recording.json`` summary. Returns the recording directory."""
    rec_dir = Path(root) / f"IBL_{pid}_{t_start:g}s_{t_start + duration:g}s"
    files = download_window(pid, rec_dir, t_start, duration, one=one)
    oebin = write_open_ephys_view(rec_dir, pid)
    (rec_dir / "recording.json").write_text(
        json.dumps(
            dict(
                pid=pid,
                t_start=t_start,
                duration=duration,
                sglx=files,
                oebin=str(oebin),
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    return rec_dir


def write_spikeglx_sim_files(rec_dir, pid: str) -> Path:
    """Files for a SpikeGLX simulated probe that replays a test recording faithfully.

    SpikeGLX (20260826-api4) recreates a 1.0 probe's LF band from the ``.lf.bin`` next to the
    ``.ap.bin``, but it steps through the LF rows with the AP channel count (384) as the row width
    (``ImSimLfDat::get_ie`` in ``CimAcqImec.cpp``), while a SpikeGLX LF file holds 385 values per
    sample (384 LF + sync): within each batch of packets the LF channels slip by one more channel
    per sample. Giving it an LF file without the sync channel, declared as such in its meta
    (``acqApLfSy=384,384,0``, ``nSavedChans=384``), makes the two widths agree.

    Writes ``<rec_dir>/sglx_sim/``: the AP ``.bin`` hard-linked and its ``.meta`` copied, and the LF
    without the sync channel. Point SpikeGLX's Simulated Probes dialog at the returned ``.ap.bin``.
    """
    import shutil

    import spikeglx

    rec_dir = Path(rec_dir)
    out = rec_dir / "sglx_sim"
    out.mkdir(exist_ok=True)
    ap_bin, ap_meta = sglx_files(rec_dir, pid, "ap")
    lf_bin, lf_meta = sglx_files(rec_dir, pid, "lf")
    sim_ap, sim_lf = out / ap_bin.name, out / lf_bin.name
    if not sim_ap.exists():
        os.link(ap_bin, sim_ap)
    shutil.copy(ap_meta, out / ap_meta.name)
    sr = spikeglx.Reader(lf_bin, ignore_warnings=True)
    n_lf = int(sr.nc - sr.nsync)
    with open(sim_lf, "wb") as f:
        step = 25000
        for a in range(0, int(sr.ns), step):
            np.ascontiguousarray(sr._raw[a : a + step, :n_lf]).tofile(f)
    text = Path(lf_meta).read_text()
    for key, value in (
        ("acqApLfSy", f"384,{n_lf},0"),
        ("nSavedChans", str(n_lf)),
        ("snsSaveChanSubset", f"384:{383 + n_lf}"),
        ("fileSizeBytes", str(sim_lf.stat().st_size)),
    ):
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    (out / lf_meta.name).write_text(text)
    return sim_ap
