"""Replay the test recordings through the C++ real-time engine of the Open Ephys plugin.

Runs ``probe-viewer-realtime-batch replay`` (Probe-Viewer plugin, ``Source/BatchLocalization``) on
each recording's Open Ephys copy at maximum speed, a 5 s snippet every ``--cadence`` seconds,
localizing after each snippet with the localizer bundle (``export_localizer_bundle.py``) and the
recording's ``plan.json``. The outputs are packed like an ``OnlinePipeline`` run (``snippets.csv``,
``updates.csv``, ``updates.npz`` with the features in the 43-column release order), so
``realtime_evaluate.py`` scores them unchanged.

Usage::

    python scripts/realtime_cpp_replay.py
    python scripts/realtime_evaluate.py cpp_speedmax_cadence10 --seeds 5
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings (each can also be given on the command line; see --help) ----------------------
DATA_ROOT = Path.home() / "OpenEphysTestData"
SELECTED = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "selected.csv"
OUT_DIR = DEFAULT_RESULTS_DIR / "realtime" / "replay"
BATCH_EXE = (
    Path.home()
    / "VisualStudioProjects/open-ephys-development/OEPlugins/probe-viewer/build_rt/Release"
    / "probe-viewer-realtime-batch.exe"
)
FEATURE_SET = "lf_ap_no_detection"
CADENCE_S = 10.0
DURATION_S = 300.0


def default_bundle() -> Path:
    """The most recent bundle exported for FEATURE_SET."""
    bundles = sorted(
        (DEFAULT_RESULTS_DIR / "realtime" / "bundles").glob(f"*_{FEATURE_SET}"),
        key=lambda p: p.stat().st_mtime,
    )
    if not bundles:
        raise FileNotFoundError(
            f"no {FEATURE_SET} bundle: run scripts/export_localizer_bundle.py"
        )
    return bundles[-1]


def pack(out: Path, feature_names: list) -> None:
    """``updates.npz`` from the engine's arrays, the features in the release's 43 columns."""
    from ephysatlas.spatial_encoder.utils import FEATURE_LIST

    f = np.load(out / "features.npy")
    features = np.full(f.shape[:2] + (len(FEATURE_LIST),), np.nan, dtype=np.float32)
    for k, name in enumerate(feature_names):
        features[..., FEATURE_LIST.index(name)] = f[..., k]
    np.savez_compressed(
        out / "updates.npz",
        channel_xyz=np.load(out / "channel_xyz.npy").astype(np.float64),
        trace_xyz=np.load(out / "trace_xyz.npy").astype(np.float64),
        features=features,
        planned_xyz=np.load(out / "planned_xyz.npy"),
        valid=np.load(out / "valid.npy").astype(bool),
    )
    for name in ("channel_xyz", "trace_xyz", "features", "planned_xyz", "valid"):
        (out / f"{name}.npy").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pids", nargs="*", help="default: all selected test insertions")
    parser.add_argument("--exe", type=Path, default=BATCH_EXE)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--cadence", type=float, default=CADENCE_S)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument(
        "--run", default=None, help="run name (default cpp_speedmax_cadence<cadence>)"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle or default_bundle()
    feature_names = json.loads((bundle / "manifest.json").read_text())["feature_names"]
    run = args.run or f"cpp_speedmax_cadence{args.cadence:g}"
    out_root = OUT_DIR / run
    out_root.mkdir(parents=True, exist_ok=True)
    pids = args.pids or pd.read_csv(SELECTED)["pid"].astype(str).tolist()
    (out_root / "config.json").write_text(
        json.dumps(
            dict(
                engine="cpp",
                exe=str(args.exe),
                bundle=str(bundle),
                feature_set=FEATURE_SET,
                cadence_s=args.cadence,
                duration_s=args.duration,
                pids=pids,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    for i, pid in enumerate(pids, 1):
        out = out_root / pid
        if (out / "updates.npz").exists() and not args.overwrite:
            print(f"[cpp-replay] {i}/{len(pids)} {pid}: done already")
            continue
        rec = args.data_root / f"IBL_{pid}_300s_600s"
        t0 = time.time()
        cmd = [
            str(args.exe),
            "replay",
            str(bundle),
            str(rec),
            str(out),
            f"{args.cadence:g}",
            f"{args.duration:g}",
        ]
        log = subprocess.run(cmd, capture_output=True, text=True)
        (out / "engine.log").write_text(log.stdout + log.stderr, encoding="utf-8")
        if log.returncode != 0:
            print(
                f"[cpp-replay] {i}/{len(pids)} {pid}: failed ({log.returncode}): {log.stdout[-300:]}"
            )
            continue
        pack(out, feature_names)
        snips = pd.read_csv(out / "snippets.csv")
        upd = pd.read_csv(out / "updates.csv")
        print(
            f"[cpp-replay] {i}/{len(pids)} {pid}: {len(upd)} updates, features "
            f"{snips['compute_s'].mean():.1f} s/snippet, localize {upd['localize_s'].mean():.2f} s, "
            f"{time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
