"""Replay the test recordings through the online localization pipeline (no SpikeGLX needed).

Each recording prepared by realtime_prepare_recordings.py is replayed by a
:class:`~ephysatlas.realtime.sources.FileReplaySource` -- at real-time speed (``--speed 1``) to
measure latencies, or with all data available (``--speed max``) for fast, identical updates --
and localized by :class:`~ephysatlas.realtime.online.OnlinePipeline` with the real-time
(no spike detection) localizer. Per recording, ``OUT_DIR/<run>/<pid>/`` gets ``snippets.csv``,
``updates.csv``, ``updates.npz`` and ``snippet_features.pqt`` (see ``OnlinePipeline.save``).
Recordings already done are skipped.

Usage::

    python scripts/realtime_replay.py --speed max
    python scripts/realtime_replay.py --speed 1 --pids 143dd7cf-6a47-47a1-906d-927ad7fe9117
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings (each can also be given on the command line; see --help) ----------------------
DATA_ROOT = Path.home() / "OpenEphysTestData"
SELECTED = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "selected.csv"
OUT_DIR = DEFAULT_RESULTS_DIR / "realtime" / "replay"
SPEED = "max"  # "max" (all data available at once) or a factor of real time, e.g. "1"
CADENCE_S = 10.0  # a 5 s snippet every CADENCE_S of acquisition
N_WORKERS = (
    0  # 0: as many as fit in the free RAM (ephysatlas.realtime.online.suggest_workers)
)
THREADS_PER_WORKER = 2
FEATURE_SET = "lf_ap_no_detection"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speed", default=SPEED)
    parser.add_argument("--cadence", type=float, default=CADENCE_S)
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS, help="0: fit to free RAM"
    )
    parser.add_argument("--threads", type=int, default=THREADS_PER_WORKER)
    parser.add_argument(
        "--pids", nargs="*", default=None, help="default: every selected insertion"
    )
    parser.add_argument(
        "--run", default=None, help="run name (default speed<speed>_cadence<cadence>)"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer
    from ephysatlas.realtime.online import OnlineConfig, OnlinePipeline, suggest_workers
    from ephysatlas.realtime.plan import load_plan
    from ephysatlas.realtime.sources import FileReplaySource

    speed = None if str(args.speed).lower() == "max" else float(args.speed)
    run = args.run or f"speed{args.speed}_cadence{args.cadence:g}"
    out_root = OUT_DIR / run
    out_root.mkdir(parents=True, exist_ok=True)
    pids = args.pids or pd.read_csv(SELECTED)["pid"].astype(str).tolist()
    n_workers = args.workers or suggest_workers()
    print(f"[replay] {n_workers} feature workers", flush=True)
    config = OnlineConfig(
        cadence_s=args.cadence, n_workers=n_workers, threads_per_worker=args.threads
    )
    (out_root / "config.json").write_text(
        json.dumps(
            dict(speed=args.speed, feature_set=FEATURE_SET, pids=pids, **vars(config)),
            indent=1,
        ),
        encoding="utf-8",
    )
    localizer = EphysOnlyLocalizer(
        ChannelModel(), AllenAtlas(), feature_set=FEATURE_SET
    )
    for i, pid in enumerate(pids, 1):
        out = out_root / pid
        if (out / "updates.csv").exists() and not args.overwrite:
            print(f"[replay] {i}/{len(pids)} {pid}: done already")
            continue
        rec = args.data_root / f"IBL_{pid}_300s_600s"
        source = FileReplaySource.from_recording_dir(rec, speed=speed)
        plan = load_plan(rec / "plan.json")["planned_xyz_m"]

        def report(u):
            print(
                f"    update {u.index}: {u.n_snippets} snippets, data to {u.acquired_s:.0f} s, "
                f"latency {u.latency_s:.1f} s (localize {u.localize_s:.1f} s)",
                flush=True,
            )

        print(f"[replay] {i}/{len(pids)} {pid}", flush=True)
        t0 = time.time()
        pipeline = OnlinePipeline(source, localizer, plan, config, on_update=report)
        pipeline.run()
        pipeline.save(out)
        snips = pipeline.snippet_table()
        print(
            f"[replay]    {len(pipeline.updates)} updates, {int(snips['dropped'].sum())} dropped "
            f"snippets, compute {snips['compute_s'].mean():.1f} s/snippet, {time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
