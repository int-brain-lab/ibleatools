"""Download the test recordings of the real-time localization (see realtime_select_pids.py).

For every selected insertion: ``DURATION`` s of raw AP and LF from ``T_START``, written once as
SpikeGLX files and viewed as an Open Ephys recording (:mod:`ephysatlas.realtime.data`), plus the
plan file the tools start from (``plan.json``, the release's planned channel positions). The
human alignment and the release features, used only to score the results, go to
``TRUTH_DIR/<pid>.npz``, away from the recordings. Insertions already prepared are skipped;
the run stops when less than ``MIN_FREE_GB`` would remain on the data disk.

Usage::

    python scripts/realtime_prepare_recordings.py
    python scripts/realtime_prepare_recordings.py --limit 1
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings ------------------------------------------------------------------------------
DATA_ROOT = Path.home() / "OpenEphysTestData"
SELECTED = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "selected.csv"
TRUTH_DIR = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "truth"
T_START = 300.0  # s; the release's first snippet starts here
DURATION = 300.0  # s; 5 minutes per insertion
MAX_PIDS = 20
EST_GB_PER_PID = (
    7.8  # AP + LF int16 (385 channels) + Open Ephys sample numbers / timestamps
)
MIN_FREE_GB = 40.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "pids", nargs="*", help="default: the selected test insertions (SELECTED)"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--duration", type=float, default=DURATION, help="seconds per insertion"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="prepare at most N insertions"
    )
    args = parser.parse_args()

    from one.api import ONE

    from ephysatlas.alignment.data import load_probe
    from ephysatlas.realtime.data import prepare_test_recording
    from ephysatlas.realtime.plan import plan_from_release, save_plan

    pids = args.pids or pd.read_csv(SELECTED)["pid"].astype(str).tolist()
    if len(pids) > MAX_PIDS:
        raise SystemExit(
            f"{len(pids)} insertions selected; at most {MAX_PIDS} are allowed"
        )
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for i, pid in enumerate(pids, 1):
        rec_dir = (
            args.data_root / f"IBL_{pid}_{T_START:g}s_{T_START + args.duration:g}s"
        )
        if (rec_dir / "recording.json").exists():
            print(f"[prepare] {i}/{len(pids)} {pid}: already prepared")
            continue
        if args.limit is not None and n_done >= args.limit:
            break
        free_gb = shutil.disk_usage(args.data_root).free / 1e9
        if free_gb - EST_GB_PER_PID < MIN_FREE_GB:
            raise SystemExit(
                f"[prepare] stopping: {free_gb:.0f} GB free, {MIN_FREE_GB} GB must remain"
            )
        print(f"[prepare] {i}/{len(pids)} {pid} ({free_gb:.0f} GB free)", flush=True)
        rec_dir = prepare_test_recording(
            pid, args.data_root, T_START, args.duration, one=one
        )
        save_plan(plan_from_release(pid), rec_dir / "plan.json")
        probe = load_probe(pid)
        np.savez(
            TRUTH_DIR / f"{pid}.npz",
            human_xyz=probe["human_xyz"],
            planned_xyz=probe["planned_xyz"],
            release_features=probe["features"],
        )
        n_done += 1
        print(f"[prepare]    -> {rec_dir}", flush=True)


if __name__ == "__main__":
    main()
