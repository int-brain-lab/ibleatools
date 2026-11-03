"""Watch the online localization in a live window: a replayed test recording or SpikeGLX.

Two sources (:mod:`ephysatlas.realtime.sources`):

- replay (default): one recording prepared by realtime_prepare_recordings.py, replayed at
  real-time speed or faster;
- live (``--sglx [ADDRESS]``): an imec probe acquired by a running SpikeGLX (its command server
  enabled), e.g. a simulated probe playing one of the test recordings.

The plan comes from ``--plan`` (a plan JSON, see :mod:`ephysatlas.realtime.plan`) or from the
test recording named by ``pid``; ``--reference`` also draws the histology alignment of that
test recording (never available live).
The run's outputs are saved in ``OUT_DIR/<name>_<time>/`` when the window closes.

Usage::

    python scripts/realtime_viewer.py 143dd7cf-6a47-47a1-906d-927ad7fe9117 --reference
    python scripts/realtime_viewer.py --sglx --plan C:/data/plan.json
    python scripts/realtime_viewer.py 143dd7cf-6a47-47a1-906d-927ad7fe9117 --sglx --reference
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings (each can also be given on the command line; see --help) ----------------------
DATA_ROOT = Path.home() / "OpenEphysTestData"
TRUTH_DIR = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "truth"
OUT_DIR = DEFAULT_RESULTS_DIR / "realtime" / "viewer_runs"
SPEED = "1"
CADENCE_S = 10.0
N_WORKERS = (
    0  # 0: as many as fit in the free RAM (ephysatlas.realtime.online.suggest_workers)
)
FEATURE_SET = "lf_ap_no_detection"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "pid",
        nargs="?",
        default=None,
        help="a test recording (replay, plan, reference)",
    )
    parser.add_argument(
        "--sglx",
        nargs="?",
        const="localhost",
        default=None,
        metavar="ADDRESS",
        help="read a running SpikeGLX instead of replaying (default address localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=4142, help="SpikeGLX command server port"
    )
    parser.add_argument(
        "--probe", type=int, default=0, help="SpikeGLX imec probe index"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="plan JSON (default: the test recording's)",
    )
    parser.add_argument(
        "--speed", default=SPEED, help='replay: factor of real time, or "max"'
    )
    parser.add_argument("--cadence", type=float, default=CADENCE_S)
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS, help="0: fit to free RAM"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="draw the test recording's histology alignment",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="save the window at the end and close",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.pid is None and (args.sglx is None or args.plan is None):
        parser.error("give a test recording pid, or --sglx with --plan")
    if args.reference and args.pid is None:
        parser.error("--reference needs the pid of a test recording")

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer
    from ephysatlas.realtime.online import OnlineConfig, OnlinePipeline, suggest_workers
    from ephysatlas.realtime.plan import load_plan
    from ephysatlas.realtime.sources import FileReplaySource, SpikeGLXSource
    from ephysatlas.realtime.viewer import LocalizationViewer

    rec = args.data_root / f"IBL_{args.pid}_300s_600s" if args.pid else None
    plan = load_plan(args.plan or rec / "plan.json")["planned_xyz_m"]
    reference = (
        np.load(TRUTH_DIR / f"{args.pid}.npz")["human_xyz"] if args.reference else None
    )
    ba = AllenAtlas()
    localizer = EphysOnlyLocalizer(ChannelModel(), ba, feature_set=FEATURE_SET)
    n_workers = args.workers or suggest_workers()
    print(f"[viewer] {n_workers} feature workers", flush=True)
    config = OnlineConfig(cadence_s=args.cadence, n_workers=n_workers)
    if args.sglx is not None:
        source = SpikeGLXSource(probe=args.probe, address=args.sglx, port=args.port)
        print(
            f"[viewer] SpikeGLX {args.sglx}:{args.port}, imec probe {args.probe}: "
            f"{source.n_channels} channels, AP {source.fs_ap:.2f} Hz, LF {source.fs_lf:.2f} Hz",
            flush=True,
        )
        label = f"SpikeGLX imec{args.probe}"
    else:
        speed = None if str(args.speed).lower() == "max" else float(args.speed)
        source = FileReplaySource.from_recording_dir(rec, speed=speed)
        label = f"replay x{args.speed}"

    def make_pipeline(on_update):
        return OnlinePipeline(source, localizer, plan, config, on_update=on_update)

    title = f"Ephys-only localization ({label})" + (
        f" - {args.pid}" if args.pid else ""
    )
    viewer = LocalizationViewer(
        make_pipeline, localizer, plan, ba, reference_xyz=reference, title=title
    )
    viewer.run(snapshot=args.snapshot)
    pipeline = getattr(viewer, "pipeline", None)
    if pipeline is not None and pipeline.updates:
        out = OUT_DIR / f"{args.pid or source.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        pipeline.save(out)
        print(f"[viewer] run saved to {out}")


if __name__ == "__main__":
    main()
