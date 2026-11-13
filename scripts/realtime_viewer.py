"""Watch the online localization in a live window: a replayed test recording or SpikeGLX.

Two sources (:mod:`ephysatlas.realtime.sources`):

- replay (default): one recording prepared by realtime_prepare_recordings.py, replayed at
  real-time speed or faster;
- live (``--sglx [ADDRESS]``): an imec probe acquired by a running SpikeGLX (its command server
  enabled), e.g. a simulated probe playing one of the test recordings.

The plan comes from ``--plan`` (a plan JSON or a Pinpoint v5 experiment file, see
:mod:`ephysatlas.realtime.plan`; ``--plan-probe`` names the probe of an experiment of several)
or from the test recording named by ``pid``. Without one (``--sglx`` without ``--plan``, or
``--no-plan``) every update searches the whole brain (:mod:`ephysatlas.realtime.whole_brain`).
``--reference`` also draws the histology alignment of that test recording (never available live).
A probe other than a Neuropixels 1.0 (e.g. a 2.0, single- or four-shank) is localized one shank at
a time: ``--shank`` (default the shank with the most channels; :mod:`ephysatlas.realtime.probe`).
The run's outputs are saved in ``OUT_DIR/<name>_<time>/`` when the window closes.

Usage::

    python scripts/realtime_viewer.py 143dd7cf-6a47-47a1-906d-927ad7fe9117 --reference
    python scripts/realtime_viewer.py --sglx --plan C:/data/plan.json
    python scripts/realtime_viewer.py --sglx --plan C:/data/experiment-001.json --plan-probe "Probe A"
    python scripts/realtime_viewer.py --sglx
    python scripts/realtime_viewer.py --sglx --shank 2
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
        "--shank",
        type=int,
        default=None,
        help="the shank to localize on a multi-shank probe (default: the one with the most "
        "channels)",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="plan JSON or Pinpoint v5 experiment file (default: the test recording's; "
        "none with --sglx: whole-brain search)",
    )
    parser.add_argument(
        "--plan-probe",
        default=None,
        metavar="NAME",
        help='the probe of a Pinpoint experiment of several, e.g. "Probe A"',
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="ignore the plan: search the whole brain at every update",
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
    if args.pid is None and args.sglx is None:
        parser.error("give a test recording pid, or --sglx (with --plan, or without one)")
    if args.reference and args.pid is None:
        parser.error("--reference needs the pid of a test recording")

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer
    from ephysatlas.realtime.online import OnlineConfig, OnlinePipeline, suggest_workers
    from ephysatlas.realtime.plan import load_plan
    from ephysatlas.realtime.sources import FileReplaySource, SpikeGLXSource
    from ephysatlas.realtime.viewer import LocalizationViewer

    rec = None
    if args.pid:  # the test recording of that pid (any time window)
        found = sorted(args.data_root.glob(f"IBL_{args.pid}_*"))
        rec = found[0] if found else args.data_root / f"IBL_{args.pid}_300s_600s"
    try:  # the probe's layout decides the shank (bad --shank, too short a shank: ValueError)
        if args.sglx is not None:
            source = SpikeGLXSource(
                probe=args.probe, address=args.sglx, port=args.port, shank=args.shank
            )
            print(
                f"[viewer] SpikeGLX {args.sglx}:{args.port}, imec probe {args.probe}: "
                f"AP {source.fs_ap:.2f} Hz"
                + (f", LF {source.fs_lf:.2f} Hz" if source.has_lf else ""),
                flush=True,
            )
            label = f"SpikeGLX imec{args.probe}"
        else:
            speed = None if str(args.speed).lower() == "max" else float(args.speed)
            source = FileReplaySource.from_recording_dir(
                rec, speed=speed, shank=args.shank
            )
            label = f"replay x{args.speed}"
    except ValueError as e:
        parser.error(str(e))
    layout = source.layout
    print(f"[viewer] {layout.describe()}", flush=True)
    if not layout.identity:
        print(
            "[viewer] the localizer was fitted on Neuropixels 1.0 recordings: its accuracy on "
            "this probe has not been measured",
            flush=True,
        )

    plan_file = None if args.no_plan else (args.plan or (rec / "plan.json" if rec else None))
    if plan_file is None:
        plan = None
        print("[viewer] no plan: every update searches the whole brain", flush=True)
    else:
        try:
            plan_doc = load_plan(plan_file, probe=args.plan_probe, shank=layout.shank)
            plan = layout.plan_rows(plan_doc["planned_xyz_m"], plan_doc.get("heights_um"))
        except ValueError as e:
            hint = " (name the probe with --plan-probe)" if "probe" in str(e) else ""
            parser.error(f"{e}{hint}")
        print(f"[viewer] plan: {plan_file} ({plan_doc.get('source', '')})", flush=True)
        if plan_doc.get("warning"):
            print(f"[viewer] warning: {plan_doc['warning']}", flush=True)
    reference = None
    if args.reference:
        reference = np.load(TRUTH_DIR / f"{args.pid}.npz")["human_xyz"]
        if not layout.identity:
            print(
                "[viewer] --reference ignored: it is for the 384 channels of a 1.0",
                flush=True,
            )
            reference = None
    ba = AllenAtlas()
    localizer = EphysOnlyLocalizer(ChannelModel(), ba, feature_set=FEATURE_SET)
    n_workers = args.workers or suggest_workers()
    print(f"[viewer] {n_workers} feature workers", flush=True)
    config = OnlineConfig(cadence_s=args.cadence, n_workers=n_workers)

    def make_pipeline(on_update):
        return OnlinePipeline(source, localizer, plan, config, on_update=on_update)

    title = f"Ephys-only localization ({label})" + (
        f" - {args.pid}" if args.pid else ""
    )
    viewer = LocalizationViewer(
        make_pipeline,
        localizer,
        plan,
        ba,
        reference_xyz=reference,
        title=title,
        n_channels=layout.n_rows,
    )
    viewer.run(snapshot=args.snapshot)
    pipeline = getattr(viewer, "pipeline", None)
    if pipeline is not None and pipeline.updates:
        out = OUT_DIR / f"{args.pid or source.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        pipeline.save(out)
        print(f"[viewer] run saved to {out}")


if __name__ == "__main__":
    main()
