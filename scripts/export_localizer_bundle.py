"""Export the fitted ephys-only localizer as a bundle of plain arrays (for the Open Ephys plugin).

Loads the released channel model and the localizer fit of the chosen feature set (fitting it
first if absent, which needs ONE credentials), then writes the bundle described in
:mod:`ephysatlas.realtime.bundle`, with a self-test on one held-out insertion.

Usage::

    python scripts/export_localizer_bundle.py
    python scripts/export_localizer_bundle.py --feature-set lf_ap --out-dir D:/bundles/lf_ap
"""

import argparse
from pathlib import Path

from ephysatlas.alignment.models import CHANNEL_MODEL_REPO_ID, CHANNEL_MODEL_REVISION

# %% User settings (each can also be given on the command line; see --help) ----------------------
FEATURE_SET = (
    "lf_ap_no_detection"  # the real-time localizer: no spike-detection features
)
# Held-out (test split) insertion used for the bundle's self-test; None: the first one.
SELFTEST_PID = None
OUT_DIR = (
    None  # default: ~/ephys-atlas/results/realtime/bundles/<model_commit>_<feature_set>
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--feature-set", default=FEATURE_SET, choices=("lf_ap", "lf_ap_no_detection")
    )
    parser.add_argument("--selftest-pid", default=SELFTEST_PID)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--revision",
        default=CHANNEL_MODEL_REVISION,
        help="channel model revision (Hub tag)",
    )
    parser.add_argument("--channel-model", default=CHANNEL_MODEL_REPO_ID)
    args = parser.parse_args()

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer, print_progress
    from ephysatlas.alignment.data import load_probe
    from ephysatlas.realtime.bundle import export_localizer_bundle

    channel_model = ChannelModel(args.channel_model, args.revision)
    test_pids = sorted(map(str, channel_model.split()["test_pids"]))
    selftest_pid = args.selftest_pid or test_pids[0]
    if selftest_pid not in test_pids:
        raise SystemExit(
            f"{selftest_pid} is not a test insertion of {args.channel_model}"
        )
    localizer = EphysOnlyLocalizer(
        channel_model,
        AllenAtlas(),
        feature_set=args.feature_set,
        progress=print_progress("  "),
    )
    out = export_localizer_bundle(
        localizer,
        args.out_dir,
        selftest_probe=load_probe(selftest_pid, channel_model.vintage),
    )
    print(f"[bundle] written to {out}")


if __name__ == "__main__":
    main()
