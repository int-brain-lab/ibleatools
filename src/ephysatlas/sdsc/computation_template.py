# Script to launch feature calculations for one pid.

from pathlib import Path
from deploy.iblsdsc import OneSdsc
import argparse
from ephysatlas.feature_computation import compute_features_from_pid

import logging

logging.basicConfig(level=logging.INFO)

# Constants
DURATION = 5  # Duration in seconds
OUTPUT_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics/output/")


def calc_features(probe_dict):
    print(f"Processing probe {probe_dict['pid'], probe_dict['t_start']}")
    one = OneSdsc(mode="local", cache_rest=None)
    # assert isinstance(one, OneSdsc), "one must be an instance of OneSdsc"
    compute_features_from_pid(
        **probe_dict,
        one=one,
        output_dir=OUTPUT_DIR,
        scratch_dir="/scratch/prai1/dartsort",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create task file for feature computation"
    )
    parser.add_argument("--pid", type=str, required=True, help="Probe ID")
    parser.add_argument("--eid", type=str, required=True, help="Experiment ID")
    parser.add_argument("--probe_name", type=str, required=True, help="Probe name")
    parser.add_argument(
        "--start_time",
        type=float,
        required=True,
        help="Start time of the passive period",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help=f"Duration of each snippet (default: {DURATION})",
    )

    args = parser.parse_args()

    # Create probe dictionary
    probe_dict = {
        "pid": args.pid,
        "eid": args.eid,
        "probe_name": args.probe_name.lower(),
        "t_start": args.start_time,
        "duration": args.duration,
    }
    # Calculate features
    calc_features(probe_dict)


if __name__ == "__main__":
    main()
