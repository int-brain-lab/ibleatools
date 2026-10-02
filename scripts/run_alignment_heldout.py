"""Align or localize every held-out (test) insertion of the released model's split.

Runs the chosen methods of :mod:`ephysatlas.alignment` on each insertion of the channel model
release's test split and scores them against the human (histology) alignment:

- ``histology``      -- channel features warped along the reconstructed histology trace;
- ``histology_unit`` -- the same with the spike-sorted units' unit-model likelihood added;
- ``ephys_only``     -- no histology: Student-t likelihood + planned-trajectory prior search.

Per insertion and method, one row of ``alignment_metrics.csv`` (Cosmos / Beryl accuracy, channel
distance, their high-/low-confidence versions and the planned trajectory's scores), plus the full
result (``results/<method>/<pid>.npz``) and optionally its figure (``figures/<method>/<pid>.png``).
Interrupted runs resume. Needs ONE credentials the first time (feature tables, histology picks).

Usage::

    python scripts/run_alignment_heldout.py
    python scripts/run_alignment_heldout.py --methods histology histology_unit --trace human
"""

import argparse
from pathlib import Path

from ephysatlas.alignment.models import CHANNEL_MODEL_REPO_ID, UNIT_MODEL_REPO_ID, VINTAGE
from ephysatlas.alignment.runner import METHODS, TRACES, RunSettings, run_alignments, summarize
from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings (each can also be given on the command line; see --help) ----------------------
METHODS_TO_RUN = ("histology", "histology_unit", "ephys_only")
# Histology trace: "picks" (the reconstructed track, as an experimenter has it) or "human" (the
# human alignment's channel positions, extended; the setting of the paper's Figure 5).
TRACE = "picks"
SPLIT = "test"  # "test" (held out), or "validation"
OUT_DIR = DEFAULT_RESULTS_DIR / "alignment" / f"heldout_{VINTAGE}"
SAVE_FIGURES = False  # one figure per insertion and method (slower)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS_TO_RUN), choices=METHODS)
    parser.add_argument("--trace", default=TRACE, choices=TRACES)
    parser.add_argument("--split", default=SPLIT, choices=("test", "validation"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: OUT_DIR, with a _human suffix for --trace human)")
    parser.add_argument("--vintage", default=VINTAGE)
    parser.add_argument("--channel-model", default=CHANNEL_MODEL_REPO_ID)
    parser.add_argument("--unit-model", default=UNIT_MODEL_REPO_ID)
    parser.add_argument("--figures", action="store_true", default=SAVE_FIGURES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only the first N insertions")
    args = parser.parse_args()

    from ephysatlas.alignment.data import split_pids
    from ephysatlas.alignment.models import ChannelModel

    out_dir = args.out_dir or (OUT_DIR if args.trace == "picks" else OUT_DIR.with_name(OUT_DIR.name + "_human"))
    pids = split_pids(ChannelModel(args.channel_model, args.vintage), args.split)
    print(f"[alignment] {len(pids)} {args.split} insertions -> {out_dir}")
    csv_path = run_alignments(
        pids,
        RunSettings(
            out_dir=out_dir,
            methods=args.methods,
            trace=args.trace,
            vintage=args.vintage,
            channel_repo_id=args.channel_model,
            unit_repo_id=args.unit_model,
            save_figures=args.figures,
            overwrite=args.overwrite,
            limit=args.limit,
        ),
        split_of={p: args.split for p in pids},
    )
    print(summarize(csv_path).round(3).to_string())


if __name__ == "__main__":
    main()
