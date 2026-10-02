"""Align or localize every insertion of the feature release (the entire dataset).

Same methods and outputs as ``run_alignment_heldout.py``, over all insertions of the release
(misaligned insertions excluded, as for model training), with each insertion's split recorded in
``alignment_metrics.csv``. For training insertions the channel model's neighbours never include the
insertion's own channels, but the models were fitted on them: their scores are in-sample and
optimistic -- use the held-out run for performance, this one for coverage (e.g. finding insertions
whose human alignment disagrees with the atlas).

Usage::

    python scripts/run_alignment_dataset.py
    python scripts/run_alignment_dataset.py --methods histology --figures
"""

import argparse
from pathlib import Path

from ephysatlas.alignment.models import CHANNEL_MODEL_REPO_ID, UNIT_MODEL_REPO_ID, VINTAGE
from ephysatlas.alignment.runner import METHODS, TRACES, RunSettings, run_alignments, summarize
from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings (each can also be given on the command line; see --help) ----------------------
METHODS_TO_RUN = ("histology", "ephys_only")
TRACE = "picks"  # "picks" (reconstructed histology track) or "human" (human channel positions)
OUT_DIR = DEFAULT_RESULTS_DIR / "alignment" / f"dataset_{VINTAGE}"
SAVE_FIGURES = False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS_TO_RUN), choices=METHODS)
    parser.add_argument("--trace", default=TRACE, choices=TRACES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--vintage", default=VINTAGE)
    parser.add_argument("--channel-model", default=CHANNEL_MODEL_REPO_ID)
    parser.add_argument("--unit-model", default=UNIT_MODEL_REPO_ID)
    parser.add_argument("--figures", action="store_true", default=SAVE_FIGURES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only the first N insertions")
    args = parser.parse_args()

    from ephysatlas.alignment.data import ChannelDataset
    from ephysatlas.alignment.models import ChannelModel

    split = ChannelModel(args.channel_model, args.vintage).split()
    split_of = {str(p): name for name, key in (("train", "train_pids"), ("validation", "validation_pids"),
                                               ("test", "test_pids")) for p in split[key]}
    pids = ChannelDataset.load(args.vintage).pids.tolist()
    print(f"[alignment] {len(pids)} insertions -> {args.out_dir}")
    csv_path = run_alignments(
        pids,
        RunSettings(
            out_dir=args.out_dir,
            methods=args.methods,
            trace=args.trace,
            vintage=args.vintage,
            channel_repo_id=args.channel_model,
            unit_repo_id=args.unit_model,
            save_figures=args.figures,
            overwrite=args.overwrite,
            limit=args.limit,
        ),
        split_of={p: split_of.get(p, "excluded") for p in pids},
    )
    print(summarize(csv_path).round(3).to_string())


if __name__ == "__main__":
    main()
