"""Run the published region classifier the way an outside user would.

Deliberately uses **no ONE account, no AWS credentials and no raw data**. It loads a model
either from the Hugging Face Hub or from a local directory, verifies the install reproduces
the model's shipped golden output, and predicts on the bundled example features.

Usage::

    # from the Hub (once published)
    python examples/inference_region_classifier_public.py \\
        --model int-brain-lab/ea-decoder-channel-xgboost --revision 2026_W32

    # from a local packaged directory
    python examples/inference_region_classifier_public.py --model /path/to/model_dir

To run it on your own data, pass ``--features`` pointing at a parquet of *denoised aggregated*
features for the vintage the model was trained on (see the model card).
"""

# %%
import argparse
import logging
from pathlib import Path

import pandas as pd

from ephysatlas import load_pretrained

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
_logger = logging.getLogger("inference_public")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id (owner/name) or a path to a local model directory",
    )
    parser.add_argument("--revision", default=None, help="Hub branch/tag to pin")
    parser.add_argument(
        "--estimator",
        default="ensemble",
        choices=["ensemble", "global"],
        help="ensemble averages the fold models (default, gives fold_agreement); "
        "global uses the single all-data model (cheaper, fold_agreement is NaN)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="parquet of your own features; defaults to the model's bundled example",
    )
    args = parser.parse_args(argv)

    # %%
    # --- load ---------------------------------------------------------------
    # load_pretrained handles both a local directory and a Hub repo id, and returns the
    # wrapper appropriate to the model's task. It is the only API a model card should name.
    clf = load_pretrained(args.model, revision=args.revision)

    index = clf.index
    _logger.info(f"model      : {index['model_id']}")
    _logger.info(f"task       : {index['task']} ({index.get('granularity')} level)")
    _logger.info(f"vintage    : {index['vintage']}")
    _logger.info(f"features   : {len(clf.inputs['features'])}")
    _logger.info(f"row index  : {clf.inputs['index']}")
    _logger.info(f"returns    : {index['outputs']['kind']} -> {index['outputs']['columns']}")
    _logger.info(f"classes    : {clf.config['class_acronyms']}")
    _logger.info(f"folds      : {len(index['artifacts']['folds'])}")

    # %%
    # --- verify the install reproduces the shipped predictions --------------
    # This is the guard against version skew: a different xgboost, or features whose units
    # changed between vintages, shows up here rather than as quietly wrong regions.
    try:
        clf.selftest()
        _logger.info("selftest PASSED -- this install reproduces the shipped output")
    except FileNotFoundError:
        _logger.warning("model ships no example/ golden files; skipping selftest")

    # %%
    # --- predict -----------------------------------------------------------
    if args.features is not None:
        df = pd.read_parquet(args.features)
        _logger.info(f"predicting on {len(df)} rows from {args.features}")
    else:
        sample = clf.path_model.joinpath("example", "features_sample.parquet")
        df = pd.read_parquet(sample)
        _logger.info(f"predicting on the bundled {len(df)}-channel example")

    _logger.info(f"estimator  : {args.estimator}")
    out = clf.predict(df, estimator=args.estimator)

    # %%
    # --- inspect -----------------------------------------------------------
    print()
    print(out[["predicted_acronym", "prediction_probability", "fold_agreement"]].head(10))
    print()
    print("predicted region counts:")
    print(out["predicted_acronym"].value_counts().to_string())
    print()
    print(
        f"mean confidence {out['prediction_probability'].mean():.3f}, "
        f"mean fold agreement {out['fold_agreement'].mean():.3f}"
    )
    # Predictions are namespaced, so they join straight onto the input without clashing with
    # any histology-derived `acronym` / `atlas_id` columns it may carry.
    joined = df.join(out)
    print(f"joined onto input features: {joined.shape[0]} rows x {joined.shape[1]} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
