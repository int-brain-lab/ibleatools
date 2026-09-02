"""Train the region classifier (Model family 1) from scratch, end to end.

This is the third and final ``training/`` entry point, alongside ``train_spatial_encoder.py`` and
``train_unit_encoder.py``. Its output **is** the publish-ready release layout: per-fold and global
XGBoost weights (``model.ubj``), the cross-validation split (``split.json``), and the manifest
(``ephysatlas_model.json``) as the single source of truth -- no ``meta.yaml`` scaffold at any level.
So the directory loads through ``load_pretrained`` unchanged and is ready for
``scripts/publish_model_to_hf.py``.

Pipeline:

    download + read features  ->  N-fold XGBClassifier.fit  ->  folds/FOLD0k/model.ubj
    (a vintage's agg_full)        global XGBClassifier.fit   ->  model.ubj  +  split.json
                                  write_manifest             ->  ephysatlas_model.json

The split is on whole **insertions** (pids), so every channel of a held-out probe stays out of that
fold's training set. The global model trains on all channels; its reported accuracy is the pooled
out-of-fold accuracy from the folds (there is no held-out set to score it against directly).

This was adapted from the flat exploratory script
``examples/training_region_predictor_gradient_boosting.py``; the downstream HMM post-processing that
lived at the bottom of that file is *analysis*, not training, and is not carried here.

It imports xgboost at module scope (as the region-classifier code always does) and never imports
torch: the two segfault together on macOS arm64, so this trainer stays torch-free. Run it in its own
process, as the rest of the region path is.
"""

# %%
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import sklearn.metrics
from iblutil.numerical import hash_uuids, ismember
from xgboost import XGBClassifier

from ephysatlas import model_registry, regionclassifier

logger = logging.getLogger(__name__)

# Defaults mirroring the historical region-classifier run.
PROJECT_DEFAULT = "ea_active"
REGION_MAP_DEFAULT = "Cosmos"
REGION_LABEL_DEFAULT = "Cosmos_id"
N_FOLDS_DEFAULT = 5
SEED_DEFAULT = 12345
# The voltage-feature families the classifier is trained on. "outside" is appended to the resolved
# column list (it is not a voltage_features_set family); see main().
FEATURE_SET_DEFAULT = ["raw_lf", "raw_lf_csd", "raw_ap", "waveforms"]


# %%
def _fit_fold(df_features, train_idx, test_idx, feature_list, region_label, rids, device, label):
    """Fit one XGBClassifier on ``train_idx`` and score it on ``test_idx``.

    The classifier is trained on the contiguous class indices (``ismember`` maps the region ids to
    ``0..K-1``); ``rids`` is the full, ordered class list the manifest records, and every fold's
    train set must contain all of them (asserted) so each fold model emits the full class vector.

    Args:
        df_features (pd.DataFrame): Feature table indexed by ``(pid, channel)``.
        train_idx, test_idx (np.ndarray): Boolean row masks over ``df_features``.
        feature_list (list): Ordered feature-column names the model consumes positionally.
        region_label (str): Target column, e.g. ``"Cosmos_id"``.
        rids (np.ndarray): The full ordered class ids.
        device (str): XGBoost device (``"cpu"``, ``"cuda"``).
        label (str): Human label for logging.

    Returns:
        tuple: ``(probas, classifier, accuracy)``. For the global model (empty ``test_idx``),
        ``probas`` is None and ``accuracy`` is NaN -- there is no held-out set to score.
    """
    x_train = df_features.loc[train_idx, feature_list].values.astype(float)
    y_train = df_features.loc[train_idx, region_label].values.astype(float)
    classes = np.unique(df_features.loc[train_idx, region_label])
    _, iy_train = ismember(y_train, classes)

    classifier = XGBClassifier(device=device)
    classifier.fit(x_train, iy_train)
    # Every fold must see every class, or its probability vector would be short and misaligned with
    # the ensemble's column order.
    np.testing.assert_array_equal(classes, rids)

    # The global model trains on every channel: no held-out set, and scoring empty arrays raises in
    # sklearn >=1.8. Its accuracy is reported as the pooled out-of-fold accuracy by the caller.
    if int(np.sum(test_idx)) == 0:
        return None, classifier, float("nan")

    x_test = df_features.loc[test_idx, feature_list].values.astype(float)
    y_test = df_features.loc[test_idx, region_label].values.astype(float)
    probas = classifier.predict_proba(x_test)
    y_pred = classes[classifier.predict(x_test)]
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    logger.info(f"{label}: accuracy={accuracy:.4f}")
    return probas, classifier, accuracy


# %%
def train_region_classifier(
    df_features,
    feature_list,
    models_dir: Path,
    *,
    vintage: str,
    region_map: str = REGION_MAP_DEFAULT,
    region_label: str = REGION_LABEL_DEFAULT,
    n_folds: int = N_FOLDS_DEFAULT,
    seed: int = SEED_DEFAULT,
    device: str = "cpu",
) -> Path:
    """Train the folds + global model and write a publish-ready region-classifier directory.

    The output is the release layout: ``model.ubj`` at the root, ``folds/FOLD0k/model.ubj``
    (weights only), ``split.json``, ``predictions.pqt`` and the manifest -- no ``meta.yaml``. The
    model directory is named ``<vintage>_<region_map>_<hash>`` by ``save_model`` and returned.

    Args:
        df_features (pd.DataFrame): Feature table indexed by ``(pid, channel)``, carrying the
            ``feature_list`` columns and the ``region_label`` target.
        feature_list (list): Ordered feature-column names (their order is hashed into the manifest).
        models_dir (Path): Base directory the named model directory is created under.
        vintage (str): Release tag, e.g. ``"2025_W39"``.
        region_map (str): Region granularity, e.g. ``"Cosmos"`` (part of the model-dir name).
        region_label (str): Target column in ``df_features``.
        n_folds (int): Number of cross-validation folds over whole insertions.
        seed (int): Shuffle seed, recorded in the manifest and ``split.json``.
        device (str): XGBoost device (``"cpu"``, ``"cuda"``).

    Returns:
        Path: The written model directory.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # The full, ordered class list the manifest records; every fold is checked against it.
    rids = np.unique(df_features.loc[:, region_label])
    all_pids = np.array(df_features.index.get_level_values(0).unique())
    # Seed kept (np.random.seed returns None) so it can be recorded in the manifest / split.
    np.random.seed(seed)
    np.random.shuffle(all_pids)
    ifold = np.floor(np.arange(len(all_pids)) / len(all_pids) * n_folds)

    import pandas as pd

    pid_level = df_features.index.get_level_values(0)
    df_predictions = pd.DataFrame(
        index=df_features.index, columns=list(rids) + ["prediction", "fold"], dtype=float
    )

    # ---- per-fold models: each holds one probe-block out ----
    path_model_fold = None
    for i in range(n_folds):
        test_pids = all_pids[ifold == i]
        train_pids = all_pids[ifold != i]
        test_idx = np.isin(pid_level, test_pids)
        logger.info(f"fold {i}: {int(np.sum(test_idx))}/{len(df_features)} channels held out")
        probas, classifier, accuracy = _fit_fold(
            df_features, ~test_idx, test_idx, feature_list, region_label, rids, device, f"fold {i}"
        )
        df_predictions.loc[test_idx, rids] = probas
        df_predictions.loc[test_idx, "fold"] = i
        df_predictions.loc[test_idx, "prediction"] = rids[np.argmax(probas, axis=1)]
        meta = dict(
            RANDOM_SEED=seed,
            VINTAGE=vintage,
            REGION_MAP=region_map,
            FEATURES=list(feature_list),
            CLASSES=[int(c) for c in rids],
            ACCURACY=accuracy,
            TRAINING=dict(
                training_size=len(train_pids),
                testing_size=len(test_pids),
                hash_training=hash_uuids(train_pids),
                hash_testing=hash_uuids(test_pids),
            ),
        )
        # Folds are saved to one shared "tmp"-identified directory so they can be moved under
        # folds/ in a single rename below; save_model writes model.ubj only (no meta.yaml).
        path_model_fold = regionclassifier.save_model(
            models_dir, classifier, meta, subfolder=f"FOLD{i:02}", identifier="tmp"
        )

    # ---- global model: trained on every channel ----
    # Pooled out-of-fold accuracy: every channel was scored in exactly one held-out fold.
    global_accuracy = sklearn.metrics.accuracy_score(
        df_features[region_label].values, df_predictions["prediction"].values.astype(int)
    )
    _, classifier, _ = _fit_fold(
        df_features,
        np.ones(len(df_features), dtype=bool),
        np.zeros(len(df_features), dtype=bool),
        feature_list,
        region_label,
        rids,
        device,
        "global",
    )
    meta = dict(
        RANDOM_SEED=seed,
        VINTAGE=vintage,
        REGION_MAP=region_map,
        FEATURES=list(feature_list),
        CLASSES=[int(c) for c in rids],
        ACCURACY=global_accuracy,
        TRAINING=dict(
            training_size=len(all_pids),
            testing_size=0,
            hash_training=hash_uuids(all_pids),
            hash_testing=None,
        ),
    )
    path_model = regionclassifier.save_model(models_dir, classifier, meta)
    logger.info(f"global model accuracy (pooled out-of-fold): {global_accuracy:.4f}")
    df_predictions.to_parquet(path_model.joinpath("predictions.pqt"))

    # Publish the split itself (which insertions were held out), not just a hash: written from the
    # shuffled all_pids/ifold, in shuffled order, since hash_uuids is order-sensitive.
    model_registry.write_split(path_model, all_pids, ifold, random_seed=seed, n_folds=n_folds)

    # Stage the folds under folds/ (one rename of the shared tmp directory), then write the manifest
    # -- which lists the folds and stamps their model_class from the global save above.
    if path_model.joinpath("folds").exists():
        shutil.rmtree(path_model.joinpath("folds"))
    shutil.move(str(path_model_fold.parent), str(path_model.joinpath("folds")))

    # The manifest is the single source of truth every loader reads, written straight from the
    # in-memory meta (which save_model has stamped with MODEL_CLASS) -- no meta.yaml round trip.
    index = model_registry.write_manifest(path_model, meta, method="xgboost")
    logger.info(f"wrote publish-ready model at {path_model} (task={index.get('task')})")
    return path_model


# %%
def load_region_features(data_dir: Path, project: str, vintage: str):
    """Download (if needed) and read a vintage's feature table, with misaligned probes dropped.

    Mirrors the data-loading head of the old flat script. Kept out of
    :func:`train_region_classifier` so the training core is testable on an in-memory DataFrame
    without ONE, S3 or the Allen atlas.

    Args:
        data_dir (Path): Base directory ``download_tables`` caches the feature tables under.
        project (str): Feature project, e.g. ``"ea_active"``.
        vintage (str): Feature vintage / release label, e.g. ``"2025_W39"``.

    Returns:
        pd.DataFrame: The feature table indexed by ``(pid, channel)``, low-quality insertions removed.
    """
    from one.api import ONE

    import ephysatlas.anatomy
    import ephysatlas.data
    from ephysatlas.fixtures import misaligned_pids

    data_dir = Path(data_dir)
    path_features = data_dir.joinpath(project, vintage, "agg_full")
    if not path_features.exists():
        one = ONE()
        ephysatlas.data.list_available_labels(one=one, project=project)
        path_features = ephysatlas.data.download_tables(
            data_dir, label=vintage, project=project, one=one
        )

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df_features = ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas)
    # Drop the low-quality (misaligned) insertions before deriving classes, so the recorded class
    # list matches what training actually saw.
    df_features = df_features[~df_features.index.get_level_values(0).isin(misaligned_pids)]
    logger.info(f"loaded {len(df_features)} channels from {path_features}")
    return df_features


# %%
def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage", required=True, help="feature vintage to train on, e.g. 2025_W39")
    parser.add_argument("--project", default=PROJECT_DEFAULT, help="feature project (default ea_active)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="base directory where the feature tables are cached / downloaded",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="base directory to create the model dir under (default: <data-dir>/models)",
    )
    parser.add_argument("--region-map", default=REGION_MAP_DEFAULT)
    parser.add_argument("--region-label", default=REGION_LABEL_DEFAULT)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", default="cpu", help="XGBoost device: cpu or cuda (default cpu)")
    parser.add_argument(
        "--feature-set",
        nargs="+",
        default=FEATURE_SET_DEFAULT,
        help="voltage_features_set families to train on (default: raw_lf raw_lf_csd raw_ap waveforms)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    args = _parse_args(argv)

    import ephysatlas.features

    df_features = load_region_features(args.data_dir, args.project, args.vintage)
    # Resolve the ordered feature-column list; "outside" is appended as in the historical run (it is
    # a column in the table, not a voltage_features_set family).
    feature_list = ephysatlas.features.voltage_features_set(args.feature_set)
    feature_list.append("outside")

    models_dir = args.models_dir or args.data_dir.joinpath("models")
    path_model = train_region_classifier(
        df_features,
        feature_list,
        models_dir,
        vintage=args.vintage,
        region_map=args.region_map,
        region_label=args.region_label,
        n_folds=args.n_folds,
        seed=args.seed,
        device=args.device,
    )
    logger.info(f"done. Model written to {path_model}")
    logger.info(
        "next: publish with `python scripts/publish_model_to_hf.py "
        f"--model-dir {path_model} --features {args.data_dir}/{args.project}/{args.vintage}/agg_full "
        "--method xgboost ...`"
    )
    return path_model


if __name__ == "__main__":
    main()
