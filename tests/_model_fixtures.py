"""Test-support: build a minimal, loadable region-classifier model directory.

The code that writes real publication manifests, checksums and release directories lives in
``paper-ephys-atlas`` alongside the training scripts (producing a model is a paper-side concern).
These helpers reproduce only the parts of a published release the *load* path reads back, so the
load/predict tests in this repo stay self-contained without importing the packaging code:

* a tiny trained ``XGBClassifier`` written as ``model.ubj`` plus per-fold weights under ``folds/``,
* the region-classifier manifest fields :class:`ephysatlas.regionclassifier.RegionClassifier`
  and :func:`ephysatlas.load_pretrained` read, and
* a ``checksums.json`` over the actual bytes, so :func:`ephysatlas.model_registry.verify_checksums`
  has something faithful to re-hash.

The checksum writer mirrors the paper-side ignore rules (the checksum file itself, the model card,
Hub-added files) so the verify-path tests behave exactly as they do against a real download.
"""

import fnmatch
import json
from pathlib import Path

import numpy as np
from iblutil.io import hashfile
from xgboost import XGBClassifier

from ephysatlas import model_registry

# The task string the paper-side writer stamps for this family; the load path never authors it,
# so it is duplicated here for the tests that assert a loaded model reports it.
TASK_REGION_CLASSIFICATION = "region-classification"
MODEL_CLASS = "xgboost.sklearn.XGBClassifier"

# Real Cosmos region ids, so the acronym lookup resolves; a handful of real feature names.
CLASSES = [315, 549, 997]  # Isocortex, TH, root
FEATURES = ["rms_ap", "rms_lf", "psd_delta", "psd_theta", "spike_count"]

# Files a published release never hashes: the checksum file cannot cover itself, the card is
# edited on the Hub after publication, and a snapshot carries .gitattributes / a .cache/ tree.
_CHECKSUM_IGNORE = (
    "predictions.pqt",
    ".DS_Store",
    "*.tmp",
    model_registry.MODEL_CHECKSUM_FILE,
    "README.md",
    "LICENSE",
    ".gitattributes",
    ".git/*",
    ".cache/*",
)


def _is_ignored(relative_posix: str, patterns=_CHECKSUM_IGNORE) -> bool:
    """True when a model-relative path matches any ignore pattern (full path or bare name)."""
    name = relative_posix.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(relative_posix, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def write_checksums(path_model: Path) -> Path:
    """Record a sha256 digest of every hashable file, as a published release ships it."""
    path_model = Path(path_model)
    files = []
    for path in sorted(path_model.rglob("*"), key=lambda p: p.relative_to(path_model).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(path_model).as_posix()
        if _is_ignored(relative):
            continue
        files.append(
            {
                "path": relative,
                "hash": hashfile.sha1(path),
                "bytes": path.stat().st_size,
            }
        )
    out = path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
    out.write_text(json.dumps({"algo": "sha1", "files": files}, indent=2) + "\n")
    return out


def write_region_manifest(
    path_model: Path,
    *,
    classes=CLASSES,
    features=FEATURES,
    region_map: str = "Cosmos",
    vintage: str = "2026_W32",
    folds=None,
) -> dict:
    """Write the region-classifier manifest fields the load path reads back.

    Only the fields the loader consumes are written -- dispatch (``model_class``), the ordered
    feature list and its digest (``inputs``), the class/acronym config, and the artifact roles.
    """
    path_model = Path(path_model)
    folds_root = path_model.joinpath("folds")
    if folds is None:
        folds = (
            sorted(p.name for p in folds_root.glob("FOLD*"))
            if folds_root.is_dir()
            else []
        )
    index = {
        "task": TASK_REGION_CLASSIFICATION,
        "model_class": MODEL_CLASS,
        "vintage": vintage,
        "granularity": "channel",
        "artifacts": {"weights": "model.ubj", "folds": folds},
        "inputs": {
            "table": "raw_ephys_features_denoised.pqt",
            "index": ["pid", "channel"],
            "features": list(features),
            "feature_order_sha256": model_registry.feature_order_sha256(features),
        },
        "outputs": {
            "kind": "categorical",
            "columns": [
                "predicted_acronym",
                "predicted_atlas_id",
                "prediction_probability",
                "fold_agreement",
            ],
        },
        "config": {
            "classes": [int(c) for c in classes],
            "class_acronyms": model_registry.class_acronyms(classes, region_map),
            "region_map": region_map,
        },
    }
    path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).write_text(
        json.dumps(index, indent=2) + "\n"
    )
    return index


def make_model_dir(
    path_models: Path,
    n_folds: int = 2,
    *,
    manifest: bool = True,
    checksums: bool = True,
) -> Path:
    """Train and save a tiny synthetic region model with folds, as a loadable release directory.

    The global model sees every row; each fold holds one contiguous block out, so the ensemble
    and the global model are genuinely different estimators, as in production.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, len(FEATURES)))
    # Balanced and deterministic, so every held-out block still contains all classes and each
    # fold model emits the full class vector.
    y = np.tile(np.arange(len(CLASSES)), 90 // len(CLASSES))

    def _fit(keep=None):
        """Fit on all rows, or on everything outside one held-out block (a real fold)."""
        mask = np.ones(len(y), bool) if keep is None else keep
        clf = XGBClassifier(n_estimators=2, max_depth=2)
        clf.fit(x[mask], y[mask])
        return clf

    path_model = Path(path_models).joinpath("2026_W32_Cosmos_test")
    path_model.mkdir(parents=True, exist_ok=True)
    _fit().save_model(path_model.joinpath("model.ubj"))

    folds = path_model.joinpath("folds")
    folds.mkdir(exist_ok=True)
    block = len(y) // n_folds
    for i in range(n_folds):
        keep = np.ones(len(y), bool)
        keep[i * block : (i + 1) * block] = False
        fold_dir = folds.joinpath(f"FOLD{i:02d}")
        fold_dir.mkdir(exist_ok=True)
        _fit(keep).save_model(fold_dir.joinpath("model.ubj"))

    if manifest:
        write_region_manifest(path_model)
    # Every published model ships checksums.json, and the load path requires it, so a faithful
    # fixture writes them last, over whatever is now on disk.
    if checksums:
        write_checksums(path_model)
    return path_model


def region_meta() -> dict:
    """The UPPER_CASE meta dict for the synthetic model, as the trainer would hold it.

    Used by the load tests that stage a legacy ``meta.yaml`` on disk to prove the manifest wins.
    """
    return dict(
        RANDOM_SEED=42,
        VINTAGE="2026_W32",
        REGION_MAP="Cosmos",
        FEATURES=FEATURES,
        CLASSES=CLASSES,
        ACCURACY=0.5,
        TRAINING=dict(training_size=10, testing_size=2),
        MODEL_CLASS=MODEL_CLASS,
    )
