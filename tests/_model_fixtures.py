"""Test-support: build a minimal, loadable region-classifier model directory.

Producing a real release -- manifests, checksums, directory layout -- is not this repo's job.
These helpers reproduce only the parts of a published release the *load* path reads back, so the
load/predict tests stay self-contained:

* a tiny trained ``XGBClassifier`` written as ``model.ubj`` plus per-fold weights under ``folds/``,
* the region-classifier manifest fields :class:`ephysatlas.regionclassifier.RegionClassifier`
  and :func:`ephysatlas.load_pretrained` read, and
* a ``checksums.json`` over the actual bytes, so :func:`ephysatlas.model_registry.verify_checksums`
  has something faithful to re-hash.

The checksum writer mirrors a real release's ignore rules (the checksum file itself, the model
card, Hub-added files) so the verify-path tests behave as they do against a real download.
"""

import fnmatch
import json
from pathlib import Path

import numpy as np
from iblutil.io import hashfile
from xgboost import XGBClassifier

from ephysatlas import model_registry

# The task string a published release stamps for this family; the load path never authors it,
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
    """Record a sha1 digest of every hashable file, as a published release ships it."""
    path_model = Path(path_model)
    files = []
    for path in sorted(
        path_model.rglob("*"), key=lambda p: p.relative_to(path_model).as_posix()
    ):
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


# ---- unit-level encoder ---------------------------------------------------------------------

# The phenotype features the released unit model's kNN stage projects (the order of
# ``waveform_feature_names.json`` in the prepared unit data).
# Spelled out rather than imported: the unit package pulls in torch, which this module must not
# (see test_unit_encoder). test_unit_waveform_features checks it against FEATURE_NAMES.
UNIT_FEATURES = [
    "depolarisation_slope",
    "recovery_slope",
    "repolarisation_slope",
    "spatial_spread_um",
    "tip_val",
    "spike_width_secs",
    "predepolarisation_width_secs",
    "spike_amplitude",
    "peak_to_trough_ratio_log",
    "polarity",
]


def unit_test_config():
    """A shrunk unit-model ``Config``: tiny modalities, few components, CPU."""
    from ephysatlas.unit_level_encoder.config import Config

    return Config(
        device="cpu",
        vintage="2026_W39",
        waveform_shape=(4, 16),
        acg_shape=(2, 8),
        stpc_shape=(6,),
        modality_latent_dim=3,
        gmm_components=3,
        knn_decoder_k=5,
        context_hidden_dim=16,
        context_layers=2,
        context_dropout=0.0,
        feature_slice_component_mc_samples=16,
        readout_neighbours=20,
        readout_shrinkage=2.0,
    )


class FakeContextManager:
    """Deterministic stand-in for ``ContextAtlasManager``: no Allen atlas download.

    Maps a position to a smooth 50 + 50 dimensional "context", so predictions vary with position
    the way they do over the real PCA volumes.
    """

    def __init__(self, n_cell_pcs=50, n_gene_pcs=50):
        rng = np.random.default_rng(3)
        self.w_cell = rng.normal(size=(3, n_cell_pcs)) * 400.0
        self.w_gene = rng.normal(size=(3, n_gene_pcs)) * 400.0

    def sample_context_numpy_m(self, xyz_m, mode="clip"):
        xyz = np.asarray(xyz_m, np.float64)
        xyz = np.column_stack([-np.abs(xyz[:, 0]), xyz[:, 1], xyz[:, 2]])
        return {
            "cell_pc": np.sin(xyz @ self.w_cell).astype(np.float32),
            "gene_pc": np.cos(xyz @ self.w_gene).astype(np.float32),
        }


def make_unit_model_dir(
    path_models: Path, *, checksums: bool = True, readout: bool = False
) -> Path:
    """Write a tiny random-init unit model in the published release layout, with its manifest.

    Every stage is real but small: a random-init :class:`UnitAutoencoder`, a latent scaler and a
    3-component full-covariance GMM fitted on random latents, a random-init context-weight net,
    and a kNN bank of random exemplars -- with a random context-local member readout when
    ``readout``, else without one, like a release made before it. No golden example is written;
    see :func:`write_unit_golden_example`.
    """
    import joblib
    import torch
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    from ephysatlas.unit_level_encoder.data import ContextTransform
    from ephysatlas.unit_level_encoder.gmm_models import (
        ContextWeightModel,
        ContextWeightNet,
        save_context_weight_bundle,
    )
    from ephysatlas.unit_level_encoder.knn_decoder import EmpiricalKNNDecoder
    from ephysatlas.unit_level_encoder.model import UnitAutoencoder
    from ephysatlas.unit_level_encoder.pipeline import (
        compute_component_features,
        save_component_features,
    )

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = unit_test_config()
    path_model = Path(path_models).joinpath("2026_W39_unit_test")
    path_model.mkdir(parents=True, exist_ok=True)

    ae = UnitAutoencoder(cfg).eval()
    torch.save(
        {
            "model_state_dict": ae.state_dict(),
            "config": cfg.to_json_dict(),
            "polarity_values": None,
        },
        path_model.joinpath(model_registry.UNIT_AE_FILE),
    )
    path_model.joinpath(model_registry.UNIT_CONFIG_FILE).write_text(
        json.dumps(cfg.to_release_dict(), indent=2)
    )

    latent_dim = cfg.latent_dim()
    scaler = StandardScaler().fit(rng.normal(size=(64, latent_dim)))
    joblib.dump(scaler, path_model.joinpath(model_registry.UNIT_SCALER_FILE))
    z_train = rng.normal(size=(60, latent_dim)).astype(np.float32)
    gmm = GaussianMixture(
        n_components=cfg.gmm_components, covariance_type="full", random_state=0
    ).fit(z_train)
    joblib.dump(gmm, path_model.joinpath(model_registry.UNIT_GMM_FILE))

    n_context = cfg.n_cell_pcs + cfg.n_gene_pcs
    transform = ContextTransform(StandardScaler().fit(rng.normal(size=(64, n_context))))
    net = ContextWeightNet(
        n_context,
        cfg.context_hidden_dim,
        cfg.context_layers,
        cfg.context_dropout,
        cfg.gmm_components,
    ).eval()
    save_context_weight_bundle(
        ContextWeightModel(net, np.zeros((0, n_context), np.float32), "cpu"),
        transform,
        path_model,
        cfg,
    )

    features = rng.normal(size=(60, len(UNIT_FEATURES))).astype(np.float32)
    knn = EmpiricalKNNDecoder.from_bank(
        z_train, features, k=cfg.knn_decoder_k, feature_names=UNIT_FEATURES
    )
    if readout:
        knn.set_context_readout(
            gmm.predict(z_train),
            rng.normal(size=(2, n_context)) * 0.3,
            np.zeros(2),
            rng.normal(size=(len(z_train), n_context)),
            key_alpha=10.0,
            void_context_pc=transform.transform(np.zeros((1, n_context))),
        )
    knn.save_bank(path_model.joinpath(model_registry.UNIT_KNN_BANK_FILE))
    save_component_features(
        path_model.joinpath(model_registry.UNIT_COMPONENT_FEATURES_FILE),
        compute_component_features(gmm, knn, cfg),
        UNIT_FEATURES,
        cfg,
    )
    path_model.joinpath("split.json").write_text(
        json.dumps({"train": ["a"], "validation": ["b"], "test": ["c"]})
    )

    index = {
        "task": "unit-encoding",
        "model_class": "UnitAutoencoder",
        "vintage": cfg.vintage,
        "granularity": "unit",
        "artifacts": {
            "autoencoder": model_registry.UNIT_AE_FILE,
            "config": model_registry.UNIT_CONFIG_FILE,
            "scaler": model_registry.UNIT_SCALER_FILE,
            "gmm": model_registry.UNIT_GMM_FILE,
            "context_transform": model_registry.UNIT_CONTEXT_TRANSFORM_FILE,
            "context_weights": model_registry.UNIT_CONTEXT_WEIGHTS_FILE,
            "knn_bank": model_registry.UNIT_KNN_BANK_FILE,
            "component_features": model_registry.UNIT_COMPONENT_FEATURES_FILE,
            "split": "split.json",
        },
        "inputs": {"index": ["pid", "cluster"], "columns": ["x", "y", "z"]},
        "outputs": {
            "kind": "continuous",
            "columns": list(UNIT_FEATURES),
            "feature_order_sha256": model_registry.feature_order_sha256(UNIT_FEATURES),
            "latent_dim": latent_dim,
        },
    }
    path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).write_text(
        json.dumps(index, indent=2)
    )
    if checksums:
        write_checksums(path_model)
    return path_model


def unit_positions(n=12, seed=1):
    """Positions (metres, IBL frame) spread over both hemispheres of a mouse brain."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    xyz = np.column_stack(
        [
            rng.uniform(-4e-3, 4e-3, n),
            rng.uniform(-7e-3, 3e-3, n),
            rng.uniform(-6e-3, 0, n),
        ]
    )
    index = pd.MultiIndex.from_arrays(
        [["pid"] * n, np.arange(n)], names=["pid", "cluster"]
    )
    return pd.DataFrame(xyz, index=index, columns=["x", "y", "z"])


def synthetic_units(cfg, n=8, seed=2):
    """Max-abs normalized random waveforms plus random ACG/stPC, shaped as the config expects."""
    rng = np.random.default_rng(seed)
    waveform = rng.normal(size=(n, *cfg.waveform_shape)).astype(np.float32)
    waveform /= np.abs(waveform).max(axis=(1, 2), keepdims=True)
    acg = np.abs(rng.normal(size=(n, *cfg.acg_shape))).astype(np.float32)
    stpc = rng.normal(size=(n, *cfg.stpc_shape)).astype(np.float32)
    return waveform, acg, stpc
