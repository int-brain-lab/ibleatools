"""Tests for ``scripts/repackage_release_from_hf.py``.

The heavy path (download + real weights) is exercised manually; these tests pin the two things
that can silently go wrong without touching the network:

1. the pure ``meta.yaml`` synthesis maps the release config onto the keys the manifest builder
   reads, and
2. that synthesized ``meta.yaml`` actually drives ``model_registry.build_model_index`` to a
   correct, per-family manifest (dummy artifact files stand in for the real weights, since the
   builder only scans for their presence).
"""

import importlib.util
from pathlib import Path

import pytest

import ephysatlas.model_registry as model_registry


def _load_script():
    """Import the repackaging script as a module (it is a script, not a package)."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root.joinpath("scripts", "repackage_release_from_hf.py")
    spec = importlib.util.spec_from_file_location("repackage_release_from_hf", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repackage = _load_script()


# A trimmed copy of the release's config.json / features.json, enough to synthesize from.
CHANNEL_CONFIG = {
    "channel_level": {
        "architecture": {"d_model": 128, "nhead": 8, "depth": 2, "drop": 0.15},
        "neighbors": {"radius_um": 500, "m_max": 8},
        "training": {"seed": 0},
    },
    "context": {"n_cell_pcs": 50, "n_gene_pcs": 50},
}
FEATURES = ["rms_lf", "psd_lfp", "spike_count"]
UNIT_CONFIG = {"shared_latent_dim": 32, "gmm_components": 16, "seed": 0}


# ---- pure synthesis -------------------------------------------------------------------------
def test_build_channel_meta_maps_release_config():
    meta = repackage.build_channel_meta(CHANNEL_CONFIG, FEATURES, "2026_W26")
    # The class name is what MODEL_CLASS_TASKS keys the spatial-encoding task on.
    assert meta["MODEL_CLASS"] == "NeighborInpaintingModel"
    assert model_registry.MODEL_CLASS_TASKS[meta["MODEL_CLASS"]] == model_registry.TASK_SPATIAL_ENCODING
    assert meta["VINTAGE"] == "2026_W26"
    assert meta["FEATURES"] == FEATURES
    assert meta["D_MODEL"] == 128 and meta["NHEAD"] == 8 and meta["DEPTH"] == 2
    assert meta["DROP"] == pytest.approx(0.15)
    assert meta["RADIUS_UM"] == pytest.approx(500.0) and meta["M_MAX"] == 8
    # Context dim = cell PCs + gene PCs.
    assert meta["F_CTX"] == 100


def test_build_unit_meta_maps_release_config():
    meta = repackage.build_unit_meta(UNIT_CONFIG, "2026_W26", "ibl_neuropixel_brainwide_01")
    assert meta["MODEL_CLASS"] == "MultimodalAutoencoder"
    assert model_registry.MODEL_CLASS_TASKS[meta["MODEL_CLASS"]] == model_registry.TASK_UNIT_ENCODING
    assert meta["VINTAGE"] == "2026_W26"
    assert meta["LATENT_DIM"] == 32
    assert meta["GMM_COMPONENTS"] == 16
    assert meta["PROJECT"] == "ibl_neuropixel_brainwide_01"


def test_channel_meta_defaults_are_sane_when_config_sparse():
    """A near-empty config must still yield a loadable meta (loader re-derives dims from buffers)."""
    meta = repackage.build_channel_meta({}, FEATURES, "vX")
    assert meta["MODEL_CLASS"] == "NeighborInpaintingModel"
    assert meta["FEATURES"] == FEATURES
    # Defaults, not crashes.
    assert meta["D_MODEL"] > 0 and meta["F_CTX"] > 0


# ---- synthesis drives the real manifest builder ---------------------------------------------
def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_channel_meta_drives_build_model_index(tmp_path):
    import yaml

    # The builder scans for artifacts by presence, so dummy files stand in for real weights.
    for name in (
        model_registry.ENCODER_WEIGHTS_FILE,
        model_registry.ENCODER_CONFIDENCE_FILE,
        model_registry.MODEL_SPLIT_FILE,
        *model_registry.ENCODER_CONTEXT_FILES,
    ):
        _touch(tmp_path.joinpath(name))
    meta = repackage.build_channel_meta(CHANNEL_CONFIG, FEATURES, "2026_W26")
    tmp_path.joinpath("meta.yaml").write_text(yaml.safe_dump(meta))

    index = model_registry.build_model_index(tmp_path, method="transformer")

    assert index["task"] == model_registry.TASK_SPATIAL_ENCODING
    assert index["model_class"] == "NeighborInpaintingModel"
    assert index["vintage"] == "2026_W26"
    # This family's ordered feature list is the OUTPUT, not the input.
    assert index["outputs"]["columns"] == FEATURES
    assert index["inputs"]["columns"] == ["x", "y", "z"]
    assert index["artifacts"]["weights"] == model_registry.ENCODER_WEIGHTS_FILE
    assert index["artifacts"]["confidence"] == model_registry.ENCODER_CONFIDENCE_FILE
    assert set(index["artifacts"]["context"]) == set(model_registry.ENCODER_CONTEXT_FILES)
    # No neighbour bank in the release -> the manifest must not claim one.
    assert "neighbor_bank" not in index["artifacts"]
    # The written manifest is what load_pretrained will read.
    assert tmp_path.joinpath(model_registry.MODEL_MANIFEST_FILE).exists()


def test_unit_meta_drives_build_model_index(tmp_path):
    import yaml

    for name in (
        model_registry.UNIT_AE_FILE,
        model_registry.UNIT_GMM_FILE,
        model_registry.UNIT_SCALER_FILE,
        model_registry.UNIT_UNCOND_GMM_FILE,
        model_registry.MODEL_SPLIT_FILE,
    ):
        _touch(tmp_path.joinpath(name))
    meta = repackage.build_unit_meta(UNIT_CONFIG, "2026_W26", "ibl_neuropixel_brainwide_01")
    tmp_path.joinpath("meta.yaml").write_text(yaml.safe_dump(meta))

    index = model_registry.build_model_index(tmp_path, method="gmm")

    assert index["task"] == model_registry.TASK_UNIT_ENCODING
    assert index["model_class"] == "MultimodalAutoencoder"
    assert index["outputs"]["kind"] == "latent"
    assert index["outputs"]["latent_dim"] == 32
    assert index["artifacts"]["autoencoder"] == model_registry.UNIT_AE_FILE
    assert index["artifacts"]["pt_gmm"] == model_registry.UNIT_GMM_FILE
    assert index["data_source"]["requires_one"] is True


# ---- context_dir: the wrapper property that removes the two-locations hack -------------------
# SpatialEncoder.context_dir derives the context-volume directory from the manifest, so context
# lives in exactly one place and both the wrapper's predict() and the figure bodies find it.
# (Importing SpatialEncoder pulls no torch -- torch is imported inside its methods -- so this is
#  safe to collect alongside the xgboost-importing tests.)
def test_context_dir_defaults_to_model_root_for_root_layout(tmp_path):
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    index = {"artifacts": {"context": ["agea_vol_pca.npy", "merfish_vol_pca.npy"]}}
    enc = SpatialEncoder(tmp_path, index=index)
    assert enc.context_dir == tmp_path


def test_context_dir_follows_subdir_layout(tmp_path):
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    index = {"artifacts": {"context": ["context/agea_vol_pca.npy", "context/merfish_vol_pca.npy"]}}
    enc = SpatialEncoder(tmp_path, index=index)
    assert enc.context_dir == tmp_path.joinpath("context")


def test_context_dir_falls_back_to_root_when_unrecorded(tmp_path):
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    enc = SpatialEncoder(tmp_path, index={"artifacts": {}})
    assert enc.context_dir == tmp_path
