"""Write a trained model out as the publish-ready release layout.

These writers serialise an already-trained model into the on-disk layout that
:mod:`ephysatlas.model_registry` (and :func:`ephysatlas.load_pretrained`) reads back. They live
next to the loader so the load->predict round trip can be tested without the training pipeline;
the training scripts in ``paper-ephys-atlas`` import them to finalise a run. ``torch`` is imported
lazily inside ``write_spatial_release`` so importing this module never pulls torch in.
"""

import json
import logging
from pathlib import Path

import numpy as np

from ephysatlas import model_registry

logger = logging.getLogger(__name__)


def write_spatial_release(
    path_model,
    base_model,
    meta: dict,
    bank_xyz,
    bank_feat,
    bank_pid,
    *,
    conf_model=None,
    split_info: dict = None,
    method: str = "transformer",
) -> dict:
    """Write the publish-ready spatial-encoder layout into ``path_model``.

    Writes canonical filenames at the model-dir root: the weights, the neighbour bank the model
    needs at inference, and the manifest (the single source of truth). Context volumes
    (``agea_vol_pca.npy`` / ``merfish_vol_pca.npy``) are assumed already at the root, where the
    trainer's ``ContextAtlasManager`` writes them.

    The bank is the exact one the model trained against -- features clipped and standardised by
    ``build_channels_plus_emptyvoxels_with_neighbors`` -- written straight from the in-memory arrays
    so the shipped bank matches the weights. It is written before ``write_manifest`` so one manifest
    pass records it.

    Args:
        path_model (Path): Model-dir root to write into.
        base_model: The trained ``NeighborInpaintingModel``.
        meta (dict): UPPER_CASE training metadata the manifest is built from (``FEATURES`` plus the
            architecture / neighbourhood / context keys ``_blocks_spatial_encoding`` reads).
        bank_xyz (np.ndarray): ``[n_bank, 3]`` positions (metres) of the training bank channels.
        bank_feat (np.ndarray): ``[n_bank, n_features]`` **already-standardised** features of those
            channels (the collate's ``bank_feat``, standardised with the model's ``e_mean``/``e_std``).
        bank_pid (array-like): ``[n_bank]`` insertion id (pid string) per bank channel.
        conf_model (optional): The trained probe-confidence model, saved when given.
        split_info (dict, optional): Train/val/test split, written as ``split.json`` when given.
        method (str, optional): Semantic label recorded in the manifest. Defaults to
            ``"transformer"``, matching the published ``ea-encoder-channel``.

    Returns:
        dict: The manifest that was written.
    """
    # Lazy import: keeps the public package torch-free at import time (the segregation tripwire).
    import torch

    path_model = Path(path_model)
    path_model.mkdir(parents=True, exist_ok=True)
    features = list(meta["FEATURES"])

    # 1. Weights at the root: the wrapped form carries the constructor args so the loader can
    #    rebuild the module; the e_mean/e_std/ctx_mean/ctx_std buffers ride in the state dict.
    architecture = dict(
        f_ctx=int(meta["F_CTX"]),
        f_ephys=len(features),
        f_out=len(features),
        d_model=int(meta["D_MODEL"]),
        nhead=int(meta["NHEAD"]),
        depth=int(meta["DEPTH"]),
        drop=float(meta["DROP"]),
    )
    torch.save(
        {"model_state": base_model.state_dict(), "architecture": architecture},
        path_model.joinpath(model_registry.ENCODER_WEIGHTS_FILE),
    )
    if conf_model is not None:
        torch.save(
            {"model_state": conf_model.state_dict()},
            path_model.joinpath(model_registry.ENCODER_CONFIDENCE_FILE),
        )

    # 2. The split (which insertions were held out) for provenance -- plain ints/pid strings.
    if split_info is not None:
        path_model.joinpath(model_registry.MODEL_SPLIT_FILE).write_text(
            json.dumps(split_info, indent=2)
        )

    # 3. The neighbour bank the model predicts from, written verbatim from the already-standardised
    #    in-memory arrays -- exactly what the model drew neighbours from -- before the manifest.
    if len(bank_xyz) != len(bank_feat) or len(bank_xyz) != len(bank_pid):
        raise ValueError(
            f"bank arrays misaligned: xyz={len(bank_xyz)}, feat={len(bank_feat)}, "
            f"pid={len(bank_pid)} -- a bank recorded from mismatched inputs cannot be trusted"
        )
    np.savez_compressed(
        path_model.joinpath(model_registry.ENCODER_BANK_FILE),
        xyz=np.asarray(bank_xyz, dtype=np.float32),
        feat=np.asarray(bank_feat, dtype=np.float32),
        pid=np.asarray(bank_pid).astype(str),
    )

    # 4. The manifest: the single source of truth every loader reads.
    return model_registry.write_manifest(
        path_model, meta, task=model_registry.TASK_SPATIAL_ENCODING, method=method
    )


def write_unit_release(
    path_model: Path,
    meta: dict,
    *,
    split_manifest: dict = None,
    method: str = "gmm",
) -> dict:
    """Write the manifest (and ``split.json``) for a unit-encoder release.

    The two-stage trainers already wrote the four checkpoints at ``path_model`` root
    (``autoencoder.pt``, ``point_transformer_gmm.pt``, ``shared_latent_scaler.joblib``,
    ``unconditional_gmm_train_only.joblib``); this finalize step adds the manifest (the single
    source of truth) and the held-out split, from the in-memory ``meta`` dict.

    Args:
        path_model (Path): Model-dir root the trainers wrote into.
        meta (dict): UPPER_CASE training metadata (see ``_build_meta``) that
            ``_blocks_unit_encoding`` reads.
        split_manifest (dict, optional): ``{"train_pids", "validation_pids", "test_pids"}``,
            written as ``split.json`` when given so held-out status is checkable against the
            channel model (``_release.check_split_agreement``).
        method (str, optional): Semantic label recorded in the manifest. Defaults to ``"gmm"``,
            matching the published ``ea-encoder-unit``.

    Returns:
        dict: The manifest that was written.
    """
    path_model = Path(path_model)
    path_model.mkdir(parents=True, exist_ok=True)

    # split.json records the held-out insertions themselves, so check_split_agreement can compare
    # them against the channel model.
    if split_manifest is not None:
        path_model.joinpath(model_registry.MODEL_SPLIT_FILE).write_text(
            json.dumps(split_manifest, indent=2) + "\n"
        )

    # write_manifest scans the root for the checkpoints actually present, so only the stages that
    # ran are recorded.
    index = model_registry.write_manifest(
        path_model, meta, task=model_registry.TASK_UNIT_ENCODING, method=method
    )
    logger.info(f"packaged unit encoder -> {path_model} (task={index.get('task')})")
    return index
