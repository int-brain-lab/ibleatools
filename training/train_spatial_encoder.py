"""Train the spatial (neighbour-inpainting) encoder from scratch and write a publish-ready model.

The output directory this writes **is** the release layout: the canonical filenames at the model
root, the neighbour bank the model needs to run, and the manifest (``ephysatlas_model.json``) as
the single source of truth. There is no ``meta.yaml`` scaffold, no ``encoding_models/``
subdirectory, and no vintage suffix -- so the directory loads through ``load_pretrained`` unchanged
and is ready for ``scripts/publish_model_to_hf.py`` (which only adds the card, the golden example
and checksums).

Reproducibility target: ``python training/train_spatial_encoder.py --vintage <W>`` downloads that
vintage's ``ea_active`` feature tables from S3 (via ``LoadInsertionData`` -> ``download_tables``),
trains, and drops a publish-ready model directory -- one command, no hand edits.

This is a training script: it pulls torch (and, through ``LoadInsertionData``, ONE/S3), so it lives
under ``training/`` rather than in the installed package.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from one.api import ONE

from ephysatlas import model_registry
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    ProbeSequenceConfidenceTransformer,
    build_probe_confidence_datasets,
    evaluate_r2_per_feature,
    train_hybrid,
    train_probe_confidence_model,
)
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    FEATURE_LIST,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    get_device,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
_logger = logging.getLogger("train_spatial_encoder")


def build_neighbor_handles(train_loader) -> dict:
    """Extract the train-neighbour bank from the DataLoader collate function."""
    collate = train_loader.collate_fn
    return {
        "bank_xyz": collate.bank_xyz,
        "bank_feat": collate.bank_feat,
        "bank_pid": collate.bank_pid,
        "nn_bank": collate.nn,
    }


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
    """Write the full publish-ready spatial-encoder layout into ``path_model``.

    The training output **is** the release layout: canonical filenames at the model-dir root, the
    neighbour bank the model needs at inference, and the manifest as the single source of truth --
    no ``meta.yaml``, no ``encoding_models/`` subdirectory. Context volumes
    (``agea_vol_pca.npy`` / ``merfish_vol_pca.npy``) are assumed to be at the root already: the
    trainer's ``ContextAtlasManager`` writes them there.

    The neighbour bank is the **exact one the model trained against** -- the train-split channels,
    with features clipped to the train percentiles and standardised by the model's ``e_mean``/
    ``e_std`` (all done in ``build_channels_plus_emptyvoxels_with_neighbors``). It is written straight
    from those in-memory arrays, so the shipped bank is consistent with the weights rather than
    re-read raw from disk (which would skip the clipping and the alpha outlier treatment).

    Ordering matters: the bank is written **before** ``write_manifest`` so a single manifest write
    records it -- there is no publish-time two-pass rebuild.

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
    path_model = Path(path_model)
    path_model.mkdir(parents=True, exist_ok=True)
    features = list(meta["FEATURES"])

    # 1. Canonical weights at the root (not encoding_models/, no _{vintage} suffix). The wrapped
    #    form carries the constructor args so the loader can rebuild the module; the normalisation
    #    buffers (e_mean/e_std/ctx_mean/ctx_std) ride inside the state dict.
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

    # 2. Publish the split itself (which insertions were held out), not just its hash, for
    #    provenance. split_info is already plain ints and pid strings, so it serialises directly.
    if split_info is not None:
        path_model.joinpath(model_registry.MODEL_SPLIT_FILE).write_text(
            json.dumps(split_info, indent=2)
        )

    # 3. The neighbour bank the model needs at inference (it predicts from nearby channels' recorded
    #    features): the in-memory training bank, written verbatim. bank_feat is already standardised,
    #    so nothing is recomputed here -- this is exactly what the model drew neighbours from. Written
    #    before the manifest so write_manifest records it in one pass.
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

    # 4. The manifest: the single source of truth every loader reads, written straight from the
    #    values in hand -- no meta.yaml round trip.
    return model_registry.write_manifest(
        path_model, meta, task=model_registry.TASK_SPATIAL_ENCODING, method=method
    )


@dataclass
class RunConfig:
    """Configuration for a from-scratch spatial-encoder training run."""

    vintage: str
    data_dir: Path = Path(".")
    model_base_dir: Path = Path(".")

    project: str = "ea_active"
    agg: str = "agg_full"

    n_cell_pcs: int = 50
    n_gene_pcs: int = 50

    radius_um: int = 500
    m_max: int = 8
    batch_size_train: int = 1024
    batch_size_eval: int = 1024

    d_model: int = 128
    nhead: int = 8
    depth: int = 2
    drop: float = 0.15
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    lambda_ctr: float = 0.1
    pos_radius_um: int = 500
    patience: int = 5

    conf_epochs: int = 50
    conf_batch_size: int = 16
    conf_samples_per_probe: int = 8

    device: torch.device = None
    seed: int = 0


def main(argv=None):
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage", required=True, help="feature vintage to train on, e.g. 2026_W26")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."), help="where download_tables caches the features"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="output directory for the publish-ready model (default: <data-dir>/<vintage>_encoder)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override base-model epochs")
    parser.add_argument("--conf-epochs", type=int, default=None, help="override confidence epochs")
    parser.add_argument(
        "--device", default=None, help="torch device (e.g. cpu, cuda, mps); default: auto-detect"
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else get_device()
    cfg = RunConfig(vintage=args.vintage, data_dir=args.data_dir, device=device)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.conf_epochs is not None:
        cfg.conf_epochs = args.conf_epochs

    path_model = args.model_dir or cfg.data_dir.joinpath(f"{cfg.vintage}_encoder")
    path_model = Path(path_model)
    path_model.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    _logger.info(f"training {cfg.vintage} on {device}; output -> {path_model}")

    one = ONE()

    # ------------------------- data/context -------------------------
    # Context volumes are (re)fit for this vintage and written to the model-dir ROOT, so they are
    # part of the release layout with no staging step.
    ctx_cfg = AtlasPCAConfig(n_cell_pcs=cfg.n_cell_pcs, n_gene_pcs=cfg.n_gene_pcs)
    ctx_manager = ContextAtlasManager(
        ctx_cfg, regenerate_context=True, output_dir=path_model
    )

    pid_names, ephys, probe_positions, probe_planned_positions = LoadInsertionData(
        project=cfg.project, agg=cfg.agg, VINTAGE=cfg.vintage, path_data=cfg.data_dir
    )
    pid_names = [str(x) for x in pid_names]

    (
        train_loader,
        conf_train_loader,
        val_loader,
        test_loader,
        e_mean,
        e_std,
        ctx_mean,
        ctx_std,
        split_info,
    ) = build_channels_plus_emptyvoxels_with_neighbors(
        ctx_manager,
        ephys,
        probe_positions,
        RADIUS_UM=cfg.radius_um,
        M_MAX=cfg.m_max,
        pid_names=pid_names,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        seed=cfg.seed,
    )

    f_ctx = int(ctx_mean.numel())
    f_e = int(e_mean.numel())
    _logger.info(f"f_ctx={f_ctx}, f_e={f_e}, n_features={len(FEATURE_LIST)}")

    # ------------------------- base model -------------------------
    base_model = NeighborInpaintingModel(
        f_ctx=f_ctx,
        f_ephys=f_e,
        f_out=f_e,
        e_mean=e_mean,
        e_std=e_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        depth=cfg.depth,
        drop=cfg.drop,
    ).to(device)

    opt = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # A rolling "best" checkpoint is early-stopping bookkeeping, not a release file -- keep it in a
    # scratch/run dir so it never lands in the model output.
    run_dir = path_model.joinpath("run")
    run_dir.mkdir(exist_ok=True)
    base_ckpt = run_dir.joinpath("base_model_best.pt")
    base_model, base_meters, best_epoch, best_value = train_hybrid(
        base_model,
        train_loader,
        val_loader,
        opt,
        epochs=cfg.epochs,
        device=device,
        lambda_sup=1.0,
        lambda_ctr=cfg.lambda_ctr,
        pos_radius_um=cfg.pos_radius_um,
        patience=cfg.patience,
        checkpoint_path=str(base_ckpt),
    )
    _logger.info(f"[base] best_epoch={best_epoch}, best_value={best_value}")

    # ------------------------- confidence model -------------------------
    conf_cfg = ProbeConfidenceTrainConfig(
        epochs=cfg.conf_epochs,
        batch_size=cfg.conf_batch_size,
        samples_per_probe=cfg.conf_samples_per_probe,
        seed=cfg.seed,
    )
    conf_model = ProbeSequenceConfidenceTransformer(
        f_ctx=f_ctx,
        f_e=f_e,
        d_model=conf_cfg.d_model,
        nhead=conf_cfg.nhead,
        depth=conf_cfg.depth,
        mlp_ratio=conf_cfg.mlp_ratio,
        drop=conf_cfg.drop,
    ).to(device)

    handles = build_neighbor_handles(train_loader)
    train_conf_ds, val_conf_ds, conf_ds_info = build_probe_confidence_datasets(
        one=one,
        pid_names=pid_names,
        ctx_manager=ctx_manager,
        ephys=ephys,
        probe_positions=probe_positions,
        split_info=split_info,
        base_model=base_model.to(device),
        cfg=conf_cfg,
        handles=handles,
    )
    _logger.info(f"[confidence dataset] {conf_ds_info}")

    conf_ckpt = run_dir.joinpath("probe_confidence_best.pt")
    conf_model, conf_info, conf_meters = train_probe_confidence_model(
        conf_model,
        train_ds=train_conf_ds,
        val_ds=val_conf_ds,
        device=device,
        f_ctx=f_ctx,
        f_e=f_e,
        cfg=conf_cfg,
        checkpoint_path=str(conf_ckpt),
    )

    # ------------------------- evaluation -------------------------
    r2 = evaluate_r2_per_feature(base_model, test_loader, e_mean, e_std, device=device)
    _logger.info(f"[base] mean test R2: {float(torch.nanmean(r2))}")

    # ------------------------- write the publish-ready release -------------------------
    meta = dict(
        RANDOM_SEED=cfg.seed,
        VINTAGE=cfg.vintage,
        MODEL_CLASS="NeighborInpaintingModel",
        FEATURES=list(FEATURE_LIST),
        F_CTX=f_ctx,
        D_MODEL=cfg.d_model,
        NHEAD=cfg.nhead,
        DEPTH=cfg.depth,
        DROP=cfg.drop,
        RADIUS_UM=float(cfg.radius_um),
        M_MAX=cfg.m_max,
        ALLOW_SAME_PROBE=False,
        N_CELL_PCS=cfg.n_cell_pcs,
        N_GENE_PCS=cfg.n_gene_pcs,
        TRAINING=dict(
            training_size=len(split_info["p_tr_names"]),
            validation_size=len(split_info["p_va_names"]),
            testing_size=len(split_info["p_te_names"]),
        ),
    )

    # Ship the exact neighbour bank the model trained against: the collate's train-split channels,
    # already clipped + standardised (build_neighbor_handles pulls bank_xyz/bank_feat/bank_pid from
    # the train loader). bank_pid holds integer probe indices; map them back to pid strings so the
    # published bank records real insertion ids (and same-probe exclusion works at inference).
    bank_pid = np.array([str(pid_names[int(i)]) for i in handles["bank_pid"]], dtype=str)

    manifest = write_spatial_release(
        path_model,
        base_model,
        meta,
        handles["bank_xyz"],
        handles["bank_feat"],
        bank_pid,
        conf_model=conf_model,
        split_info=split_info,
    )
    _logger.info(f"wrote publish-ready model at {path_model} (task={manifest['task']})")
    end = time.time()
    _logger.info(f"Training time: {end - start} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
