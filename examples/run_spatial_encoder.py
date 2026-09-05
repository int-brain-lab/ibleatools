from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from one.api import ONE

import ephysatlas.fixtures
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    ProbeSequenceConfidenceTransformer,
    build_probe_confidence_datasets,
    evaluate_probe_confidence_model,
    evaluate_r2_per_feature,
    train_hybrid,
    train_probe_confidence_model,
)
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
    RegistryError,
    load_json,
    save_json,
    split_manifest_to_builder_format,
)
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    FEATURE_LIST,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    get_device,
)


def build_neighbor_handles(train_loader) -> dict:
    """Extract the TRAIN-only neighbor bank and its inference settings."""
    collate = train_loader.collate_fn
    return {
        "bank_xyz": collate.bank_xyz,
        "bank_feat": collate.bank_feat,
        "bank_pid": collate.bank_pid,
        "nn_bank": collate.nn,
        "radius_um": int(getattr(collate, "radius_um", round(collate.r_m * 1e6))),
        "m_max": int(getattr(collate, "m_max", collate.M)),
    }


@dataclass
class RunConfig:
    # Raw/table data cache. This remains independent from the model registry.
    data_dir: Path = Path(".")

    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W26"

    # Modes:
    #   train_models=True  -> train/re-train this vintage.
    #   train_models=False -> load released weights and evaluate this vintage.
    train_models: bool = False

    # Local HF staging/registry directory. By default this is OUTSIDE the git repo:
    # ~/.ephysatlas/model_registry/ephys-atlas-models/releases/<vintage>/
    registry_root: Path = DEFAULT_REGISTRY_ROOT

    # Optional Hub repository, e.g. "internationalbrainlab/ephys-atlas-models".
    # If a requested vintage is missing locally, it is downloaded using
    # revision=<vintage>.
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None

    # If True after training, upload the complete release directory to HF and
    # create a tag whose name is exactly `vintage`.
    upload_after_training: bool = False
    hf_private_repo: bool = False
    publish_existing_release: bool = False
    replace_existing_hf_tag: bool = False

    # Evaluation controls.
    evaluate_channel_model: bool = True
    evaluate_confidence_model: bool = True

    # Context / channel model.
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

    # Confidence model.
    conf_epochs: int = 50
    conf_batch_size: int = 16
    conf_samples_per_probe: int = 8

    device: torch.device = get_device()
    seed: int = 0

    # Path used only to record the source-code git commit in metadata.
    code_repo_dir: Path = Path(".")


def _channel_release_config(cfg: RunConfig, conf_cfg: ProbeConfidenceTrainConfig) -> dict:
    return {
        "format_version": 1,
        "data": {
            "project": cfg.project,
            "agg": cfg.agg,
            "vintage": cfg.vintage,
        },
        "coordinate_convention": {
            "units": "meters",
            "hemisphere_policy": "mirror_x_to_left",
            "mirror_expression": "x = -abs(x)",
        },
        "context": {
            "n_cell_pcs": cfg.n_cell_pcs,
            "n_gene_pcs": cfg.n_gene_pcs,
            "gene_volume": "context/agea_vol_pca.npy",
            "cell_type_volume": "context/merfish_vol_pca.npy",
        },
        "channel_level": {
            "neighbors": {
                "radius_um": cfg.radius_um,
                "m_max": cfg.m_max,
            },
            "architecture": {
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "depth": cfg.depth,
                "drop": cfg.drop,
            },
            "training": {
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "lambda_ctr": cfg.lambda_ctr,
                "pos_radius_um": cfg.pos_radius_um,
                "patience": cfg.patience,
                "seed": cfg.seed,
            },
            "confidence_model": asdict(conf_cfg),
            "preprocessing": {
                "clip_percentiles": [0.5, 99.5],
                "stats_file": "preprocessing/channel_stats.npz",
                "context_stats_source": "saved release statistics",
                "ephys_stats_source": "clipped training channels",
            },
        },
        "unit_level": {
            "directory": "models/unit",
            "config": "models/unit/config.json",
            "note": "Unit-level availability is recorded in metadata.json and managed separately.",
        },
    }


def _validate_release_against_requested_data(cfg: RunConfig, release_config: dict) -> None:
    data_cfg = release_config.get("data", {})
    for key, requested in [
        ("project", cfg.project),
        ("agg", cfg.agg),
        ("vintage", cfg.vintage),
    ]:
        saved = data_cfg.get(key)
        if saved is not None and str(saved) != str(requested):
            raise RegistryError(
                f"Release mismatch for {key}: release={saved!r}, requested={requested!r}."
            )


def _apply_saved_architecture(cfg: RunConfig, release_config: dict) -> None:
    """Use release-defining architecture/preprocessing settings when loading/retraining."""
    context = release_config.get("context", {})
    channel = release_config.get("channel_level", {})
    architecture = channel.get("architecture", {})
    neighbors = channel.get("neighbors", {})

    cfg.n_cell_pcs = int(context.get("n_cell_pcs", cfg.n_cell_pcs))
    cfg.n_gene_pcs = int(context.get("n_gene_pcs", cfg.n_gene_pcs))
    cfg.radius_um = int(neighbors.get("radius_um", cfg.radius_um))
    cfg.m_max = int(neighbors.get("m_max", cfg.m_max))
    cfg.d_model = int(architecture.get("d_model", cfg.d_model))
    cfg.nhead = int(architecture.get("nhead", cfg.nhead))
    cfg.depth = int(architecture.get("depth", cfg.depth))
    cfg.drop = float(architecture.get("drop", cfg.drop))


def _make_conf_cfg(cfg: RunConfig, release_config: Optional[dict] = None) -> ProbeConfidenceTrainConfig:
    conf_cfg = ProbeConfidenceTrainConfig(
        epochs=cfg.conf_epochs,
        batch_size=cfg.conf_batch_size,
        samples_per_probe=cfg.conf_samples_per_probe,
        seed=cfg.seed,
    )
    if release_config:
        saved = release_config.get("channel_level", {}).get("confidence_model", {})
        valid_fields = set(conf_cfg.__dataclass_fields__)
        for key, value in saved.items():
            if key in valid_fields:
                setattr(conf_cfg, key, value)
    return conf_cfg


def _save_channel_checkpoints(
    release_dir: Path,
    *,
    base_model,
    base_meters,
    best_epoch,
    best_value,
    conf_model,
    conf_info,
    conf_meters,
    f_ctx: int,
    f_e: int,
    cfg: RunConfig,
    conf_cfg: ProbeConfidenceTrainConfig,
) -> None:
    channel_dir = release_dir / "models" / "channel"
    channel_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format_version": 1,
            "model_state": base_model.state_dict(),
            "architecture": {
                "f_ctx": f_ctx,
                "f_ephys": f_e,
                "f_out": f_e,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "depth": cfg.depth,
                "drop": cfg.drop,
            },
            "training": {
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "lambda_ctr": cfg.lambda_ctr,
                "pos_radius_um": cfg.pos_radius_um,
                "patience": cfg.patience,
                "seed": cfg.seed,
            },
            "meters": base_meters,
            "best_epoch": best_epoch,
            "best_value": best_value,
        },
        channel_dir / "spatial_encoder.pt",
    )

    torch.save(
        {
            "format_version": 1,
            "model_state": conf_model.state_dict(),
            "architecture": {
                "f_ctx": f_ctx,
                "f_e": f_e,
                "d_model": conf_cfg.d_model,
                "nhead": conf_cfg.nhead,
                "depth": conf_cfg.depth,
                "mlp_ratio": conf_cfg.mlp_ratio,
                "drop": conf_cfg.drop,
            },
            "training": asdict(conf_cfg),
            "info": conf_info,
            "meters": conf_meters,
        },
        channel_dir / "confidence_model.pt",
    )


def _load_channel_models(
    release_dir: Path,
    *,
    base_model,
    conf_model,
    device: torch.device,
):
    base_ckpt_path = release_dir / "models" / "channel" / "spatial_encoder.pt"
    conf_ckpt_path = release_dir / "models" / "channel" / "confidence_model.pt"

    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    conf_ckpt = torch.load(conf_ckpt_path, map_location=device)

    base_model.load_state_dict(base_ckpt["model_state"], strict=True)
    conf_model.load_state_dict(conf_ckpt["model_state"], strict=True)
    base_model.to(device).eval()
    conf_model.to(device).eval()
    return base_ckpt, conf_ckpt


def _save_channel_r2_results(release_dir: Path, r2: torch.Tensor) -> None:
    r2_np = r2.detach().cpu().numpy()
    payload = {
        "split": "test",
        "mean_r2": float(np.nanmean(r2_np)),
        "per_feature": {
            feature: (None if not np.isfinite(value) else float(value))
            for feature, value in zip(FEATURE_LIST, r2_np)
        },
    }
    save_json(release_dir / "results" / "channel" / "test_r2.json", payload)


def _save_confidence_results(release_dir: Path, conf_eval: dict) -> None:
    payload = {
        "split": "validation_synthetic",
        "accuracy": float(conf_eval["acc"]),
        "confusion_matrix": np.asarray(conf_eval["cm"]).tolist(),
        "n_valid_channels": int(len(conf_eval["labels"])),
    }
    save_json(
        release_dir / "results" / "channel" / "confidence_validation.json",
        payload,
    )


def main(cfg: Optional[RunConfig] = None):
    cfg = cfg or RunConfig(
        vintage="2026_W26",
        train_models=False,
        publish_existing_release=False,
        hf_repo_id="AlonSaguy/ephys-atlas-models",
        replace_existing_hf_tag=False,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    registry = EphysAtlasReleaseRegistry(cfg.registry_root)
    device = cfg.device
    print(f"Using device: {device}")
    print(f"Model registry: {registry.project_root}")

    # ============================================================
    # PUBLISH-ONLY MODE
    # ============================================================
    if cfg.publish_existing_release:

        if cfg.train_models:
            raise ValueError(
                "publish_existing_release=True and train_models=True "
                "are mutually exclusive."
            )

        if not cfg.hf_repo_id:
            raise ValueError(
                "publish_existing_release=True requires hf_repo_id."
            )

        if not registry.has_release(
                cfg.vintage,
                require_weights=True,
        ):
            raise RegistryError(
                f"Cannot publish {cfg.vintage}: "
                "no complete local release exists."
            )

        print(
            f"[registry] publishing existing release "
            f"{cfg.vintage}"
        )

        # Repair/finalize manifest.
        registry.write_checksums(
            cfg.vintage
        )

        registry.verify_checksums(
            cfg.vintage
        )

        registry.upload_release_to_hf(
            cfg.vintage,
            repo_id=cfg.hf_repo_id,
            private=cfg.hf_private_repo,
            token=cfg.hf_token,
            replace_existing_tag=cfg.replace_existing_hf_tag,
        )

        print(
            f"[huggingface] published {cfg.vintage} "
            f"to {cfg.hf_repo_id}"
        )

        return

    # ------------------------------------------------------------
    # Resolve release state.
    # ------------------------------------------------------------
    release_exists = registry.has_release(cfg.vintage, require_weights=False)

    if not release_exists and cfg.hf_repo_id:
        try:
            registry.download_release_from_hf(
                cfg.vintage,
                repo_id=cfg.hf_repo_id,
                token=cfg.hf_token,
            )
            release_exists = True
        except Exception:
            if not cfg.train_models:
                raise
            print(
                f"[registry] No existing HF release found for {cfg.vintage}; "
                "creating a new local release."
            )

    if not cfg.train_models and not registry.has_release(
        cfg.vintage, require_weights=True
    ):
        raise RegistryError(
            f"Evaluation requires a complete release for {cfg.vintage}. "
            "Set hf_repo_id to download it, or train it first."
        )

    release_dir = registry.ensure_release_layout(cfg.vintage)

    # Existing releases are authoritative for architecture, features, split,
    # PCA volumes, and preprocessing statistics.
    split_manifest = None
    preprocessing_stats = None
    release_config = None

    if release_exists:
        release_config = registry.load_config(cfg.vintage)
        _validate_release_against_requested_data(cfg, release_config)
        _apply_saved_architecture(cfg, release_config)
        registry.validate_feature_order(cfg.vintage, FEATURE_LIST)
        split_manifest = split_manifest_to_builder_format(
            registry.load_split(cfg.vintage)
        )
        preprocessing_stats = registry.load_channel_preprocessing_stats(cfg.vintage)

    conf_cfg = _make_conf_cfg(cfg, release_config)

    # ------------------------------------------------------------
    # Context. A first training creates and freezes the PCA volumes.
    # Re-training/evaluation loads the release volumes.
    # ------------------------------------------------------------
    ctx_cfg = AtlasPCAConfig(
        n_cell_pcs=cfg.n_cell_pcs,
        n_gene_pcs=cfg.n_gene_pcs,
    )
    ctx_manager = ContextAtlasManager(
        ctx_cfg,
        regenerate_context=not release_exists,
        output_dir=release_dir / "context",
    )

    # ------------------------------------------------------------
    # Data. For an existing release, split.json is authoritative.
    # ------------------------------------------------------------
    pid_names, ephys, probe_positions, probe_planned_positions = LoadInsertionData(
        project=cfg.project,
        agg=cfg.agg,
        VINTAGE=cfg.vintage,
        path_data=cfg.data_dir,
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
        preprocessing_stats_out,
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
        split_manifest=split_manifest,
        preprocessing_stats=preprocessing_stats,
        return_preprocessing_stats=True,
    )

    f_ctx = int(ctx_mean.numel())
    f_e = int(e_mean.numel())
    if f_e != len(FEATURE_LIST):
        raise ValueError(
            f"Feature dimension mismatch: data has {f_e}, FEATURE_LIST has {len(FEATURE_LIST)}."
        )
    print(f"f_ctx={f_ctx}, f_e={f_e}, n_features={len(FEATURE_LIST)}")
    print(
        f"split source={split_info['source']} "
        f"train={len(split_info['p_tr_names'])} "
        f"val={len(split_info['p_va_names'])} "
        f"test={len(split_info['p_te_names'])}"
    )

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

    conf_model = ProbeSequenceConfidenceTransformer(
        f_ctx=f_ctx,
        f_e=f_e,
        d_model=conf_cfg.d_model,
        nhead=conf_cfg.nhead,
        depth=conf_cfg.depth,
        mlp_ratio=conf_cfg.mlp_ratio,
        drop=conf_cfg.drop,
    ).to(device)

    one = ONE()

    # ------------------------------------------------------------
    # TRAIN / RETRAIN
    # ------------------------------------------------------------
    val_conf_ds = None
    if cfg.train_models:
        # First training of a vintage freezes all data-defining artifacts.
        # Re-training uses the existing split/context/stats instead.
        if not release_exists:
            registry.write_features(cfg.vintage, FEATURE_LIST)
            registry.write_split_manifest(
                cfg.vintage,
                split_info,
                seed=cfg.seed,
                excluded_pids=ephysatlas.fixtures.misaligned_pids,
            )
            registry.write_channel_preprocessing_stats(
                cfg.vintage,
                preprocessing_stats_out,
            )

        registry.write_config(
            cfg.vintage,
            _channel_release_config(cfg, conf_cfg),
        )
        registry.write_metadata(
            cfg.vintage,
            code_repo_dir=cfg.code_repo_dir,
            extra={
                "dataset_summary": {
                    "n_loaded_pids_after_filtering": len(pid_names),
                    "n_channel_features": len(FEATURE_LIST),
                }
            },
        )
        registry.write_readme(cfg.vintage)

        opt = torch.optim.AdamW(
            base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # Optimization/resume checkpoints are intentionally kept OUTSIDE the
        # release directory so they are not uploaded to Hugging Face.
        training_dir = (
            registry.project_root / "training_checkpoints" / cfg.vintage / "channel"
        )
        training_dir.mkdir(parents=True, exist_ok=True)

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
            checkpoint_path=str(training_dir / "base_model_best.pt"),
        )
        print(f"[base] best_epoch={best_epoch}, best_value={best_value}")

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
        print("[confidence dataset]", conf_ds_info)

        conf_model, conf_info, conf_meters = train_probe_confidence_model(
            conf_model,
            train_ds=train_conf_ds,
            val_ds=val_conf_ds,
            device=device,
            f_ctx=f_ctx,
            f_e=f_e,
            cfg=conf_cfg,
            checkpoint_path=str(training_dir / "probe_confidence_best.pt"),
        )

        _save_channel_checkpoints(
            release_dir,
            base_model=base_model,
            base_meters=base_meters,
            best_epoch=best_epoch,
            best_value=best_value,
            conf_model=conf_model,
            conf_info=conf_info,
            conf_meters=conf_meters,
            f_ctx=f_ctx,
            f_e=f_e,
            cfg=cfg,
            conf_cfg=conf_cfg,
        )

    # ------------------------------------------------------------
    # LOAD RELEASED MODELS
    # ------------------------------------------------------------
    else:
        _load_channel_models(
            release_dir,
            base_model=base_model,
            conf_model=conf_model,
            device=device,
        )

    # ------------------------------------------------------------
    # EVALUATION. test_loader is guaranteed to use split.json for
    # existing releases/retraining.
    # ------------------------------------------------------------
    if cfg.evaluate_channel_model:
        r2 = evaluate_r2_per_feature(
            base_model,
            test_loader,
            e_mean,
            e_std,
            device=device,
        )
        print("[base] mean test R2:", float(torch.nanmean(r2)))
        print("[base] test R2 per feature:", r2)
        _save_channel_r2_results(release_dir, r2)

    if cfg.evaluate_confidence_model:
        if val_conf_ds is None:
            handles = build_neighbor_handles(train_loader)
            _, val_conf_ds, conf_ds_info = build_probe_confidence_datasets(
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
            print("[confidence dataset]", conf_ds_info)

        conf_eval = evaluate_probe_confidence_model(
            conf_model=conf_model,
            dataset=val_conf_ds,
            device=device,
            batch_size=conf_cfg.batch_size,
            title="Validation synthetic probe confidence",
        )
        _save_confidence_results(release_dir, conf_eval)

    registry.write_checksums(cfg.vintage)
    print(f"[registry] release ready: {release_dir}")

    if cfg.train_models and cfg.upload_after_training:
        if not cfg.hf_repo_id:
            raise ValueError("upload_after_training=True requires hf_repo_id.")
        registry.upload_release_to_hf(
            cfg.vintage,
            repo_id=cfg.hf_repo_id,
            private=cfg.hf_private_repo,
            token=cfg.hf_token,
        )
        print(
            f"[huggingface] uploaded {cfg.vintage} to {cfg.hf_repo_id} "
            f"and created tag {cfg.vintage}"
        )


if __name__ == "__main__":
    main()
