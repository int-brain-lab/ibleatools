from pathlib import Path

import numpy as np
import torch

from one.api import ONE
from dataclasses import dataclass

from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    FEATURE_LIST,
    get_device,
)
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    build_probe_confidence_datasets,
    train_hybrid,
    evaluate_r2_per_feature,
    train_probe_confidence_model,
    evaluate_probe_confidence_model,
    predict_probe_confidence_classes,
    ProbeSequenceConfidenceTransformer,
)

def build_neighbor_handles(train_loader) -> dict:
    """Extract the train-neighbor bank from the DataLoader collate function."""
    collate = train_loader.collate_fn
    return {
        "bank_xyz": collate.bank_xyz,
        "bank_feat": collate.bank_feat,
        "bank_pid": collate.bank_pid,
        "nn_bank": collate.nn,
    }

@torch.no_grad()
def run_base_inference(model, data_loader, device: torch.device, output_dir: Path):
    """Run one full pass of base-model inference on a loader and save standardized predictions/targets."""
    model.eval().to(device)
    preds, targets, xyzs = [], [], []
    device_type = device.type
    use_autocast = device_type == "cuda"

    for batch in data_loader:
        ctx_q, p_q, e_n, p_n, mask, has_ephys, y_e, *_ = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu = model(ctx_q, p_q, e_n, p_n, mask)
        if has_ephys.any():
            preds.append(mu[has_ephys].float().cpu())
            targets.append(y_e[has_ephys].float().cpu())
            xyzs.append(p_q[has_ephys].float().cpu())

    if len(preds) == 0:
        print("[base inference] No recorded samples found in this loader.")
        return None

    out = {
        "pred_std": torch.cat(preds, dim=0),
        "target_std": torch.cat(targets, dim=0),
        "xyz_m": torch.cat(xyzs, dim=0),
    }
    save_path = output_dir / "base_inference_test.pt"
    torch.save(out, save_path)
    print(f"[base inference] saved {save_path}")
    return out

@dataclass
class RunConfig:
    data_dir: Path = Path(".")
    model_base_dir: Path = Path(".")

    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W12"

    train_models: bool = False
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

    device: torch.device = get_device()
    seed: int = 0


def main():
    cfg = RunConfig(train_models=False)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.model_base_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_base_dir / f"encoding_models/{cfg.vintage}").mkdir(parents=True, exist_ok=True)

    device = cfg.device
    print(f"Using device: {device}")

    one = ONE()

    if not cfg.train_models:
        from ephysatlas.regionclassifier import download_model
        model_path = download_model(cfg.model_base_dir, f"encoding_models/{cfg.vintage}", one=one)

    # ------------------------- data/context -------------------------
    ctx_cfg = AtlasPCAConfig(n_cell_pcs=cfg.n_cell_pcs, n_gene_pcs=cfg.n_gene_pcs)
    ctx_manager = ContextAtlasManager(
        ctx_cfg,
        regenerate_context=cfg.train_models,
        output_dir=cfg.model_base_dir / f"encoding_models/{cfg.vintage}",
    )

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
    print(f"f_ctx={f_ctx}, f_e={f_e}, n_features={len(FEATURE_LIST)}")

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

    if cfg.train_models:
        opt = torch.optim.AdamW(base_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        base_ckpt = cfg.model_base_dir / "base_model_best.pt"
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
        torch.save({"model_state": base_model.state_dict(), "meters": base_meters, "split_info": split_info}, cfg.model_base_dir / f"encoding_models/SE_model_{cfg.vintage}.pt")
        print(f"[base] best_epoch={best_epoch}, best_value={best_value}")

        # ------------------------- confidence model -------------------------
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

        conf_ckpt = cfg.model_base_dir / "probe_confidence_best.pt"
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
        torch.save({"model_state": conf_model.state_dict(), "info": conf_info, "meters": conf_meters}, cfg.model_base_dir / f"encoding_models/Confidence_model_{cfg.vintage}.pt")

        conf_eval = evaluate_probe_confidence_model(
            conf_model=conf_model,
            dataset=val_conf_ds,
            device=device,
            batch_size=cfg.conf_batch_size,
            title="Validation synthetic probe confidence",
        )
        torch.save(conf_eval, cfg.model_base_dir / "probe_confidence_val_eval.pt")
    else:
        base_ckpt = torch.load(
            model_path / f"SE_model_{cfg.vintage}.pt",
            map_location=device,
        )
        conf_ckpt = torch.load(
            model_path / f"Confidence_model_{cfg.vintage}.pt",
            map_location=device,
        )

        base_model.load_state_dict(base_ckpt["model_state"])
        conf_model.load_state_dict(conf_ckpt["model_state"])

    # Evaluation
    r2 = evaluate_r2_per_feature(base_model, test_loader, e_mean, e_std, device=device)
    print("[base] mean test R2:", float(torch.nanmean(r2)))
    print("[base] test R2 per feature:", r2)

    # run_base_inference(base_model, test_loader, device, cfg.model_base_dir)
    #
    # # One-sample confidence inference example.
    # sample = val_conf_ds[0]
    # logits, probs, conf_scalar = predict_probe_confidence_classes(
    #     conf_model=conf_model,
    #     rec_std=sample["rec"],
    #     pred_std=sample["pred"],
    #     ctx_std=sample["ctx"],
    #     valid_mask=sample["valid"],
    #     device=device,
    # )


if __name__ == "__main__":
    main()
