"""Inference example for the published spatial (neighbour-inpainting) encoder.

Downloads a published spatial encoder and evaluates its per-feature R2 on the held-out test set.
Training from scratch lives in ``training/train_spatial_encoder.py``, which writes the canonical
publish-ready layout directly -- this file only demonstrates loading and evaluating.
"""

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
    evaluate_r2_per_feature,
    predict_probe_confidence_classes,
    ProbeSequenceConfidenceTransformer,
)

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
    cfg = RunConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.model_base_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_base_dir / f"encoding_models/{cfg.vintage}").mkdir(parents=True, exist_ok=True)

    device = cfg.device
    print(f"Using device: {device}")

    one = ONE()

    from ephysatlas.regionclassifier import download_model
    model_path = download_model(cfg.model_base_dir, f"encoding_models/{cfg.vintage}", one=one)

    # ------------------------- data/context -------------------------
    ctx_cfg = AtlasPCAConfig(n_cell_pcs=cfg.n_cell_pcs, n_gene_pcs=cfg.n_gene_pcs)
    ctx_manager = ContextAtlasManager(
        ctx_cfg,
        regenerate_context=False,
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

    # Load the published checkpoints. Training now lives in
    # training/train_spatial_encoder.py, which writes the canonical publish-ready layout;
    # this example only demonstrates inference.
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
