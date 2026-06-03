import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from typing import Optional

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from ephysatlas.spatial_encoder.utils import (
    _build_shift_based_synthetic_probe_sample,
    _make_histology_probe_bank,
)


# =========================== prediction model ============================
def mlp(d_in, d_hidden, d_out, n_layers=2, drop=0.0):
    layers = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(drop)]
    layers += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*layers)


class PosEnc3D(nn.Module):
    """Encode absolute and relative 3D positions."""

    def __init__(self, d_out):
        super().__init__()
        self.pe = mlp(6, max(64, d_out), d_out)  # [xyz_abs(3), xyz_rel(3)] -> d_out

    def forward(self, p_abs, p_rel):
        # p_abs, p_rel: [B, M, 3] (for neighbors) or [B, 1, 3] (for query)
        x = torch.cat([p_abs, p_rel], dim=-1)
        return self.pe(x)


class NeighborEncoder(nn.Module):
    def __init__(self, f_ephys, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ephys + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, e_n, p_n_abs, p_n_rel, mask):  # e_n: [B,M,Fe]
        pos = self.pos(p_n_abs, p_n_rel)  # [B,M,d_pos]
        x = torch.cat([e_n, pos], dim=-1)
        h = self.embed(x)  # [B,M,d_model]
        h = h * mask[..., None]  # zero out pads
        return h


class QueryEncoder(nn.Module):
    def __init__(self, f_ctx, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ctx + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, ctx_q, p_q_abs):
        # broadcast rel=0 for the query token
        B = ctx_q.size(0)
        p_rel0 = torch.zeros(B, 1, 3, device=ctx_q.device, dtype=ctx_q.dtype)
        p_abs = p_q_abs[:, None, :]
        pos = self.pos(p_abs, p_rel0)  # [B,1,d_pos]
        x = torch.cat([ctx_q[:, None, :], pos], dim=-1)
        h = self.embed(x)  # [B,1,d_model]
        return h


class CrossBlock(nn.Module):
    """Optional neighbor self-attn, then query->neighbor cross-attn."""

    def __init__(self, d_model, nhead=8, drop=0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            d_model, nhead, dropout=drop, batch_first=True
        )
        self.ff = mlp(d_model, 4 * d_model, d_model, n_layers=2, drop=drop)
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)

    def forward(self, h_q, h_n, mask_nei):
        # h_q: [B,1,D], h_n: [B,M,D], mask_nei: [B,M] (True=real, False=pad)
        B, M, D = h_n.shape
        no_nei = ~mask_nei.any(dim=1)  # [B]
        if no_nei.any():
            # append a dummy zero neighbor and mark it valid ONLY for empty rows
            dummy = h_n.new_zeros(B, 1, D)
            h_n = torch.cat([h_n, dummy], dim=1)  # [B, M+1, D]
            pad = mask_nei.new_zeros(B, 1)
            pad[no_nei, 0] = True
            mask_nei = torch.cat([mask_nei, pad], dim=1)  # [B, M+1]

        # query <- neighbors cross-attn
        key_padding_mask = ~mask_nei.bool()
        h_q2, _ = self.cross(h_q, h_n, h_n, key_padding_mask=key_padding_mask)
        h_q = self.norm_q1(h_q + h_q2)
        h_q = self.norm_q2(h_q + self.ff(h_q))
        return h_q


class EphysPredictor(nn.Module):
    def __init__(self, d_model, f_out):
        super().__init__()
        self.mu_head = mlp(d_model, 2 * d_model, f_out, n_layers=2, drop=0.0)

    def forward(self, h_q):  # [B,1,D]
        h = h_q.squeeze(1)  # [B,D]
        mu = self.mu_head(h)
        return mu


class NeighborInpaintingModel(nn.Module):
    """
    Predict ephys for a single query channel using context of that channel
    and a variable-size set of neighbor ephys from *other* probes.
    """

    def __init__(
        self,
        f_ctx,
        f_ephys,
        f_out,
        e_mean=None,
        e_std=None,
        ctx_mean=None,
        ctx_std=None,
        d_model=256,
        nhead=8,
        depth=2,
        drop=0.1,
    ):
        super().__init__()
        self.qenc = QueryEncoder(f_ctx, d_model, drop=drop)
        self.nenc = NeighborEncoder(f_ephys, d_model, drop=drop)
        self.blocks = nn.ModuleList(
            [CrossBlock(d_model, nhead=nhead, drop=drop) for _ in range(depth)]
        )
        self.pred = EphysPredictor(d_model, f_out)

        if (
            e_mean is not None
            and e_std is not None
            and ctx_mean is not None
            and ctx_std is not None
        ):
            self.register_buffer("e_mean", e_mean.clone().detach())
            self.register_buffer("e_std", e_std.clone().detach())
            self.register_buffer("ctx_mean", ctx_mean.clone().detach())
            self.register_buffer("ctx_std", ctx_std.clone().detach())

    def forward(self, ctx_q, p_q, e_n, p_n, mask_nei):
        # relative pos of neighbors to query
        p_q_b = p_q[:, None, :].expand_as(p_n)
        p_rel = p_n - p_q_b

        h_q = self.qenc(ctx_q, p_q)  # [B,1,D]
        h_n = self.nenc(e_n, p_n, p_rel, mask_nei)  # [B,M,D]
        for blk in self.blocks:
            h_q = blk(h_q, h_n, mask_nei)

        mu = self.pred(h_q)  # [B,F_out], [B,F_out]

        return h_q.squeeze(1), mu


@torch.no_grad()
def mean_feature_corr(mu, y, mask=None):
    if mask is not None:
        mu = mu[mask]
        y = y[mask]
    if mu.numel() == 0:
        return mu.new_tensor(float("nan"))
    mu = mu - mu.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    mu = F.normalize(mu, dim=1)
    y = F.normalize(y, dim=1)
    return (mu * y).sum(dim=1).mean()


def info_nce_multi_positive(z, xyz, pos_radius_m: float, tau: float = 0.2):
    """
    Multi-positive InfoNCE on a set of embeddings.
    z:   [N, D] (will be L2-normalized)
    xyz: [N, 3] (meters)
    pos_radius_m: positives are pairs with distance <= radius (excluding self)
    Returns: scalar loss and #anchors used
    """
    N = z.size(0)
    if N < 2:
        return torch.tensor(0.0, device=z.device), 0

    # L2-normalize embeddings
    z = F.normalize(z, dim=1)

    # Pairwise cosine similarities
    sim = z @ z.t()  # [N,N]
    sim = sim / tau

    # Pairwise distances (no sqrt for speed is fine, but we keep meters here)
    # Build distances in meters
    with torch.no_grad():
        diffs = xyz[:, None, :] - xyz[None, :, :]
        dist = torch.linalg.norm(diffs, dim=-1)  # [N,N]
        pos_mask = (dist <= pos_radius_m) & (
            ~torch.eye(N, dtype=torch.bool, device=z.device)
        )  # exclude self
        # We will use ALL other samples (except self) as the denominator
        denom_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

        # anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)

    if not has_pos.any():
        return torch.tensor(0.0, device=z.device), 0

    # For numerical stability, subtract row-wise max over the denominator set
    sim_denom = sim.masked_fill(~denom_mask, float("-inf"))
    row_max, _ = torch.max(sim_denom, dim=1, keepdim=True)
    sim = sim - row_max  # [N,N]

    # Numerator: sum over positives
    num = torch.logsumexp(sim.masked_fill(~pos_mask, float("-inf")), dim=1)  # [N]
    # Denominator: sum over all j != i
    den = torch.logsumexp(sim.masked_fill(~denom_mask, float("-inf")), dim=1)  # [N]

    loss_vec = -(num - den)  # [N]
    loss = loss_vec[has_pos].mean()
    return loss, int(has_pos.sum().item())


def train_hybrid(
    model,
    train_dl,
    val_dl,
    optimizer,
    epochs=10,
    device=torch.device("cuda"),
    lambda_sup=1.0,
    lambda_ctr=0.2,
    tau=0.2,
    pos_radius_um=600.0,
    grad_clip=1.0,
    log_every=50,
    *,
    # Early stopping knobs
    early_stopping=True,
    patience: int = 10,
    min_delta: float = 0.0,
    ephys_drop: float = 0.0,
    monitor: str = "val/sup",  # "val/sup" (min) or "val/corr" (max) etc.
    mode: str = "min",  # "min" or "max"
    checkpoint_path: Optional[
        str
    ] = None,  # if set, saves best state dict here on improvement
):
    """
    Trains with supervised + contrastive losses and early stopping on a validation metric.
    Restores best weights at the end (based on `monitor`/`mode`).

    Returns:
        meters: dict of learning curves
        best_epoch: epoch index (1-based) with best monitored metric
        best_value: best monitored metric value
    """
    device_type = device.type
    model.to(device)
    pos_radius_m = pos_radius_um * 1e-6
    meters = {
        "train/sup": [],
        "train/ctr": [],
        "train/total": [],
        "val/sup": [],
        "val/corr": [],
    }
    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Early stopping state
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    best_val = math.inf if mode == "min" else -math.inf
    best_state = None
    best_epoch = 0
    num_bad_epochs = 0

    def _is_improvement(current, best):
        if current is None or math.isnan(current):
            return False
        return (
            (current < best - min_delta)
            if mode == "min"
            else (current > best + min_delta)
        )

    for ep in range(1, epochs + 1):
        # -------------------- Train --------------------
        model.train()
        r_sup = r_ctr = r_tot = 0.0
        n_steps = 0

        for step, batch in enumerate(train_dl, 1):
            (ctx_q, p_q, e_n, p_n, mask, has_ephys, y_e, *_) = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                mask_dropped = mask.clone()
                n_drop = int(ephys_drop * mask.shape[0])
                if n_drop > 0:
                    drop_idx = torch.randperm(mask.shape[0], device=mask.device)[
                        :n_drop
                    ]
                    mask_dropped[drop_idx] = False
                h_q, mu = model(ctx_q, p_q, e_n, p_n, mask_dropped)

                # supervised
                if has_ephys.any():
                    sup = F.mse_loss(mu[has_ephys], y_e[has_ephys], reduction="mean")
                else:
                    sup = torch.zeros((), device=device)

                # contrastive
                ctr = torch.tensor(0.0, device=device)
                if (~has_ephys).sum() >= 2:
                    z = h_q[~has_ephys]
                    xyz = p_q[~has_ephys]
                    ctr, _ = info_nce_multi_positive(z, xyz, pos_radius_m, tau=tau)

                loss = lambda_sup * sup + lambda_ctr * ctr

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            r_sup += sup.item()
            r_ctr += ctr.item()
            r_tot += loss.item()
            n_steps += 1
            if (step % log_every) == 0:
                print(
                    f"[ep {ep} step {step}] sup={r_sup / n_steps:.4f} ctr={r_ctr / n_steps:.4f} tot={r_tot / n_steps:.4f}"
                )

        meters["train/sup"].append(r_sup / max(1, n_steps))
        meters["train/ctr"].append(r_ctr / max(1, n_steps))
        meters["train/total"].append(r_tot / max(1, n_steps))

        # -------------------- Validation --------------------
        current_val = None  # value of the monitored metric this epoch
        if val_dl is not None:
            model.eval()
            vs = vc = 0.0
            m = 0
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type=device_type, enabled=use_amp),
            ):
                for batch in val_dl:
                    (ctx_q, p_q, e_n, p_n, mask, has_ephys, y_e, *_) = [
                        x.to(device) if torch.is_tensor(x) else x for x in batch
                    ]
                    h_q, mu = model(ctx_q, p_q, e_n, p_n, mask)
                    val_sup = F.mse_loss(
                        mu[has_ephys], y_e[has_ephys], reduction="mean"
                    )

                    vs += val_sup.item()
                    vc += mean_feature_corr(mu, y_e, has_ephys).item()
                    m += 1

            val_sup_mean = vs / max(1, m)
            val_corr_mean = vc / max(1, m)
            meters["val/sup"].append(val_sup_mean)
            meters["val/corr"].append(val_corr_mean)
            print(f"[ep {ep}] VAL sup={val_sup_mean:.4f} corr={val_corr_mean:.4f}")

            # pick metric to monitor
            if monitor == "val/sup":
                current_val = val_sup_mean
            elif monitor == "val/corr":
                current_val = val_corr_mean
            else:
                # allow monitoring of any key in meters if you add more later
                if monitor in meters and len(meters[monitor]) > 0:
                    current_val = meters[monitor][-1]

        # -------------------- Early stopping --------------------
        if early_stopping and (val_dl is not None):
            if _is_improvement(current_val, best_val):
                best_val = current_val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep
                num_bad_epochs = 0
                if checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state": best_state,
                            "optimizer_state": optimizer.state_dict(),
                            "meters": meters,
                            "best_value": best_val,
                            "monitor": monitor,
                            "mode": mode,
                        },
                        checkpoint_path,
                    )
                print(f"✓ Improvement on {monitor}: {best_val:.6f} (epoch {ep})")
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    print(
                        f"⏹ Early stopping at epoch {ep} (no improvement in {patience} epochs)."
                    )
                    break

    # Restore best weights (if any) at the end
    if early_stopping and (val_dl is not None) and (best_state is not None):
        model.load_state_dict(best_state)
        print(
            f"Restored best model from epoch {best_epoch} with {monitor}={best_val:.6f}"
        )

    return model, meters, best_epoch, best_val


# ============================================================
# Config
# ============================================================
@dataclass
class ProbeConfidenceTrainConfig:
    d_model: int = 64
    nhead: int = 4
    depth: int = 2
    mlp_ratio: float = 2.0
    drop: float = 0.1

    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: Optional[float] = 1.0

    batch_size: int = 16

    # how many random synthetic variants per probe
    samples_per_probe: int = 8

    patience: int = 4
    min_delta: float = 1e-4

    max_len: Optional[int] = None
    seed: int = 0

    # -------- shift-based synthetic generation --------
    # keep shift logic the same
    max_abs_shift_channels: int = 64
    extra_trace_channels_each_side: int = 128
    channel_step_um: float = 20.0

    # lower perturbation probability than before
    suspicious_probe_prob: float = 0.15

    suspicious_perturb_min_um: float = 150.0
    suspicious_perturb_max_um: float = 500.0

    suspicious_chunk_frac_min: float = 0.10
    suspicious_chunk_frac_max: float = 1.00
    suspicious_n_chunks_min: int = 1
    suspicious_n_chunks_max: int = 3

    perturb_n_anchors_min: int = 3
    perturb_n_anchors_max: int = 6
    perturb_smooth_kernel_min: int = 9
    perturb_smooth_kernel_max: int = 31
    perturb_taper_frac: float = 0.20


# ============================================================
# Dataset
# ============================================================
class SyntheticProbeConfidenceDataset(Dataset):
    """
    Shift-based synthetic dataset.

    samples_per_probe = how many random candidate shifts we sample per probe.
    """

    def __init__(
        self,
        bank: list[dict],
        cfg: ProbeConfidenceTrainConfig,
        *,
        ctx_manager,
        base_model,
        handles,
        seed: int = 0,
    ):
        self.bank = bank
        self.cfg = cfg
        self.ctx_manager = ctx_manager
        self.base_model = base_model
        self.handles = handles
        self.seed = int(seed)

        self.n_probes = len(bank)
        self.samples_per_probe = int(max(1, cfg.samples_per_probe))
        self.n = self.n_probes * self.samples_per_probe

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        # deterministic mapping from idx -> base probe, but still randomized inside
        probe_idx = int(idx % self.n_probes)

        rec, ctx, pred, labels, valid = _build_shift_based_synthetic_probe_sample(
            probe_idx=probe_idx,
            bank=self.bank,
            cfg=self.cfg,
            rng=rng,
            ctx_manager=self.ctx_manager,
            base_model=self.base_model,
            handles=self.handles,
        )

        return {
            "rec": torch.from_numpy(rec).float(),  # [C,F_e]
            "ctx": torch.from_numpy(ctx).float(),  # [C,F_ctx]
            "pred": torch.from_numpy(pred).float(),  # [C,F_e]
            "labels": torch.from_numpy(labels).long(),  # [C]
            "valid": torch.from_numpy(valid).bool(),  # [C]
        }


# ============================================================
# Public confidence-dataset builder
# ============================================================
def build_probe_confidence_datasets(
    *,
    one,
    pid_names: list[str],
    ctx_manager,
    ephys: np.ndarray,
    probe_positions: np.ndarray,
    split_info: dict,
    base_model,
    cfg: ProbeConfidenceTrainConfig,
    handles,
):
    """
    Build synthetic train/validation datasets for the probe-level confidence model.

    This is kept in model.py because it instantiates SyntheticProbeConfidenceDataset,
    while the lower-level histology-bank construction lives in utils.py.
    """
    train_ids = list(split_info["p_tr_ids"])
    val_ids = (
        list(split_info["p_va_ids"])
        if len(split_info["p_va_ids"]) > 0
        else list(split_info["p_te_ids"])
    )

    train_bank = _make_histology_probe_bank(
        one=one,
        ephys=ephys,
        probe_positions=probe_positions,
        probe_ids=train_ids,
        pid_names=pid_names,
        base_model=base_model,
        ctx_manager=ctx_manager,
        handles=handles,
        cfg=cfg,
    )

    val_bank = _make_histology_probe_bank(
        one=one,
        ephys=ephys,
        probe_positions=probe_positions,
        probe_ids=val_ids,
        pid_names=pid_names,
        base_model=base_model,
        ctx_manager=ctx_manager,
        handles=handles,
        cfg=cfg,
    )

    train_ds = SyntheticProbeConfidenceDataset(
        train_bank,
        cfg,
        ctx_manager=ctx_manager,
        base_model=base_model,
        handles=handles,
        seed=cfg.seed,
    )

    val_ds = SyntheticProbeConfidenceDataset(
        val_bank,
        cfg,
        ctx_manager=ctx_manager,
        base_model=base_model,
        handles=handles,
        seed=cfg.seed + 10_000,
    )

    info = {
        "n_train_probes": len(train_bank),
        "n_val_probes": len(val_bank),
        "train_probe_ids": train_ids,
        "val_probe_ids": val_ids,
    }
    return train_ds, val_ds, info


# ============================================================
# Probe-level transformer confidence model
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, mlp_ratio: float = 2.0, drop: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ProbeSequenceConfidenceTransformer(nn.Module):
    """
    Inputs per channel:
      rec   [B,C,F_e]
      pred  [B,C,F_e]
      ctx   [B,C,F_ctx]

    Output:
      logits [B,C,2] for classes:
        0 = good
        1 = suspicious
    """

    def __init__(
        self,
        *,
        f_ctx: int,
        f_e: int,
        d_model: int = 64,
        nhead: int = 4,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
    ):
        super().__init__()
        self.f_ctx = int(f_ctx)
        self.f_e = int(f_e)
        self.d_model = int(d_model)

        fin = self.f_ctx + self.f_e * 4  # rec, pred, diff, absdiff, ctx
        self.inp = nn.Sequential(
            nn.LayerNorm(fin),
            nn.Linear(fin, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, mlp_ratio=mlp_ratio, drop=drop)
                for _ in range(depth)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        *,
        rec: torch.Tensor,  # [B,C,F_e]
        pred: torch.Tensor,  # [B,C,F_e]
        ctx: torch.Tensor,  # [B,C,F_ctx]
        valid: torch.Tensor,  # [B,C] bool
    ):
        diff = pred - rec
        x = torch.cat([rec, pred, diff, torch.abs(diff), ctx], dim=-1)
        x = self.inp(x)

        pad_mask = ~valid.bool()
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)

        logits = self.head(x)  # [B,C,2]
        return logits


# ============================================================
# Training / evaluation
# ============================================================
def _flatten_valid_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
):
    m = valid.bool().reshape(-1)
    logits_f = logits.reshape(-1, logits.shape[-1])[m]
    labels_f = labels.reshape(-1)[m]
    return logits_f, labels_f


def train_probe_confidence_model(
    conf_model,
    *,
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    f_ctx: int,
    f_e: int,
    cfg: ProbeConfidenceTrainConfig,
    checkpoint_path: Optional[str] = None,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    opt = torch.optim.AdamW(
        conf_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    device_type = device.type
    use_autocast = device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)

    best_val = math.inf
    best_state = None
    bad = 0
    meters = {
        "train/loss": [],
        "val/loss": [],
        "val/acc": [],
    }

    # binary class weights: [good, suspicious]
    class_weights = torch.tensor([1.0, 3.0], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
        reduction="mean",
    )

    @torch.no_grad()
    def _eval():
        conf_model.eval()
        tot = 0.0
        n = 0
        n_correct = 0

        for batch in val_loader:
            rec = batch["rec"].to(device)
            ctx = batch["ctx"].to(device)
            pred = batch["pred"].to(device)
            labels = batch["labels"].to(device)
            valid = batch["valid"].to(device)

            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                logits = conf_model(rec=rec, pred=pred, ctx=ctx, valid=valid)

            logits_f, labels_f = _flatten_valid_logits_and_labels(logits, labels, valid)
            if logits_f.numel() == 0:
                continue

            loss = criterion(logits_f.float(), labels_f)
            bs = int(labels_f.numel())
            tot += float(loss.item()) * bs
            n += bs

            pred_cls = logits_f.argmax(dim=-1)
            n_correct += int((pred_cls == labels_f).sum().item())

        return tot / max(1, n), n_correct / max(1, n)

    for ep in range(1, cfg.epochs + 1):
        conf_model.train()
        tot = 0.0
        n = 0

        for batch in tqdm(train_loader):
            rec = batch["rec"].to(device)
            ctx = batch["ctx"].to(device)
            pred = batch["pred"].to(device)
            labels = batch["labels"].to(device)
            valid = batch["valid"].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                logits = conf_model(rec=rec, pred=pred, ctx=ctx, valid=valid)
                logits_f, labels_f = _flatten_valid_logits_and_labels(
                    logits, labels, valid
                )
                if logits_f.numel() == 0:
                    continue
                loss = criterion(logits_f.float(), labels_f)

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(conf_model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = int(labels_f.numel())
            tot += float(loss.item()) * bs
            n += bs

        tr = tot / max(1, n)
        va, va_acc = _eval()

        meters["train/loss"].append(tr)
        meters["val/loss"].append(va)
        meters["val/acc"].append(va_acc)

        print(
            f"[probe-conf ep {ep}] train_loss={tr:.5f} val_loss={va:.5f} val_acc={va_acc:.4f}"
        )

        if va < best_val - cfg.min_delta:
            best_val = va
            best_state = copy.deepcopy(conf_model.state_dict())
            bad = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "conf_model_state": best_state,
                        "cfg": cfg.__dict__,
                        "meters": meters,
                        "best_val": best_val,
                        "best_epoch": ep,
                    },
                    checkpoint_path,
                )
        else:
            bad += 1
            if bad >= cfg.patience:
                print(
                    f"[probe-conf] Early stopping at ep={ep} (no improvement {cfg.patience} epochs)"
                )
                break

    if best_state is not None:
        conf_model.load_state_dict(best_state)

    conf_model.eval()
    info = {"best_val": best_val}
    return conf_model, info, meters


@torch.no_grad()
def evaluate_probe_confidence_model(
    *,
    conf_model,
    dataset,
    device,
    batch_size: int = 16,
    title: str = "Probe-level confidence diagnostics",
):
    conf_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_pred = []
    all_prob = []

    for batch in loader:
        rec = batch["rec"].to(device)
        ctx = batch["ctx"].to(device)
        pred = batch["pred"].to(device)
        labels = batch["labels"].to(device)
        valid = batch["valid"].to(device)

        logits = conf_model(rec=rec, pred=pred, ctx=ctx, valid=valid)
        probs = torch.softmax(logits, dim=-1)

        logits_f, labels_f = _flatten_valid_logits_and_labels(logits, labels, valid)
        probs_f = probs.reshape(-1, 2)[valid.reshape(-1)]

        pred_cls = probs_f.argmax(dim=-1)

        all_labels.append(labels_f.cpu().numpy())
        all_pred.append(pred_cls.cpu().numpy())
        all_prob.append(probs_f.cpu().numpy())

    y = np.concatenate(all_labels)
    yhat = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, yhat):
        cm[int(t), int(p)] += 1

    acc = float((y == yhat).mean())
    print(f"\n===== {title} =====")
    print(f"accuracy: {acc:.4f}")
    print("confusion matrix rows=true cols=pred")
    print(cm)

    class_names = ["good", "suspicious"]
    for k, name in enumerate(class_names):
        m = y == k
        if m.any():
            print(
                f"class={name:10s} n={m.sum():6d} mean_pred_prob={prob[m, k].mean():.4f}"
            )

    return {
        "labels": y,
        "pred": yhat,
        "prob": prob,
        "cm": cm,
        "acc": acc,
    }


@torch.no_grad()
def visualize_probe_confidence_prediction(
    *,
    conf_model,
    rec_std: torch.Tensor | np.ndarray,  # [C, F_e]
    pred_std: torch.Tensor | np.ndarray,  # [C, F_e]
    ctx_std: torch.Tensor | np.ndarray,  # [C, F_ctx]
    valid_mask: torch.Tensor | np.ndarray,  # [C] bool
    device: torch.device,
    title: str = "Probe confidence prediction",
    feature_names: Optional[list[str]] = None,
    unstandardize: bool = False,
    e_mean: Optional[torch.Tensor | np.ndarray] = None,
    e_std: Optional[torch.Tensor | np.ndarray] = None,
    figsize=(14, 8),
):
    def _to_torch(x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.detach().cpu().to(dtype=dtype)
        return torch.as_tensor(x, dtype=dtype)

    rec_std_t = _to_torch(rec_std, torch.float32)
    pred_std_t = _to_torch(pred_std, torch.float32)
    ctx_std_t = _to_torch(ctx_std, torch.float32)
    valid_t = _to_torch(valid_mask, torch.bool)

    assert rec_std_t.ndim == 2
    assert pred_std_t.ndim == 2
    assert ctx_std_t.ndim == 2
    assert valid_t.ndim == 1

    C, F_e = rec_std_t.shape
    assert pred_std_t.shape == (C, F_e)
    assert ctx_std_t.shape[0] == C
    assert valid_t.shape[0] == C

    conf_model = conf_model.to(device).eval()

    rec_b = rec_std_t[None].to(device)
    pred_b = pred_std_t[None].to(device)
    ctx_b = ctx_std_t[None].to(device)
    valid_b = valid_t[None].to(device)

    logits = conf_model(rec=rec_b, pred=pred_b, ctx=ctx_b, valid=valid_b)[0]  # [C,2]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [C,2]
    pred_cls = probs.argmax(axis=-1).astype(int)  # [C]

    # confidence = p(good)
    conf_scalar = probs[:, 0]

    rec_disp = rec_std_t.numpy().copy()
    pred_disp = pred_std_t.numpy().copy()

    if unstandardize:
        if e_mean is None or e_std is None:
            raise ValueError(
                "If unstandardize=True, you must provide e_mean and e_std."
            )
        e_mean_np = _to_torch(e_mean, torch.float32).numpy().reshape(1, -1)
        e_std_np = _to_torch(e_std, torch.float32).numpy().reshape(1, -1)
        rec_disp = rec_disp * (e_std_np + 1e-8) + e_mean_np
        pred_disp = pred_disp * (e_std_np + 1e-8) + e_mean_np

    valid_np = valid_t.numpy().astype(bool)
    rec_disp_plot = rec_disp.copy()
    pred_disp_plot = pred_disp.copy()
    rec_disp_plot[~valid_np] = np.nan
    pred_disp_plot[~valid_np] = np.nan

    finite_vals = np.concatenate(
        [
            rec_disp_plot[np.isfinite(rec_disp_plot)],
            pred_disp_plot[np.isfinite(pred_disp_plot)],
        ]
    )
    if finite_vals.size > 0:
        vmin = np.percentile(finite_vals, 1)
        vmax = np.percentile(finite_vals, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [2.4, 2.4, 1.0, 1.4]},
    )

    ax_rec, ax_pred, ax_cls, ax_prob = axes
    extent = (0, F_e, C - 1, 0)

    ax_rec.imshow(
        rec_disp_plot,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax_rec.set_title("Recorded ephys")
    ax_rec.set_xlabel("Feature")
    ax_rec.set_ylabel("Channel")

    im1 = ax_pred.imshow(
        pred_disp_plot,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax_pred.set_title("Predicted ephys")
    ax_pred.set_xlabel("Feature")

    if feature_names is not None and len(feature_names) == F_e and F_e <= 20:
        xt = np.arange(F_e) + 0.5
        ax_rec.set_xticks(xt)
        ax_pred.set_xticks(xt)
        ax_rec.set_xticklabels(feature_names, rotation=90, fontsize=8)
        ax_pred.set_xticklabels(feature_names, rotation=90, fontsize=8)

    y = np.arange(C)
    cls_plot = pred_cls.astype(float).copy()
    cls_plot[~valid_np] = np.nan

    ax_cls.step(cls_plot, y, where="mid", linewidth=1.8)
    ax_cls.set_title("Predicted class")
    ax_cls.set_xlabel("Class")
    ax_cls.set_xlim(-0.4, 1.4)
    ax_cls.set_xticks([0, 1])
    ax_cls.set_xticklabels(["good", "susp."], rotation=45)
    ax_cls.set_ylim(C - 1, 0)
    ax_cls.grid(alpha=0.25)

    p_good = probs[:, 0].copy()
    p_susp = probs[:, 1].copy()
    p_good[~valid_np] = np.nan
    p_susp[~valid_np] = np.nan

    ax_prob.plot(p_good, y, label="good", linewidth=1.5)
    ax_prob.plot(p_susp, y, label="suspicious", linewidth=1.5)
    ax_prob.set_title("Class probabilities")
    ax_prob.set_xlabel("Probability")
    ax_prob.set_xlim(0.0, 1.0)
    ax_prob.set_ylim(C - 1, 0)
    ax_prob.grid(alpha=0.25)
    ax_prob.legend(fontsize=8, loc="best")

    cbar = fig.colorbar(im1, ax=[ax_rec, ax_pred], fraction=0.025, pad=0.02)
    cbar.set_label("Ephys value")

    invalid_idx = np.where(~valid_np)[0]
    if invalid_idx.size > 0:
        for ax in axes:
            for ch in invalid_idx:
                ax.axhline(ch, linewidth=0.3, alpha=0.15)

    mean_conf = (
        float(np.nanmean(conf_scalar[valid_np])) if valid_np.any() else float("nan")
    )
    fig.suptitle(f"{title}\nmean good probability = {mean_conf:.3f}", y=0.98)
    plt.tight_layout()
    plt.show()

    return {
        "logits": logits.detach().cpu().numpy(),
        "probs": probs,
        "pred_cls": pred_cls,
        "conf_scalar": conf_scalar,
    }


# ============================================================
# Inference helper for real probes
# ============================================================
@torch.no_grad()
def predict_probe_confidence_classes(
    *,
    conf_model,
    rec_std: torch.Tensor,  # [C,F_e]
    pred_std: torch.Tensor,  # [C,F_e]
    ctx_std: torch.Tensor,  # [C,F_ctx]
    valid_mask: torch.Tensor,  # [C] bool
    device: torch.device,
):
    """
    Returns
    -------
    logits : [C,2]
    probs  : [C,2]
    conf_scalar : [C]
        scalar confidence = p(good)
    """
    conf_model.eval()

    rec = rec_std[None].to(device)
    pred = pred_std[None].to(device)
    ctx = ctx_std[None].to(device)
    valid = valid_mask[None].to(device)

    logits = conf_model(rec=rec, pred=pred, ctx=ctx, valid=valid)[0]
    probs = torch.softmax(logits, dim=-1)
    conf_scalar = probs[:, 0]

    return logits.detach().cpu(), probs.detach().cpu(), conf_scalar.detach().cpu()


# ============================== evaluation ===============================
@torch.no_grad()
def evaluate_r2_per_feature(
    model, test_dl, ephys_mean, ephys_std, device=torch.device("cuda")
):
    print("Evaluating R2 for each feature")

    model.eval()
    device_type = device.type
    use_autocast = device_type == "cuda"

    ephys_mean = ephys_mean.to(device)
    ephys_std = ephys_std.to(device)

    F_dim = ephys_mean.numel()
    ss_res = torch.zeros(F_dim, device=device)
    sum_y = torch.zeros(F_dim, device=device)
    sum_y2 = torch.zeros(F_dim, device=device)
    n_obs = 0

    for batch in tqdm(test_dl):
        (ctx_q, p_q, e_n, p_n, mask, has_ephys, y_std, *_) = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu_std = model(ctx_q, p_q, e_n, p_n, mask)

        m = has_ephys
        if not m.any():
            continue

        # Convert BOTH prediction and target to original ephys scale
        mu = mu_std.float() * ephys_std + ephys_mean
        y = y_std.float() * ephys_std + ephys_mean

        y_m = y[m]
        mu_m = mu[m]

        ss_res += ((y_m - mu_m) ** 2).sum(dim=0)
        sum_y += y_m.sum(dim=0)
        sum_y2 += (y_m**2).sum(dim=0)
        n_obs += y_m.shape[0]

    if n_obs == 0:
        return torch.full((F_dim,), float("nan"))

    ss_tot = sum_y2 - (sum_y**2) / n_obs
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)

    return r2.detach().cpu()


def unstandardize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return X * std + mean
