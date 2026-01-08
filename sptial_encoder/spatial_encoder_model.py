import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from typing import Optional

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
    def __init__(self, f_ephys, f_region, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ephys + f_region + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, e_n, reg_n, p_n_abs, p_n_rel, mask):  # e_n: [B,M,Fe]
        pos = self.pos(p_n_abs, p_n_rel)                     # [B,M,d_pos]
        x = torch.cat([e_n, reg_n, pos], dim=-1)
        h = self.embed(x)                                    # [B,M,d_model]
        h = h * mask[..., None]                              # zero out pads
        return h

class QueryEncoder(nn.Module):
    def __init__(self, f_ctx, f_region, d_model, d_pos=64, drop=0.1):
        super().__init__()
        self.pos = PosEnc3D(d_pos)
        self.embed = mlp(f_ctx + f_region + d_pos, d_model, d_model, n_layers=2, drop=drop)

    def forward(self, ctx_q, reg_q, p_q_abs):
        # broadcast rel=0 for the query token
        B = ctx_q.size(0)
        p_rel0 = torch.zeros(B, 1, 3, device=ctx_q.device, dtype=ctx_q.dtype)
        p_abs = p_q_abs[:, None, :]
        pos = self.pos(p_abs, p_rel0)                        # [B,1,d_pos]
        x = torch.cat([ctx_q[:, None, :], reg_q[:, None, :], pos], dim=-1)
        h = self.embed(x)                                    # [B,1,d_model]
        return h

class CrossBlock(nn.Module):
    """Optional neighbor self-attn, then query->neighbor cross-attn."""
    def __init__(self, d_model, nhead=8, drop=0.1, neighbor_self_attn=True):
        super().__init__()
        self.nei_self = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=drop, batch_first=True
        ) if neighbor_self_attn else None
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=drop, batch_first=True)
        self.ff = mlp(d_model, 4*d_model, d_model, n_layers=2, drop=drop)
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)

    def forward(self, h_q, h_n, mask_nei):
        # h_q: [B,1,D], h_n: [B,M,D], mask_nei: [B,M] (True=real, False=pad)
        B, M, D = h_n.shape
        no_nei = ~mask_nei.any(dim=1)          # [B]
        if no_nei.any():
            # append a dummy zero neighbor and mark it valid ONLY for empty rows
            dummy = h_n.new_zeros(B, 1, D)
            h_n = torch.cat([h_n, dummy], dim=1)               # [B, M+1, D]
            pad = mask_nei.new_zeros(B, 1)
            pad[no_nei, 0] = True
            mask_nei = torch.cat([mask_nei, pad], dim=1)       # [B, M+1]

        # neighbor self-attn (optional)
        if self.nei_self is not None:
            key_padding_mask = (~mask_nei.bool())              # True=ignore
            h_n = self.nei_self(h_n, src_key_padding_mask=key_padding_mask)

        # query <- neighbors cross-attn
        key_padding_mask = (~mask_nei.bool())
        h_q2, _ = self.cross(h_q, h_n, h_n, key_padding_mask=key_padding_mask)
        h_q = self.norm_q1(h_q + h_q2)
        h_q = self.norm_q2(h_q + self.ff(h_q))
        return h_q

class EphysPredictor(nn.Module):
    def __init__(self, d_model, f_out, heteroscedastic=True):
        super().__init__()
        self.mu_head = mlp(d_model, 2*d_model, f_out, n_layers=2, drop=0.0)
        self.het = heteroscedastic
        if heteroscedastic:
            self.logvar_head = mlp(d_model, 2*d_model, f_out, n_layers=2, drop=0.0)

    def forward(self, h_q):  # [B,1,D]
        h = h_q.squeeze(1)   # [B,D]
        mu = self.mu_head(h)
        if self.het:
            logvar = self.logvar_head(h).clamp(-6.0, 4.0)
            return mu, logvar
        return mu, None

class NeighborInpaintingModel(nn.Module):
    """
    Predict ephys for a single query channel using context of that channel
    and a variable-size set of neighbor ephys from *other* probes.
    """
    def __init__(self, f_ctx, f_ephys, f_region, f_out, e_mean=None, e_std=None, ctx_mean=None, ctx_std=None,
                 d_model=256, nhead=8, depth=2, neighbor_self_attn=True, heteroscedastic=True, drop=0.1):
        super().__init__()
        self.qenc = QueryEncoder(f_ctx, f_region, d_model, drop=drop)
        self.nenc = NeighborEncoder(f_ephys, f_region, d_model, drop=drop)
        self.blocks = nn.ModuleList([
            CrossBlock(d_model, nhead=nhead, drop=drop, neighbor_self_attn=neighbor_self_attn)
            for _ in range(depth)
        ])
        self.pred = EphysPredictor(d_model, f_out, heteroscedastic=heteroscedastic)

        if(e_mean is not None and e_std is not None and ctx_mean is not None and ctx_std is not None):
            # register as buffers so they're saved/loaded with state_dict
            self.register_buffer("e_mean", e_mean.clone().detach())
            self.register_buffer("e_std",  e_std.clone().detach())
            self.register_buffer("ctx_mean", ctx_mean.clone().detach())
            self.register_buffer("ctx_std", ctx_std.clone().detach())

    def forward(self, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_nei):
        """
        ctx_q: [B, F_ctx]
        reg_q: [B, F_reg]   (e.g., one-hot Allen/Cosmos or learned emb)
        p_q:   [B, 3]       absolute (voxel/world) coords
        e_n:   [B, M, F_e]  neighbor ephys
        reg_n: [B, M, F_reg]
        p_n:   [B, M, 3]    neighbor absolute coords
        mask_nei: [B, M]    1 = real neighbor, 0 = pad
        """
        # relative pos of neighbors to query
        p_q_b = p_q[:, None, :].expand_as(p_n)
        p_rel = p_n - p_q_b

        h_q = self.qenc(ctx_q, reg_q, p_q)                   # [B,1,D]
        h_n = self.nenc(e_n, reg_n, p_n, p_rel, mask_nei)    # [B,M,D]

        for blk in self.blocks:
            h_q = blk(h_q, h_n, mask_nei)

        mu, logvar = self.pred(h_q)                          # [B,F_out], [B,F_out] or None
        return mu, logvar

# ---------- losses & training ----------
def gaussian_nll(mu_std: torch.Tensor, logvar: Optional[torch.Tensor], y_std: torch.Tensor):
    """ Sum NLL over channels & features where mask==True. If logvar is None -> falls back to MSE. """
    mu = mu_std
    y = y_std
    if mu.numel() == 0:
        return torch.inf
    if logvar is None:
        return torch.mean((mu - y)**2)
    lv = logvar.clamp(-6.0, 6.0)
    inv_var = torch.exp(-lv)
    nll = 0.5 * ((mu - y)**2 * inv_var + lv + np.log(2*np.pi))
    return nll.mean()

def masked_mse(mu, y, mask):
    if mask.sum() == 0: return mu.new_tensor(0.0)
    d = (mu - y)[mask]
    return (d*d).mean()

@torch.no_grad()
def mean_feature_corr(mu, y, mask=None):
    if mask is not None:
        mu = mu[mask]; y = y[mask]
    if mu.numel()==0: return mu.new_tensor(float('nan'))
    mu = mu - mu.mean(dim=1, keepdim=True)
    y  = y  - y.mean(dim=1, keepdim=True)
    mu = F.normalize(mu, dim=1); y = F.normalize(y, dim=1)
    return (mu*y).sum(dim=1).mean()

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
        pos_mask = (dist <= pos_radius_m) & (~torch.eye(N, dtype=torch.bool, device=z.device))  # exclude self
        # We will use ALL other samples (except self) as the denominator
        denom_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

        # anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)

    if not has_pos.any():
        return torch.tensor(0.0, device=z.device), 0

    # For numerical stability, subtract row-wise max over the denominator set
    sim_denom = sim.masked_fill(~denom_mask, float('-inf'))
    row_max, _ = torch.max(sim_denom, dim=1, keepdim=True)
    sim = sim - row_max  # [N,N]

    # Numerator: sum over positives
    num = torch.logsumexp(sim.masked_fill(~pos_mask, float('-inf')), dim=1)  # [N]
    # Denominator: sum over all j != i
    den = torch.logsumexp(sim.masked_fill(~denom_mask, float('-inf')), dim=1)  # [N]

    loss_vec = -(num - den)  # [N]
    loss = loss_vec[has_pos].mean()
    return loss, int(has_pos.sum().item())

def get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_nei):
    p_rel = p_n - p_q[:, None, :]
    h_q = model.qenc(ctx_q, reg_q, p_q)                    # [B,1,D]
    h_n = model.nenc(e_n, reg_n, p_n, p_rel, mask_nei)     # [B,M,D]
    for blk in model.blocks:
        h_q = blk(h_q, h_n, mask_nei)                      # [B,1,D]
    mu, logvar = model.pred(h_q)                           # [B,F], [B,F] or None
    return h_q.squeeze(1), mu, logvar

def train_hybrid(
    model, train_dl, val_dl, optimizer, epochs=10, device=torch.device("cuda"),
    lambda_sup=1.0, lambda_ctr=0.2, tau=0.2, pos_radius_um=600.0,
    heteroscedastic=True, grad_clip=1.0, log_every=50,
    *,
    # Early stopping knobs
    early_stopping=True,
    patience: int = 10,
    min_delta: float = 0.0,
    ephys_drop: float = 0.0,
    monitor: str = "val/sup",     # "val/sup" (min) or "val/corr" (max) etc.
    mode: str = "min",            # "min" or "max"
    checkpoint_path: Optional[str] = None,  # if set, saves best state dict here on improvement
    lr_scheduler: Optional[object] = None,  # e.g., torch.optim.lr_scheduler.ReduceLROnPlateau
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
    meters = {"train/sup":[], "train/ctr":[], "train/total":[], "val/sup":[], "val/corr":[]}
    use_grad_scaler = device_type in ['cuda', 'mps']
    scaler = torch.amp.GradScaler(device_type, enabled=use_grad_scaler)

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
        return (current < best - min_delta) if mode == "min" else (current > best + min_delta)

    for ep in range(1, epochs + 1):
        # -------------------- Train --------------------
        model.train()
        r_sup = r_ctr = r_tot = 0.0
        n_steps = 0

        for step, batch in enumerate(train_dl, 1):
            (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y_e, vox_count, *_) = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=use_grad_scaler):
                mask_dropped = mask.clone()
                mask_dropped[np.random.permutation(np.arange(mask.shape[0]))[:int(ephys_drop * len(mask))]] = False
                h_q, mu, logvar = get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask_dropped)

                # supervised
                if (heteroscedastic and (logvar is not None) and has_ephys.any()):
                    sup = gaussian_nll(mu[has_ephys], logvar[has_ephys], y_e[has_ephys])
                else:
                    sup = masked_mse(mu, y_e, has_ephys)

                # contrastive
                ctr = torch.tensor(0.0, device=device)
                if (~has_ephys).sum() >= 2:
                    z = h_q[~has_ephys]; xyz = p_q[~has_ephys]
                    ctr, _ = info_nce_multi_positive(z, xyz, pos_radius_m, tau=tau)

                loss = lambda_sup * sup + lambda_ctr * ctr

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            r_sup += sup.item(); r_ctr += ctr.item(); r_tot += loss.item()
            n_steps += 1
            if (step % log_every) == 0:
                print(f"[ep {ep} step {step}] sup={r_sup/n_steps:.4f} ctr={r_ctr/n_steps:.4f} tot={r_tot/n_steps:.4f}")

        meters["train/sup"].append(r_sup / max(1, n_steps))
        meters["train/ctr"].append(r_ctr / max(1, n_steps))
        meters["train/total"].append(r_tot / max(1, n_steps))

        # -------------------- Validation --------------------
        current_val = None  # value of the monitored metric this epoch
        if val_dl is not None:
            model.eval()
            vs = vc = 0.0
            m = 0
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_grad_scaler):
                for batch in val_dl:
                    (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y_e, vox_count, *_) = [
                        x.to(device) if torch.is_tensor(x) else x for x in batch
                    ]
                    h_q, mu, logvar = get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask)
                    val_sup = (gaussian_nll(mu[has_ephys], logvar[has_ephys], y_e[has_ephys])
                               if (heteroscedastic and (logvar is not None) and has_ephys.any())
                               else masked_mse(mu, y_e, has_ephys))
                    vs += val_sup.item()
                    # mean feature corr (only where supervised)
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

            # Step LR scheduler if provided
            if lr_scheduler is not None:
                try:
                    # ReduceLROnPlateau expects a metric
                    lr_scheduler.step(current_val if current_val is not None else val_sup_mean)
                except TypeError:
                    # other schedulers (e.g., CosineAnnealingLR) don't take metrics
                    lr_scheduler.step()

        # -------------------- Early stopping --------------------
        if early_stopping and (val_dl is not None):
            if _is_improvement(current_val, best_val):
                best_val = current_val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep
                num_bad_epochs = 0
                if checkpoint_path is not None:
                    torch.save({"epoch": ep,
                                "model_state": best_state,
                                "optimizer_state": optimizer.state_dict(),
                                "meters": meters,
                                "best_value": best_val,
                                "monitor": monitor,
                                "mode": mode},
                               checkpoint_path)
                print(f"✓ Improvement on {monitor}: {best_val:.6f} (epoch {ep})")
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    print(f"⏹ Early stopping at epoch {ep} (no improvement in {patience} epochs).")
                    break

    # Restore best weights (if any) at the end
    if early_stopping and (val_dl is not None) and (best_state is not None):
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} with {monitor}={best_val:.6f}")

    return model, meters, best_epoch, best_val

# ---------- R^2 (per feature) on test ----------
@torch.no_grad()
def evaluate_r2_per_feature(model, test_dl, ephys_mean, ephys_std, device=torch.device("cuda")):
    print("Evaluating R2 for each feature")
    device_type = device.type
    use_autocast = device_type in ['cuda', 'mps']
    model.eval()
    F = ephys_mean.numel()
    # accumulate in ORIGINAL scale
    ss_res = torch.zeros(F, device=device)
    sum_y  = torch.zeros(F, device=device)
    n_obs  = 0
    with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
        for batch in tqdm(test_dl):
            (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y, vox_count, *_) = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            h_q, mu_std, _ = get_query_repr_and_pred(model, ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask)
            # unstandardize
            mu = unstandardize(mu_std, ephys_mean, ephys_std)
            # mask (test should be all supervised, but keep mask)
            m = has_ephys
            if m.any():
                y_m  = y[m]
                mu_m = mu[m]
                ss_res += ((y_m - mu_m)**2).sum(dim=0)
                sum_y  += y_m.sum(dim=0)
                n_obs  += y_m.shape[0]
    if n_obs == 0:
        return torch.full((F,), float('nan'))
    ybar = sum_y / n_obs
    # need one more pass for ss_tot (or store sumsq). Let's store sumsq:
    ss_tot = torch.zeros(F, device=device)
    with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
        for batch in test_dl:
            (ctx_q, reg_q, p_q, e_n, reg_n, p_n, mask, has_ephys, y, vox_count, *_) = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            m = has_ephys
            if m.any():
                ss_tot += ((y[m] - ybar)**2).sum(dim=0)
    r2 = 1.0 - (ss_res / ss_tot.clamp_min(1e-12))
    return r2.detach().cpu()

def unstandardize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return X * std + mean