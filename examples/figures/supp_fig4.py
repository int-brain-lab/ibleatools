from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy.ma as ma
import pandas as pd

from typing import Optional
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice
from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig

from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors,
    FEATURE_LIST,
    get_device,
    region_ids_from_xyz,
)
from ephysatlas.spatial_encoder.model import (
    NeighborInpaintingModel,
    ProbeConfidenceTrainConfig,
    evaluate_r2_per_feature,
    ProbeSequenceConfidenceTransformer,
)

# Loader swapped to this branch's publishing system (examples/figures/_release.py); this figure had
# diverged from the shared convention (S3 download_model + raw torch.load), so it is brought in line.
from _release import channel_release

from torch.utils.data import DataLoader, ConcatDataset, Dataset

TARGET_PID = "2c7b7191-cb21-4f4e-ac8f-3345803337f7"

def _concat_context(cell_pc: np.ndarray, gene_pc: np.ndarray) -> np.ndarray:
    return np.concatenate([cell_pc, gene_pc], axis=1).astype(np.float32)

@torch.no_grad()
def _sample_and_standardize_ctx_for_xyz(
    ctx_manager,
    xyz_m: np.ndarray,
    ctx_mean: torch.Tensor,
    ctx_std: torch.Tensor,
    *,
    chunk: int = 8192,
) -> torch.Tensor:
    assert xyz_m.ndim == 2 and xyz_m.shape[1] == 3

    ctx_list = []
    for s in range(0, xyz_m.shape[0], chunk):
        xyz_chunk = xyz_m[s : s + chunk].astype(np.float32, copy=False)
        pack = ctx_manager.sample_context_numpy_m(xyz_chunk, mode='clip')
        ctx_chunk = _concat_context(pack['cell_pc'], pack['gene_pc'])
        ctx_list.append(ctx_chunk)

    ctx = np.concatenate(ctx_list, axis=0).astype(np.float32)
    ctx_t = torch.from_numpy(ctx).float()

    ctx_mean = ctx_mean.detach().cpu()
    ctx_std = ctx_std.detach().cpu()

    has_ctx = ctx_t.abs().sum(dim=1) != 0
    ctx_t[has_ctx] = (ctx_t[has_ctx] - ctx_mean) / (ctx_std + 1e-8)

    return ctx_t

class GridDS(Dataset):
    """Grid-only voxels: (context, xyz_m, empty ephys, pid=0, has_ephys=False)."""

    def __init__(self, ctx, xyz_m, f_e):
        self.ctx, self.xyz = ctx, xyz_m
        self._empty = torch.zeros(f_e, dtype=torch.float32)
        self._empty_pid = torch.tensor(0.0, dtype=torch.float32)  # scalar, not [1]
        self.has = torch.zeros(len(self.ctx), dtype=torch.bool)

    def __len__(self):
        return self.ctx.shape[0]

    def __getitem__(self, i):
        return (self.ctx[i], self.xyz[i], self._empty, self._empty_pid, self.has[i])

class NeighborCollate:
    """
    Takes per-sample (idx, ctx, allen, xyz_m, ephys, has_ephys) and adds:
      - e_n [B,M,F_e], p_n [B,M,3], mask [B,M]
      - y_e [B,F_e] from dataset
    Uses a TRAIN-ONLY neighbor bank and excludes same-probe neighbors for recorded voxels.
    Assumes inputs are already standardized.
    """

    def __init__(
        self,
        ctx_manager,
        bank_xyz_m,
        bank_feat_stdzd,
        bank_pid,
        kdtree_bank,
        e_feat_dim: int,
        M_max=64,
        radius_um=600.0,
        allow_same_probe=False,
    ):
        self.ctx_manager = ctx_manager
        self.bank_xyz = bank_xyz_m
        self.bank_feat = bank_feat_stdzd
        self.bank_pid = bank_pid
        self.nn = kdtree_bank
        self.F_e = int(e_feat_dim)
        self.M = int(M_max)
        self.r_m = float(radius_um) * 1e-6
        self.allow_same_probe = allow_same_probe

    def __call__(self, batch_items):
        # unpack
        (ctxs, xyzs, ephys, pids, has) = zip(*batch_items)

        B = len(ctxs)
        ctx_q = torch.stack(ctxs, dim=0)  # [B,F_ctx] (already standardized)
        p_q = torch.stack(xyzs, dim=0)  # [B,3] m
        y_e = torch.stack(
            [
                t if t.numel() else torch.zeros(self.F_e, dtype=torch.float32)
                for t in ephys
            ],
            dim=0,
        )  # [B,F_e]
        has_ephys = torch.stack(has, dim=0).bool()  # [B]

        # placeholders
        e_n = torch.zeros(B, self.M, self.F_e, dtype=torch.float32)
        p_n = torch.zeros(B, self.M, 3, dtype=torch.float32)
        mask = torch.zeros(B, self.M, dtype=torch.bool)

        # voxel keys for exclusion / target lookup
        xi, zi, yi = downsample_keys_from_xyz(self.ctx_manager, p_q.numpy())

        # neighbor candidates from train bank
        neigh_lists = self.nn.query_radius(p_q.numpy(), r_m=self.r_m, k_cap=8 * self.M)

        for b in range(B):
            _ = (xi[b], zi[b], yi[b])

            # Exclude same-probe neighbors for recorded voxels
            exclude_pids = set()
            if has_ephys[b] and not self.allow_same_probe:
                exclude_pids = {pids[b].item()}

            # build neighbor set
            cand = [
                ci
                for ci in neigh_lists[b]
                if int(self.bank_pid[ci]) not in exclude_pids
            ]
            L = len(cand)

            if L > self.M:
                # random subset, not just first-M
                sel = np.random.choice(cand, size=self.M, replace=False)
                cand = sel.tolist()

            L = len(cand)

            if L > 0:
                e_n[b, :L] = torch.from_numpy(self.bank_feat[cand])
                p_n[b, :L] = torch.from_numpy(self.bank_xyz[cand])
                mask[b, :L] = True

        batch = (ctx_q, p_q, e_n, p_n, mask, has_ephys, y_e, pids)
        return batch

def mirror_xyz_to_left(xyz_m: np.ndarray) -> np.ndarray:
    """Return a copy where x is reflected to the left hemisphere (x<=0 in world coords)."""
    out = xyz_m.copy()
    mirror_ind = np.where(out[..., 0] > 0)[0]
    out[..., 0][mirror_ind] = -np.abs(out[..., 0][mirror_ind])
    return out


def mirror_x_indices_to_left(xi: np.ndarray, Xh: int) -> np.ndarray:
    """Mirror an atlas x-index array into the left half (index space)."""
    xi = xi.copy()
    right = xi >= (Xh // 2)
    xi[right] = Xh - xi[right] - 1
    return xi

def downsample_keys_from_xyz(ctx_manager, xyz_m, ds_rate=8):
    Xh, Zh, Yh = ctx_manager.cell_pca.shape[1:]
    xyz_m = mirror_xyz_to_left(xyz_m)  # <<< add
    ijk = ctx_manager.bc.xyz2i(xyz_m, mode="clip")
    xi = np.clip(np.round(ijk[:, 0] / ds_rate).astype(int), 0, Xh - 1)
    yi = np.clip(np.round(ijk[:, 1] / ds_rate).astype(int), 0, Yh - 1)
    zi = np.clip(np.round(ijk[:, 2] / ds_rate).astype(int), 0, Zh - 1)
    # ensure xi is mirrored in index space too (defensive, though mirror_xyz already did it)
    xi = mirror_x_indices_to_left(xi, Xh)
    return xi, zi, yi

@torch.no_grad()
def predict_features_at_xyz(
    model,
    ctx_manager,
    handles: dict,
    xyz_m: np.ndarray,
    *,
    batch_size: int = 512,
    radius_um: float,
    M_max: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    xyz_t = torch.from_numpy(xyz_m).float()

    ctx_q = _sample_and_standardize_ctx_for_xyz(
        ctx_manager,
        xyz_m,
        model.ctx_mean,
        model.ctx_std,
        chunk=8192,
    )

    F_e = int(model.e_mean.numel())
    qds = GridDS(ctx_q, xyz_t, F_e)

    collate = NeighborCollate(
        ctx_manager,
        handles['bank_xyz'],
        handles['bank_feat'],
        handles['bank_pid'],
        handles['nn_bank'],
        e_feat_dim=F_e,
        M_max=M_max,
        radius_um=radius_um,
    )

    dl = DataLoader(
        qds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    mu_all = []
    device_type = device.type
    use_autocast = device_type == 'cuda'

    for batch in dl:
        ctx_b, p_b, e_n, p_n, mask, *_ = [x.to(device) if torch.is_tensor(x) else x for x in batch]

        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu = model(ctx_b, p_b, e_n, p_n, mask)

        mu_all.append(mu.detach().cpu())

    return torch.cat(mu_all, dim=0)

def build_cost_matrix(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    AB = A @ B.T

    return (AA + BB - 2.0 * AB).clip(min=0.0)

def dynamic_time_warping_debug(C, lam_d=0.0, lam_u=0.1, lam_l=0.1, band=None, open_begin=True):
    C = np.asarray(C, dtype=np.float64)
    C = np.where(np.isfinite(C), C, np.inf)

    N, M = C.shape
    D = np.full((N, M), np.inf, dtype=np.float64)
    P = np.full((N, M), -1, dtype=np.int8)

    if band is None:
        band = np.ones((N, M), dtype=bool)
    else:
        band = np.asarray(band, dtype=bool)

    if band[0, 0]:
        D[0, 0] = C[0, 0]

    for j in range(1, M):
        if not band[0, j]:
            continue
        if open_begin:
            D[0, j] = C[0, j]
            P[0, j] = -1
        else:
            D[0, j] = C[0, j] + D[0, j - 1] + lam_l
            P[0, j] = 2

    for i in range(1, N):
        if not band[i, 0]:
            continue
        D[i, 0] = C[i, 0] + D[i - 1, 0] + lam_u
        P[i, 0] = 1

    for i in range(1, N):
        for j in range(1, M):
            if not band[i, j]:
                continue

            candidates = [
                D[i - 1, j - 1] + lam_d,
                D[i - 1, j] + lam_u,
                D[i, j - 1] + lam_l,
            ]

            k = int(np.argmin(candidates))
            D[i, j] = C[i, j] + candidates[k]
            P[i, j] = k

    j_end = int(np.nanargmin(D[N - 1]))
    total = float(D[N - 1, j_end])

    i, j = N - 1, j_end
    path = [(i, j)]

    while i > 0 or (not open_begin and j > 0):
        k = P[i, j]

        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i, j = i - 1, j
        elif k == 2:
            i, j = i, j - 1
        else:
            break

        path.append((i, j))

    path.reverse()
    j_start = path[0][1]

    return j_start, j_end, path, total, D, P

def rigid_assignment(A, B):
    best_k, best_mse = 0, np.inf
    Nr = A.shape[0]

    for k in range(0, B.shape[0] - Nr + 1):
        m = ((B[k : k + Nr] - A) ** 2).mean()
        if m < best_mse:
            best_mse, best_k = m, k

    j_start = best_k
    j_end = best_k + Nr - 1
    path = [(i, best_k + i) for i in range(Nr)]

    return j_start, j_end, path

def _scatter_recorded_onto_trace(
    recorded_full: np.ndarray,
    j_map_all_i: np.ndarray,
    trace_len: int,
    *,
    kp_mask: Optional[np.ndarray] = None,
):
    recorded_full = np.asarray(recorded_full)
    j_map_all_i = np.asarray(j_map_all_i, dtype=int)

    C_rec, F = recorded_full.shape
    L = int(trace_len)

    if kp_mask is None:
        kp_mask = np.ones((C_rec,), dtype=bool)
    else:
        kp_mask = np.asarray(kp_mask, dtype=bool)

    sums = np.zeros((L, F), dtype=np.float64)
    counts = np.zeros((L,), dtype=np.int64)

    for c in range(C_rec):
        if not kp_mask[c]:
            continue
        j = int(j_map_all_i[c])
        if 0 <= j < L:
            sums[j] += recorded_full[c]
            counts[j] += 1

    recorded_on_trace_raw = np.full((L, F), np.nan, dtype=np.float64)
    recorded_on_trace_filled = np.zeros((L, F), dtype=np.float64)

    hit = counts > 0
    recorded_on_trace_raw[hit] = sums[hit] / counts[hit, None]
    recorded_on_trace_filled[hit] = sums[hit] / counts[hit, None]

    return recorded_on_trace_raw, recorded_on_trace_filled, counts

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

@torch.no_grad()
def classify_aligned_probe_channels(
    *,
    conf_model,
    model,
    ctx_manager,
    recorded_full: np.ndarray,
    est_xyz: np.ndarray,
    mu_std_est: np.ndarray | torch.Tensor,
    device: torch.device,
):
    conf_model.eval()

    rec_raw = np.asarray(recorded_full, dtype=np.float32)
    xyz_np = np.asarray(est_xyz, dtype=np.float32)

    C, F_e = rec_raw.shape

    if torch.is_tensor(mu_std_est):
        pred_std_np = mu_std_est.detach().cpu().numpy().astype(np.float32)
    else:
        pred_std_np = np.asarray(mu_std_est, dtype=np.float32)

    rec_is_finite = np.isfinite(rec_raw).all(axis=1)
    rec_has_signal = ~np.all(np.nan_to_num(rec_raw, nan=0.0) == 0.0, axis=1)
    xyz_is_finite = np.isfinite(xyz_np).all(axis=1)
    pred_is_finite = np.isfinite(pred_std_np).all(axis=1)

    valid_mask = rec_is_finite & rec_has_signal & xyz_is_finite & pred_is_finite
    valid_t = torch.from_numpy(valid_mask).bool()

    e_mean = model.e_mean.detach().cpu().numpy().astype(np.float32)
    e_std = model.e_std.detach().cpu().numpy().astype(np.float32)

    rec_raw_safe = np.nan_to_num(rec_raw, nan=0.0, posinf=0.0, neginf=0.0)
    rec_std = (rec_raw_safe - e_mean) / (e_std + 1e-8)
    rec_std[~valid_mask] = 0.0

    pred_std_np = np.nan_to_num(pred_std_np, nan=0.0, posinf=0.0, neginf=0.0)
    pred_std_np[~valid_mask] = 0.0

    ctx_std_t = _sample_and_standardize_ctx_for_xyz(
        ctx_manager,
        xyz_np,
        model.ctx_mean,
        model.ctx_std,
        chunk=8192,
    ).float()
    ctx_std_t[~valid_t] = 0.0

    logits, probs, _ = predict_probe_confidence_classes(
        conf_model=conf_model,
        rec_std=torch.from_numpy(rec_std).float(),
        pred_std=torch.from_numpy(pred_std_np).float(),
        ctx_std=ctx_std_t,
        valid_mask=valid_t,
        device=device,
    )

    probs_cpu = probs.detach().cpu().float()
    pred_cls = probs_cpu.argmax(dim=1).numpy().astype(np.int64)
    pred_cls[~valid_mask] = -1

    probs_np = probs_cpu.numpy().astype(np.float32)
    probs_np[~valid_mask] = np.nan

    return pred_cls, probs_np

@torch.no_grad()
def align(
    model,
    ctx_manager,
    xyz_samples_ext,
    recorded_full,
    handles,
    optimization_features,
    RADIUS_UM,
    M_MAX,
    device,
    conf_model=None,
    return_debug: bool = True,
    brain_atlas=None,
):
    C_full = recorded_full.shape[0]
    L_trace = xyz_samples_ext.shape[0]

    kp_mask = ~np.all(recorded_full == 0.0, axis=1)
    if kp_mask.sum() < 2:
        print(
            'Need at least 2 recorded (non-zero) channels with non-zero features for spatial encoding.'
        )
        return None

    recorded_std = (
        (
            (torch.from_numpy(recorded_full.copy()) - model.e_mean.cpu())
            / (model.e_std.cpu() + 1e-8)
        )
        .numpy()
        .astype(np.float64)
    )
    recorded_opt = recorded_std[kp_mask][:, optimization_features]

    # full-trace prediction
    pred_std_full = predict_features_at_xyz(
        model,
        ctx_manager,
        handles,
        xyz_samples_ext,
        batch_size=512,
        radius_um=RADIUS_UM,
        M_max=M_MAX,
        device=device,
    )
    pred_std_full_np = pred_std_full.detach().cpu().numpy().astype(np.float64)
    pred_std_opt = pred_std_full_np[:, optimization_features]

    ephys_cost_matrix = build_cost_matrix(recorded_opt, pred_std_opt)

    cost_matrix = ephys_cost_matrix

    finite_cost = cost_matrix[np.isfinite(cost_matrix)]
    max_cost = float(np.median(np.nan_to_num(finite_cost)))

    lam_u = 0.5 * max_cost
    lam_l = 0.1 * max_cost

    j_start, j_end, path, total_cost, D, P = dynamic_time_warping_debug(
        cost_matrix,
        lam_d=0.0,
        lam_u=lam_u,
        lam_l=lam_l,
        open_begin=True,
    )

    min_overlap_channels = int(0.9 * int(kp_mask.sum()))
    if (j_end - j_start + 1) < min_overlap_channels:
        print(f'Trace too short - resorting to rigid optimization')
        j_start, j_end, path = rigid_assignment(recorded_opt, pred_std_opt)

    i_seq, j_seq = np.array(path, dtype=int).T
    j_for_i = np.full(recorded_opt.shape[0], np.nan)
    j_for_i[i_seq] = j_seq
    j_for_i = (
        pd.Series(j_for_i)
        .ffill()
        .bfill()
        .astype(int)
        .clip(0, pred_std_opt.shape[0] - 1)
        .to_numpy()
    )

    # map from ALL recorded channels -> full trace indices
    j_map = np.interp(np.arange(C_full), np.where(kp_mask)[0], j_for_i.astype(float))
    j_map_i = np.clip(np.round(j_map).astype(int), 0, pred_std_opt.shape[0] - 1)

    est_xyz = xyz_samples_ext[j_map_i]

    if not return_debug:
        return est_xyz

    # aligned-window prediction as before
    mu_std_est = pred_std_full_np[j_map_i]

    # create full-trace recorded array with NaNs outside aligned channels
    recorded_on_trace_raw, recorded_on_trace_filled, recorded_on_trace_counts = (
        _scatter_recorded_onto_trace(
            recorded_full=recorded_full,
            j_map_all_i=j_map_i,
            trace_len=L_trace,
            kp_mask=kp_mask,
        )
    )

    pred_cls_est = None
    cls_probs_est = None
    pred_cls_trace = None
    cls_probs_trace = None

    if conf_model is not None:
        # per-channel class/confidence on aligned estimated probe (same as before)
        pred_cls_est, cls_probs_est = classify_aligned_probe_channels(
            conf_model=conf_model,
            model=model,
            ctx_manager=ctx_manager,
            recorded_full=recorded_full,
            est_xyz=est_xyz,
            mu_std_est=mu_std_est,
            device=device,
        )

        # full-trace class/confidence
        # Use the NaN-padded trace for plotting and the zero-filled trace for inference.
        pred_cls_trace, cls_probs_trace = classify_aligned_probe_channels(
            conf_model=conf_model,
            model=model,
            ctx_manager=ctx_manager,
            recorded_full=recorded_on_trace_raw,  # not recorded_on_trace_filled
            est_xyz=xyz_samples_ext,
            mu_std_est=pred_std_full_np,
            device=device,
        )

    return dict(
        est_xyz=est_xyz,
        kp_mask=kp_mask,
        j_map_all_i=j_map_i,
        cost_matrix=cost_matrix,
        path=np.array(path, dtype=int),
        total_cost=float(total_cost),
        j_start=int(j_start),
        j_end=int(j_end),
        pred_cls_est=pred_cls_est,
        cls_probs_est=cls_probs_est,
        mu_std_est=mu_std_est,
        # full-trace outputs
        xyz_samples_ext=xyz_samples_ext,
        mu_std_trace=pred_std_full_np,
        pred_cls_trace=pred_cls_trace,
        cls_probs_trace=cls_probs_trace,
        recorded_on_trace_raw=recorded_on_trace_raw,
        recorded_on_trace_counts=recorded_on_trace_counts,
        ephys_cost_matrix=ephys_cost_matrix,
    )

def _to_um(x):
    x = np.asarray(x, dtype=float)
    if np.nanmax(np.abs(x)) < 50:
        x = x * 1e6
    return x


def _safe_limits(vals, q=(1, 99)):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, 1
    vmin, vmax = np.nanpercentile(vals, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmin == vmax:
        vmax = vmin + 1
    return float(vmin), float(vmax)


def _feature_idx(feature_list, name):
    return list(feature_list).index(name)


def _unstandardize(mu_std, model):
    e_mean = model.e_mean.detach().cpu().numpy()
    e_std = model.e_std.detach().cpu().numpy()
    return np.asarray(mu_std) * (e_std + 1e-8) + e_mean


def _stripe(vals, width=14):
    vals = np.asarray(vals, float).reshape(-1)
    img = np.repeat(vals[:, None], width, axis=1)
    return ma.masked_invalid(img)


def _plot_scalar_stripe(ax, vals, title, *, vmin=None, vmax=None, cmap="viridis"):
    img = _stripe(vals)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")

    im = ax.imshow(
        img,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=2, fontsize=5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _plot_region_stripe(ax, xyz_m, brain_atlas, title, mapping="Cosmos", width=16):
    xyz_m = np.asarray(xyz_m, dtype=np.float32)

    idx = brain_atlas.bc.xyz2i(xyz_m, mode="clip")
    inds = brain_atlas._lookup_inds(idx)
    rids = brain_atlas._get_mapping(mapping=mapping)[brain_atlas.label.flat[inds]]
    rids = np.asarray(rids).astype(int).reshape(-1)

    rgb = np.asarray(brain_atlas.regions.rgb[rids], dtype=float)
    if rgb.max() > 1:
        rgb = rgb / 255.0

    img = np.repeat(rgb[:, None, :3], width, axis=1)
    img[rids == 0] = 0

    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_title(title, pad=2, fontsize=5.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)


def extend_xyz_samples_to_brain(
    xyz_samples: np.ndarray,
    *,
    n_edge: int = 100,
    max_extra: int = 4096,
    brain_atlas: AllenAtlas,
    region_mapping_for_boundary: str = "Allen",
) -> np.ndarray:
    """Linearly extend a valid histology trace in both directions until leaving the brain."""
    xyz = np.asarray(xyz_samples, dtype=np.float64)
    valid = np.ones(xyz.shape[0])
    if valid.sum() < 2:
        return xyz.astype(np.float32)

    idx = np.where(valid)[0]
    xyzv = xyz[int(idx[0]): int(idx[-1]) + 1]
    if xyzv.shape[0] < 2:
        return xyz.astype(np.float32)

    n_edge = int(min(n_edge, xyzv.shape[0]))

    def first_zero_region(xarr: np.ndarray) -> bool:
        rids = region_ids_from_xyz(
            brain_atlas,
            np.atleast_2d(xarr),
            mapping=region_mapping_for_boundary,
            mode="clip",
        )
        return bool(np.any(rids == 0))

    def estimate_step(edge_xyz: np.ndarray) -> np.ndarray:
        d = edge_xyz[1:] - edge_xyz[:-1]
        mag = np.linalg.norm(d, axis=1)
        nz = mag > 0
        if np.any(nz):
            return d[nz].mean(axis=0)
        return (edge_xyz[-1] - edge_xyz[0]) / max(1, edge_xyz.shape[0] - 1)

    step_top = estimate_step(xyzv[:n_edge])
    step_bot = estimate_step(xyzv[-n_edge:])
    if np.linalg.norm(step_top) < 1e-12:
        step_top = step_bot.copy()
    if np.linalg.norm(step_bot) < 1e-12:
        step_bot = step_top.copy()
    if np.linalg.norm(step_top) < 1e-12 and np.linalg.norm(step_bot) < 1e-12:
        return xyzv.astype(np.float32)

    pre = []
    cur = xyzv[0].copy()
    for _ in range(max_extra):
        cur = cur - step_top
        if first_zero_region(cur):
            break
        pre.append(cur.copy())

    post = []
    cur = xyzv[-1].copy()
    for _ in range(max_extra):
        cur = cur + step_bot
        if first_zero_region(cur):
            break
        post.append(cur.copy())

    pre_arr = np.asarray(pre[::-1], dtype=np.float64).reshape(-1, 3)
    post_arr = np.asarray(post, dtype=np.float64).reshape(-1, 3)
    return np.concatenate([pre_arr, xyzv, post_arr], axis=0).astype(np.float32)


def prepare_auto_alignment_debug_for_pid(
    *,
    pid,
    pid_names,
    ephys,
    probe_positions,
    model,
    ctx_manager,
    handles,
    optimization_features,
    radius_um,
    m_max,
    device,
    brain_atlas=None,
    conf_model=None,
):
    if brain_atlas is None:
        brain_atlas = AllenAtlas()

    pid_names = np.asarray(pid_names).astype(str)
    idx = np.where(pid_names == str(pid))[0]
    if len(idx) == 0:
        raise ValueError(f"PID not found: {pid}")
    pidx = int(idx[0])

    recorded_full = np.asarray(ephys[pidx], dtype=np.float32)
    xyz_histology = np.asarray(probe_positions[pidx], dtype=np.float32)

    xyz_traj_ext = extend_xyz_samples_to_brain(
        xyz_histology,
        brain_atlas=brain_atlas,
    )

    out = align(
        model,
        ctx_manager,
        xyz_traj_ext,
        recorded_full,
        handles,
        optimization_features,
        radius_um,
        m_max,
        device,
        conf_model=conf_model,
        return_debug=True,
        brain_atlas=brain_atlas,
    )

    if out is None:
        raise RuntimeError(f"Alignment failed for pid={pid}")

    return {
        "pid": str(pid),
        "probe_index": pidx,
        "recorded_full": recorded_full,
        "xyz_histology": xyz_histology,
        "xyz_traj_ext": xyz_traj_ext,
        "alignment_out": out,
    }

def _panel_label_row(fig, subplot_spec, label):
    bbox = subplot_spec.get_position(fig)
    fig.text(
        bbox.x0 - 0.015,
        bbox.y1 + 0.005,
        label,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

def _nearest_fill_internal_nans_only(x):
    """
    Fill only NaNs between the first and last valid sample.
    Leave leading/trailing NaNs untouched.
    """
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)

    if good.sum() == 0:
        return x

    idx = np.arange(len(x))
    first = idx[good][0]
    last = idx[good][-1]

    internal = (idx >= first) & (idx <= last)
    fill_mask = internal & ~good

    if not fill_mask.any():
        return x

    good_idx = idx[good]
    insert = np.searchsorted(good_idx, idx)

    left = np.clip(insert - 1, 0, len(good_idx) - 1)
    right = np.clip(insert, 0, len(good_idx) - 1)

    left_idx = good_idx[left]
    right_idx = good_idx[right]

    nearest_idx = np.where(
        np.abs(idx - left_idx) <= np.abs(idx - right_idx),
        left_idx,
        right_idx,
    )

    x[fill_mask] = x[nearest_idx[fill_mask]]
    return x


def _nearest_fill_internal_nans_2d(X):
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        X[:, j] = _nearest_fill_internal_nans_only(X[:, j])
    return X


def _true_region_ids_on_full_trace(
    *,
    debug,
    out,
    brain_atlas,
    mapping="Cosmos",
):
    """
    True region ids only where recorded probe samples were mapped onto the
    extended histology trace. Leading/trailing extended trace regions remain NaN.
    """
    xyz_trace = np.asarray(out["xyz_samples_ext"])
    trace_len = len(xyz_trace)

    true_rids = np.full(trace_len, np.nan, dtype=float)

    recorded_xyz = np.asarray(debug["xyz_histology"])
    j_map = np.asarray(out["j_map_all_i"], dtype=int)

    idx = brain_atlas.bc.xyz2i(recorded_xyz, mode="clip")
    inds = brain_atlas._lookup_inds(idx)
    rids = brain_atlas._get_mapping(mapping=mapping)[brain_atlas.label.flat[inds]]
    rids = np.asarray(rids).astype(float)

    for ch, j in enumerate(j_map):
        if 0 <= j < trace_len and np.isfinite(rids[ch]):
            true_rids[j] = rids[ch]

    # Fill only internal stretch gaps; keep extended edges as NaN.
    true_rids = _nearest_fill_internal_nans_only(true_rids)
    return true_rids


def _predicted_region_ids_on_full_trace(
    *,
    out,
    brain_atlas,
    mapping="Cosmos",
):
    """
    Predicted/anatomical region ids for the full extended histological trace.
    """
    xyz_trace = np.asarray(out["xyz_samples_ext"])

    idx = brain_atlas.bc.xyz2i(xyz_trace, mode="clip")
    inds = brain_atlas._lookup_inds(idx)
    rids = brain_atlas._get_mapping(mapping=mapping)[brain_atlas.label.flat[inds]]

    return np.asarray(rids).astype(float)


def plot_auto_alignment_figure(
    *,
    debug,
    model,
    feature_list,
    brain_atlas=None,
    save_path=None,
    dpi=600,
    cosmos_accuracy=None,
    alignment_distance_um=None,
    acc_good=None,
    acc_suspicious=None,
):
    figure_style()

    if brain_atlas is None:
        brain_atlas = AllenAtlas()

    pid = debug["pid"]
    out = debug["alignment_out"]
    xyz_trace = np.asarray(out["xyz_samples_ext"])
    recorded_full = np.asarray(debug["recorded_full"])
    recorded_on_trace = np.asarray(out["recorded_on_trace_raw"])

    pred_raw = _unstandardize(out["mu_std_trace"], model)

    rms_lf_idx = _feature_idx(feature_list, "rms_lf")
    rms_ap_idx = _feature_idx(feature_list, "rms_ap")

    pred_rms_lf = pred_raw[:, rms_lf_idx]
    rec_rms_lf = recorded_full[:, rms_lf_idx]

    pred_rms_ap = pred_raw[:, rms_ap_idx]
    recorded_on_trace_internal_filled = _nearest_fill_internal_nans_2d(recorded_on_trace)

    rec_rms_ap_trace = recorded_on_trace_internal_filled[:, rms_ap_idx]
    rec_rms_lf_trace = recorded_on_trace_internal_filled[:, rms_lf_idx]

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0], 10.2)

    outer = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[0.5, 0.5, 1.0, 1.0, 1.0],
        hspace=0.4,
    )

    # ------------------------------------------------------------------
    # Panel a + b, same row
    # ------------------------------------------------------------------
    gs_top = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[0:2],
        width_ratios=[0.8, 1.0],
        wspace=0.15,
    )

    gs_a = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs_top[0, 0],
        width_ratios=[1.0, 0.5, 0.5],
        wspace=0.35,
    )

    ax_trace = fig.add_subplot(gs_a[0, 0])
    ax_pred_lf = fig.add_subplot(gs_a[0, 1])
    ax_rec_lf = fig.add_subplot(gs_a[0, 2])

    _plot_histology_trace(ax_trace, xyz_trace, brain_atlas)

    lf_vmin, lf_vmax = _safe_limits(np.r_[pred_rms_lf, rec_rms_lf])

    # Make recorded stripe shorter by placing it at the top of a longer trace-length vector.
    rec_rms_lf_top_aligned = np.full(len(pred_rms_lf), np.nan, dtype=float)
    rec_rms_lf_top_aligned[:len(rec_rms_lf)] = rec_rms_lf

    _plot_scalar_stripe(
        ax_pred_lf,
        pred_rms_lf,
        "Predicted\nRMS LF",
        vmin=lf_vmin,
        vmax=lf_vmax,
    )

    _plot_scalar_stripe(
        ax_rec_lf,
        rec_rms_lf_top_aligned,
        "Recorded\nRMS LF",
        vmin=lf_vmin,
        vmax=lf_vmax,
    )

    # Force the stripe panels to match the visible height of the brain slice panel.
    trace_pos = ax_trace.get_position()

    pos = ax_pred_lf.get_position()
    ax_pred_lf.set_position([pos.x0, trace_pos.y0 - trace_pos.height * 0.5, pos.width, 2 * trace_pos.height])
    ax_pred_lf.set_ylim(len(pred_rms_lf) - 0.5, -0.5)

    pos = ax_rec_lf.get_position()
    ax_rec_lf.set_position([pos.x0, trace_pos.y0 - trace_pos.height * 0.1, pos.width, trace_pos.height * 1.6])
    ax_rec_lf.set_ylim(len(rec_rms_lf) - 0.5, -0.5)

    ax_b = fig.add_subplot(gs_top[0, 1])
    _draw_panel_b_first_window(
        ax_b,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )
    _panel_label_row(fig, gs_top[0, 0], "a")
    _panel_label(ax_b, "b")

    # ------------------------------------------------------------------
    # Panel c
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[2])
    _draw_panel_c_shift_schematic(
        ax_c,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )
    _panel_label(ax_c, "c")

    # ------------------------------------------------------------------
    # Panel d
    # ------------------------------------------------------------------
    gs_d = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[3],
        width_ratios=[1.0, 1.5],
        wspace=0.25,
    )

    ax_cost = fig.add_subplot(gs_d[0, 0])
    _plot_cost_matrix(ax_cost, out["cost_matrix"], out.get("path"))
    _panel_label(ax_cost, "d")

    gs_stripes = GridSpecFromSubplotSpec(
        1, 9,
        subplot_spec=gs_d[0, 1],
        width_ratios=[0.75, 0.75, 0.68, 0.68, 0.68, 0.68, 0.68, 0.18, 0.18],
        wspace=0.48,
    )

    true_rids_trace = _true_region_ids_on_full_trace(
        debug=debug,
        out=out,
        brain_atlas=brain_atlas,
        mapping="Cosmos",
    )

    pred_rids_trace = _predicted_region_ids_on_full_trace(
        out=out,
        brain_atlas=brain_atlas,
        mapping="Cosmos",
    )

    ax_true_reg = fig.add_subplot(gs_stripes[0, 0])
    _plot_region_id_stripe(
        ax_true_reg,
        true_rids_trace,
        brain_atlas,
        "True\nCosmos",
    )

    ax_pred_reg = fig.add_subplot(gs_stripes[0, 1])
    _plot_region_id_stripe(
        ax_pred_reg,
        pred_rids_trace,
        brain_atlas,
        "Predicted\nCosmos",
    )

    ax_conf = fig.add_subplot(gs_stripes[0, 2])
    if out.get("cls_probs_trace") is not None:
        probs = np.asarray(out["cls_probs_trace"], dtype=float)
        conf = np.full(probs.shape[0], np.nan, dtype=float)
        valid_rows = np.isfinite(probs).any(axis=1)
        conf[valid_rows] = np.nanmax(probs[valid_rows], axis=1)
        conf = _nearest_fill_internal_nans_only(conf)
    else:
        conf = np.full(len(xyz_trace), np.nan)
    _plot_scalar_stripe(ax_conf, 2 * conf - 1, "Confidence", vmin=0, vmax=1, cmap="RdYlGn")

    ax_lf_rec = fig.add_subplot(gs_stripes[0, 3])
    ax_lf_pred = fig.add_subplot(gs_stripes[0, 4])
    _plot_scalar_stripe(ax_lf_rec, rec_rms_lf_trace, "RMS LF\nrecorded", vmin=lf_vmin, vmax=lf_vmax)
    _plot_scalar_stripe(ax_lf_pred, pred_rms_lf, "RMS LF\npredicted", vmin=lf_vmin, vmax=lf_vmax)

    ap_vmin, ap_vmax = _safe_limits(np.r_[rec_rms_ap_trace, pred_rms_ap])
    ax_ap_rec = fig.add_subplot(gs_stripes[0, 5])
    ax_ap_pred = fig.add_subplot(gs_stripes[0, 6])
    _plot_scalar_stripe(ax_ap_rec, rec_rms_ap_trace, "RMS AP\nrecorded", vmin=ap_vmin, vmax=ap_vmax)
    _plot_scalar_stripe(ax_ap_pred, pred_rms_ap, "RMS AP\npredicted", vmin=ap_vmin, vmax=ap_vmax)

    ax_dots = fig.add_subplot(gs_stripes[0, 7:])
    ax_dots.axis("off")
    ax_dots.text(0.5, 0.5, "⋯", ha="center", va="center", fontsize=14)

    # ------------------------------------------------------------------
    # Panel e
    # ------------------------------------------------------------------
    gs_e = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[4],
        width_ratios=[1, 1],
        wspace=0.28,
    )
    ax_e1 = fig.add_subplot(gs_e[0, 0])
    ax_e2 = fig.add_subplot(gs_e[0, 1])

    _plot_panel_e_histograms(
        ax_e1,
        ax_e2,
        cosmos_accuracy=cosmos_accuracy,
        alignment_distance_um=alignment_distance_um,
        acc_good=acc_good,
        acc_suspicious=acc_suspicious,
    )
    _panel_label(ax_e1, "e")

    fig.suptitle(f"Automatic ephys-based alignment | pid={pid}", y=0.995)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        return save_path

    return fig


def _plot_histology_trace(ax, xyz_trace_m, brain_atlas):
    xyz_um = _to_um(xyz_trace_m)
    coord_um = int(np.nanmean(xyz_um[:, 1]))

    empty = np.zeros((0, 3))
    plot_points_on_slice(
        empty,
        coord=coord_um,
        slice="coronal",
        ax=ax,
        cmap="Greys",
    )

    ax.plot(
        xyz_um[:, 0],
        xyz_um[:, 2],
        color="red",
        lw=1.4,
        label="histological trace",
    )

    ax.set_title("Histological trace", pad=2)
    ax.set_xlabel("ML")
    ax.set_ylabel("DV")
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)

    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    ax.set_aspect("equal", adjustable="box")


def _draw_panel_b_first_window(ax, *, pred_vec, rec_vec, step=20):
    ax.axis("off")

    pred_vec = np.asarray(pred_vec, float)
    rec_vec = np.asarray(rec_vec, float)

    n_pred = 12
    n_rec = 7

    pred_idx = np.linspace(0, min(len(pred_vec) - 1, step * (n_pred - 1)), n_pred).astype(int)
    rec_idx = np.linspace(0, min(len(rec_vec) - 1, step * (n_rec - 1)), n_rec).astype(int)

    pred_s = pred_vec[pred_idx]
    rec_s = rec_vec[rec_idx]

    vmin, vmax = _safe_limits(np.r_[pred_s, rec_s])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")

    dist = (pred_s[:n_rec] - rec_s) ** 2
    dmin, dmax = _safe_limits(dist)
    dnorm = plt.Normalize(dmin, dmax)
    dcmap = plt.get_cmap("inferno")

    # compressed vertical range
    x0 = 0.03
    y_long = 0.60
    y_short = 0.45
    w = 0.030
    h = 0.050
    gap = 0.004

    ax.text(x0, y_long + 0.085, "Predicted feature values (N)", ha="left", va="center")
    ax.text(x0, y_short + 0.085, "Recorded feature values (M)", ha="left", va="center")

    for i, val in enumerate(pred_s):
        ax.add_patch(Rectangle(
            (x0 + i * (w + gap), y_long), w, h,
            facecolor=cmap(norm(val)), edgecolor="black", lw=0.35,
        ))

    for i, val in enumerate(rec_s):
        ax.add_patch(Rectangle(
            (x0 + i * (w + gap), y_short), w, h,
            facecolor=cmap(norm(val)), edgecolor="black", lw=0.35,
        ))

    ax.plot(
        [0.55, 0.55],
        [0.0, 1.05],
        transform=ax.transAxes,
        color="0.45",
        lw=0.8,
        ls="--",
    )

    x_cmp = 0.56
    x_dist = 0.86
    y0 = 0.63
    dy = 0.065

    ax.text(x_cmp + 0.0175, 0.73, "pred", ha="center", va="center")
    ax.text(x_cmp + 0.0975, 0.73, "true", ha="center", va="center")
    ax.text(x_dist + 0.0275, 0.73, "$L_2$ \n distance", ha="center", va="center")

    for i in range(n_rec):
        y = y0 - i * dy

        ax.add_patch(Rectangle(
            (x_cmp, y), 0.035, 0.040,
            facecolor=cmap(norm(pred_s[i])), edgecolor="black", lw=0.35,
        ))
        ax.text(x_cmp + 0.055, y + 0.020, "│", ha="center", va="center")
        ax.add_patch(Rectangle(
            (x_cmp + 0.08, y), 0.035, 0.040,
            facecolor=cmap(norm(rec_s[i])), edgecolor="black", lw=0.35,
        ))

        ax.add_patch(Rectangle(
            (x_dist, y), 0.055, 0.040,
            facecolor=dcmap(dnorm(dist[i])), edgecolor="black", lw=0.35,
        ))

    ax.annotate(
        "",
        xy=(x_dist - 0.018, 0.50),
        xytext=(x_cmp + 0.150, 0.50),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=0.75,
            shrinkA=4,
            shrinkB=4,
            color="black",
        ),
    )


def _draw_panel_c_shift_schematic(ax, *, pred_vec, rec_vec, step=20):
    ax.axis("off")

    pred_vec = np.asarray(pred_vec, float)
    rec_vec = np.asarray(rec_vec, float)

    n_pred = 12
    n_rec = 7
    n_shifts = 4

    pred_idx = np.linspace(0, min(len(pred_vec) - 1, step * (n_pred - 1)), n_pred).astype(int)
    rec_idx = np.linspace(0, min(len(rec_vec) - 1, step * (n_rec - 1)), n_rec).astype(int)

    pred_s = pred_vec[pred_idx]
    rec_s = rec_vec[rec_idx]

    vmin, vmax = _safe_limits(np.r_[pred_s, rec_s])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")

    shift_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    x0 = 0.035
    y_long = 0.74
    w = 0.030
    h = 0.052
    gap = 0.004

    ax.text(x0, y_long + 0.105, "Predicted trace", ha="left", va="center")
    for i, val in enumerate(pred_s):
        ax.add_patch(Rectangle(
            (x0 + i * (w + gap), y_long), w, h,
            facecolor=cmap(norm(val)), edgecolor="black", lw=0.3,
        ))

    ax.text(x0, 0.59, "Recorded trace shifted along predicted trace", ha="left", va="center")

    for s, color in enumerate(shift_colors[:4]):
        y = 0.48 - s * 0.095
        dx = s * (w + gap)
        for i, val in enumerate(rec_s):
            ax.add_patch(Rectangle(
                (x0 + dx + i * (w + gap), y), w, h,
                facecolor=cmap(norm(val)), edgecolor=color, lw=1.0,
            ))

    ax.text(0.28, 0.12, "⋯", fontsize=12, ha="center", va="center")

    mat_x, mat_y = 0.64, 0.30
    mat_w, mat_h = 0.30, 0.43

    n_rows = n_rec
    n_cols = 8
    cell_w = mat_w / n_cols
    cell_h = mat_h / n_rows

    ax.text(
        mat_x + mat_w / 2,
        mat_y + mat_h + 0.075,
        "Cost matrix computation",
        ha="center",
        va="center",
        fontweight="bold",
    )

    for c in range(n_cols):
        for r in range(n_rows):
            ax.add_patch(Rectangle(
                (mat_x + c * cell_w, mat_y + (n_rows - 1 - r) * cell_h),
                cell_w,
                cell_h,
                facecolor="white",
                edgecolor="0.75",
                lw=0.35,
            ))

    all_d = []
    for c in range(n_shifts):
        vals = (pred_s[c:c + n_rec] - rec_s[:min(n_rec, len(pred_s[c:c + n_rec]))]) ** 2
        all_d.extend(vals.tolist())

    dmin, dmax = _safe_limits(np.asarray(all_d))
    dnorm = plt.Normalize(dmin, dmax)
    dcmap = plt.get_cmap("inferno")

    for c, color in enumerate(shift_colors[:n_shifts]):
        vals = (pred_s[c:c + n_rec] - rec_s[:min(n_rec, len(pred_s[c:c + n_rec]))]) ** 2

        for r in range(n_rows):
            x = mat_x + c * cell_w
            y = mat_y + (n_rows - 1 - r) * cell_h

            fc = "white"
            if r < len(vals):
                fc = dcmap(dnorm(vals[r]))

            ax.add_patch(Rectangle(
                (x, y),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor=color,
                lw=1.0,
            ))

    ax.text(
        mat_x + (n_shifts + 1.2) * cell_w,
        mat_y + mat_h / 2,
        "⋯",
        fontsize=14,
        ha="center",
        va="center",
    )

    for c in range(n_shifts):
        ax.text(
            mat_x + (c + 0.5) * cell_w,
            mat_y - 0.070,
            str(c),
            ha="center",
            va="center",
        )

    ax.text(
        mat_x + (n_shifts / 2) * cell_w,
        mat_y - 0.155,
        "shift [#channels]",
        ha="center",
        va="center",
    )


def _plot_cost_matrix(ax, cost_matrix, path=None):
    C = np.asarray(cost_matrix, float)
    vals = C[np.isfinite(C)]
    vmin, vmax = _safe_limits(vals)

    im = ax.imshow(
        C,
        aspect="auto",
        interpolation="nearest",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    if path is not None:
        p = np.asarray(path, int)
        if p.ndim == 2 and p.shape[1] == 2:
            ax.plot(p[:, 1], p[:, 0], color="white", lw=1.0)

    ax.set_title("Alignment cost matrix", pad=2)
    ax.set_xlabel("Histology trace position")
    ax.set_ylabel("Recorded channel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return im


def _plot_region_id_stripe(ax, rids, brain_atlas, title, width=16):
    rids = np.asarray(rids, dtype=float).reshape(-1)

    valid = np.isfinite(rids) & (rids > 0)
    rids_safe = np.zeros_like(rids, dtype=int)
    rids_safe[valid] = rids[valid].astype(int)

    rgb = np.asarray(brain_atlas.regions.rgb[rids_safe], dtype=float)
    if rgb.max() > 1:
        rgb = rgb / 255.0

    img = np.repeat(rgb[:, None, :3], width, axis=1)
    img[~valid] = 0

    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_title(title, pad=2, fontsize=5.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)


def _plot_panel_e_histograms(
    ax_acc,
    ax_dist,
    *,
    cosmos_accuracy=None,
    alignment_distance_um=None,
    acc_good=None,
    acc_suspicious=None,
):
    cosmos_accuracy = np.asarray([] if cosmos_accuracy is None else cosmos_accuracy, dtype=float).ravel()
    cosmos_accuracy = cosmos_accuracy[np.isfinite(cosmos_accuracy)]

    alignment_distance_um = np.asarray([] if alignment_distance_um is None else alignment_distance_um, dtype=float).ravel()
    alignment_distance_um = alignment_distance_um[np.isfinite(alignment_distance_um)]

    acc_good = np.asarray([] if acc_good is None else acc_good, dtype=float).ravel()
    acc_good = acc_good[np.isfinite(acc_good)]

    acc_suspicious = np.asarray([] if acc_suspicious is None else acc_suspicious, dtype=float).ravel()
    acc_suspicious = acc_suspicious[np.isfinite(acc_suspicious)]

    if acc_good.size or acc_suspicious.size:
        bins = np.linspace(0, 1, 26)

        if acc_good.size:
            m_good = float(np.nanmean(acc_good))
            ax_acc.hist(acc_good, bins=bins, alpha=0.65, label=f"high confidence, mean={m_good:.2f}")
            ax_acc.axvline(m_good, color="black", lw=1.0, ls="-")

        if acc_suspicious.size:
            m_susp = float(np.nanmean(acc_suspicious))
            ax_acc.hist(acc_suspicious, bins=bins, alpha=0.65, label=f"low confidence, mean={m_susp:.2f}")
            ax_acc.axvline(m_susp, color="black", lw=1.0, ls="--")

        ax_acc.set_xlabel("Cosmos region classification accuracy")
        ax_acc.set_ylabel("Number of probes")
        ax_acc.set_xlim(0, 1)
        ax_acc.legend(frameon=False, fontsize=5.5)

    elif cosmos_accuracy.size:
        m = float(np.nanmean(cosmos_accuracy))
        ax_acc.hist(cosmos_accuracy, bins=np.linspace(0, 1, 26))
        ax_acc.axvline(m, color="black", lw=1.0)
        ax_acc.text(0.04, 0.92, f"mean={m:.2f}", transform=ax_acc.transAxes, ha="left", va="top")
        ax_acc.set_xlabel("Cosmos region classification accuracy")
        ax_acc.set_ylabel("Number of probes")
        ax_acc.set_xlim(0, 1)

    else:
        ax_acc.text(0.5, 0.5, "Cosmos accuracy\nnot provided", ha="center", va="center")
        ax_acc.set_xticks([])
        ax_acc.set_yticks([])

    if alignment_distance_um.size:
        m_dist = float(np.nanmean(alignment_distance_um))
        ax_dist.hist(alignment_distance_um, bins=25)
        ax_dist.axvline(m_dist, color="black", lw=1.0)
        ax_dist.text(
            0.96,
            0.92,
            f"mean={m_dist:.1f} µm",
            transform=ax_dist.transAxes,
            ha="right",
            va="top",
        )
        ax_dist.set_xlabel("Mean channel error (µm)")
        ax_dist.set_ylabel("Number of probes")
    else:
        ax_dist.text(0.5, 0.5, "Alignment distance\nnot provided", ha="center", va="center")
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])

    for ax in [ax_acc, ax_dist]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def load_alignment_summary_metrics(summary_csv):
    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)

    required = [
        "region_acc_cosmos",
        "mean_abs_channel_error_um",
        "acc_good",
        "acc_suspicious",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {summary_csv}: {missing}")

    df = df.copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ok = df["ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    df_ok = df[ok].copy()

    return {
        "region_acc_cosmos": df_ok["region_acc_cosmos"].to_numpy(float),
        "mean_abs_channel_error_um": df_ok["mean_abs_channel_error_um"].to_numpy(float),
        "acc_good": df_ok["acc_good"].to_numpy(float),
        "acc_suspicious": df_ok["acc_suspicious"].to_numpy(float),
    }



def _sample_cartoon_vectors(pred_vec, rec_vec, step=20):
    pred_vec = np.asarray(pred_vec, float)
    rec_vec = np.asarray(rec_vec, float)

    n_pred = 12
    n_rec = 7

    pred_idx = np.linspace(
        0,
        min(len(pred_vec) - 1, step * (n_pred - 1)),
        n_pred,
    ).astype(int)
    rec_idx = np.linspace(
        0,
        min(len(rec_vec) - 1, step * (n_rec - 1)),
        n_rec,
    ).astype(int)

    return pred_vec[pred_idx], rec_vec[rec_idx]


def _draw_vertical_vector(
    ax,
    values,
    *,
    title,
    norm,
    cmap,
    x_center=0.5,
    width=0.32,
):
    values = np.asarray(values, float)
    n = len(values)
    cell_h = 0.72 / max(1, n)
    y_top = 0.84

    ax.text(
        x_center,
        0.96,
        title,
        ha="center",
        va="top",
    )

    for i, value in enumerate(values):
        y = y_top - (i + 1) * cell_h
        ax.add_patch(
            Rectangle(
                (x_center - width / 2, y),
                width,
                cell_h * 0.90,
                facecolor=cmap(norm(value)),
                edgecolor="black",
                lw=0.4,
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_supp_panel_a(ax, *, pred_vec, rec_vec):
    """Two vertical feature vectors shown side by side."""
    pred_s, rec_s = _sample_cartoon_vectors(pred_vec, rec_vec)
    vmin, vmax = _safe_limits(np.r_[pred_s, rec_s])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")

    ax.axis("off")

    n_max = max(len(pred_s), len(rec_s))
    cell_h = 0.72 / n_max
    y_top = 0.82
    box_w = 0.18

    ax.text(
        0.29,
        0.96,
        "Predicted feature\nvalues (N)",
        ha="center",
        va="top",
    )
    ax.text(
        0.71,
        0.96,
        "Recorded feature\nvalues (M)",
        ha="center",
        va="top",
    )

    for i, value in enumerate(pred_s):
        y = y_top - (i + 1) * cell_h
        ax.add_patch(
            Rectangle(
                (0.29 - box_w / 2, y),
                box_w,
                cell_h * 0.9,
                facecolor=cmap(norm(value)),
                edgecolor="black",
                lw=0.4,
            )
        )

    for i, value in enumerate(rec_s):
        y = y_top - (i + 1) * cell_h
        ax.add_patch(
            Rectangle(
                (0.71 - box_w / 2, y),
                box_w,
                cell_h * 0.9,
                facecolor=cmap(norm(value)),
                edgecolor="black",
                lw=0.4,
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_supp_panel_b(ax, *, pred_vec, rec_vec):
    """Elementwise L2 computation for one candidate alignment."""
    pred_s, rec_s = _sample_cartoon_vectors(pred_vec, rec_vec)
    n = min(len(rec_s), len(pred_s))
    pred_s = pred_s[:n]
    rec_s = rec_s[:n]

    vmin, vmax = _safe_limits(np.r_[pred_s, rec_s])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")

    dist = (pred_s - rec_s) ** 2
    dmin, dmax = _safe_limits(dist)
    dnorm = plt.Normalize(dmin, dmax)
    dcmap = plt.get_cmap("inferno")

    ax.axis("off")
    ax.plot(
        [0.02, 0.02],
        [0.02, 0.98],
        transform=ax.transAxes,
        color="0.45",
        lw=0.9,
        ls="--",
    )

    ax.text(0.22, 0.96, "pred", ha="center", va="top")
    ax.text(0.48, 0.96, "true", ha="center", va="top")
    ax.text(0.80, 0.96, "$L_2$\ndistance", ha="center", va="top")

    cell_h = 0.78 / max(1, n)
    y_top = 0.86

    for i in range(n):
        y = y_top - (i + 1) * cell_h

        ax.add_patch(
            Rectangle(
                (0.17, y),
                0.10,
                cell_h * 0.78,
                facecolor=cmap(norm(pred_s[i])),
                edgecolor="black",
                lw=0.35,
            )
        )
        ax.text(0.35, y + cell_h * 0.39, "−", ha="center", va="center")
        ax.add_patch(
            Rectangle(
                (0.43, y),
                0.10,
                cell_h * 0.78,
                facecolor=cmap(norm(rec_s[i])),
                edgecolor="black",
                lw=0.35,
            )
        )

        ax.annotate(
            "",
            xy=(0.69, y + cell_h * 0.39),
            xytext=(0.57, y + cell_h * 0.39),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->",
                lw=0.65,
                color="black",
            ),
        )

        ax.add_patch(
            Rectangle(
                (0.73, y),
                0.15,
                cell_h * 0.78,
                facecolor=dcmap(dnorm(dist[i])),
                edgecolor="black",
                lw=0.35,
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_supp_panel_c(ax, *, pred_vec, rec_vec, step=20):
    """
    Predicted trace and vertically shifted recorded windows.

    The shifted recorded vectors are narrower and spaced farther apart so
    their colored outlines do not overlap.
    """
    pred_s, rec_s = _sample_cartoon_vectors(pred_vec, rec_vec, step=step)
    n_pred = len(pred_s)
    n_rec = len(rec_s)
    n_shifts = min(4, max(1, n_pred - n_rec + 1))

    vmin, vmax = _safe_limits(np.r_[pred_s, rec_s])
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")
    shift_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    ax.axis("off")
    ax.text(0.20, 0.96, "Predicted trace", ha="center", va="top")
    ax.text(
        0.67,
        0.96,
        "Recorded trace shifted\nalong predicted trace",
        ha="center",
        va="top",
    )

    y_top = 0.88
    cell_h = 0.72 / max(1, n_pred)

    # Keep the predicted trace clearly visible.
    pred_w = 0.16

    # Smaller recorded vectors with wider spacing.
    rec_w = 0.085
    x_centers = np.linspace(0.47, 0.88, n_shifts)

    for i, value in enumerate(pred_s):
        y = y_top - (i + 1) * cell_h
        ax.add_patch(
            Rectangle(
                (0.20 - pred_w / 2, y),
                pred_w,
                cell_h * 0.88,
                facecolor=cmap(norm(value)),
                edgecolor="black",
                lw=0.35,
            )
        )

    for shift, (x_center, color) in enumerate(
        zip(x_centers, shift_colors[:n_shifts])
    ):
        for i, value in enumerate(rec_s):
            y_idx = i + shift
            y = y_top - (y_idx + 1) * cell_h

            ax.add_patch(
                Rectangle(
                    (x_center - rec_w / 2, y),
                    rec_w,
                    cell_h * 0.78,
                    facecolor=cmap(norm(value)),
                    edgecolor=color,
                    lw=0.9,
                )
            )

        ax.text(
            x_center,
            0.105,
            f"shift {shift}",
            rotation=90,
            ha="center",
            va="center",
            color=color,
            fontsize=5.2,
        )

    ax.text(0.68, 0.035, "⋯", fontsize=13, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_supp_panel_d(ax, *, pred_vec, rec_vec, step=20):
    """
    Cost-matrix construction with columns corresponding to shifts.

    Each constructed column is outlined using the same color assigned to
    that shift in panel c.
    """
    pred_s, rec_s = _sample_cartoon_vectors(pred_vec, rec_vec, step=step)

    n_pred = len(pred_s)
    n_rec = len(rec_s)
    n_cols = max(1, n_pred - n_rec + 1)

    matrix = np.full((n_rec, n_cols), np.nan)
    for shift in range(n_cols):
        matrix[:, shift] = (
            pred_s[shift : shift + n_rec] - rec_s
        ) ** 2

    vals = matrix[np.isfinite(matrix)]
    vmin, vmax = _safe_limits(vals)

    im = ax.imshow(
        matrix,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )

    shift_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
    ]

    # Add a colored outline around the first constructed columns.
    # Rectangle coordinates are in image data units, where each pixel spans
    # [column-0.5, column+0.5] and [row-0.5, row+0.5].
    n_colored = min(len(shift_colors), n_cols)
    for col in range(n_colored):
        ax.add_patch(
            Rectangle(
                (col - 0.5, -0.5),
                1.0,
                n_rec,
                fill=False,
                edgecolor=shift_colors[col],
                linewidth=1.5,
                zorder=5,
            )
        )

    ax.set_title("Cost matrix construction", pad=2)
    ax.set_xlabel("Shift [# channels]")
    ax.set_ylabel("Recorded feature index")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return im


def plot_auto_alignment_supplementary_figure(
    *,
    debug,
    model,
    feature_list,
    save_path=None,
    dpi=600,
):
    figure_style()

    out = debug["alignment_out"]
    recorded_full = np.asarray(debug["recorded_full"])
    pred_raw = _unstandardize(out["mu_std_trace"], model)

    rms_lf_idx = _feature_idx(feature_list, "rms_lf")
    pred_rms_lf = pred_raw[:, rms_lf_idx]
    rec_rms_lf = recorded_full[:, rms_lf_idx]

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0], 6.5)

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.2],
        hspace=0.20,
    )

    # Top row: a = vectors, b = one-shift L2 computation.
    gs_top = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[0.82, 1.18],
        wspace=0.08,
    )
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])

    _draw_supp_panel_a(
        ax_a,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )
    _draw_supp_panel_b(
        ax_b,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")

    # Bottom row: c = shifted windows, d = cost matrix.
    gs_bottom = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[1.18, 0.92],
        wspace=0.10,
    )
    ax_c = fig.add_subplot(gs_bottom[0, 0])
    ax_d = fig.add_subplot(gs_bottom[0, 1])

    _draw_supp_panel_c(
        ax_c,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )
    im = _draw_supp_panel_d(
        ax_d,
        pred_vec=pred_rms_lf,
        rec_vec=rec_rms_lf,
    )

    _panel_label(ax_c, "c")
    _panel_label(ax_d, "d")

    cbar = fig.colorbar(
        im,
        ax=ax_d,
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label(r"$L_2$ distance", labelpad=2)
    cbar.ax.tick_params(length=2, pad=1)

    fig.subplots_adjust(left=0.04, right=0.975, top=0.975, bottom=0.06)

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
        return save_path

    return fig

@dataclass
class RunConfig:
    data_dir: Path = Path("../")
    model_base_dir: Path = Path("../")

    # Released model: set to your HF repo id (or a local model directory).
    hf_repo_id: Optional[str] = "int-brain-lab/ea-encoder-channel"

    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W26"

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

def build_neighbor_handles(train_loader) -> dict:
    """Extract the train-neighbor bank from the DataLoader collate function."""
    collate = train_loader.collate_fn
    return {
        "bank_xyz": collate.bank_xyz,
        "bank_feat": collate.bank_feat,
        "bank_pid": collate.bank_pid,
        "nn_bank": collate.nn,
    }

def main():
    cfg = RunConfig(train_models=False)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.model_base_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_base_dir / f"encoding_models/{cfg.vintage}").mkdir(parents=True, exist_ok=True)

    device = cfg.device
    print(f"Using device: {device}")

    # Resolve the released channel + confidence models through load_pretrained.
    release = channel_release(cfg.hf_repo_id, cfg.vintage, device=str(device))

    # ------------------------- data/context -------------------------
    ctx_cfg = AtlasPCAConfig(n_cell_pcs=cfg.n_cell_pcs, n_gene_pcs=cfg.n_gene_pcs)
    ctx_manager = ContextAtlasManager(
        ctx_cfg,
        regenerate_context=cfg.train_models,
        output_dir=release.context_dir,
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
        split_manifest=release.split,
        preprocessing_stats=release.stats,
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

    # Weights come from the released model directory (root-level files), not the old
    # encoding_models/<vintage>/SE_model_<vintage>.pt layout.
    base_ckpt = torch.load(
        release.dir / "spatial_encoder.pt", map_location=device, weights_only=False
    )
    conf_ckpt = torch.load(
        release.dir / "confidence_model.pt", map_location=device, weights_only=False
    )
    base_model.load_state_dict(base_ckpt["model_state"])
    conf_model.load_state_dict(conf_ckpt["model_state"])

    handles = build_neighbor_handles(train_loader)

    debug = prepare_auto_alignment_debug_for_pid(
        pid=TARGET_PID,
        pid_names=pid_names,
        ephys=ephys,
        probe_positions=probe_positions,
        model=base_model,
        ctx_manager=ctx_manager,
        handles=handles,
        optimization_features=np.arange(len(FEATURE_LIST)),
        radius_um=500,
        m_max=8,
        device=device,
        brain_atlas=AllenAtlas(),
        conf_model=conf_model,  # or None
    )

    plot_auto_alignment_supplementary_figure(
        debug=debug,
        model=base_model,
        feature_list=FEATURE_LIST,
        save_path="supp_figure4_cost_matrix_construction.pdf",
    )


    print("all done")

if __name__ == "__main__":
    main()
