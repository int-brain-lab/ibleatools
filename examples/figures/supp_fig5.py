from one.api import ONE
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
from ibl_alignment_gui.loaders.histology_loader import (
    download_histology_data,
    NrrdSliceLoader,
)
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
    ProbeSequenceConfidenceTransformer,
)
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
    RegistryError,
    split_manifest_to_builder_format,
)

from torch.utils.data import DataLoader, ConcatDataset, Dataset

TARGET_PID = "5a4740a3-8e93-495d-b6cf-83dccba01d7a"

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

    min_overlap_channels = int(0.75 * int(kp_mask.sum()))
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


def _plot_scalar_stripe(
    ax,
    vals,
    title,
    *,
    vmin=None,
    vmax=None,
    cmap="viridis",
    nan_color="black",
):
    img = _stripe(vals)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

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


def _region_indices_for_xyz(
    xyz_m: np.ndarray,
    *,
    brain_atlas,
    mapping: str = "Cosmos",
) -> np.ndarray:
    """
    Return atlas region-table indices for xyz locations.

    ``brain_atlas._get_mapping`` returns indices into ``brain_atlas.regions``;
    these are exactly what ``_plot_region_id_stripe`` expects when indexing
    ``brain_atlas.regions.rgb``.
    """
    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    idx = brain_atlas.bc.xyz2i(xyz_m, mode="clip")
    inds = brain_atlas._lookup_inds(idx)
    mapped = brain_atlas._get_mapping(mapping=mapping)[
        brain_atlas.label.flat[inds]
    ]
    return np.asarray(mapped, dtype=float).reshape(-1)


def _true_region_ids_on_full_trace(
    *,
    out,
    brain_atlas,
    mapping="Cosmos",
):
    """
    Ground-truth Cosmos anatomy along the ENTIRE histological trace.

    The histological trace itself defines the ground-truth anatomical path, so
    every position in ``xyz_samples_ext`` receives its atlas region identity.
    """
    return _region_indices_for_xyz(
        np.asarray(out["xyz_samples_ext"]),
        brain_atlas=brain_atlas,
        mapping=mapping,
    )


def _predicted_region_ids_on_probe_only(
    *,
    out,
    brain_atlas,
    mapping="Cosmos",
):
    """
    Predicted Cosmos regions only over the aligned Neuropixels probe length.

    ``est_xyz`` contains one predicted anatomical location per recording
    channel. Those region identities are scattered back onto the full
    histological-trace coordinate axis using ``j_map_all_i``. Positions outside
    the predicted probe extent remain NaN, so the stripe is blank there.
    """
    trace_len = len(out["xyz_samples_ext"])
    j_map = np.asarray(out["j_map_all_i"], dtype=int)
    est_xyz = np.asarray(out["est_xyz"], dtype=np.float32)

    pred_per_channel = _region_indices_for_xyz(
        est_xyz,
        brain_atlas=brain_atlas,
        mapping=mapping,
    )

    pred_on_trace = np.full(trace_len, np.nan, dtype=float)

    # If local DTW stretching maps multiple recording channels to the same
    # histology position, all such channels necessarily share the same xyz
    # sample. Assigning repeatedly is therefore safe.
    for ch, j in enumerate(j_map):
        if 0 <= j < trace_len and np.isfinite(pred_per_channel[ch]):
            pred_on_trace[j] = pred_per_channel[ch]

    # Fill gaps only BETWEEN the first and last predicted probe positions.
    # Leading/trailing histological-trace positions intentionally stay NaN.
    pred_on_trace = _nearest_fill_internal_nans_only(pred_on_trace)
    return pred_on_trace



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
    """
    Main Figure 6.

    Panel a is a compact overview cartoon:
      histological trace -> predicted RMS LF -> recorded RMS LF
    with vertical arrows indicating that the recorded trace is shifted to
    find the optimal alignment.

    The former panels d/e are retained and relabeled b/c.
    """
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
    recorded_on_trace_internal_filled = _nearest_fill_internal_nans_2d(
        recorded_on_trace
    )
    rec_rms_ap_trace = recorded_on_trace_internal_filled[:, rms_ap_idx]
    rec_rms_lf_trace = recorded_on_trace_internal_filled[:, rms_lf_idx]

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0], 8.4)

    outer = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.15, 1.35, 1.0],
        hspace=0.32,
    )

    # ==============================================================
    # Panel a: overview cartoon
    # ==============================================================
    gs_a = GridSpecFromSubplotSpec(
        1,
        6,
        subplot_spec=outer[0],
        width_ratios=[1.35, 0.18, 0.55, 0.16, 0.55, 0.15],
        wspace=0.12,
    )

    ax_trace = fig.add_subplot(gs_a[0, 0])
    ax_arrow1 = fig.add_subplot(gs_a[0, 1])
    ax_pred = fig.add_subplot(gs_a[0, 2])
    ax_arrow2 = fig.add_subplot(gs_a[0, 3])
    ax_rec = fig.add_subplot(gs_a[0, 4])
    ax_shift = fig.add_subplot(gs_a[0, 5])

    _plot_histology_trace(ax_trace, xyz_trace, brain_atlas)

    lf_vmin, lf_vmax = _safe_limits(np.r_[pred_rms_lf, rec_rms_lf])

    rec_rms_lf_top_aligned = np.full(len(pred_rms_lf), np.nan, dtype=float)
    rec_rms_lf_top_aligned[: len(rec_rms_lf)] = rec_rms_lf

    _plot_scalar_stripe(
        ax_pred,
        pred_rms_lf,
        "Predicted RMS LF",
        vmin=lf_vmin,
        vmax=lf_vmax,
    )
    _plot_scalar_stripe(
        ax_rec,
        rec_rms_lf_top_aligned,
        "Recorded RMS LF",
        vmin=lf_vmin,
        vmax=lf_vmax,
        nan_color="white",
    )

    for ax in (ax_pred, ax_rec):
        ax.set_anchor("C")

    for ax in (ax_arrow1, ax_arrow2):
        ax.axis("off")
        ax.annotate(
            "",
            xy=(0.96, 0.5),
            xytext=(0.04, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.6,
                mutation_scale=13,
                color="black",
            ),
        )

    ax_shift.axis("off")
    ax_shift.annotate(
        "",
        xy=(0.5, 0.92),
        xytext=(0.5, 0.58),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.25,
            mutation_scale=11,
            color="black",
        ),
    )
    ax_shift.annotate(
        "",
        xy=(0.5, 0.08),
        xytext=(0.5, 0.42),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.25,
            mutation_scale=11,
            color="black",
        ),
    )
    ax_shift.text(
        0.5,
        0.5,
        "shift",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6,
    )

    _panel_label(ax_trace, "a")

    # ==============================================================
    # Panel b: alignment result / feature stripes
    # ==============================================================
    gs_b = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[1.0, 1.55],
        wspace=0.25,
    )

    ax_cost = fig.add_subplot(gs_b[0, 0])
    _plot_cost_matrix(ax_cost, out["cost_matrix"], out.get("path"))
    _panel_label(ax_cost, "b")

    gs_stripes = GridSpecFromSubplotSpec(
        1,
        9,
        subplot_spec=gs_b[0, 1],
        width_ratios=[0.75, 0.75, 0.68, 0.68, 0.68, 0.68, 0.68, 0.18, 0.18],
        wspace=0.48,
    )

    # Ground truth spans the complete histological trace; the prediction is
    # displayed only over the aligned physical probe extent.
    true_rids_trace = _true_region_ids_on_full_trace(
        out=out,
        brain_atlas=brain_atlas,
        mapping="Cosmos",
    )
    pred_rids_trace = _predicted_region_ids_on_probe_only(
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

    _plot_scalar_stripe(
        ax_conf,
        2 * conf - 1,
        "Confidence",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )

    ax_lf_rec = fig.add_subplot(gs_stripes[0, 3])
    ax_lf_pred = fig.add_subplot(gs_stripes[0, 4])
    _plot_scalar_stripe(
        ax_lf_rec,
        rec_rms_lf_trace,
        "RMS LF\nrecorded",
        vmin=lf_vmin,
        vmax=lf_vmax,
    )
    _plot_scalar_stripe(
        ax_lf_pred,
        pred_rms_lf,
        "RMS LF\npredicted",
        vmin=lf_vmin,
        vmax=lf_vmax,
    )

    ap_vmin, ap_vmax = _safe_limits(np.r_[rec_rms_ap_trace, pred_rms_ap])
    ax_ap_rec = fig.add_subplot(gs_stripes[0, 5])
    ax_ap_pred = fig.add_subplot(gs_stripes[0, 6])
    _plot_scalar_stripe(
        ax_ap_rec,
        rec_rms_ap_trace,
        "RMS AP\nrecorded",
        vmin=ap_vmin,
        vmax=ap_vmax,
    )
    _plot_scalar_stripe(
        ax_ap_pred,
        pred_rms_ap,
        "RMS AP\npredicted",
        vmin=ap_vmin,
        vmax=ap_vmax,
    )

    ax_dots = fig.add_subplot(gs_stripes[0, 7:])
    ax_dots.axis("off")
    ax_dots.text(0.5, 0.5, "⋯", ha="center", va="center", fontsize=14)

    # ==============================================================
    # Panel c: held-out test-set alignment evaluation
    # ==============================================================
    gs_c = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[2],
        width_ratios=[1, 1],
        wspace=0.28,
    )
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    _plot_panel_e_histograms(
        ax_c1,
        ax_c2,
        cosmos_accuracy=cosmos_accuracy,
        alignment_distance_um=alignment_distance_um,
        acc_good=acc_good,
        acc_suspicious=acc_suspicious,
    )
    _panel_label(ax_c1, "c")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.06)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
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


def compute_single_alignment_metrics(
    *,
    debug,
    brain_atlas,
    confidence_threshold: float = 0.5,
    mapping: str = "Cosmos",
):
    """
    Compute Figure 6 evaluation quantities directly from the alignment that was
    just performed for TARGET_PID.

    No CSV or precomputed alignment result is read.

    Returns
    -------
    dict with:
        region_acc_cosmos:
            one-element array containing channel-wise Cosmos accuracy
        mean_abs_channel_error_um:
            one-element array containing mean Euclidean channel localization error
        acc_good:
            one-element array containing Cosmos accuracy among channels classified
            as high-confidence by the confidence model, when available
        acc_suspicious:
            one-element array containing Cosmos accuracy among low-confidence
            channels, when available
    """
    out = debug["alignment_out"]

    true_xyz = np.asarray(debug["xyz_histology"], dtype=np.float32)
    pred_xyz = np.asarray(out["est_xyz"], dtype=np.float32)

    if true_xyz.shape != pred_xyz.shape:
        raise ValueError(
            f"True/predicted channel xyz shape mismatch: "
            f"{true_xyz.shape} vs {pred_xyz.shape}"
        )

    valid_true = np.isfinite(true_xyz).all(axis=1)
    valid_true &= ~np.all(true_xyz == 0.0, axis=1)

    valid_pred = np.isfinite(pred_xyz).all(axis=1)
    valid = valid_true & valid_pred

    if valid.sum() == 0:
        raise RuntimeError(
            "No valid channels are available for evaluating the target PID alignment."
        )

    # --------------------------------------------------------------
    # Channel localization error.
    # --------------------------------------------------------------
    channel_error_um = (
        np.linalg.norm(pred_xyz[valid] - true_xyz[valid], axis=1) * 1e6
    )
    mean_channel_error_um = float(np.nanmean(channel_error_um))

    # --------------------------------------------------------------
    # Cosmos-region accuracy.
    #
    # Use region_ids_from_xyz here rather than the plotting helper because
    # this function returns anatomical IDs suitable for direct comparison.
    # --------------------------------------------------------------
    true_cosmos = np.asarray(
        region_ids_from_xyz(
            brain_atlas,
            true_xyz,
            mapping=mapping,
            mode="clip",
        )
    ).reshape(-1)

    pred_cosmos = np.asarray(
        region_ids_from_xyz(
            brain_atlas,
            pred_xyz,
            mapping=mapping,
            mode="clip",
        )
    ).reshape(-1)

    valid_region = valid.copy()
    valid_region &= np.isfinite(true_cosmos)
    valid_region &= np.isfinite(pred_cosmos)
    valid_region &= true_cosmos != 0
    valid_region &= pred_cosmos != 0

    if valid_region.any():
        region_correct = pred_cosmos == true_cosmos
        cosmos_acc = float(np.mean(region_correct[valid_region]))
    else:
        region_correct = np.zeros(len(true_xyz), dtype=bool)
        cosmos_acc = np.nan

    # --------------------------------------------------------------
    # Confidence-stratified accuracy for this same probe.
    #
    # The confidence transformer uses class 0 as "good"; therefore p(good)
    # is cls_probs_est[:, 0]. Channels at/above confidence_threshold are
    # treated as high-confidence.
    # --------------------------------------------------------------
    acc_good = np.nan
    acc_suspicious = np.nan

    probs_est = out.get("cls_probs_est")
    if probs_est is not None:
        probs_est = np.asarray(probs_est, dtype=float)

        if probs_est.ndim != 2 or probs_est.shape[0] != len(true_xyz):
            raise ValueError(
                "cls_probs_est has unexpected shape: "
                f"{probs_est.shape}; expected [n_channels, n_classes]."
            )

        p_good = probs_est[:, 0]
        valid_conf = valid_region & np.isfinite(p_good)

        high = valid_conf & (p_good >= float(confidence_threshold))
        low = valid_conf & (p_good < float(confidence_threshold))

        if high.any():
            acc_good = float(np.mean(region_correct[high]))
        if low.any():
            acc_suspicious = float(np.mean(region_correct[low]))

    print("\n[Figure 6 single-alignment evaluation]")
    print(f"PID: {debug['pid']}")
    print(f"Valid channels: {int(valid.sum())}/{len(valid)}")
    print(f"Cosmos accuracy: {cosmos_acc:.3f}")
    print(f"Mean channel error: {mean_channel_error_um:.1f} um")
    if np.isfinite(acc_good):
        print(
            f"High-confidence Cosmos accuracy "
            f"(p_good >= {confidence_threshold:.2f}): {acc_good:.3f}"
        )
    if np.isfinite(acc_suspicious):
        print(
            f"Low-confidence Cosmos accuracy "
            f"(p_good < {confidence_threshold:.2f}): {acc_suspicious:.3f}"
        )

    return {
        "region_acc_cosmos": np.asarray([cosmos_acc], dtype=float),
        "mean_abs_channel_error_um": np.asarray(
            [mean_channel_error_um], dtype=float
        ),
        "acc_good": (
            np.asarray([acc_good], dtype=float)
            if np.isfinite(acc_good)
            else np.asarray([], dtype=float)
        ),
        "acc_suspicious": (
            np.asarray([acc_suspicious], dtype=float)
            if np.isfinite(acc_suspicious)
            else np.asarray([], dtype=float)
        ),
        # Useful raw values if you later want a different panel-c summary.
        "channel_error_um": channel_error_um,
        "true_cosmos": true_cosmos,
        "pred_cosmos": pred_cosmos,
        "valid_region_mask": valid_region,
    }



def _extract_test_pids_from_split_manifest(
    split_manifest: dict,
    *,
    all_pid_names: Optional[list[str]] = None,
) -> list[str]:
    """
    Extract held-out test PIDs from the split manifest stored with the release.

    ``split_manifest_to_builder_format`` has changed slightly across versions,
    so this helper accepts the common layouts rather than hard-coding one
    schema. The release manifest remains the source of truth.

    Supported examples include:
        {"test_pids": [...]}
        {"test": [...]}
        {"pids_test": [...]}
        {"splits": {"test": [...]}}
        {"test": {"pids": [...]}}
        {"pid_to_split": {pid: "train"/"val"/"test"}}
    """
    if split_manifest is None:
        raise RuntimeError("Release does not contain a split manifest.")

    def _as_pid_list(value):
        if value is None:
            return None

        if isinstance(value, dict):
            for key in ("pids", "pid", "ids"):
                if key in value:
                    return _as_pid_list(value[key])
            return None

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]

        return None

    # Direct/common keys.
    for key in (
        "test_pids",
        "pids_test",
        "test",
        "heldout_pids",
        "held_out_pids",
    ):
        if key in split_manifest:
            pids = _as_pid_list(split_manifest[key])
            if pids is not None:
                return sorted(set(pids))

    # Nested split dictionary.
    splits = split_manifest.get("splits")
    if isinstance(splits, dict):
        for key in ("test", "heldout", "held_out"):
            if key in splits:
                pids = _as_pid_list(splits[key])
                if pids is not None:
                    return sorted(set(pids))

    # PID -> split label mapping.
    for key in ("pid_to_split", "split_by_pid", "pid_split"):
        mapping = split_manifest.get(key)
        if isinstance(mapping, dict):
            pids = [
                str(pid)
                for pid, split_name in mapping.items()
                if str(split_name).lower() in ("test", "heldout", "held_out")
            ]
            if pids:
                return sorted(set(pids))

    # Some builder-format manifests store index arrays. Convert them only if
    # all_pid_names was supplied.
    if all_pid_names is not None:
        for key in ("test_indices", "test_idx", "idx_test"):
            if key in split_manifest:
                idx = np.asarray(split_manifest[key], dtype=int).reshape(-1)
                all_pid_names = np.asarray(all_pid_names).astype(str)
                if np.any(idx < 0) or np.any(idx >= len(all_pid_names)):
                    raise RuntimeError(
                        f"{key} contains indices outside the PID array."
                    )
                return sorted(set(all_pid_names[idx].tolist()))

    raise RuntimeError(
        "Could not identify held-out test PIDs from the released split manifest. "
        f"Top-level keys are: {sorted(split_manifest.keys())}"
    )


def evaluate_heldout_test_set_alignments(
    *,
    test_pids,
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
    brain_atlas,
    conf_model,
    confidence_threshold: float = 0.5,
    mapping: str = "Cosmos",
    fail_fast: bool = False,
):
    """
    Run the complete automatic alignment independently for every held-out test
    probe and aggregate the per-probe metrics used in Figure 6 panel c.

    Importantly, this does NOT reuse any previously saved alignment results.
    Every PID is aligned from its recorded electrophysiological features against
    the atlas prediction using the same ``align`` function used for panel b.

    Returns
    -------
    metrics : dict
        Arrays with one value per successfully evaluated test probe:
            region_acc_cosmos
            mean_abs_channel_error_um
            acc_good
            acc_suspicious
        plus ``per_probe`` and ``failed`` diagnostic lists.
    """
    pid_names_arr = np.asarray(pid_names).astype(str)
    available = set(pid_names_arr.tolist())

    test_pids = [str(pid) for pid in test_pids]
    missing = [pid for pid in test_pids if pid not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)} test PIDs from the released split are absent from "
            f"the loaded dataset. Examples: {missing[:5]}"
        )

    region_acc = []
    mean_errors = []
    acc_good = []
    acc_suspicious = []
    per_probe = []
    failed = []

    n_total = len(test_pids)
    print(f"\n[Figure 6] aligning {n_total} held-out test probes...")

    for i, pid in enumerate(test_pids, start=1):
        print(f"[Figure 6 test alignment] {i}/{n_total} PID={pid}")

        try:
            debug = prepare_auto_alignment_debug_for_pid(
                pid=pid,
                pid_names=pid_names_arr,
                ephys=ephys,
                probe_positions=probe_positions,
                model=model,
                ctx_manager=ctx_manager,
                handles=handles,
                optimization_features=optimization_features,
                radius_um=radius_um,
                m_max=m_max,
                device=device,
                brain_atlas=brain_atlas,
                conf_model=conf_model,
            )

            one = compute_single_alignment_metrics(
                debug=debug,
                brain_atlas=brain_atlas,
                confidence_threshold=confidence_threshold,
                mapping=mapping,
            )

            cosmos_acc = (
                float(one["region_acc_cosmos"][0])
                if len(one["region_acc_cosmos"])
                else np.nan
            )
            mean_err = (
                float(one["mean_abs_channel_error_um"][0])
                if len(one["mean_abs_channel_error_um"])
                else np.nan
            )
            good_acc = (
                float(one["acc_good"][0])
                if len(one["acc_good"])
                else np.nan
            )
            suspicious_acc = (
                float(one["acc_suspicious"][0])
                if len(one["acc_suspicious"])
                else np.nan
            )

            region_acc.append(cosmos_acc)
            mean_errors.append(mean_err)

            if np.isfinite(good_acc):
                acc_good.append(good_acc)
            if np.isfinite(suspicious_acc):
                acc_suspicious.append(suspicious_acc)

            per_probe.append(
                {
                    "pid": pid,
                    "region_acc_cosmos": cosmos_acc,
                    "mean_abs_channel_error_um": mean_err,
                    "acc_good": good_acc,
                    "acc_suspicious": suspicious_acc,
                }
            )

        except Exception as exc:
            failed.append({"pid": pid, "error": repr(exc)})
            print(f"[Figure 6] WARNING: alignment failed for PID={pid}: {exc}")
            if fail_fast:
                raise

    if len(per_probe) == 0:
        raise RuntimeError(
            "Alignment failed for every held-out test PID; panel c cannot be generated."
        )

    region_acc = np.asarray(region_acc, dtype=float)
    mean_errors = np.asarray(mean_errors, dtype=float)
    acc_good = np.asarray(acc_good, dtype=float)
    acc_suspicious = np.asarray(acc_suspicious, dtype=float)

    print("\n[Figure 6 held-out alignment summary]")
    print(f"Requested test probes: {n_total}")
    print(f"Successful alignments: {len(per_probe)}")
    print(f"Failed alignments: {len(failed)}")
    if np.isfinite(region_acc).any():
        print(
            f"Mean Cosmos accuracy: "
            f"{np.nanmean(region_acc):.3f}"
        )
    if np.isfinite(mean_errors).any():
        print(
            f"Mean probe-wise channel error: "
            f"{np.nanmean(mean_errors):.1f} um"
        )
    if acc_good.size:
        print(
            f"Mean high-confidence Cosmos accuracy: "
            f"{np.nanmean(acc_good):.3f}"
        )
    if acc_suspicious.size:
        print(
            f"Mean low-confidence Cosmos accuracy: "
            f"{np.nanmean(acc_suspicious):.3f}"
        )

    return {
        "region_acc_cosmos": region_acc,
        "mean_abs_channel_error_um": mean_errors,
        "acc_good": acc_good,
        "acc_suspicious": acc_suspicious,
        "per_probe": per_probe,
        "failed": failed,
        "test_pids": test_pids,
    }


# =============================================================================
# Released model loading (local registry -> Hugging Face fallback)
# =============================================================================

def _apply_release_config_to_cfg(cfg, release_config: dict) -> None:
    """
    Make the released model authoritative for context, neighborhood, and data
    identifiers. This prevents Figure 6 from silently using settings that differ
    from the uploaded model.
    """
    data_cfg = release_config.get("data", {})
    context_cfg = release_config.get("context", {})
    channel_cfg = release_config.get("channel_level", {})
    neighbors_cfg = channel_cfg.get("neighbors", {})

    saved_vintage = str(data_cfg.get("vintage", cfg.vintage))
    if saved_vintage != str(cfg.vintage):
        raise RegistryError(
            f"Requested vintage={cfg.vintage!r}, but release config contains "
            f"vintage={saved_vintage!r}."
        )

    cfg.project = str(data_cfg.get("project", cfg.project))
    cfg.agg = str(data_cfg.get("agg", cfg.agg))
    cfg.n_cell_pcs = int(context_cfg.get("n_cell_pcs", cfg.n_cell_pcs))
    cfg.n_gene_pcs = int(context_cfg.get("n_gene_pcs", cfg.n_gene_pcs))
    cfg.radius_um = int(neighbors_cfg.get("radius_um", cfg.radius_um))
    cfg.m_max = int(neighbors_cfg.get("m_max", cfg.m_max))


def resolve_alignment_release(
    *,
    cfg,
    device: torch.device,
):
    """
    Resolve Figure 6's released channel models.

    Resolution order:
      1. local Ephys Atlas model registry
      2. Hugging Face ``hf_repo_id`` at revision/tag ``cfg.vintage``

    Returns the release artifacts needed to reconstruct the exact preprocessing,
    data split, context PCA volumes, interpolation model, and confidence model.
    """
    registry = EphysAtlasReleaseRegistry(cfg.registry_root)

    release_dir = registry.resolve_release(
        cfg.vintage,
        repo_id=cfg.hf_repo_id,
        token=cfg.hf_token,
        require_weights=True,
    )
    registry.verify_checksums(cfg.vintage)
    registry.validate_feature_order(cfg.vintage, FEATURE_LIST)

    release_config = registry.load_config(cfg.vintage)
    release_features = registry.load_features(cfg.vintage)
    split_manifest = split_manifest_to_builder_format(
        registry.load_split(cfg.vintage)
    )
    preprocessing_stats = registry.load_channel_preprocessing_stats(cfg.vintage)

    _apply_release_config_to_cfg(cfg, release_config)

    base_path = release_dir / "models" / "channel" / "spatial_encoder.pt"
    conf_path = release_dir / "models" / "channel" / "confidence_model.pt"

    if not base_path.exists():
        raise FileNotFoundError(f"Released spatial encoder not found: {base_path}")
    if not conf_path.exists():
        raise FileNotFoundError(f"Released confidence model not found: {conf_path}")

    base_ckpt = torch.load(
        base_path,
        map_location=device,
        weights_only=False,
    )
    conf_ckpt = torch.load(
        conf_path,
        map_location=device,
        weights_only=False,
    )

    print(f"[figure 6] resolved release vintage={cfg.vintage}")
    print(f"[figure 6] release directory: {release_dir}")

    return {
        "registry": registry,
        "release_dir": release_dir,
        "features": release_features,
        "config": release_config,
        "split_manifest": split_manifest,
        "preprocessing_stats": preprocessing_stats,
        "base_ckpt": base_ckpt,
        "conf_ckpt": conf_ckpt,
    }


def construct_released_models(
    *,
    release,
    device: torch.device,
):
    """
    Construct both channel-level models directly from checkpoint architecture
    metadata and frozen preprocessing statistics.
    """
    stats = release["preprocessing_stats"]
    base_ckpt = release["base_ckpt"]
    conf_ckpt = release["conf_ckpt"]

    base_arch = base_ckpt.get("architecture", {})
    required_base = (
        "f_ctx", "f_ephys", "f_out", "d_model", "nhead", "depth", "drop"
    )
    missing = [key for key in required_base if key not in base_arch]
    if missing:
        raise RuntimeError(
            f"Released spatial encoder is missing architecture fields: {missing}"
        )

    def stat_tensor(name: str) -> torch.Tensor:
        if name not in stats:
            raise RuntimeError(
                f"Release preprocessing/channel_stats.npz is missing {name!r}"
            )
        return torch.as_tensor(stats[name], dtype=torch.float32)

    base_model = NeighborInpaintingModel(
        f_ctx=int(base_arch["f_ctx"]),
        f_ephys=int(base_arch["f_ephys"]),
        f_out=int(base_arch["f_out"]),
        e_mean=stat_tensor("e_mean"),
        e_std=stat_tensor("e_std"),
        ctx_mean=stat_tensor("ctx_mean"),
        ctx_std=stat_tensor("ctx_std"),
        d_model=int(base_arch["d_model"]),
        nhead=int(base_arch["nhead"]),
        depth=int(base_arch["depth"]),
        drop=float(base_arch["drop"]),
    ).to(device)
    base_model.load_state_dict(base_ckpt["model_state"], strict=True)
    base_model.eval()

    conf_arch = conf_ckpt.get("architecture", {})
    required_conf = (
        "f_ctx", "f_e", "d_model", "nhead", "depth", "mlp_ratio", "drop"
    )
    missing = [key for key in required_conf if key not in conf_arch]
    if missing:
        raise RuntimeError(
            f"Released confidence model is missing architecture fields: {missing}"
        )

    conf_model = ProbeSequenceConfidenceTransformer(
        f_ctx=int(conf_arch["f_ctx"]),
        f_e=int(conf_arch["f_e"]),
        d_model=int(conf_arch["d_model"]),
        nhead=int(conf_arch["nhead"]),
        depth=int(conf_arch["depth"]),
        mlp_ratio=float(conf_arch["mlp_ratio"]),
        drop=float(conf_arch["drop"]),
    ).to(device)
    conf_model.load_state_dict(conf_ckpt["model_state"], strict=True)
    conf_model.eval()

    return base_model, conf_model


@dataclass
class RunConfig:
    # Raw/table data cache. Kept separate from the model registry.
    data_dir: Path = Path("../")

    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W26"

    # Released-model registry / Hugging Face interface.
    registry_root: Path = DEFAULT_REGISTRY_ROOT
    hf_repo_id: Optional[str] = "AlonSaguy/ephys-atlas-models"
    hf_token: Optional[str] = None

    # Context / neighborhood defaults. For a released model these are replaced
    # by the authoritative values saved in config.json.
    n_cell_pcs: int = 50
    n_gene_pcs: int = 50
    radius_um: int = 500
    m_max: int = 8

    batch_size_train: int = 1024
    batch_size_eval: int = 1024

    # Confidence threshold used for the panel-c high/low-confidence split.
    # The confidence transformer's class 0 is "good".
    confidence_threshold: float = 0.5

    # If False, a problematic held-out probe is reported and skipped. If True,
    # the script stops immediately on the first alignment failure.
    fail_fast_test_alignment: bool = False

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


# =============================================================================
# Supplementary Figure 5: misaligned-probe correction example
# =============================================================================

SUPP5_FEATURE_GROUPS = {
    "LFP": [
        "rms_lf",
        "psd_alpha",
        "psd_gamma",
    ],
    "AP": [
        "rms_ap",
        "alpha_mean",
        "alpha_std",
    ],
    "Spike detection": [
        "repolarisation_slope",
        "peak_val",
        "peak_time_secs",
    ],
}

SUPP5_DISPLAY_NAMES = {
    "rms_lf": "RMS LF",
    "psd_alpha": "PSD alpha",
    "psd_gamma": "PSD gamma",
    "rms_ap": "RMS AP",
    "alpha_mean": "Alpha mean",
    "alpha_std": "Alpha std",
    "repolarisation_slope": "Repolarization",
    "peak_val": "Peak value",
    "peak_time_secs": "Peak time",
}


def _resolve_lab_name(session: dict) -> str:
    lab = session.get("lab")
    if isinstance(lab, dict):
        name = lab.get("name") or lab.get("nickname")
        if name:
            return str(name)
    if isinstance(lab, str):
        return lab.rstrip("/").split("/")[-1]
    raise RuntimeError(f"Could not determine laboratory from session: {lab}")


def _densify_xyz_track(
    xyz_m: np.ndarray,
    *,
    spacing_um: float = 25.0,
) -> np.ndarray:
    """Resample a 3-D polyline at approximately uniform spacing."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N, 3), got {xyz.shape}")
    if len(xyz) < 2:
        return xyz.copy()

    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(seg)]
    total = float(cumulative[-1])
    if total <= 0:
        return xyz.copy()

    n = max(2, int(np.ceil(total / (spacing_um * 1e-6))) + 1)
    d = np.linspace(0.0, total, n)
    return np.column_stack(
        [np.interp(d, cumulative, xyz[:, k]) for k in range(3)]
    ).astype(np.float32)


def _robust_normalize_gray(
    image: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x, dtype=np.float32)

    vals = x[good]
    lo = float(np.percentile(vals, low_percentile))
    hi = float(np.percentile(vals, high_percentile))
    if not np.isfinite(lo):
        lo = float(np.nanmin(vals))
    if not np.isfinite(hi):
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0

    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    x[~good] = 0.0
    return x


def load_registered_histology_for_pid(
    *,
    pid: str,
    one: ONE,
    brain_atlas: AllenAtlas,
):
    """
    Resolve the PID to subject/lab, download the registered IBL histology
    volumes when not already cached, and extract the red-channel plane passing
    through the manually traced probe trajectory.
    """
    insertion = one.alyx.rest("insertions", "read", id=str(pid))
    if insertion is None:
        raise RuntimeError(f"Could not retrieve insertion PID={pid}")

    eid = insertion["session"]
    session = one.alyx.rest("sessions", "read", id=eid)

    subject = str(session["subject"])
    laboratory = _resolve_lab_name(session)
    probe_name = str(insertion.get("name", ""))

    ins_json = insertion.get("json") or {}
    xyz_picks = ins_json.get("xyz_picks")
    if xyz_picks is None:
        raise RuntimeError(
            f"Insertion {pid} has no xyz_picks; a traced histology trajectory is required."
        )

    # Alyx stores picks in micrometres.
    xyz_picks_m = np.asarray(xyz_picks, dtype=np.float64) * 1e-6
    xyz_dense_m = _densify_xyz_track(xyz_picks_m, spacing_um=25.0)

    print(
        f"[supp fig 5] histology: subject={subject}, lab={laboratory}, "
        f"probe={probe_name}; {len(xyz_picks_m)} picks -> "
        f"{len(xyz_dense_m)} dense samples"
    )

    paths, histology_dir = download_histology_data(
        subject=subject,
        laboratory=laboratory,
    )
    if paths is None:
        raise RuntimeError(
            f"No registered histology found for subject={subject}, lab={laboratory}"
        )

    histology_dir = Path(histology_dir)
    loader = NrrdSliceLoader(histology_dir, brain_atlas)
    slices = loader.get_slices(xyz_dense_m)

    red_info = slices.get("Histology red")
    if red_info is None:
        raise RuntimeError(
            f"No red histology channel was returned. Available slices: {list(slices)}"
        )

    red = np.asarray(red_info["slice"], dtype=np.float32)
    red = _robust_normalize_gray(red)

    # Requested visual orientation:
    #   1. rotate 90 degrees clockwise
    #   2. flip the displayed y direction
    #   3. flip left-right
    red = np.rot90(red, k=-1)
    red = np.flipud(red)
    red = np.fliplr(red)

    return {
        "pid": str(pid),
        "eid": str(eid),
        "subject": subject,
        "laboratory": laboratory,
        "probe_name": probe_name,
        "xyz_picks_m": xyz_picks_m.astype(np.float32),
        "xyz_dense_m": xyz_dense_m,
        "histology_dir": histology_dir,
        "red": red,
        "slices": slices,
    }


def _valid_xyz_mask(xyz) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    valid &= ~np.all(xyz == 0.0, axis=1)
    return valid


def _valid_xyz(xyz) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz[_valid_xyz_mask(xyz)]


def _top_to_bottom_order(xyz_m: np.ndarray) -> np.ndarray:
    """
    Return indices that order a probe trajectory from dorsal/top to
    ventral/bottom in Allen world coordinates.

    For the coronal plots used here the vertical axis is z, so sorting by
    decreasing z gives top -> bottom.  The same permutation must be applied to
    any per-channel feature array paired with these xyz coordinates.
    """
    xyz = np.asarray(xyz_m, dtype=np.float32)
    valid = _valid_xyz_mask(xyz)

    idx_valid = np.flatnonzero(valid)
    idx_invalid = np.flatnonzero(~valid)

    if len(idx_valid) == 0:
        return np.arange(len(xyz), dtype=int)

    # Stable sort so ties preserve the original channel order.
    local = np.argsort(-xyz[idx_valid, 2], kind="stable")
    return np.concatenate([idx_valid[local], idx_invalid]).astype(int)


def _order_xyz_top_to_bottom(xyz_m: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float32)
    order = _top_to_bottom_order(xyz)
    return xyz[order]


def _slice_coord_um(*xyz_arrays) -> int:
    vals = []
    for xyz in xyz_arrays:
        xyz = _valid_xyz(xyz)
        if len(xyz):
            vals.append(xyz[:, 1])
    if not vals:
        return 0
    return int(np.nanmedian(np.concatenate(vals)) * 1e6)


def _draw_empty_coronal_slice(
    ax,
    *,
    brain_atlas,
    coord_um: int,
):
    """
    Draw an empty Allen coronal slice exactly as in the Figure 6 reference:
    plot_points_on_slice is called with an empty point array and Greys cmap.
    Trajectory xyz coordinates are then drawn on top with ordinary matplotlib
    scatter calls.
    """
    empty = np.zeros((0, 3), dtype=float)

    plot_points_on_slice(
        empty,
        coord=coord_um,
        slice="coronal",
        ax=ax,
        cmap="Greys",
        brain_atlas=brain_atlas,
    )

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def _scatter_xyz_on_coronal(
    ax,
    xyz_m,
    *,
    color,
    label: Optional[str] = None,
    s: float = 7.0,
    alpha: float = 0.95,
):
    """
    Scatter x/z coordinates on the already-created coronal slice.

    The xyz points are ordered dorsal/top -> ventral/bottom before plotting,
    but the coordinates themselves are not modified.
    """
    xyz = _order_xyz_top_to_bottom(xyz_m)
    xyz = _valid_xyz(xyz)
    if len(xyz) == 0:
        return

    xyz_um = _to_um(xyz)

    ax.scatter(
        xyz_um[:, 0],
        xyz_um[:, 2],
        s=s,
        color=color,
        edgecolors="none",
        alpha=alpha,
        label=label,
        zorder=10,
    )


def _plot_points_trajectory(
    ax,
    xyz_m,
    *,
    brain_atlas,
    coord_um: int,
    title: str,
    color,
    label: Optional[str] = None,
):
    """
    Empty Allen coronal slice + scatter plot of trajectory xyz coordinates.
    """
    xyz = _valid_xyz(xyz_m)
    if len(xyz) == 0:
        ax.text(0.5, 0.5, "No valid trajectory", ha="center", va="center")
        ax.set_title(title)
        return

    _draw_empty_coronal_slice(
        ax,
        brain_atlas=brain_atlas,
        coord_um=coord_um,
    )

    _scatter_xyz_on_coronal(
        ax,
        xyz,
        color=color,
        label=label,
        s=7.0,
    )

    ax.set_title(title, pad=3)
    ax.set_aspect("equal", adjustable="box")

    if label:
        ax.legend(frameon=False, loc="lower left", fontsize=5.5)


def _plot_two_trajectories(
    ax,
    xyz_a,
    xyz_b,
    *,
    brain_atlas,
    coord_um: int,
    title: str,
):
    """
    Empty Allen coronal slice + two trajectory scatter plots.
    """
    _draw_empty_coronal_slice(
        ax,
        brain_atlas=brain_atlas,
        coord_um=coord_um,
    )

    _scatter_xyz_on_coronal(
        ax,
        xyz_a,
        color="tab:blue",
        label="True trajectory",
        s=7.0,
    )
    _scatter_xyz_on_coronal(
        ax,
        xyz_b,
        color="tab:orange",
        label="Planned trajectory",
        s=7.0,
    )

    ax.set_title(title, pad=3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)


def _ensure_trace_top_to_bottom(
    xyz_trace: np.ndarray,
    *paired_arrays: np.ndarray,
):
    """
    Ensure a full histological trace is displayed dorsal/top -> ventral/bottom.

    ``extend_xyz_samples_to_brain`` preserves the ordering of the original IBL
    trace.  If that ordering runs bottom -> top, reverse the trace and every
    paired trace-level array together.  This is a visualization-only operation;
    the alignment itself is not changed.
    """
    xyz = np.asarray(xyz_trace)

    valid = _valid_xyz_mask(xyz)
    idx = np.flatnonzero(valid)

    if len(idx) < 2:
        return (xyz,) + tuple(np.asarray(a) for a in paired_arrays)

    first_z = xyz[idx[0], 2]
    last_z = xyz[idx[-1], 2]

    reverse = first_z < last_z

    if reverse:
        xyz = xyz[::-1]
        paired = tuple(np.asarray(a)[::-1] for a in paired_arrays)
    else:
        paired = tuple(np.asarray(a) for a in paired_arrays)

    return (xyz,) + paired


def _nearest_fill_internal_nans_1d(x):
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if not good.any():
        return x
    idx = np.arange(len(x))
    first, last = idx[good][0], idx[good][-1]
    target = (~good) & (idx >= first) & (idx <= last)
    if not target.any():
        return x

    good_idx = idx[good]
    ins = np.searchsorted(good_idx, idx)
    left = np.clip(ins - 1, 0, len(good_idx) - 1)
    right = np.clip(ins, 0, len(good_idx) - 1)
    li, ri = good_idx[left], good_idx[right]
    nearest = np.where(np.abs(idx - li) <= np.abs(idx - ri), li, ri)
    x[target] = x[nearest[target]]
    return x


def _unstandardize_array(mu_std: np.ndarray, model) -> np.ndarray:
    e_mean = model.e_mean.detach().cpu().numpy()
    e_std = model.e_std.detach().cpu().numpy()
    return np.asarray(mu_std, dtype=float) * (e_std + 1e-8) + e_mean


def _plot_feature_pair(
    fig,
    sub_spec,
    *,
    recorded_on_trace: np.ndarray,
    predicted_on_trace: np.ndarray,
    title: str,
):
    """
    Compare one feature over the FULL histological trace.

    Both stripes have the same number of rows and therefore the same vertical
    coordinate system. The atlas prediction spans the complete histological
    trace. The recorded feature is NaN outside its inferred probe position,
    so it is visibly shorter and shifted to the model-inferred location along
    the histological trace.
    """
    rec = np.asarray(recorded_on_trace, dtype=float).reshape(-1)
    pred = np.asarray(predicted_on_trace, dtype=float).reshape(-1)

    if len(rec) != len(pred):
        raise ValueError(
            f"Trace-length mismatch for {title}: "
            f"recorded={len(rec)}, predicted={len(pred)}"
        )

    vals = np.r_[rec[np.isfinite(rec)], pred[np.isfinite(pred)]]
    vmin, vmax = _safe_limits(vals)

    gs = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=sub_spec,
        width_ratios=[1, 1],
        wspace=0.12,
    )
    ax_rec = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])

    _plot_scalar_stripe(
        ax_rec,
        rec,
        "rec.",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        nan_color="white",
    )
    _plot_scalar_stripe(
        ax_pred,
        pred,
        "pred.",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        nan_color="white",
    )

    # Same explicit y-range guarantees positional agreement along the trace.
    n_trace = len(pred)
    for ax in (ax_rec, ax_pred):
        ax.set_ylim(n_trace - 0.5, -0.5)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("black")
            s.set_linewidth(0.35)

    bb1 = ax_rec.get_position()
    bb2 = ax_pred.get_position()
    fig.text(
        (bb1.x0 + bb2.x1) / 2,
        max(bb1.y1, bb2.y1) + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=5.5,
    )

    return ax_rec, ax_pred


def plot_supp_figure_5(
    *,
    histology,
    debug,
    planned_xyz,
    model,
    feature_list,
    brain_atlas,
    save_path: Optional[Path] = None,
    dpi: int = 600,
):
    """
    Supplementary Figure 5: example correction of a misaligned probe.

      a. Registered red-channel histology image (grayscale; 2:1 display ratio).
      b. Manually inferred histological trace as xyz scatter on CCF.
      c. True channel trajectory and planned trajectory as xyz scatters.
      d. Ephys-Atlas model-inferred trajectory as xyz scatter.
      e. Recorded vs atlas-predicted electrophysiological signatures along
         the full histological trace for three features from each group.
    """
    figure_style()

    out = debug["alignment_out"]

    # Per-channel xyz arrays for panels c/d.
    true_xyz_raw = np.asarray(debug["xyz_histology"], dtype=float)
    model_xyz_raw = np.asarray(out["est_xyz"], dtype=float)

    # Use the same dorsal->ventral visualization convention for all xyz panels.
    true_xyz = _order_xyz_top_to_bottom(true_xyz_raw)
    model_xyz = _order_xyz_top_to_bottom(model_xyz_raw)
    inferred_xyz = _order_xyz_top_to_bottom(
        np.asarray(histology["xyz_dense_m"], dtype=float)
    )
    planned_xyz = _order_xyz_top_to_bottom(
        np.asarray(planned_xyz, dtype=float)
    )

    # ------------------------------------------------------------------
    # Panel e uses the FULL histological trace.
    #
    # mu_std_trace:
    #   model prediction at every point of the extended histology trace.
    #
    # recorded_on_trace_raw:
    #   recorded channel features scattered onto their model-inferred trace
    #   locations through j_map_all_i; outside the inferred probe position the
    #   array is NaN.
    #
    # Therefore the recorded stripe is naturally shorter and shifted to the
    # inferred putative position, while the predicted stripe spans the complete
    # histological trace.
    # ------------------------------------------------------------------
    xyz_trace_raw = np.asarray(out["xyz_samples_ext"], dtype=float)

    pred_trace_raw = _unstandardize_array(
        out["mu_std_trace"],
        model,
    )
    rec_trace_raw = np.asarray(
        out["recorded_on_trace_raw"],
        dtype=float,
    )

    if not (
        len(xyz_trace_raw)
        == len(pred_trace_raw)
        == len(rec_trace_raw)
    ):
        raise RuntimeError(
            "Full-trace xyz/feature arrays do not have identical lengths: "
            f"xyz={len(xyz_trace_raw)}, predicted={len(pred_trace_raw)}, "
            f"recorded={len(rec_trace_raw)}"
        )

    # Fill only gaps INSIDE the recorded probe's inferred interval.
    # Leading/trailing NaNs stay NaN and therefore remain white.
    rec_trace_filled = _nearest_fill_internal_nans_2d(rec_trace_raw)

    # Ensure the trace is displayed top -> bottom and reverse every paired
    # feature matrix together if necessary.
    xyz_trace, pred_trace, rec_trace_filled = _ensure_trace_top_to_bottom(
        xyz_trace_raw,
        pred_trace_raw,
        rec_trace_filled,
    )

    # Validate the nine requested features against the released feature order.
    selected = [
        feat
        for group in SUPP5_FEATURE_GROUPS.values()
        for feat in group
    ]
    feature_list = list(map(str, feature_list))
    missing = [f for f in selected if f not in feature_list]
    if missing:
        raise RuntimeError(
            "The released model does not contain all requested panel-e features: "
            f"{missing}. Available features: {feature_list}"
        )

    feature_to_idx = {name: i for i, name in enumerate(feature_list)}

    print("[supp fig 5] panel-e feature indices:")
    for group_name, feats in SUPP5_FEATURE_GROUPS.items():
        print(f"  {group_name}:")
        for feat in feats:
            print(f"    {feature_to_idx[feat]:02d} -> {feat}")

    coord_um = _slice_coord_um(
        inferred_xyz,
        true_xyz,
        planned_xyz,
        model_xyz,
    )

    fig = double_column_fig()
    fig.set_size_inches(fig.get_size_inches()[0], 7.7)

    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.25],
        width_ratios=[1, 1],
        hspace=0.32,
        wspace=0.20,
    )

    # ------------------------------------------------------------------
    # a. Red histology channel in grayscale; requested 2:1 W:H panel.
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.imshow(
        histology["red"],
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax_a.set_box_aspect(0.5)
    ax_a.set_title("Registered red-channel histology", pad=3)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for s in ax_a.spines.values():
        s.set_visible(False)
    _panel_label(ax_a, "a")

    # Yellow left-pointing arrow near the top of panel a.
    # Arrowhead ends around 2/3 of the panel width.
    ax_a.annotate(
        "",
        xy=(0.6, 0.88),
        xytext=(0.7, 0.88),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="red",
            lw=2.0,
            mutation_scale=14,
        ),
        zorder=20,
    )

    ax_a.annotate(
        "",
        xy=(0.6, 0.5),
        xytext=(0.5, 0.4),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="red",
            lw=2.0,
            mutation_scale=14,
        ),
        zorder=20,
    )

    # ------------------------------------------------------------------
    # b. Manually inferred histological track.
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[0, 1])
    _plot_points_trajectory(
        ax_b,
        inferred_xyz,
        brain_atlas=brain_atlas,
        coord_um=coord_um,
        title="Inferred histological trace",
        color="tab:red",
    )
    _panel_label(ax_b, "b")

    # ------------------------------------------------------------------
    # c. True vs planned trajectories.
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 0])
    _plot_two_trajectories(
        ax_c,
        true_xyz,
        planned_xyz,
        brain_atlas=brain_atlas,
        coord_um=coord_um,
        title="True and planned trajectories",
    )
    _panel_label(ax_c, "c")

    # ------------------------------------------------------------------
    # d. Ephys Atlas inferred trajectory.
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(outer[1, 1])
    _plot_points_trajectory(
        ax_d,
        model_xyz,
        brain_atlas=brain_atlas,
        coord_um=coord_um,
        title="Ephys Atlas inferred trajectory",
        color="tab:green",
    )
    _panel_label(ax_d, "d")

    # ------------------------------------------------------------------
    # e. Nine feature pairs: 3 LFP + 3 AP + 3 spike-detection.
    # ------------------------------------------------------------------
    gs_e = GridSpecFromSubplotSpec(
        1,
        9,
        subplot_spec=outer[2, :],
        wspace=0.45,
    )

    first_e_ax = None
    for col, feat in enumerate(selected):
        idx = feature_to_idx[feat]

        rec = np.asarray(rec_trace_filled[:, idx], dtype=float)
        pred = np.asarray(pred_trace[:, idx], dtype=float)

        rec[~np.isfinite(rec)] = np.nan
        pred[~np.isfinite(pred)] = np.nan

        ax_rec, ax_pred = _plot_feature_pair(
            fig,
            gs_e[0, col],
            recorded_on_trace=rec,
            predicted_on_trace=pred,
            title=SUPP5_DISPLAY_NAMES.get(feat, feat),
        )
        if first_e_ax is None:
            first_e_ax = ax_rec

    # Group labels above groups of three pairs.
    for group_i, (group_name, feats) in enumerate(SUPP5_FEATURE_GROUPS.items()):
        first_col = group_i * 3
        last_col = first_col + 2

        # Derive group center from the subplots belonging to first/last feature.
        # Each feature has two tiny axes; find all axes from the figure that overlap
        # the corresponding GridSpec slot by using slot bounding boxes.
        bb_first = gs_e[0, first_col].get_position(fig)
        bb_last = gs_e[0, last_col].get_position(fig)
        fig.text(
            (bb_first.x0 + bb_last.x1) / 2,
            bb_first.y1 + 0.055,
            group_name,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    if first_e_ax is not None:
        _panel_label(first_e_ax, "e")

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.965,
        bottom=0.055,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"[supp fig 5] saved {save_path.resolve()}")

    return fig


def main():
    cfg = RunConfig()

    # Supplementary Figure 5 uses a single, deliberately chosen misalignment
    # example. Change TARGET_PID at the top of this file for another insertion.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    print(f"Using device: {device}")

    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    brain_atlas = AllenAtlas()

    # ------------------------------------------------------------------
    # 1. Resolve the exact released Ephys Atlas model. The registry uses
    #    the local cached release when available and otherwise downloads
    #    the requested Hugging Face revision/tag.
    # ------------------------------------------------------------------
    release = resolve_alignment_release(
        cfg=cfg,
        device=device,
    )
    release_dir = release["release_dir"]

    # ------------------------------------------------------------------
    # 2. Load the versioned context atlases saved with that release.
    # ------------------------------------------------------------------
    ctx_cfg = AtlasPCAConfig(
        n_cell_pcs=cfg.n_cell_pcs,
        n_gene_pcs=cfg.n_gene_pcs,
    )
    ctx_manager = ContextAtlasManager(
        ctx_cfg,
        regenerate_context=False,
        output_dir=release_dir / "context",
    )

    # ------------------------------------------------------------------
    # 3. Load the channel-level Ephys Atlas data.
    # ------------------------------------------------------------------
    pid_names, ephys, probe_positions, probe_planned_positions = LoadInsertionData(
        project=cfg.project,
        agg=cfg.agg,
        VINTAGE=cfg.vintage,
        path_data=cfg.data_dir,
    )
    pid_names = [str(x) for x in pid_names]

    # ------------------------------------------------------------------
    # 4. Reconstruct the exact train-neighbor bank from the frozen release
    #    split/preprocessing statistics. This prevents information leakage
    #    from the example probe into its own atlas prediction.
    # ------------------------------------------------------------------
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
        split_manifest=release["split_manifest"],
        preprocessing_stats=release["preprocessing_stats"],
        return_preprocessing_stats=True,
    )

    base_model, conf_model = construct_released_models(
        release=release,
        device=device,
    )

    # The ephys tensor returned by LoadInsertionData is defined in FEATURE_LIST
    # order.  The released checkpoint contains its own stored feature order.
    # Refuse to continue unless they are exactly identical; this prevents a
    # visually plausible but incorrectly permuted panel-e comparison.
    release_features = list(map(str, release["features"]))
    table_features = list(map(str, FEATURE_LIST))

    if release_features != table_features:
        raise RuntimeError(
            "Feature-order mismatch between the Hugging Face release and "
            "LoadInsertionData/FEATURE_LIST.\n"
            f"Release features: {release_features}\n"
            f"Table features:   {table_features}"
        )

    feature_to_idx = {name: i for i, name in enumerate(release_features)}
    print("[supp fig 5] verified released/table feature order:")
    for i, name in enumerate(release_features):
        print(f"  {i:02d}: {name}")

    handles = build_neighbor_handles(train_loader)

    # ------------------------------------------------------------------
    # 5. Actually run the Ephys Atlas alignment for TARGET_PID.
    #    No stored alignment result/CSV is used.
    # ------------------------------------------------------------------
    debug = prepare_auto_alignment_debug_for_pid(
        pid=TARGET_PID,
        pid_names=pid_names,
        ephys=ephys,
        probe_positions=probe_positions,
        model=base_model,
        ctx_manager=ctx_manager,
        handles=handles,
        optimization_features=np.arange(len(release["features"])),
        radius_um=cfg.radius_um,
        m_max=cfg.m_max,
        device=device,
        brain_atlas=brain_atlas,
        conf_model=conf_model,
    )

    pidx = int(debug["probe_index"])
    planned_xyz = np.asarray(
        probe_planned_positions[pidx],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # 6. Download/cache the registered subject histology and extract the
    #    red fluorescence channel along the traced path.
    # ------------------------------------------------------------------
    histology = load_registered_histology_for_pid(
        pid=TARGET_PID,
        one=one,
        brain_atlas=brain_atlas,
    )

    # ------------------------------------------------------------------
    # 7. Plot.
    # ------------------------------------------------------------------
    plot_supp_figure_5(
        histology=histology,
        debug=debug,
        planned_xyz=planned_xyz,
        model=base_model,
        feature_list=release["features"],
        brain_atlas=brain_atlas,
        save_path=Path("supp_fig5_misaligned_probe_correction.pdf"),
        dpi=600,
    )
    print("all done")

if __name__ == "__main__":
    main()