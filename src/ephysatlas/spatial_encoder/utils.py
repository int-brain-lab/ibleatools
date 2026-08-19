import numpy as np

from pathlib import Path
from typing import Tuple, Any
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import ephysatlas.fixtures
from ephysatlas.data import download_tables, read_features_from_disk

from iblatlas.genomics import agea, merfish
from iblatlas.atlas import AllenAtlas

from one.api import ONE

from tqdm import tqdm
import scipy.interpolate

FEATURE_LIST = [
    "rms_lf",
    "psd_lfp",
    "psd_alpha",
    "psd_beta",
    "psd_gamma",
    "psd_delta",
    "psd_theta",
    "psd_lfp_csd_diff1",
    "psd_alpha_csd_diff1",
    "psd_beta_csd_diff1",
    "psd_gamma_csd_diff1",
    "psd_delta_csd_diff1",
    "psd_theta_csd_diff1",
    "rms_lf_csd_diff1",
    "psd_residual_lfp",
    "psd_residual_alpha",
    "psd_residual_beta",
    "psd_residual_gamma",
    "psd_residual_delta",
    "psd_residual_theta",
    "decay_fit_error",
    "decay_fit_r_squared",
    "decay_n_peaks",
    "aperiodic_exponent",
    "aperiodic_offset",
    "cor_ratio",
    "rms_ap",
    "alpha_mean",
    "alpha_std",
    "spike_count",
    "tip_time_secs",
    "recovery_time_secs",
    "peak_time_secs",
    "trough_time_secs",
    "trough_val",
    "tip_val",
    "peak_val",
    "recovery_slope",
    "depolarisation_slope",
    "repolarisation_slope",
    "polarity",
    #'channel_labels'
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================= data handling =============================
@dataclass
class AtlasPCAConfig:
    n_cell_pcs: int = 50  # 338
    n_gene_pcs: int = 50  # 4345
    no_pca: bool = False


class ContextAtlasManager:
    """
    Handles loading raw atlases, computing PCs, caching transforms, sampling,
    and now: saving/loading precomputed context to/from disk.
    """

    def __init__(
        self,
        cfg: AtlasPCAConfig,
        regenerate_context: bool = False,
        output_dir: Path = Path("."),
    ):
        brain_atlas = AllenAtlas()
        self.bc = brain_atlas.bc
        self.cfg = cfg

        # Save scales
        self.xscale = np.asarray(self.bc.xscale)
        self.yscale = np.asarray(self.bc.yscale)
        self.zscale = np.asarray(self.bc.zscale)

        # -------- Load brain regions --------
        Allen_regions = brain_atlas._get_mapping(mapping="Allen")[brain_atlas.label]

        if regenerate_context:
            # -------- AGEA → PCA --------
            _, gene_vols, _ = agea.load(label="processed")  # [G, Xc, Zc, Yc]
            size_x, size_z, size_y = gene_vols.shape[1:]
            xgenes = gene_vols.reshape(gene_vols.shape[0], -1).T.astype(np.float32)
            scaler = StandardScaler().fit(xgenes)
            xgenes = scaler.transform(xgenes)
            pca = PCA(n_components=cfg.n_gene_pcs).fit(xgenes)
            X_pca = pca.transform(xgenes).reshape(
                size_x, size_z, size_y, cfg.n_gene_pcs
            )
            gene_exp_vol = np.moveaxis(X_pca, -1, 0).astype(
                np.float32
            )  # [P_gene, Xc, Zc, Yc]

            # -------- MERFISH → PCA --------
            merfish.load()
            LEVEL = "subclass"
            path = AllenAtlas._get_cache_dir().joinpath("merfish")
            cell_type_vol = torch.tensor(np.load(path.joinpath(f"merfish_{LEVEL}.npy")))
            zero_ind = torch.where(cell_type_vol.sum(dim=0) == 0)
            cell_type_vol = cell_type_vol.numpy()
            size_x, size_z, size_y = cell_type_vol.shape[1:]

            xcells = cell_type_vol.reshape(cell_type_vol.shape[0], -1).T.astype(
                np.float32
            )
            scaler = StandardScaler().fit(xcells)
            xcells = scaler.transform(xcells)
            pca = PCA(n_components=cfg.n_cell_pcs).fit(xcells)
            X_pca = pca.transform(xcells).reshape(
                size_x, size_z, size_y, cfg.n_cell_pcs
            )
            cell_type_vol = np.moveaxis(X_pca, -1, 0).astype(
                np.float32
            )  # [P_cell, Xc, Zc, Yc]

            # zero out empty MERFISH sites in both modalities
            cell_type_vol[:, zero_ind[0], zero_ind[1], zero_ind[2]] = 0
            gene_exp_vol[:, zero_ind[0], zero_ind[1], zero_ind[2]] = 0

            np.save(output_dir / "agea_vol_pca", gene_exp_vol)
            np.save(output_dir / "merfish_vol_pca", cell_type_vol)
        else:
            gene_exp_vol = np.load(output_dir / "agea_vol_pca.npy")
            cell_type_vol = np.load(output_dir / "merfish_vol_pca.npy")

        self.cell_pca = cell_type_vol  # [P_cell, Xh, Zh, Yh]
        self.gene_pca = gene_exp_vol  # [P_gene, Xh, Zh, Yh]
        self.allen_idx = Allen_regions  # [Yh, Xh, Zh]
        assert self.cell_pca.ndim == 4 and self.gene_pca.ndim == 4
        assert self.allen_idx.ndim == 3

    def sample_context_numpy_m(self, xyz_m: np.ndarray, mode="raise"):
        xyz_m = xyz_m.copy()
        xyz_m[:, 0] = -np.abs(xyz_m[:, 0])  # mirror to left
        indices = self.bc.xyz2i(xyz_m, mode=mode)  # fractional (x,y,z) high-res

        Yh, Xh, Zh = self.allen_idx.shape  # note: allen_idx is [Y, X, Z]
        xi = np.clip(np.round(indices[:, 0] / 8).astype(int), 0, Xh - 1)
        yi = np.clip(np.round(indices[:, 1] / 8).astype(int), 0, Yh - 1)
        zi = np.clip(np.round(indices[:, 2] / 8).astype(int), 0, Zh - 1)

        cell_pc = self.cell_pca[:, xi, zi, yi].T.astype(np.float32)  # [N, P_cell]
        gene_pc = self.gene_pca[:, xi, zi, yi].T.astype(np.float32)  # [N, P_gene]
        return {
            "cell_pc": cell_pc,
            "gene_pc": gene_pc,
            "allen_ix": self.allen_idx[yi, xi, zi].astype(np.int32),
        }

    def sample_context_numpy_i(
        self, xyz_i: np.ndarray, s_xyz: np.ndarray = np.array([8, 8, 8])
    ):
        Xh, Zh, Yh = self.cell_pca.shape[1:]
        xyz_i = xyz_i.copy()
        # xyz_i columns: [xi, yi, zi] in downsampled grid
        xyz_i[:, 0] = mirror_x_indices_to_left(xyz_i[:, 0], Xh)  # mirror x index

        cell_pc = self.cell_pca[:, xyz_i[:, 0], xyz_i[:, 2], xyz_i[:, 1]].T.astype(
            np.float32
        )
        gene_pc = self.gene_pca[:, xyz_i[:, 0], xyz_i[:, 2], xyz_i[:, 1]].T.astype(
            np.float32
        )

        iy = np.clip(xyz_i[:, 1] * s_xyz[1], 0, len(self.yscale) - 1)
        ix = np.clip(xyz_i[:, 0] * s_xyz[0], 0, len(self.xscale) - 1)
        iz = np.clip(xyz_i[:, 2] * s_xyz[2], 0, len(self.zscale) - 1)
        return {
            "cell_pc": cell_pc,
            "gene_pc": gene_pc,
            "allen_ix": self.allen_idx[iy, ix, iz].astype(np.int32),
        }


def LoadInsertionData(
    project: str = "ea_active",
    agg: str = "agg_full",
    VINTAGE: str = "",
    path_data: Path = Path("."),
):
    """
    Loads table-based ephys features and concatenates per-channel averaged waveform latents
    assigned by nearest channel in xyz for each probe.

    Returns:
      unique_pids, context [N,C,(cell_pc+gene_pc)], allen_ix [N,C],
      ephys_concat [N,C,F+L], probe_positions [N,C,3], probe_planned_positions [N,C,3], filter_indices
    """

    print("Loading ephys features")
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    # path_data = Path('../ephys-atlas-decoding/features')
    path_data = download_tables(
        path_data, label=VINTAGE, project=project, one=one, agg_level=agg
    )
    df_features = read_features_from_disk(path_data, strict=False)

    # Pre-allocate containers
    probe_positions = []
    probe_planned_positions = []
    ephys_per_probe = []

    # Iterate probes
    for pid, df_pid in df_features.groupby(level="pid"):
        C = df_pid["x"].shape[0]
        # --- Prepare channel xyz (actual + planned), preserving your up->down reversal ---
        xyz = np.zeros((C, 3), dtype=np.float32)
        xyz_planned = np.zeros((C, 3), dtype=np.float32)

        channel_indices = df_pid.index.get_level_values("channel").to_numpy()
        xyz_values = df_pid[["x", "y", "z"]].values
        xyz_planned_values = df_pid[["x_target", "y_target", "z_target"]].values

        # Reverse order to be up -> down (same as your existing code)
        xyz[channel_indices] = xyz_values[::-1, :].copy()
        xyz_planned[channel_indices] = xyz_planned_values[::-1, :].copy()

        probe_positions.append(xyz)
        probe_planned_positions.append(xyz_planned)

        # --- Table features per probe ---
        ephys_probe = np.zeros((C, len(FEATURE_LIST)), dtype=np.float32)
        channel_idx = df_pid.index.get_level_values("channel").to_numpy()
        values = np.stack([df_pid[feat].values for feat in FEATURE_LIST], axis=-1)
        ephys_probe[channel_idx] = values

        # Keep your final reversal (up->down)
        ephys_per_probe.append(ephys_probe[::-1, :].copy())

    # Stack all probes
    ephys = np.stack(ephys_per_probe)  # [N, C, F(+L)]
    ephys[np.where(np.isinf(ephys))] = 0.0
    probe_positions = np.stack(probe_positions)  # [N, C, 3]
    probe_planned_positions = np.stack(probe_planned_positions)

    # PIDs in the df order
    unique_pids = df_features.index.get_level_values("pid").unique()

    # Filter bad/misaligned
    MISALIGNED_PIDS = ephysatlas.fixtures.misaligned_pids
    block_set = set(MISALIGNED_PIDS)

    filter_indices = [i for i, item in enumerate(unique_pids) if item not in block_set]

    filter_pids = unique_pids[filter_indices]
    filter_ephys = ephys[filter_indices]
    filter_probe_positions = probe_positions[filter_indices]
    filter_probe_planned_positions = probe_planned_positions[filter_indices]

    return (
        filter_pids,
        filter_ephys,
        filter_probe_positions,
        filter_probe_planned_positions,
    )


def region_ids_from_xyz(
    brain_atlas, xyz_m_np: np.ndarray, mapping: str = "Cosmos", mode: str = "raise"
) -> np.ndarray:
    """
    xyz_m_np: [C, 3] in meters
    mapping: "Cosmos" or "Allen" (must be supported by your xyz_to_region_ids)
    Returns: np.ndarray [C] of region IDs (int)
    """
    idx = brain_atlas.bc.xyz2i(xyz_m_np, mode=mode)  # meters → voxel indices
    # Assumes you have xyz_to_region_ids(idx, brain_atlas, mapping=...) in your codebase
    return xyz_to_region_ids(idx, brain_atlas, mapping=mapping)


def xyz_to_region_ids(xyz_i, brain_atlas, mapping="Cosmos"):
    """
    xyz_i: [C, 3]
    """
    inds = brain_atlas._lookup_inds(xyz_i)
    regions = brain_atlas._get_mapping(mapping=mapping)[brain_atlas.label.flat[inds]]

    return regions


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


def _axis_step_in_indices(bc, axis: int) -> float:
    """
    Estimate physical spacing (meters) per +1 index step along a given axis in the high-res grid.
    axis: 0->x, 1->y, 2->z
    """
    a0 = np.zeros((1, 3), dtype=np.int64)
    a1 = a0.copy()
    a1[0, axis] = 1
    p0 = bc.i2xyz(a0)  # [1,3], meters
    p1 = bc.i2xyz(a1)
    return float(np.linalg.norm(p1 - p0))


def compute_grid_strides_200um(bc) -> Tuple[int, int, int]:
    """
    Convert 200 µm into strides in index units along x,y,z, rounding to at least 1.
    """
    target_m = 200e-6
    dx = _axis_step_in_indices(bc, 0)
    dy = _axis_step_in_indices(bc, 1)
    dz = _axis_step_in_indices(bc, 2)

    sx = max(1, int(round(target_m / max(dx, 1e-12))))
    sy = max(1, int(round(target_m / max(dy, 1e-12))))
    sz = max(1, int(round(target_m / max(dz, 1e-12))))
    return sx, sy, sz


def concat_context(cell_pc: np.ndarray, gene_pc: np.ndarray) -> np.ndarray:
    return np.concatenate([cell_pc, gene_pc], axis=-1)


def build_channels_plus_emptyvoxels_with_neighbors(
    ctx_manager: ContextAtlasManager,
    ephys: np.ndarray,
    probe_positions: np.ndarray,
    RADIUS_UM: int,
    M_MAX: int,
    *,
    pid_names=None,
    batch_size_train: int = 1024,
    batch_size_eval: int = 1024,
    shuffle_train: bool = True,
    seed: int = 0,
    split_manifest: dict = None,
    preprocessing_stats: dict = None,
):
    """
      • GRID DATASET = voxels that do NOT contain any ephys channels.
      • RECORDED DATASET = per-channel samples (no voxel averaging):
            one sample per (probe, channel) with valid xyz; context/allen are sampled at that xyz.

    Context standardization: mean/std over ALL grid voxels (full atlas grid, same as before).
    Ephys standardization: mean/std over TRAIN split of recorded channels only.

    Optional (figure reproduction against a released model):
      split_manifest: dict with ``train_pids``/``validation_pids``/``test_pids`` -- use the model's
        own saved split instead of a fresh random one, so evaluation runs on its real held-out set.
      preprocessing_stats: dict with ``e_mean``/``e_std``/``ctx_mean``/``ctx_std`` -- use the
        model's train-time standardization instead of recomputing it, so inputs match the weights.
      Both default to None, which preserves the original random-split, recomputed-stats behavior.

    Returns:
      train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std
    """

    # ----- 200 µm grid over the whole atlas -----
    P_cell, Xh, Zh, Yh = ctx_manager.cell_pca.shape
    sx, sy, sz = compute_grid_strides_200um(ctx_manager.bc)

    xs = np.arange(0, Xh, dtype=int)
    ys = np.arange(0, Yh, dtype=int)
    zs = np.arange(0, Zh, dtype=int)
    XX, ZZ, YY = np.meshgrid(xs, zs, ys, indexing="ij")
    xi = XX.reshape(-1)
    zi = ZZ.reshape(-1)
    yi = YY.reshape(-1)
    N = xi.size

    ijk = np.stack([xi, yi, zi], axis=1)
    xyz_m = ctx_manager.bc.i2xyz(ijk * 8).astype(np.float32)  # [N,3] meters
    xyz_m = mirror_xyz_to_left(xyz_m)

    # Sample context for ALL voxels
    allen_all, ctx_all = [], []
    for i in range(N):
        ctx = ctx_manager.sample_context_numpy_i(
            np.array((xi[i], yi[i], zi[i]))[None, :], np.array((sx, sy, sz))
        )
        ctx_all.append(concat_context(ctx["cell_pc"], ctx["gene_pc"])[0])
        allen_all.append(ctx["allen_ix"][0])

    ctx_all = np.asarray(ctx_all, dtype=np.float32)  # [N, F_ctx]
    F_e = int(ephys.shape[-1])

    # --- mark voxels that have any ephys channels, to keep only "empty" ones for grid_ds ---
    has_eph = compute_voxel_with_ephys(ctx_manager, probe_positions, xi, yi, zi)
    has_ctx = ~(ctx_all.sum(axis=1) == 0)

    grid_mask = ~has_eph & has_ctx

    # Context stats over ALL grid voxels (rec + non-rec) per your original rule
    ctx_all_t = torch.from_numpy(ctx_all).float()
    ctx_mean = ctx_all_t[grid_mask].mean(dim=0)
    ctx_std = ctx_all_t[grid_mask].std(dim=0, unbiased=False).clamp_min(1e-6)

    # Prefer the released model's own train-time context stats when supplied, so standardization
    # matches what the weights were trained with rather than being recomputed from this data.
    if preprocessing_stats is not None and "ctx_mean" in preprocessing_stats:
        ctx_mean = torch.as_tensor(preprocessing_stats["ctx_mean"], dtype=torch.float32)
        ctx_std = torch.as_tensor(
            preprocessing_stats["ctx_std"], dtype=torch.float32
        ).clamp_min(1e-6)

    def _stdz_ctx(t):
        mask = np.where(t.sum(axis=1) != 0)[0]
        t_clone = t.clone()
        t_clone[mask] = (t[mask] - ctx_mean) / ctx_std
        return t_clone

    # GRID DATASET (only voxels WITHOUT ephys)
    ctx_grid = _stdz_ctx(torch.from_numpy(ctx_all[grid_mask]).float())
    xyz_grid = torch.from_numpy(xyz_m[grid_mask]).float()
    grid_ds = GridDS(ctx_grid, xyz_grid, F_e)

    # ----- RECORDED CHANNEL DATASET (per-channel; NO voxel averaging) -----
    P, C, _ = probe_positions.shape
    rec_ctx_list, rec_xyz_list, rec_ephys_list, rec_pid_list = [], [], [], []

    for p in range(P):
        xyz_p = probe_positions[p].astype(np.float32)  # [C,3]
        eph_p = ephys[p].astype(np.float32)  # [C,F]
        valid = ~(np.all(xyz_p == 0.0, axis=1))
        if not valid.any():
            continue

        xyz_valid = xyz_p[valid]  # [C_valid,3]
        xyz_valid = mirror_xyz_to_left(xyz_valid)  # <<< add this

        pack = ctx_manager.sample_context_numpy_m(xyz_valid, mode="clip")

        ctx_p = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1).astype(
            np.float32
        )
        eph_valid = eph_p[valid]

        rec_ctx_list.append(ctx_p)
        rec_xyz_list.append(xyz_valid)
        rec_ephys_list.append(eph_valid)
        rec_pid_list.append(p * np.ones(valid.sum(), dtype=np.float32))

    if len(rec_ctx_list) == 0:
        raise RuntimeError(
            "No valid recorded channels found to build recorded dataset."
        )

    rec_ctx = torch.from_numpy(
        np.concatenate(rec_ctx_list, axis=0)
    ).float()  # [Nc,F_ctx]
    rec_xyz = torch.from_numpy(np.concatenate(rec_xyz_list, axis=0)).float()  # [Nc,3]
    rec_ephys = torch.from_numpy(
        np.concatenate(rec_ephys_list, axis=0)
    ).float()  # [Nc,F_e]
    rec_pids = torch.from_numpy(np.concatenate(rec_pid_list, axis=0)).float()  # [Nc,]

    # Standardize context (use global grid stats)
    rec_ctx_std = _stdz_ctx(rec_ctx)

    # ----- Split RECORDED by PROBE (probe-level split) -----
    # rec_pids is [Nc] float right now; convert to int probe ids
    rec_pids_i = rec_pids.to(torch.int64).cpu().numpy()  # [Nc]

    uniq_p = np.array(sorted(np.unique(rec_pids_i).astype(int)))

    if pid_names is not None:
        pid_names = np.asarray(pid_names).astype(str)
        if len(pid_names) < int(uniq_p.max()) + 1:
            raise ValueError(
                f"pid_names length={len(pid_names)} is too short for max probe index={uniq_p.max()}"
            )
    else:
        pid_names = np.asarray(
            [str(i) for i in range(int(uniq_p.max()) + 1)], dtype=str
        )

    def _read_pid_txt(path):
        path = Path(path)
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _pid_strings_to_probe_indices(pid_list, *, split_name):
        pid_to_probe_idx = {str(pid_names[i]): int(i) for i in uniq_p}

        ids = []
        missing = []

        for pid in pid_list:
            pid = str(pid).strip()
            if pid in pid_to_probe_idx:
                ids.append(pid_to_probe_idx[pid])
            else:
                missing.append(pid)

        if len(missing) > 0:
            print(
                f"[warn] {split_name}: {len(missing)} pids from txt were not found "
                f"in loaded data. First few: {missing[:5]}"
            )

        return set(ids)

    if split_manifest is not None:
        # Use the model's authoritative saved split (figure reproduction): map its train/val/test
        # insertion pids onto the probe indices loaded here. Insertions absent from the loaded data
        # (e.g. excluded pids) are dropped with a warning by the helper.
        p_tr_ids = _pid_strings_to_probe_indices(
            split_manifest.get("train_pids", []), split_name="train"
        )
        p_va_ids = _pid_strings_to_probe_indices(
            split_manifest.get("validation_pids", []), split_name="validation"
        )
        p_te_ids = _pid_strings_to_probe_indices(
            split_manifest.get("test_pids", []), split_name="test"
        )
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(uniq_p)

        p_tr = 0.7
        p_va = 0.1

        nP = len(shuffled)
        n_tr_p = int(round(p_tr * nP))
        n_va_p = int(round(p_va * nP))

        n_tr_p = int(np.clip(n_tr_p, 1, nP))
        n_va_p = int(np.clip(n_va_p, 0, nP - n_tr_p))

        p_tr_ids = set(shuffled[:n_tr_p].astype(int).tolist())
        p_va_ids = set(shuffled[n_tr_p : n_tr_p + n_va_p].astype(int).tolist())
        p_te_ids = set(shuffled[n_tr_p + n_va_p :].astype(int).tolist())

    # safety checks
    all_split_ids = set(p_tr_ids) | set(p_va_ids) | set(p_te_ids)

    if len(p_tr_ids) == 0:
        raise ValueError("Train split is empty.")
    if len(p_te_ids) == 0:
        print("[warn] Test split is empty.")

    overlap_tv = set(p_tr_ids) & set(p_va_ids)
    overlap_tt = set(p_tr_ids) & set(p_te_ids)
    overlap_vt = set(p_va_ids) & set(p_te_ids)

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split overlap detected: "
            f"train/val={len(overlap_tv)}, train/test={len(overlap_tt)}, val/test={len(overlap_vt)}"
        )

    missing_loaded = set(uniq_p.tolist()) - all_split_ids
    if len(missing_loaded) > 0:
        print(
            f"[warn] {len(missing_loaded)} loaded probes are not assigned to any split."
        )

    # map probe split -> row indices
    I_tr = np.flatnonzero(np.isin(rec_pids_i, list(p_tr_ids)))
    I_va = np.flatnonzero(np.isin(rec_pids_i, list(p_va_ids)))
    I_te = np.flatnonzero(np.isin(rec_pids_i, list(p_te_ids)))

    # Compute clipping thresholds from TRAIN only
    rec_ephys_low_pctl = torch.tensor(
        [
            np.percentile(rec_ephys[I_tr, feat_ind].cpu().numpy(), 0.5)
            for feat_ind in range(rec_ephys.shape[1])
        ],
        dtype=rec_ephys.dtype,
    )

    rec_ephys_high_pctl = torch.tensor(
        [
            np.percentile(rec_ephys[I_tr, feat_ind].cpu().numpy(), 99.5)
            for feat_ind in range(rec_ephys.shape[1])
        ],
        dtype=rec_ephys.dtype,
    )

    # Clip all splits using train thresholds
    rec_ephys = torch.maximum(
        torch.minimum(rec_ephys, rec_ephys_high_pctl), rec_ephys_low_pctl
    )

    # Now compute ephys stats from CLIPPED TRAIN data
    e_mean = rec_ephys[I_tr].mean(dim=0)
    e_std = rec_ephys[I_tr].std(dim=0, unbiased=False).clamp_min(1e-6)

    # As with context, prefer the released model's own train-time ephys stats when supplied. The
    # train-split clipping above still runs, matching how those published stats were produced.
    if preprocessing_stats is not None and "e_mean" in preprocessing_stats:
        e_mean = torch.as_tensor(preprocessing_stats["e_mean"], dtype=torch.float32)
        e_std = torch.as_tensor(
            preprocessing_stats["e_std"], dtype=torch.float32
        ).clamp_min(1e-6)

    # Standardize all splits
    rec_ephys_std = (rec_ephys - e_mean) / e_std

    rec_train = RecDS(
        rec_ctx_std[I_tr], rec_xyz[I_tr], rec_ephys_std[I_tr], rec_pids[I_tr]
    )
    rec_val = RecDS(
        rec_ctx_std[I_va], rec_xyz[I_va], rec_ephys_std[I_va], rec_pids[I_va]
    )
    rec_test = RecDS(
        rec_ctx_std[I_te], rec_xyz[I_te], rec_ephys_std[I_te], rec_pids[I_te]
    )

    train_concat = ConcatDataset([rec_train, grid_ds])

    # =========================
    # Neighbor bank (TRAIN ONLY) built from REC arrays (same indexing!)
    # =========================
    bank_xyz = rec_xyz[I_tr].cpu().numpy()
    bank_feat_std = rec_ephys_std[I_tr].cpu().numpy()
    bank_pid = rec_pids[I_tr].cpu().numpy()

    nn_bank = ChannelNN(bank_xyz)

    collate = NeighborCollate(
        ctx_manager,
        bank_xyz,
        bank_feat_std,
        bank_pid,
        nn_bank,
        e_feat_dim=F_e,
        M_max=M_MAX,
        radius_um=RADIUS_UM,
    )

    split_info = dict(
        p_tr_ids=sorted([int(x) for x in p_tr_ids]),
        p_va_ids=sorted([int(x) for x in p_va_ids]),
        p_te_ids=sorted([int(x) for x in p_te_ids]),
        p_tr_names=[str(pid_names[int(i)]) for i in sorted(p_tr_ids)],
        p_va_names=[str(pid_names[int(i)]) for i in sorted(p_va_ids)],
        p_te_names=[str(pid_names[int(i)]) for i in sorted(p_te_ids)],
    )

    # original mixed train loader for base model
    train_loader = DataLoader(
        train_concat,
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    # recorded-only train loader for confidence training
    conf_train_loader = DataLoader(
        rec_train,
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        rec_val,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        rec_test,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    return (
        train_loader,  # mixed, for base model
        conf_train_loader,  # recorded-only, for confidence model
        val_loader,
        test_loader,
        e_mean,
        e_std,
        ctx_mean,
        ctx_std,
        split_info,
    )


class RecDS(Dataset):
    """Recorded voxels: (context, xyz_m, ephys, pid, has_ephys=True)."""

    def __init__(self, ctx, xyz_m, ephys, pid):
        self.ctx, self.xyz = ctx, xyz_m
        self.ephys, self.pid = ephys, pid
        self.has = torch.ones(len(self.ctx), dtype=torch.bool)

    def __len__(self):
        return self.ctx.shape[0]

    def __getitem__(self, i):
        return (self.ctx[i], self.xyz[i], self.ephys[i], self.pid[i], self.has[i])


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


def compute_voxel_with_ephys(ctx_manager, probe_positions, xi, yi, zi):
    N = xi.size

    ch_xyz = (
        probe_positions
        if len(probe_positions.shape) == 2
        else probe_positions.reshape(-1, 3)
    )
    ch_xyz = mirror_xyz_to_left(ch_xyz)  # <<< add
    xic, zic, yic = downsample_keys_from_xyz(ctx_manager, ch_xyz)

    has = np.zeros(N, dtype=bool)

    # Map grid tuple -> flat index
    key2flat = {(int(xi[i]), int(zi[i]), int(yi[i])): i for i in range(N)}

    for x, z, y in zip(xic, zic, yic):
        if (x, z, y) in key2flat:
            has[key2flat[(x, z, y)]] = True
    return has


try:
    from sklearn.neighbors import KDTree

    _HAS_KDT = True
except Exception:
    _HAS_KDT = False


class ChannelNN:
    def __init__(self, ch_xyz_m: np.ndarray):
        self.X = ch_xyz_m.astype(np.float64)
        self.tree = (
            KDTree(self.X, leaf_size=40) if (self.X.shape[0] and _HAS_KDT) else None
        )

    def query_radius(self, q_xyz_m: np.ndarray, r_m: float, k_cap: int = 8):
        if self.tree is not None:
            inds, _ = self.tree.query_radius(
                q_xyz_m, r=r_m, return_distance=True, sort_results=True
            )
            return [ii[:k_cap] for ii in inds]
        # brute force
        out = []
        X = self.X
        for q in q_xyz_m:
            if X.shape[0] == 0:
                out.append(np.array([], dtype=int))
                continue
            d2 = np.sum((X - q[None, :]) ** 2, axis=1)
            filtered_indices = np.where(d2 <= (r_m**2))[0]
            if filtered_indices.size > k_cap:
                J = np.argpartition(d2[filtered_indices], k_cap)[:k_cap]
                filtered_indices = filtered_indices[J]
            out.append(filtered_indices)
        return out


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


# ============================================================
# Synthetic sample generation
# ============================================================
def _build_shift_based_synthetic_probe_sample(
    *,
    probe_idx: int,
    bank: list[dict],
    cfg: Any,
    rng: np.random.Generator,
    ctx_manager,
    base_model,
    handles,
):
    """
    Binary labels only:
      0 = good
      1 = suspicious

    Logic:
      1) sample a shifted candidate window along the histology trace
      2) compare shifted Cosmos region vs original aligned Cosmos region
         -> different region => suspicious
         -> same region      => good
      3) optionally apply smooth xyz perturbations to the current candidate window
         and recompute the label by region change relative to the original aligned channel

    This keeps the shift logic, keeps perturbations, and collapses the old
    suspicious+wrong semantics into a single 'suspicious' class.
    """
    item = bank[probe_idx]

    rec = item["rec_std"].copy()  # original recorded data stays the input
    valid = item["valid"].copy()
    hist_xyz_full = item["hist_xyz_full"]
    hist_ctx_full = item["hist_ctx_full"]
    hist_pred_full = item["hist_pred_full"]
    true_start = int(item["true_start"])

    C, F_e = rec.shape
    labels = np.zeros((C,), dtype=np.int64)  # 0=good, 1=suspicious

    valid_idx = np.where(valid)[0]
    Nv = len(valid_idx)

    if Nv < 8:
        F_ctx = hist_ctx_full.shape[1]
        return (
            rec.astype(np.float32),
            np.zeros((C, F_ctx), dtype=np.float32),
            np.zeros((C, F_e), dtype=np.float32),
            labels.astype(np.int64),
            valid.astype(bool),
        )

    # -------------------------------------------------
    # sample one random shift along the histology trace
    # -------------------------------------------------
    min_shift_allowed = -true_start
    max_shift_allowed = hist_xyz_full.shape[0] - (true_start + C)

    max_abs = int(cfg.max_abs_shift_channels)
    min_shift = max(min_shift_allowed, -max_abs)
    max_shift = min(max_shift_allowed, max_abs)

    if min_shift > max_shift:
        shift = 0
    else:
        all_shifts = np.arange(min_shift, max_shift + 1, dtype=int)
        shift = int(rng.choice(all_shifts)) if len(all_shifts) > 0 else 0

    start = true_start + shift
    stop = start + C

    query_xyz = hist_xyz_full[start:stop].copy()
    ctx = hist_ctx_full[start:stop].copy()
    pred = hist_pred_full[start:stop].copy()

    # -------------------------------------------------
    # get original/shifted Cosmos labels
    # -------------------------------------------------
    hist_cosmos_full = item.get("hist_cosmos_full", None)
    use_cosmos_labeling = True

    if hist_cosmos_full is None:
        try:
            brain_atlas = AllenAtlas()
            hist_cosmos_full = region_ids_from_xyz(
                brain_atlas, hist_xyz_full, mapping="Cosmos"
            )
            hist_cosmos_full = np.asarray(hist_cosmos_full).reshape(-1)
        except Exception:
            hist_cosmos_full = None
            use_cosmos_labeling = False
    else:
        hist_cosmos_full = np.asarray(hist_cosmos_full).reshape(-1)
        if hist_cosmos_full.shape[0] != hist_xyz_full.shape[0]:
            use_cosmos_labeling = False

    if use_cosmos_labeling and hist_cosmos_full is not None:
        cosmos_true = np.asarray(hist_cosmos_full[true_start : true_start + C]).reshape(
            -1
        )
        cosmos_shifted = np.asarray(hist_cosmos_full[start:stop]).reshape(-1)

        if cosmos_true.shape[0] != C or cosmos_shifted.shape[0] != C:
            use_cosmos_labeling = False
    else:
        cosmos_true = None
        cosmos_shifted = None

    # -------------------------------------------------
    # initial labels from the shifted window
    # -------------------------------------------------
    if use_cosmos_labeling and cosmos_true is not None:
        same_region = (
            (cosmos_true == cosmos_shifted) & (cosmos_true != 0) & (cosmos_shifted != 0)
        )
        labels[valid] = np.where(same_region[valid], 0, 1)
    else:
        # fallback if Cosmos labeling fails
        labels[valid] = 0 if shift == 0 else 1

    # -------------------------------------------------
    # optionally perturb the candidate window itself
    # then relabel by comparing perturbed Cosmos region to original aligned region
    # -------------------------------------------------
    if rng.random() < float(cfg.suspicious_probe_prob):
        valid_local_idx = np.where(valid)[0]
        Nvalid = len(valid_local_idx)

        if Nvalid > 0:
            frac = float(
                rng.uniform(
                    cfg.suspicious_chunk_frac_min, cfg.suspicious_chunk_frac_max
                )
            )
            total_pert = max(1, int(round(frac * Nvalid)))
            total_pert = min(total_pert, Nvalid)

            pert_blocks_local = _choose_nonoverlapping_blocks_covering_total_len(
                Nvalid,
                total_pert,
                rng,
                n_chunks_min=cfg.suspicious_n_chunks_min,
                n_chunks_max=cfg.suspicious_n_chunks_max,
            )

            query_xyz_pert = query_xyz.copy()

            for s_local, e_local in pert_blocks_local:
                selected_idx = valid_local_idx[s_local:e_local]
                L = len(selected_idx)
                if L <= 0:
                    continue

                field = _make_smooth_xyz_perturbation(
                    L,
                    rng,
                    mag_min_um=cfg.suspicious_perturb_min_um,
                    mag_max_um=cfg.suspicious_perturb_max_um,
                    n_anchors_min=cfg.perturb_n_anchors_min,
                    n_anchors_max=cfg.perturb_n_anchors_max,
                    smooth_kernel_min=cfg.perturb_smooth_kernel_min,
                    smooth_kernel_max=cfg.perturb_smooth_kernel_max,
                    taper_frac=cfg.perturb_taper_frac,
                )

                query_xyz_pert[selected_idx] = query_xyz_pert[selected_idx] + field

            # recompute ctx/pred for perturbed positions
            ctx, pred = _predict_ctx_and_ephys_for_xyz(
                xyz_m=query_xyz_pert,
                ctx_manager=ctx_manager,
                base_model=base_model,
                handles=handles,
                batch_size=64,
            )

            # relabel perturbed channels by comparing to original aligned Cosmos region
            if use_cosmos_labeling and cosmos_true is not None:
                try:
                    brain_atlas = AllenAtlas()
                    cosmos_after = region_ids_from_xyz(
                        brain_atlas, query_xyz_pert, mapping="Cosmos"
                    )
                    cosmos_after = np.asarray(cosmos_after).reshape(-1)

                    same_region_after = (
                        (cosmos_true == cosmos_after)
                        & (cosmos_true != 0)
                        & (cosmos_after != 0)
                    )
                    labels[valid] = np.where(same_region_after[valid], 0, 1)
                except Exception:
                    # keep previous labels if atlas lookup fails
                    pass

    pred[~valid] = 0.0
    ctx[~valid] = 0.0

    return (
        rec.astype(np.float32),
        ctx.astype(np.float32),
        pred.astype(np.float32),
        labels.astype(np.int64),
        valid.astype(bool),
    )


# ============================================================
# Utilities
# ============================================================
def _smooth1d_reflect(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Reflective moving-average smoothing along channel axis only.
    x: [L] or [L, F]
    """
    x = np.asarray(x, dtype=np.float32)
    k = int(max(1, kernel_size))
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1

    pad = k // 2
    ker = np.ones((k,), dtype=np.float32) / float(k)

    if x.ndim == 1:
        xp = np.pad(x, (pad, pad), mode="reflect")
        y = np.convolve(xp, ker, mode="valid")
        return y.astype(np.float32)

    if x.ndim == 2:
        out = np.empty_like(x, dtype=np.float32)
        for f in range(x.shape[1]):
            xp = np.pad(x[:, f], (pad, pad), mode="reflect")
            out[:, f] = np.convolve(xp, ker, mode="valid").astype(np.float32)
        return out

    raise ValueError(f"x must be 1D or 2D, got shape={x.shape}")


def _valid_xyz_mask_np(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.isfinite(xyz).all(axis=1) & ~(np.all(xyz == 0.0, axis=1))


def _sample_ctx_for_probe_xyz_std(
    ctx_manager,
    xyz_m: np.ndarray,  # [C,3]
    ctx_mean: torch.Tensor,  # [F_ctx]
    ctx_std: torch.Tensor,  # [F_ctx]
) -> np.ndarray:
    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    valid = _valid_xyz_mask_np(xyz_m)

    ctx = np.zeros((xyz_m.shape[0], int(ctx_mean.numel())), dtype=np.float32)
    if valid.any():
        xyz_use = xyz_m[valid].copy()
        try:
            xyz_use = mirror_xyz_to_left(xyz_use)
        except Exception:
            pass

        pack = ctx_manager.sample_context_numpy_m(xyz_use, mode="clip")
        ctx_valid = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1).astype(
            np.float32
        )

        ctx_mean_np = ctx_mean.detach().cpu().numpy().astype(np.float32)
        ctx_std_np = ctx_std.detach().cpu().numpy().astype(np.float32)
        ctx_valid = (ctx_valid - ctx_mean_np) / (ctx_std_np + 1e-8)
        ctx[valid] = ctx_valid

    return ctx


def _cosine_taper(length: int, taper_frac: float = 0.20) -> np.ndarray:
    L = int(length)
    if L <= 1:
        return np.ones((L,), dtype=np.float32)

    taper_frac = float(np.clip(taper_frac, 0.0, 0.49))
    n_edge = max(1, int(round(L * taper_frac)))
    n_edge = min(n_edge, L // 2)

    w = np.ones((L,), dtype=np.float32)
    if n_edge > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_edge, dtype=np.float32)))
        w[:n_edge] = ramp
        w[-n_edge:] = ramp[::-1]
    return w.astype(np.float32)


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype(np.float32)
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return (v / norm).astype(np.float32)


def _make_smooth_xyz_perturbation(
    length: int,
    rng: np.random.Generator,
    *,
    mag_min_um: float,
    mag_max_um: float,
    n_anchors_min: int = 3,
    n_anchors_max: int = 6,
    smooth_kernel_min: int = 9,
    smooth_kernel_max: int = 31,
    taper_frac: float = 0.20,
) -> np.ndarray:
    """
    Smooth [L,3] perturbation field in meters.
    """
    L = int(length)
    if L <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    if L == 1:
        d = _random_unit_vectors(1, rng)[0]
        m = float(rng.uniform(mag_min_um, mag_max_um)) * 1e-6
        return (d[None, :] * m).astype(np.float32)

    n_anchors = int(rng.integers(n_anchors_min, n_anchors_max + 1))
    n_anchors = max(2, min(n_anchors, L))

    anchor_pos = np.linspace(0, L - 1, n_anchors, dtype=np.float32)
    anchor_dir = _random_unit_vectors(n_anchors, rng)
    anchor_mag = (
        rng.uniform(mag_min_um, mag_max_um, size=(n_anchors, 1)).astype(np.float32)
        * 1e-6
    )
    anchor_vec = anchor_dir * anchor_mag

    x = np.arange(L, dtype=np.float32)
    field = np.zeros((L, 3), dtype=np.float32)
    for d in range(3):
        field[:, d] = np.interp(x, anchor_pos, anchor_vec[:, d]).astype(np.float32)

    k = int(rng.integers(smooth_kernel_min, smooth_kernel_max + 1))
    if k % 2 == 0:
        k += 1
    k = min(k, L if L % 2 == 1 else max(1, L - 1))
    k = max(1, k)
    field = _smooth1d_reflect(field, kernel_size=k)

    taper = _cosine_taper(L, taper_frac=taper_frac)[:, None]
    field = field * taper
    return field.astype(np.float32)


def _choose_nonoverlapping_blocks_covering_total_len(
    n: int,
    total_len: int,
    rng: np.random.Generator,
    n_chunks_min: int = 1,
    n_chunks_max: int = 3,
):
    """
    Choose 1..3 non-overlapping contiguous blocks in local coordinates [0..n-1]
    whose total covered length is approximately total_len.
    """
    n = int(n)
    total_len = int(np.clip(total_len, 1, n))
    n_chunks = int(rng.integers(n_chunks_min, n_chunks_max + 1))
    n_chunks = max(1, min(n_chunks, total_len, n))

    # split total_len across chunks
    if n_chunks == 1:
        chunk_lens = [total_len]
    else:
        cuts = np.sort(
            rng.choice(np.arange(1, total_len), size=n_chunks - 1, replace=False)
        )
        chunk_lens = np.diff(np.r_[0, cuts, total_len]).tolist()

    # place chunks greedily with random gaps
    blocks = []
    remaining_space = n - total_len
    if remaining_space < 0:
        remaining_space = 0

    gap_slots = n_chunks + 1
    if remaining_space > 0:
        gap_cuts = np.sort(
            rng.choice(
                np.arange(remaining_space + gap_slots - 1),
                size=gap_slots - 1,
                replace=False,
            )
        )
        gap_sizes = np.diff(np.r_[-1, gap_cuts, remaining_space + gap_slots - 1]) - 1
    else:
        gap_sizes = np.zeros((gap_slots,), dtype=int)

    pos = int(gap_sizes[0])
    for i, L in enumerate(chunk_lens):
        s = pos
        e = s + int(L)
        blocks.append((s, e))
        pos = e + int(gap_sizes[i + 1])

    return blocks


def _infer_xyz_unit_and_convert_to_meters(xyz: np.ndarray) -> np.ndarray:
    """
    Best-effort helper:
      - if coordinates look like micrometers, convert to meters
      - if they already look like meters, keep as-is
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    max_abs = np.nanmax(np.abs(xyz))
    # very rough heuristic
    if max_abs > 10.0:
        return xyz * 1e-6
    return xyz


def _predict_ctx_and_ephys_for_xyz(
    *,
    xyz_m: np.ndarray,  # [N,3]
    ctx_manager,
    base_model,
    handles,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict standardized ctx and ephys at arbitrary xyz locations.
    """
    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    valid = _valid_xyz_mask_np(xyz_m)

    F_e = int(base_model.e_mean.numel())

    ctx = _sample_ctx_for_probe_xyz_std(
        ctx_manager=ctx_manager,
        xyz_m=xyz_m,
        ctx_mean=base_model.ctx_mean,
        ctx_std=base_model.ctx_std,
    ).astype(np.float32)

    pred = np.zeros((xyz_m.shape[0], F_e), dtype=np.float32)
    if not valid.any():
        return ctx, pred

    device = base_model.e_mean.device
    base_model.eval()

    xyz_t = torch.from_numpy(xyz_m).float()
    qds = GridDS(torch.from_numpy(ctx), xyz_t, F_e)

    collate = NeighborCollate(
        ctx_manager,
        handles["bank_xyz"],
        handles["bank_feat"],
        handles["bank_pid"],
        handles["nn_bank"],
        e_feat_dim=F_e,
        M_max=8,
        radius_um=500,
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
    use_autocast = device_type == "cuda"

    for batch in dl:
        (ctx_b, p_b, e_n, p_n, mask, has_ephys, y_e, *_) = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu = base_model(ctx_b, p_b, e_n, p_n, mask)
        mu_all.append(mu.detach())

    pred = torch.cat(mu_all, dim=0).cpu().numpy().astype(np.float32)
    pred[~valid] = 0.0
    ctx[~valid] = 0.0
    return ctx.astype(np.float32), pred.astype(np.float32)


def _build_histology_trace_extended_and_aligned_window(
    one,
    pid_name,
    true_xyz: np.ndarray,  # [C,3] meters, actual probe positions
    channel_step_um: float,
    extra_channels_each_side: int = 128,
    ba=None,
    max_extension_um: float = 7000.0,
):
    """
    Build a full trace whose core is EXACTLY the provided true_xyz, and whose before/after
    extensions are guided by the histology picks.

    Returns
    -------
    full_xyz_ext : [L_full, 3] float32
        Extended trace in channel coordinates.
    true_xyz : [C, 3] float32
        The provided true xyz, copied into the center of the full trace.
    true_start : int
        Start index such that full_xyz_ext[true_start:true_start+C] == true_xyz
    """
    if ba is None:
        ba = AllenAtlas()

    recs = one.alyx.rest("insertions", "list", id=str(pid_name))
    if recs is None or len(recs) == 0:
        raise ValueError(f"No insertion found in Alyx for pid={pid_name}")

    js = recs[0].get("json", {})
    xyz_picks = js.get("xyz_picks", None)
    if xyz_picks is None:
        raise ValueError(f"No xyz_picks found in Alyx for pid={pid_name}")

    xyz_picks = np.asarray(xyz_picks, dtype=np.float64)
    if xyz_picks.ndim != 2 or xyz_picks.shape[1] != 3:
        raise ValueError(
            f"xyz_picks has invalid shape for pid={pid_name}: {xyz_picks.shape}"
        )

    xyz_picks = _infer_xyz_unit_and_convert_to_meters(xyz_picks)
    xyz_picks = xyz_picks[_valid_xyz_mask_np(xyz_picks)]
    if xyz_picks.shape[0] < 2:
        raise ValueError(f"Need at least 2 valid xyz_picks for pid={pid_name}")

    true_xyz = np.asarray(true_xyz, dtype=np.float64)
    if true_xyz.ndim != 2 or true_xyz.shape[1] != 3:
        raise ValueError(f"true_xyz must have shape [C,3], got {true_xyz.shape}")

    true_valid = _valid_xyz_mask_np(true_xyz)
    if true_valid.sum() < 2:
        raise ValueError(f"Need at least 2 valid true_xyz rows for pid={pid_name}")

    C = true_xyz.shape[0]

    base_channel_step_um = (
        float(channel_step_um)
        if np.isfinite(channel_step_um) and channel_step_um > 0
        else 20.0
    )
    row_step_um = base_channel_step_um
    row_step_m = row_step_um * 1e-6

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _ensure_2d(xyz_m):
        xyz_m = np.asarray(xyz_m, dtype=np.float64)
        if xyz_m.ndim == 1:
            xyz_m = xyz_m[None, :]
        return xyz_m

    if ba is not None:
        xlo, xhi = np.min(ba.bc.xlim), np.max(ba.bc.xlim)
        ylo, yhi = np.min(ba.bc.ylim), np.max(ba.bc.ylim)
        zlo, zhi = np.min(ba.bc.zlim), np.max(ba.bc.zlim)
    else:
        xlo = ylo = zlo = -np.inf
        xhi = yhi = zhi = np.inf

    def _inside_bbox(xyz_m: np.ndarray) -> np.ndarray:
        xyz_m = _ensure_2d(xyz_m)
        return (
            (xyz_m[:, 0] >= xlo)
            & (xyz_m[:, 0] <= xhi)
            & (xyz_m[:, 1] >= ylo)
            & (xyz_m[:, 1] <= yhi)
            & (xyz_m[:, 2] >= zlo)
            & (xyz_m[:, 2] <= zhi)
        )

    def _inside_regions(xyz_m: np.ndarray) -> np.ndarray:
        xyz_m = _ensure_2d(xyz_m)
        if ba is None:
            return np.ones((xyz_m.shape[0],), dtype=bool)
        try:
            region_ids = region_ids_from_xyz(ba, xyz_m)
            region_ids = np.asarray(region_ids).reshape(-1)
            return region_ids > 0
        except Exception:
            return np.ones((xyz_m.shape[0],), dtype=bool)

    def _is_in_brain(xyz_m: np.ndarray) -> np.ndarray:
        xyz_m = _ensure_2d(xyz_m)
        return _inside_bbox(xyz_m) & _inside_regions(xyz_m)

    def _cumlen_um(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[0] < 2:
            return np.array([0.0], dtype=np.float64)
        seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1e6
        return np.cumsum(np.r_[0.0, seg])

    def _resample_curve_equal_arclength(
        xyz_pts: np.ndarray, step_um: float
    ) -> np.ndarray:
        xyz_pts = np.asarray(xyz_pts, dtype=np.float64)
        if xyz_pts.shape[0] < 2:
            return xyz_pts.copy()

        s_pts = _cumlen_um(xyz_pts)
        s_pts = s_pts + np.arange(len(s_pts)) * 1e-9

        interp_kind = "cubic" if xyz_pts.shape[0] >= 4 else "linear"

        dense_step_um = min(2.0, max(0.5, step_um / 10.0))
        s_dense = np.arange(
            0.0, s_pts[-1] + dense_step_um, dense_step_um, dtype=np.float64
        )
        if s_dense.size < 2:
            s_dense = np.array([0.0, s_pts[-1]], dtype=np.float64)

        dense_xyz = np.zeros((len(s_dense), 3), dtype=np.float64)
        for d in range(3):
            f = scipy.interpolate.interp1d(
                s_pts,
                xyz_pts[:, d],
                kind=interp_kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            dense_xyz[:, d] = f(s_dense)

        s_dense_true = _cumlen_um(dense_xyz)
        s_dense_true = s_dense_true + np.arange(len(s_dense_true)) * 1e-12

        s_target = np.arange(0.0, s_dense_true[-1] + step_um, step_um, dtype=np.float64)
        if s_target.size < 2:
            s_target = np.array([0.0, s_dense_true[-1]], dtype=np.float64)

        out = np.zeros((len(s_target), 3), dtype=np.float64)
        for d in range(3):
            out[:, d] = np.interp(s_target, s_dense_true, dense_xyz[:, d])

        return out

    def _fit_endpoint_direction(
        xyz_rows: np.ndarray, side: str, k: int = 12
    ) -> np.ndarray:
        """
        Stable endpoint tangent from multiple nearby points.
        Returns outward direction from the chosen endpoint.
        """
        xyz_rows = np.asarray(xyz_rows, dtype=np.float64)
        N = xyz_rows.shape[0]
        if N < 2:
            raise ValueError(f"Need at least 2 points, got {N}")

        k = int(max(3, min(k, N)))

        if side == "start":
            pts = xyz_rows[:k].copy()
            t = np.arange(k, dtype=np.float64)
            inward_ref = pts[min(1, k - 1)] - pts[0]
        elif side == "end":
            pts = xyz_rows[-k:].copy()
            t = np.arange(k, dtype=np.float64)
            inward_ref = pts[max(0, k - 2)] - pts[-1]
        else:
            raise ValueError(f"side must be 'start' or 'end', got {side}")

        dir_vec = np.zeros(3, dtype=np.float64)
        t0 = t - t.mean()
        denom = np.dot(t0, t0) + 1e-12
        for d in range(3):
            y0 = pts[:, d] - pts[:, d].mean()
            dir_vec[d] = np.dot(t0, y0) / denom

        if side == "start":
            dir_vec = -dir_vec

        if np.dot(dir_vec, inward_ref) > 0:
            dir_vec = -dir_vec

        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
        return dir_vec

    def _append_linear_steps_inside(
        start_xyz: np.ndarray, direction: np.ndarray, n_steps_max: int
    ):
        rows = []
        cur = np.asarray(start_xyz, dtype=np.float64).copy()
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        for _ in range(n_steps_max):
            cand = cur + direction * row_step_m
            if not bool(_is_in_brain(cand)[0]):
                break
            rows.append(cand.copy())
            cur = cand

        if len(rows) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(rows, axis=0)

    # ---------------------------------------------------------
    # core comes directly from TRUE xyz
    # ---------------------------------------------------------
    core_xyz_ext = true_xyz.astype(np.float64).copy()

    # histology is used only to estimate extension directions
    hist_xyz_rows = _resample_curve_equal_arclength(xyz_picks, row_step_um)
    if hist_xyz_rows.shape[0] < 2:
        raise ValueError(
            f"Could not build histology resampled curve for pid={pid_name}"
        )

    fit_k_hist = min(40, max(4, hist_xyz_rows.shape[0] // 10))
    hist_start_dir = _fit_endpoint_direction(hist_xyz_rows, side="start", k=fit_k_hist)
    hist_end_dir = _fit_endpoint_direction(hist_xyz_rows, side="end", k=fit_k_hist)

    # orient histology directions to agree with true core directions
    core_valid_idx = np.where(true_valid)[0]
    first_i = int(core_valid_idx[0])
    last_i = int(core_valid_idx[-1])

    if len(core_valid_idx) >= 2:
        core_inward_start = (
            core_xyz_ext[core_valid_idx[min(1, len(core_valid_idx) - 1)]]
            - core_xyz_ext[first_i]
        )
        core_outward_start = -core_inward_start

        core_inward_end = (
            core_xyz_ext[last_i]
            - core_xyz_ext[core_valid_idx[max(0, len(core_valid_idx) - 2)]]
        )
        core_outward_end = core_inward_end
    else:
        core_outward_start = hist_start_dir.copy()
        core_outward_end = hist_end_dir.copy()

    if np.dot(hist_start_dir, core_outward_start) < 0:
        hist_start_dir = -hist_start_dir
    if np.dot(hist_end_dir, core_outward_end) < 0:
        hist_end_dir = -hist_end_dir

    # optionally blend histology and true-core directions so joins are smoother
    def _unit(v):
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    start_dir = _unit(0.5 * _unit(hist_start_dir) + 0.5 * _unit(core_outward_start))
    end_dir = _unit(0.5 * _unit(hist_end_dir) + 0.5 * _unit(core_outward_end))

    fallback_extra_rows_each_side = int(np.ceil(extra_channels_each_side / 2.0))
    max_steps = int(np.ceil(max_extension_um / row_step_um))
    n_extend = max_steps if ba is not None else fallback_extra_rows_each_side

    before_rows = _append_linear_steps_inside(
        core_xyz_ext[first_i], start_dir, n_extend
    )
    after_rows = _append_linear_steps_inside(core_xyz_ext[last_i], end_dir, n_extend)

    if before_rows.shape[0] > 0:
        before_rows = before_rows[::-1].copy()

    full_xyz_ext = np.concatenate(
        [before_rows, core_xyz_ext, after_rows], axis=0
    ).astype(np.float32)

    true_start = int(before_rows.shape[0])
    true_xyz_out = full_xyz_ext[true_start : true_start + C].copy()

    # exact overwrite for absolute safety
    full_xyz_ext[true_start : true_start + C] = true_xyz.astype(np.float32)

    return (
        full_xyz_ext.astype(np.float32),
        true_xyz_out.astype(np.float32),
        int(true_start),
    )


# ============================================================
# Bank creation
# ============================================================
def _make_histology_probe_bank(
    *,
    one,
    ephys: np.ndarray,  # [P,C,F_e] raw
    probe_positions: np.ndarray,  # [P,C,3] m
    probe_ids: list[int],  # indices into arrays
    pid_names: list[str],  # alyx pid name/id for each array index
    base_model,
    ctx_manager,
    handles,
    cfg: Any,
):
    """
    One entry per probe:
      {
        "pid": int,
        "pid_name": str,
        "rec_std": [C,F_e],
        "valid": [C],
        "true_xyz": [C,3],          # aligned histology window
        "hist_xyz_full": [L,3],     # extended dense trace
        "hist_ctx_full": [L,F_ctx],
        "hist_pred_full": [L,F_e],
        "hist_cosmos_full": [L],    # Cosmos region id per dense-trace point
        "true_start": int,
      }
    """
    e_mean = base_model.e_mean.detach().cpu().numpy().astype(np.float32)
    e_std = base_model.e_std.detach().cpu().numpy().astype(np.float32)

    brain_atlas = AllenAtlas()
    bank = []

    for p in tqdm(probe_ids):
        rec = ephys[p].astype(np.float32, copy=True)
        xyz = probe_positions[p].astype(np.float32, copy=True)
        valid = _valid_xyz_mask_np(xyz)

        rec_std = (rec - e_mean) / (e_std + 1e-8)
        rec_std[~valid] = 0.0

        pid_name = pid_names[p]

        hist_xyz_full, true_xyz, true_start = (
            _build_histology_trace_extended_and_aligned_window(
                one=one,
                pid_name=pid_name,
                true_xyz=xyz,  # use the actual probe positions as the core
                channel_step_um=cfg.channel_step_um,
                extra_channels_each_side=cfg.extra_trace_channels_each_side,
            )
        )

        hist_ctx_full, hist_pred_full = _predict_ctx_and_ephys_for_xyz(
            xyz_m=hist_xyz_full,
            ctx_manager=ctx_manager,
            base_model=base_model,
            handles=handles,
            batch_size=64,
        )

        # NEW: precompute Cosmos labels for the full dense trace
        try:
            hist_cosmos_full = region_ids_from_xyz(
                brain_atlas, hist_xyz_full, mapping="Cosmos"
            )
            hist_cosmos_full = np.asarray(hist_cosmos_full).reshape(-1)
        except Exception:
            hist_cosmos_full = np.zeros((hist_xyz_full.shape[0],), dtype=np.int64)

        if hist_cosmos_full.shape[0] != hist_xyz_full.shape[0]:
            raise ValueError(
                f"hist_cosmos_full shape mismatch for pid={pid_name}: "
                f"{hist_cosmos_full.shape} vs hist_xyz_full {hist_xyz_full.shape}"
            )

        bank.append(
            {
                "pid": int(p),
                "pid_name": str(pid_name),
                "rec_std": rec_std.astype(np.float32),
                "valid": valid.astype(bool),
                "true_xyz": true_xyz.astype(np.float32),
                "hist_xyz_full": hist_xyz_full.astype(np.float32),
                "hist_ctx_full": hist_ctx_full.astype(np.float32),
                "hist_pred_full": hist_pred_full.astype(np.float32),
                "hist_cosmos_full": hist_cosmos_full.astype(np.int64),  # NEW
                "true_start": int(true_start),
            }
        )

    return bank
