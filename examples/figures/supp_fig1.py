"""Supplementary Figure 1 and held-out interpolation ablation scores.

Outputs
-------
interpolation_model_ablation_scores.csv
supplementary_figure1_interpolation_ablation_coronal.pdf
supplementary_figure1_interpolation_ablation_sagittal.pdf
<model-registry>/ephys-atlas-models/ablation_checkpoints/<vintage>/
    SE_model_merfish_only_bilateral.pt
    SE_model_agea_only_bilateral.pt

The combined MERFISH + AGEA interpolation model is loaded from the canonical
Hugging Face/local registry release at revision=<vintage>.

All classical baselines are fitted only on training probes. All scores are
computed only on held-out test probes.
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_points_on_slice
from ibl_style.style import figure_style

from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig, ContextAtlasManager, LoadInsertionData,
    build_channels_plus_emptyvoxels_with_neighbors, FEATURE_LIST, get_device,
    region_ids_from_xyz,
)
from ephysatlas.spatial_encoder.model import NeighborInpaintingModel, train_hybrid
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
    RegistryError,
    split_manifest_to_builder_format,
)
from figure2 import _safe_percentile_limits, _to_um, _panel_label

METHODS = [
    "Region mean baseline",
    "Gaussian KDE 200 µm",
    "Gaussian KDE 500 µm",
    "Ridge regression",
    "Interpolation: MERFISH only",
    "Interpolation: AGEA only",
    "Interpolation: MERFISH + AGEA",
]
DISPLAY_FEATURES = ("psd_delta", "rms_ap", "peak_val")
DISPLAY_TITLES = (r"PSD delta [$\mu V^2$/Hz]", "RMS AP [µV]", "Peak value [µV]")


@dataclass
class Config:
    data_dir: Path = Path("../")

    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W26"

    # Hugging Face / local release registry.
    registry_root: Path = DEFAULT_REGISTRY_ROOT
    hf_repo_id: str | None = "AlonSaguy/ephys-atlas-models"
    hf_token: str | None = None

    # Ablation checkpoints are deliberately NOT stored in the release bundle.
    # They are supplementary-analysis artifacts rather than released models.
    ablation_checkpoint_root: Path = (
        DEFAULT_REGISTRY_ROOT / "ephys-atlas-models" / "ablation_checkpoints"
    )

    n_cell_pcs: int = 50
    n_gene_pcs: int = 50
    radius_um: int = 500
    m_max: int = 8
    batch_size_train: int = 1024
    batch_size_eval: int = 1024

    # Used only for the two modality-ablation models.
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

    ridge_alpha: float = 10.0
    min_insertions_per_slice: int = 10
    voxel_size_um: int = 200
    slice_thickness_um: int = 200
    observed_slice_thickness_um: int = 200

    seed: int = 0
    device: torch.device = get_device()

    # MERFISH-only / AGEA-only models are supplementary ablations.
    # The combined model is ALWAYS loaded from the released HF/local registry.
    force_retrain_ablation_models: bool = False
    train_merfish_only_model: bool = True
    train_agea_only_model: bool = True

    scores_csv: Path = Path("interpolation_model_ablation_scores.csv")
    coronal_figure_path: Path = Path(
        "supplementary_figure1_interpolation_ablation_coronal.pdf"
    )
    sagittal_figure_path: Path = Path(
        "supplementary_figure1_interpolation_ablation_sagittal.pdf"
    )


class BilateralAverageContextManager:
    """Average MERFISH and AGEA PCs across homologous hemispheres."""

    def __init__(self, base_manager):
        self.base = base_manager

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _sample_without_mirroring(self, xyz_m, mode="clip"):
        xyz_m = np.asarray(xyz_m, dtype=np.float32)
        indices = self.bc.xyz2i(xyz_m, mode=mode)
        Yh, Xh, Zh = self.allen_idx.shape
        xi = np.clip(np.round(indices[:, 0] / 8).astype(int), 0, Xh - 1)
        yi = np.clip(np.round(indices[:, 1] / 8).astype(int), 0, Yh - 1)
        zi = np.clip(np.round(indices[:, 2] / 8).astype(int), 0, Zh - 1)
        return {
            "cell_pc": self.cell_pca[:, xi, zi, yi].T.astype(np.float32),
            "gene_pc": self.gene_pca[:, xi, zi, yi].T.astype(np.float32),
            "allen_ix": self.allen_idx[yi, xi, zi].astype(np.int32),
        }

    def sample_context_numpy_m(self, xyz_m, mode="clip"):
        xyz_m = np.asarray(xyz_m, dtype=np.float32)
        xyz_left = xyz_m.copy(); xyz_left[:, 0] = -np.abs(xyz_left[:, 0])
        xyz_right = xyz_m.copy(); xyz_right[:, 0] = np.abs(xyz_right[:, 0])
        left = self._sample_without_mirroring(xyz_left, mode=mode)
        right = self._sample_without_mirroring(xyz_right, mode=mode)
        return {
            "cell_pc": 0.5 * (left["cell_pc"] + right["cell_pc"]),
            "gene_pc": 0.5 * (left["gene_pc"] + right["gene_pc"]),
            "allen_ix": left["allen_ix"],
        }

    def sample_context_numpy_i(self, xyz_i, s_xyz=np.array([8, 8, 8])):
        xyz_i = np.asarray(xyz_i, dtype=int)
        Xh, Zh, Yh = self.cell_pca.shape[1:]
        xi = np.clip(xyz_i[:, 0], 0, Xh - 1)
        yi = np.clip(xyz_i[:, 1], 0, Yh - 1)
        zi = np.clip(xyz_i[:, 2], 0, Zh - 1)
        xi_pair = Xh - xi - 1
        xi_left = np.minimum(xi, xi_pair)
        xi_right = np.maximum(xi, xi_pair)
        cell_left = self.cell_pca[:, xi_left, zi, yi].T.astype(np.float32)
        cell_right = self.cell_pca[:, xi_right, zi, yi].T.astype(np.float32)
        gene_left = self.gene_pca[:, xi_left, zi, yi].T.astype(np.float32)
        gene_right = self.gene_pca[:, xi_right, zi, yi].T.astype(np.float32)
        ix_hr = np.clip(xi_left * int(s_xyz[0]), 0, len(self.xscale) - 1)
        iy_hr = np.clip(yi * int(s_xyz[1]), 0, len(self.yscale) - 1)
        iz_hr = np.clip(zi * int(s_xyz[2]), 0, len(self.zscale) - 1)
        return {
            "cell_pc": 0.5 * (cell_left + cell_right),
            "gene_pc": 0.5 * (gene_left + gene_right),
            "allen_ix": self.allen_idx[iy_hr, ix_hr, iz_hr].astype(np.int32),
        }


def mirror_xyz_to_left(xyz_m):
    out = np.asarray(xyz_m, dtype=np.float32).copy()
    out[..., 0] = -np.abs(out[..., 0])
    return out


class ContextMaskedModel(torch.nn.Module):
    """Keep the full input shape while hiding one context modality."""
    def __init__(self, model, keep_slice: slice):
        super().__init__()
        self.model = model
        self.keep_slice = keep_slice

    def forward(self, ctx_q, p_q, e_n, p_n, mask_nei):
        masked = torch.zeros_like(ctx_q)
        masked[:, self.keep_slice] = ctx_q[:, self.keep_slice]
        return self.model(masked, p_q, e_n, p_n, mask_nei)


class Predictor:
    def predict(self, xyz_m: np.ndarray, ctx_raw: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class RegionMeanPredictor(Predictor):
    def __init__(self, xyz_m, y, brain_atlas):
        rid = region_ids_from_xyz(brain_atlas, xyz_m, mapping="Cosmos", mode="clip")
        self.brain_atlas = brain_atlas
        self.global_mean = np.nanmean(y, axis=0)
        self.means = {}
        for r in np.unique(rid):
            vals = y[rid == r]
            if vals.size:
                self.means[int(r)] = np.nanmean(vals, axis=0)

    def predict(self, xyz_m, ctx_raw=None):
        rid = region_ids_from_xyz(self.brain_atlas, xyz_m, mapping="Cosmos", mode="clip")
        return np.stack([self.means.get(int(r), self.global_mean) for r in rid]).astype(np.float32)


class GaussianKDEPredictor(Predictor):
    def __init__(self, xyz_m, y, sigma_um):
        self.xyz = np.asarray(xyz_m, np.float64)
        self.y = np.asarray(y, np.float64)
        self.sigma_m = float(sigma_um) * 1e-6
        self.tree = KDTree(self.xyz)
        self.global_mean = np.nanmean(self.y, axis=0)

    def predict(self, xyz_m, ctx_raw=None):
        q = np.asarray(xyz_m, np.float64)
        out = np.empty((len(q), self.y.shape[1]), dtype=np.float32)
        inds, dists = self.tree.query_radius(q, r=3*self.sigma_m, return_distance=True, sort_results=True)
        for i, (ii, dd) in enumerate(zip(inds, dists)):
            if len(ii) == 0:
                _, nearest = self.tree.query(q[i:i+1], k=1)
                out[i] = self.y[nearest[0, 0]]
                continue
            w = np.exp(-0.5 * (dd / self.sigma_m) ** 2)
            vals = self.y[ii]
            good = np.isfinite(vals)
            num = np.nansum(vals * w[:, None], axis=0)
            den = np.sum(good * w[:, None], axis=0)
            pred = np.divide(num, den, out=self.global_mean.copy(), where=den > 0)
            out[i] = pred
        return out


class RidgePredictor(Predictor):
    def __init__(self, ctx, xyz_m, y, alpha):
        self.x_mean = np.nanmean(ctx, axis=0)
        self.x_std = np.nanstd(ctx, axis=0) + 1e-6
        self.xyz_mean = np.nanmean(xyz_m, axis=0)
        self.xyz_std = np.nanstd(xyz_m, axis=0) + 1e-9
        X = self._design(ctx, xyz_m)
        self.model = Ridge(alpha=alpha)
        self.model.fit(X, y)

    def _design(self, ctx, xyz_m):
        c = (np.asarray(ctx) - self.x_mean) / self.x_std
        p = (np.asarray(xyz_m) - self.xyz_mean) / self.xyz_std
        return np.concatenate([c, p], axis=1)

    def predict(self, xyz_m, ctx_raw=None):
        if ctx_raw is None:
            raise ValueError("Ridge prediction requires raw context.")
        return self.model.predict(self._design(ctx_raw, xyz_m)).astype(np.float32)


class NeuralPredictor(Predictor):
    """Dense inference with the same context normalization used in training."""
    def __init__(self, model, ctx_manager, train_xyz_m, train_y_std, e_mean, e_std,
                 ctx_mean, ctx_std, device, context_slice=None,
                 radius_um=500, m_max=8, batch_size=2048):
        self.model = model.eval().to(device)
        self.ctx_manager = ctx_manager
        self.train_xyz = np.asarray(train_xyz_m, np.float64)
        self.train_y_std = np.asarray(train_y_std, np.float32)
        self.tree = KDTree(self.train_xyz)
        self.e_mean = np.asarray(e_mean, np.float32)
        self.e_std = np.asarray(e_std, np.float32)
        self.ctx_mean = np.asarray(ctx_mean, np.float32)
        self.ctx_std = np.asarray(ctx_std, np.float32)
        self.device = device
        self.context_slice = context_slice
        self.radius_m = radius_um * 1e-6
        self.m_max = m_max
        self.batch_size = batch_size

    def _prepare_context(self, ctx_raw):
        ctx_raw = np.asarray(ctx_raw, dtype=np.float32)
        ctx_std = (ctx_raw - self.ctx_mean[None]) / (self.ctx_std[None] + 1e-8)
        if self.context_slice is not None:
            masked = np.zeros_like(ctx_std)
            masked[:, self.context_slice] = ctx_std[:, self.context_slice]
            ctx_std = masked
        return ctx_std.astype(np.float32)

    @torch.no_grad()
    def predict(self, xyz_m, ctx_raw=None):
        xyz_m = np.asarray(xyz_m, np.float32)
        if ctx_raw is None:
            pack = self.ctx_manager.sample_context_numpy_m(xyz_m, mode="clip")
            ctx_raw = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1)
        ctx_model = self._prepare_context(ctx_raw)
        out = []
        for i0 in range(0, len(xyz_m), self.batch_size):
            q = xyz_m[i0:i0+self.batch_size]
            c = ctx_model[i0:i0+self.batch_size]
            neigh = self.tree.query_radius(q, r=self.radius_m, return_distance=False)
            B = len(q); F = self.train_y_std.shape[1]
            e_n = np.zeros((B, self.m_max, F), np.float32)
            p_n = np.zeros((B, self.m_max, 3), np.float32)
            mask = np.zeros((B, self.m_max), bool)
            for b, ii in enumerate(neigh):
                if len(ii) > self.m_max:
                    d2 = np.sum((self.train_xyz[ii] - q[b][None])**2, axis=1)
                    ii = ii[np.argsort(d2)[:self.m_max]]
                L = len(ii)
                if L:
                    e_n[b, :L] = self.train_y_std[ii]
                    p_n[b, :L] = self.train_xyz[ii]
                    mask[b, :L] = True
            tensors = [
                torch.as_tensor(c, dtype=torch.float32, device=self.device),
                torch.as_tensor(q, dtype=torch.float32, device=self.device),
                torch.as_tensor(e_n, dtype=torch.float32, device=self.device),
                torch.as_tensor(p_n, dtype=torch.float32, device=self.device),
                torch.as_tensor(mask, dtype=torch.bool, device=self.device),
            ]
            _, mu_std = self.model(*tensors)
            out.append(mu_std.float().cpu().numpy() * self.e_std[None] + self.e_mean[None])
        return np.concatenate(out, axis=0)


def loader_arrays(loader, e_mean, e_std, ctx_manager):
    xyzs, ys, ctxs = [], [], []
    for batch in loader:
        ctx_q, p_q, _, _, _, has, y_std, *_ = batch
        m = has.bool()
        xyz = p_q[m].cpu().numpy()
        y = y_std[m].cpu().numpy() * e_std[None] + e_mean[None]
        pack = ctx_manager.sample_context_numpy_m(xyz, mode="clip")
        ctx = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1)
        xyzs.append(xyz); ys.append(y); ctxs.append(ctx)
    return np.concatenate(xyzs), np.concatenate(ys), np.concatenate(ctxs)


@torch.no_grad()
def evaluate_neural_model_on_loader(model, loader, e_mean, e_std, device):
    """Authoritative held-out evaluation using standardized loader inputs."""
    model.eval().to(device)
    device_type = device.type
    use_autocast = device_type == "cuda"
    e_mean_t = torch.as_tensor(e_mean, dtype=torch.float32, device=device)
    e_std_t = torch.as_tensor(e_std, dtype=torch.float32, device=device)
    ys, preds = [], []
    for batch in loader:
        ctx_q, p_q, e_n, p_n, mask, has, y_std, *_ = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu_std = model(ctx_q, p_q, e_n, p_n, mask)
        m = has.bool()
        if not m.any():
            continue
        ys.append((y_std[m].float() * e_std_t + e_mean_t).cpu().numpy())
        preds.append((mu_std[m].float() * e_std_t + e_mean_t).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def r2_per_feature(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ss_res = np.nansum((y-pred)**2, axis=0)
    mean = np.nanmean(y, axis=0)
    ss_tot = np.nansum((y-mean)**2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def build_base_model(cfg, f_ctx, f_e, e_mean_t, e_std_t, ctx_mean, ctx_std):
    return NeighborInpaintingModel(
        f_ctx=f_ctx, f_ephys=f_e, f_out=f_e,
        e_mean=e_mean_t, e_std=e_std_t, ctx_mean=ctx_mean, ctx_std=ctx_std,
        d_model=cfg.d_model, nhead=cfg.nhead, depth=cfg.depth, drop=cfg.drop,
    ).to(cfg.device)


def _resolve_verified_release(cfg):
    """
    Resolve the requested vintage from the local registry or Hugging Face.

    If a stale/corrupted local copy exists, force-refresh it once from HF.
    """
    registry = EphysAtlasReleaseRegistry(cfg.registry_root)

    release_dir = registry.resolve_release(
        cfg.vintage,
        repo_id=cfg.hf_repo_id,
        token=cfg.hf_token,
        require_weights=True,
    )

    try:
        registry.verify_checksums(cfg.vintage)
    except RegistryError:
        if not cfg.hf_repo_id:
            raise
        print(
            f"[registry] local {cfg.vintage} failed checksum verification; "
            "refreshing from Hugging Face."
        )
        release_dir = registry.download_release_from_hf(
            cfg.vintage,
            repo_id=cfg.hf_repo_id,
            token=cfg.hf_token,
            force=True,
        )
        registry.verify_checksums(cfg.vintage)

    registry.validate_feature_order(cfg.vintage, FEATURE_LIST)

    return (
        registry,
        release_dir,
        registry.load_config(cfg.vintage),
        split_manifest_to_builder_format(registry.load_split(cfg.vintage)),
        registry.load_channel_preprocessing_stats(cfg.vintage),
    )


def _apply_release_settings(cfg, release_config):
    """Make data/context/neighborhood settings authoritative from the release."""
    data_cfg = release_config.get("data", {})
    context_cfg = release_config.get("context", {})
    channel_cfg = release_config.get("channel_level", {})
    neighbor_cfg = channel_cfg.get("neighbors", {})

    saved_vintage = str(data_cfg.get("vintage", cfg.vintage))
    if saved_vintage != cfg.vintage:
        raise RegistryError(
            f"Requested vintage={cfg.vintage}, release contains {saved_vintage}."
        )

    cfg.project = str(data_cfg.get("project", cfg.project))
    cfg.agg = str(data_cfg.get("agg", cfg.agg))
    cfg.n_cell_pcs = int(context_cfg.get("n_cell_pcs", cfg.n_cell_pcs))
    cfg.n_gene_pcs = int(context_cfg.get("n_gene_pcs", cfg.n_gene_pcs))
    cfg.radius_um = int(neighbor_cfg.get("radius_um", cfg.radius_um))
    cfg.m_max = int(neighbor_cfg.get("m_max", cfg.m_max))


def load_released_combined_model(
    *,
    release_dir,
    preprocessing_stats,
    device,
):
    """Load the exact released MERFISH+AGEA spatial encoder."""
    ckpt_path = release_dir / "models" / "channel" / "spatial_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    arch = ckpt.get("architecture", {})
    required = ("f_ctx", "f_ephys", "f_out", "d_model", "nhead", "depth", "drop")
    missing = [key for key in required if key not in arch]
    if missing:
        raise RuntimeError(
            f"Released checkpoint is missing architecture fields: {missing}"
        )

    def stat(name):
        if name not in preprocessing_stats:
            raise RuntimeError(
                f"Released preprocessing statistics are missing {name!r}."
            )
        return torch.as_tensor(preprocessing_stats[name], dtype=torch.float32)

    model = NeighborInpaintingModel(
        f_ctx=int(arch["f_ctx"]),
        f_ephys=int(arch["f_ephys"]),
        f_out=int(arch["f_out"]),
        e_mean=stat("e_mean"),
        e_std=stat("e_std"),
        ctx_mean=stat("ctx_mean"),
        ctx_std=stat("ctx_std"),
        d_model=int(arch["d_model"]),
        nhead=int(arch["nhead"]),
        depth=int(arch["depth"]),
        drop=float(arch["drop"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"[model] loaded released combined model: {ckpt_path}")
    return model


def train_or_load_ablation_model(
    cfg,
    *,
    name,
    train_loader,
    val_loader,
    f_ctx,
    f_e,
    e_mean_t,
    e_std_t,
    ctx_mean,
    ctx_std,
    keep_slice,
    enabled=True,
):
    """
    Train/load a supplementary modality-ablation model.

    These checkpoints are intentionally kept outside the Hugging Face release
    because the released combined model is the canonical model.
    """
    out_dir = cfg.ablation_checkpoint_root / cfg.vintage
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"SE_model_{name}.pt"

    seed_offset = {
        "merfish_only_bilateral": 0,
        "agea_only_bilateral": 1,
    }.get(name, 0)
    model_seed = int(cfg.seed + seed_offset)

    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    base = build_base_model(
        cfg,
        f_ctx,
        f_e,
        e_mean_t,
        e_std_t,
        ctx_mean,
        ctx_std,
    )
    model = ContextMaskedModel(base, keep_slice).to(cfg.device)

    should_train = bool(
        cfg.force_retrain_ablation_models
        or not ckpt_path.exists()
    )

    if not should_train:
        print(f"[model] loading ablation {name}: {ckpt_path}")
        state = torch.load(
            ckpt_path,
            map_location=cfg.device,
            weights_only=False,
        )
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        return model

    if not enabled:
        raise FileNotFoundError(
            f"Ablation checkpoint does not exist for {name}: {ckpt_path}. "
            "Enable training for this ablation."
        )

    print(f"\n[model] training ablation {name}")
    print(f"[model] checkpoint: {ckpt_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    model, meters, best_epoch, best_value = train_hybrid(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=cfg.epochs,
        device=cfg.device,
        lambda_sup=1.0,
        lambda_ctr=cfg.lambda_ctr,
        pos_radius_um=cfg.pos_radius_um,
        patience=cfg.patience,
        checkpoint_path=None,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "meters": meters,
            "best_epoch": best_epoch,
            "best_value": best_value,
            "name": name,
            "keep_slice": {
                "start": keep_slice.start,
                "stop": keep_slice.stop,
                "step": keep_slice.step,
            },
            "data_vintage": cfg.vintage,
            "config": {
                "f_ctx": f_ctx,
                "f_e": f_e,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "depth": cfg.depth,
                "drop": cfg.drop,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "lambda_ctr": cfg.lambda_ctr,
                "pos_radius_um": cfg.pos_radius_um,
                "patience": cfg.patience,
                "seed": model_seed,
            },
        },
        ckpt_path,
    )

    print(
        f"[model] saved {name}: best_epoch={best_epoch}, "
        f"best_value={best_value:.6g}"
    )
    return model


def save_scores(path, score_map):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "feature", "feature_index", "r2"])
        w.writeheader()
        for method, scores in score_map.items():
            for i, (feature, score) in enumerate(zip(FEATURE_LIST, scores)):
                w.writerow({"method": method, "feature": feature, "feature_index": i, "r2": float(score)})



def _axis_scale_um(brain_atlas, axis):
    scale = {
        "x": brain_atlas.bc.xscale,
        "y": brain_atlas.bc.yscale,
        "z": brain_atlas.bc.zscale,
    }[axis]
    return _to_um(np.asarray(scale, dtype=float))


def _valid_probe_xyz_um(probe_positions):
    xyz_um = _to_um(probe_positions)
    valid = np.isfinite(xyz_um).all(axis=-1) & ~np.all(xyz_um == 0, axis=-1)
    return xyz_um, valid


def _count_insertions_in_slice(probe_positions_um, valid, *, view, coord_um, thickness_um):
    """Count unique probes with at least one valid channel inside the slab."""
    axis = 1 if view == "coronal" else 0
    in_slab = valid & (
        np.abs(probe_positions_um[..., axis] - float(coord_um))
        <= float(thickness_um) / 2
    )
    return int(np.sum(np.any(in_slab, axis=1)))


def _choose_slice_near_center(
    *,
    probe_positions,
    brain_atlas,
    view,
    thickness_um,
    voxel_size_um,
    min_insertions=10,
):
    """
    Select the qualifying slab closest to the anatomical center.

    Coronal candidates vary along AP (y). Sagittal candidates vary along ML
    (x), but only the mirrored left hemisphere (x <= 0) is considered.
    """
    if view not in {"coronal", "sagittal"}:
        raise ValueError(f"Unsupported view: {view}")

    pos_um, valid = _valid_probe_xyz_um(probe_positions)
    pos_um = mirror_xyz_to_left(pos_um / 1e6) * 1e6

    axis_name = "y" if view == "coronal" else "x"
    axis_vals = _axis_scale_um(brain_atlas, axis_name)

    if view == "sagittal":
        axis_vals = axis_vals[axis_vals <= 0]

    lo = np.ceil(np.nanmin(axis_vals) / voxel_size_um) * voxel_size_um
    hi = np.floor(np.nanmax(axis_vals) / voxel_size_um) * voxel_size_um
    candidates = np.arange(lo, hi + voxel_size_um, voxel_size_um, dtype=float)

    # The sagittal center is the midpoint of the canonical left hemisphere,
    # not the x=0 midline. The coronal center is the AP midpoint.
    center_um = 0.5 * (float(np.nanmin(axis_vals)) + float(np.nanmax(axis_vals)))

    records = []
    for coord in candidates:
        n = _count_insertions_in_slice(
            pos_um,
            valid,
            view=view,
            coord_um=coord,
            thickness_um=thickness_um,
        )
        records.append((float(coord), int(n), abs(float(coord) - center_um)))

    qualifying = [r for r in records if r[1] >= int(min_insertions)]
    if qualifying:
        chosen = min(qualifying, key=lambda r: (r[2], -r[1]))
    else:
        # Graceful fallback: maximize insertion count, then prefer the center.
        chosen = min(records, key=lambda r: (-r[1], r[2]))
        print(
            f"[slice selection warning] No {view} slab contained "
            f"{min_insertions} held-out insertions. Using the best available "
            f"slice with {chosen[1]} insertions."
        )

    print(
        f"[slice selection] {view}: coord={chosen[0]:.0f} µm, "
        f"held-out insertions={chosen[1]}, center={center_um:.0f} µm"
    )
    return chosen[0], chosen[1]


def _build_slice_grid(
    *, brain_atlas, view, coord_um, voxel_size_um, slice_thickness_um
):
    """Build either a coronal (x-z) or sagittal (y-z) 200 µm grid."""
    if view not in {"coronal", "sagittal"}:
        raise ValueError(f"Unsupported view: {view}")

    x_atlas = _axis_scale_um(brain_atlas, "x")
    y_atlas = _axis_scale_um(brain_atlas, "y")
    z_atlas = _axis_scale_um(brain_atlas, "z")
    label_raw = brain_atlas.label

    if label_raw.shape != (len(y_atlas), len(x_atlas), len(z_atlas)):
        raise ValueError(
            "Expected brain_atlas.label layout [Y, X, Z], "
            f"got {label_raw.shape}."
        )

    xo, yo, zo = np.argsort(x_atlas), np.argsort(y_atlas), np.argsort(z_atlas)
    x_atlas, y_atlas, z_atlas = x_atlas[xo], y_atlas[yo], z_atlas[zo]
    label = label_raw[np.ix_(yo, xo, zo)]

    def nearest_indices(axis, values):
        idx = np.searchsorted(axis, values)
        idx = np.clip(idx, 1, len(axis) - 1)
        left, right = axis[idx - 1], axis[idx]
        return np.where(np.abs(values - left) <= np.abs(values - right), idx - 1, idx)

    if view == "coronal":
        fixed_axis = y_atlas
        fixed_inds = np.where(
            np.abs(fixed_axis - coord_um) <= slice_thickness_um / 2
        )[0]
        if len(fixed_inds) == 0:
            fixed_inds = np.array([np.argmin(np.abs(fixed_axis - coord_um))])
        coord_actual = float(np.mean(fixed_axis[fixed_inds]))

        h_grid = np.arange(
            np.floor(x_atlas.min() / voxel_size_um) * voxel_size_um,
            np.ceil(x_atlas.max() / voxel_size_um) * voxel_size_um + voxel_size_um,
            voxel_size_um,
        )
        v_grid = np.arange(
            np.floor(z_atlas.min() / voxel_size_um) * voxel_size_um,
            np.ceil(z_atlas.max() / voxel_size_um) * voxel_size_um + voxel_size_um,
            voxel_size_um,
        )
        H, V = np.meshgrid(h_grid, v_grid, indexing="ij")
        hi = nearest_indices(x_atlas, H.ravel())
        vi = nearest_indices(z_atlas, V.ravel())
        inside = np.any(label[fixed_inds, :, :][:, hi, vi] > 0, axis=0)
        mask = inside.reshape(H.shape)
        xyz_um = np.column_stack(
            [H[mask], np.full(mask.sum(), coord_actual), V[mask]]
        )
        horizontal_name = "ML"
    else:
        fixed_axis = x_atlas
        fixed_inds = np.where(
            np.abs(fixed_axis - coord_um) <= slice_thickness_um / 2
        )[0]
        if len(fixed_inds) == 0:
            fixed_inds = np.array([np.argmin(np.abs(fixed_axis - coord_um))])
        coord_actual = float(np.mean(fixed_axis[fixed_inds]))

        h_grid = np.arange(
            np.floor(y_atlas.min() / voxel_size_um) * voxel_size_um,
            np.ceil(y_atlas.max() / voxel_size_um) * voxel_size_um + voxel_size_um,
            voxel_size_um,
        )
        v_grid = np.arange(
            np.floor(z_atlas.min() / voxel_size_um) * voxel_size_um,
            np.ceil(z_atlas.max() / voxel_size_um) * voxel_size_um + voxel_size_um,
            voxel_size_um,
        )
        H, V = np.meshgrid(h_grid, v_grid, indexing="ij")
        hi = nearest_indices(y_atlas, H.ravel())
        vi = nearest_indices(z_atlas, V.ravel())
        inside = np.any(label[:, fixed_inds, :][hi, :, vi] > 0, axis=1)
        mask = inside.reshape(H.shape)
        xyz_um = np.column_stack(
            [np.full(mask.sum(), coord_actual), H[mask], V[mask]]
        )
        horizontal_name = "AP"

    return {
        "view": view,
        "h_grid": h_grid,
        "v_grid": v_grid,
        "rid_mask": mask,
        "slice_xyz_um": xyz_um,
        "coord_um_actual": coord_actual,
        "horizontal_name": horizontal_name,
    }


def _observed_voxel_average_image_view(
    *, ephys, probe_positions_um, feature_list, feature,
    h_grid, v_grid, coord_um_actual, observed_slice_thickness_um, view
):
    xyz = np.asarray(probe_positions_um).reshape(-1, 3)
    eph = np.asarray(ephys).reshape(-1, np.asarray(ephys).shape[-1])
    valid = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0, axis=1)

    fixed_axis = 1 if view == "coronal" else 0
    horizontal_axis = 0 if view == "coronal" else 1
    in_slice = valid & (
        np.abs(xyz[:, fixed_axis] - coord_um_actual)
        <= observed_slice_thickness_um / 2
    )
    xyz, eph = xyz[in_slice], eph[in_slice]

    sums = np.zeros((len(h_grid), len(v_grid)), dtype=float)
    counts = np.zeros_like(sums)
    if len(xyz) == 0:
        return np.full_like(sums, np.nan)

    def nearest(axis, vals):
        idx = np.searchsorted(axis, vals)
        idx = np.clip(idx, 1, len(axis) - 1)
        return np.where(
            np.abs(vals - axis[idx - 1]) <= np.abs(vals - axis[idx]),
            idx - 1,
            idx,
        )

    hi = nearest(h_grid, xyz[:, horizontal_axis])
    vi = nearest(v_grid, xyz[:, 2])
    vals = eph[:, feature_list.index(feature)]
    good = np.isfinite(vals)
    np.add.at(sums, (hi[good], vi[good]), vals[good])
    np.add.at(counts, (hi[good], vi[good]), 1)
    out = np.full_like(sums, np.nan)
    out[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return out


def _draw_slice_heatmap_view(
    *, fig, ax, img, grid, voxel_size_um, title, cmap, vmin, vmax
):
    plot_points_on_slice(
        np.zeros((0, 3), dtype=float),
        coord=int(grid["coord_um_actual"]),
        slice=grid["view"],
        ax=ax,
        cmap="Greys",
    )
    im = ax.imshow(
        img.T,
        origin="lower",
        extent=[
            grid["h_grid"].min() - voxel_size_um / 2,
            grid["h_grid"].max() + voxel_size_um / 2,
            grid["v_grid"].min() - voxel_size_um / 2,
            grid["v_grid"].max() + voxel_size_um / 2,
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    return im


def plot_supplementary_view(
    cfg,
    predictors,
    score_map,
    test_ephys,
    test_positions,
    ctx_manager,
    brain_atlas,
    *,
    view,
    coord_um,
    n_insertions,
    save_path,
):
    figure_style()
    grid = _build_slice_grid(
        brain_atlas=brain_atlas,
        view=view,
        coord_um=coord_um,
        voxel_size_um=cfg.voxel_size_um,
        slice_thickness_um=cfg.slice_thickness_um,
    )

    xyz_um = grid["slice_xyz_um"]
    xyz_m_left = mirror_xyz_to_left(xyz_um / 1e6)
    # Ridge uses the bilateral-average context used for the ablation analysis.
    # NeuralPredictor instances sample from their own context manager:
    #   - modality ablations -> bilateral-average manager
    #   - released combined model -> exact release manager
    pack = ctx_manager.sample_context_numpy_m(xyz_m_left, mode="clip")
    ctx_left_bilateral = np.concatenate(
        [pack["cell_pc"], pack["gene_pc"]],
        axis=1,
    )

    pred_all = {}
    for name, predictor in predictors.items():
        if isinstance(predictor, RidgePredictor):
            pred_all[name] = predictor.predict(
                xyz_m_left,
                ctx_left_bilateral,
            )
        else:
            pred_all[name] = predictor.predict(
                xyz_m_left,
                None,
            )

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(
        len(DISPLAY_FEATURES),
        len(METHODS) + 2,
        width_ratios=[1] * len(METHODS) + [1, 0.05],
        hspace=0.13,
        wspace=0.025,
    )

    test_positions_left_um = _to_um(mirror_xyz_to_left(test_positions))
    for row, (feature, row_title) in enumerate(zip(DISPLAY_FEATURES, DISPLAY_TITLES)):
        fi = FEATURE_LIST.index(feature)
        obs = _observed_voxel_average_image_view(
            ephys=test_ephys,
            probe_positions_um=test_positions_left_um,
            feature_list=FEATURE_LIST,
            feature=feature,
            h_grid=grid["h_grid"],
            v_grid=grid["v_grid"],
            coord_um_actual=grid["coord_um_actual"],
            observed_slice_thickness_um=cfg.observed_slice_thickness_um,
            view=view,
        )

        imgs = []
        for method in METHODS:
            img = np.full(grid["rid_mask"].shape, np.nan)
            img[grid["rid_mask"]] = pred_all[method][:, fi]
            imgs.append(img)

        finite = [im[np.isfinite(im)] for im in imgs + [obs] if np.isfinite(im).any()]
        vals = np.concatenate(finite)
        vmin, vmax = _safe_percentile_limits(vals)

        last = None
        for col, (method, img) in enumerate(zip(METHODS, imgs)):
            ax = fig.add_subplot(gs[row, col])
            title = f"{method}\n$R^2$ = {float(score_map[method][fi]):.3f}"
            last = _draw_slice_heatmap_view(
                fig=fig,
                ax=ax,
                img=img,
                grid=grid,
                voxel_size_um=cfg.voxel_size_um,
                title=title,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            if col == 0:
                ax.text(
                    -0.04, 0.5, row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                )
                if row == 0:
                    _panel_label(ax, "a")

        ax_obs = fig.add_subplot(gs[row, len(METHODS)])
        last = _draw_slice_heatmap_view(
            fig=fig,
            ax=ax_obs,
            img=obs,
            grid=grid,
            voxel_size_um=cfg.voxel_size_um,
            title="Held-out observations\nmirrored to x < 0",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        cax = fig.add_subplot(gs[row, len(METHODS) + 1])
        fig.colorbar(last, cax=cax)

    axis_label = "AP" if view == "coronal" else "ML"
    fig.suptitle(
        f"{view.capitalize()} view: {axis_label} = {grid['coord_um_actual']:.0f} µm "
        f"({n_insertions} held-out insertions)",
        y=0.995,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.025)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    cfg = Config(
        # False = reuse existing supplementary ablation checkpoints when present.
        force_retrain_ablation_models=False,
        train_merfish_only_model=True,
        train_agea_only_model=True,
        seed=0,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    one = ONE()
    brain_atlas = AllenAtlas()

    # ================================================================
    # Resolve the canonical combined model release.
    # ================================================================
    (
        registry,
        release_dir,
        release_config,
        split_manifest,
        release_preprocessing_stats,
    ) = _resolve_verified_release(cfg)

    _apply_release_settings(cfg, release_config)

    print(f"[registry] release: {release_dir}")
    print(f"[registry] vintage: {cfg.vintage}")

    released_combined = load_released_combined_model(
        release_dir=release_dir,
        preprocessing_stats=release_preprocessing_stats,
        device=cfg.device,
    )

    # ================================================================
    # Frozen PCA volumes from the release.
    # ================================================================
    base_ctx_manager = ContextAtlasManager(
        AtlasPCAConfig(
            n_cell_pcs=cfg.n_cell_pcs,
            n_gene_pcs=cfg.n_gene_pcs,
        ),
        regenerate_context=False,
        output_dir=release_dir / "context",
    )

    # The supplementary ablations intentionally average homologous
    # hemispheres, preserving the behavior of the previous analysis.
    bilateral_ctx_manager = BilateralAverageContextManager(
        base_ctx_manager
    )

    # ================================================================
    # Source data from the same vintage.
    # ================================================================
    pid_names, ephys, positions, _ = LoadInsertionData(
        project=cfg.project,
        agg=cfg.agg,
        VINTAGE=cfg.vintage,
        path_data=cfg.data_dir,
    )
    pid_names = [str(x) for x in pid_names]

    # ================================================================
    # A. Exact released-model loaders.
    #
    # These use:
    #   - exact saved PID split
    #   - exact saved context normalization
    #   - exact saved ephys clipping/normalization
    #   - standard mirrored-left context used by the released model
    # ================================================================
    release_loaders = build_channels_plus_emptyvoxels_with_neighbors(
        base_ctx_manager,
        ephys,
        positions,
        RADIUS_UM=cfg.radius_um,
        M_MAX=cfg.m_max,
        pid_names=pid_names,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        seed=cfg.seed,
        split_manifest=split_manifest,
        preprocessing_stats=release_preprocessing_stats,
    )

    (
        release_train_loader,
        _,
        _,
        release_test_loader,
        release_e_mean_t,
        release_e_std_t,
        release_ctx_mean_t,
        release_ctx_std_t,
        release_split_info,
    ) = release_loaders

    # ================================================================
    # B. Bilateral-context loaders for the supplementary ablations.
    #
    # We preserve the released split and all EPHYS preprocessing, but
    # recompute context mean/std because bilateral averaging changes the
    # context distribution relative to the canonical released model.
    # ================================================================
    bilateral_stats = {
        key: value
        for key, value in release_preprocessing_stats.items()
        if key not in {"ctx_mean", "ctx_std"}
    }

    bilateral_loaders = build_channels_plus_emptyvoxels_with_neighbors(
        bilateral_ctx_manager,
        ephys,
        positions,
        RADIUS_UM=cfg.radius_um,
        M_MAX=cfg.m_max,
        pid_names=pid_names,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        seed=cfg.seed,
        split_manifest=split_manifest,
        preprocessing_stats=bilateral_stats,
    )

    (
        train_loader,
        _,
        val_loader,
        test_loader,
        e_mean_t,
        e_std_t,
        ctx_mean,
        ctx_std,
        split_info,
    ) = bilateral_loaders

    # Both loader families MUST correspond to the same held-out PIDs.
    if (
        list(split_info["p_te_names"])
        != list(release_split_info["p_te_names"])
    ):
        raise RuntimeError(
            "Released and bilateral loaders do not use the same test PIDs."
        )

    e_mean = e_mean_t.cpu().numpy()
    e_std = e_std_t.cpu().numpy()

    release_e_mean = release_e_mean_t.cpu().numpy()
    release_e_std = release_e_std_t.cpu().numpy()

    # Classical baselines + Ridge use the bilateral-analysis loaders.
    train_xyz, train_y, train_ctx = loader_arrays(
        train_loader,
        e_mean,
        e_std,
        bilateral_ctx_manager,
    )
    test_xyz, test_y, test_ctx = loader_arrays(
        test_loader,
        e_mean,
        e_std,
        bilateral_ctx_manager,
    )

    if (
        np.nanmax(train_xyz[:, 0]) > 1e-10
        or np.nanmax(test_xyz[:, 0]) > 1e-10
    ):
        raise RuntimeError(
            "Training/test coordinates were not fully mirrored to x <= 0."
        )

    print(
        "[mirroring] all ephys coordinates use x <= 0; "
        "ablation/Ridge context PCs are bilateral averages"
    )

    # Neighbor bank for bilateral ablation models.
    collate = train_loader.collate_fn
    bank_xyz = np.asarray(collate.bank_xyz, np.float32)
    bank_y_std = np.asarray(collate.bank_feat, np.float32)

    # Exact released-model train bank for dense combined-model inference.
    release_collate = release_train_loader.collate_fn
    release_bank_xyz = np.asarray(
        release_collate.bank_xyz,
        np.float32,
    )
    release_bank_y_std = np.asarray(
        release_collate.bank_feat,
        np.float32,
    )

    f_ctx = int(ctx_mean.numel())
    f_e = int(e_mean_t.numel())
    n_cell = cfg.n_cell_pcs

    if f_e != len(FEATURE_LIST):
        raise RuntimeError(
            f"Feature mismatch: loader={f_e}, FEATURE_LIST={len(FEATURE_LIST)}"
        )

    # Context concatenation:
    # [MERFISH PCs (0:n_cell), AGEA PCs (n_cell:f_ctx)]
    merfish_model = train_or_load_ablation_model(
        cfg,
        name="merfish_only_bilateral",
        keep_slice=slice(0, n_cell),
        enabled=cfg.train_merfish_only_model,
        train_loader=train_loader,
        val_loader=val_loader,
        f_ctx=f_ctx,
        f_e=f_e,
        e_mean_t=e_mean_t,
        e_std_t=e_std_t,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
    )

    agea_model = train_or_load_ablation_model(
        cfg,
        name="agea_only_bilateral",
        keep_slice=slice(n_cell, f_ctx),
        enabled=cfg.train_agea_only_model,
        train_loader=train_loader,
        val_loader=val_loader,
        f_ctx=f_ctx,
        f_e=f_e,
        e_mean_t=e_mean_t,
        e_std_t=e_std_t,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
    )

    # ================================================================
    # Predictors
    # ================================================================
    predictors = {
        METHODS[0]: RegionMeanPredictor(
            train_xyz,
            train_y,
            brain_atlas,
        ),
        METHODS[1]: GaussianKDEPredictor(
            train_xyz,
            train_y,
            200,
        ),
        METHODS[2]: GaussianKDEPredictor(
            train_xyz,
            train_y,
            500,
        ),
        METHODS[3]: RidgePredictor(
            train_ctx,
            train_xyz,
            train_y,
            cfg.ridge_alpha,
        ),
        METHODS[4]: NeuralPredictor(
            merfish_model,
            bilateral_ctx_manager,
            bank_xyz,
            bank_y_std,
            e_mean,
            e_std,
            ctx_mean.cpu().numpy(),
            ctx_std.cpu().numpy(),
            cfg.device,
            context_slice=slice(0, n_cell),
            radius_um=cfg.radius_um,
            m_max=cfg.m_max,
        ),
        METHODS[5]: NeuralPredictor(
            agea_model,
            bilateral_ctx_manager,
            bank_xyz,
            bank_y_std,
            e_mean,
            e_std,
            ctx_mean.cpu().numpy(),
            ctx_std.cpu().numpy(),
            cfg.device,
            context_slice=slice(n_cell, f_ctx),
            radius_um=cfg.radius_um,
            m_max=cfg.m_max,
        ),
        METHODS[6]: NeuralPredictor(
            released_combined,
            base_ctx_manager,
            release_bank_xyz,
            release_bank_y_std,
            release_e_mean,
            release_e_std,
            release_ctx_mean_t.cpu().numpy(),
            release_ctx_std_t.cpu().numpy(),
            cfg.device,
            context_slice=None,
            radius_um=cfg.radius_um,
            m_max=cfg.m_max,
        ),
    }

    # ================================================================
    # Held-out scores
    # ================================================================
    score_map = {}

    for method in METHODS[:4]:
        print(f"Evaluating {method}")
        pred = predictors[method].predict(
            test_xyz,
            test_ctx if method == METHODS[3] else None,
        )
        score_map[method] = r2_per_feature(
            test_y,
            pred,
        )
        print(
            f"  mean R2 = {np.nanmean(score_map[method]):.4f}"
        )

    # Bilateral modality-ablation models.
    for method, model in {
        METHODS[4]: merfish_model,
        METHODS[5]: agea_model,
    }.items():
        print(f"Evaluating {method} with bilateral test_loader")
        y_eval, pred_eval = evaluate_neural_model_on_loader(
            model,
            test_loader,
            e_mean,
            e_std,
            cfg.device,
        )
        score_map[method] = r2_per_feature(
            y_eval,
            pred_eval,
        )
        print(
            f"  mean R2 = {np.nanmean(score_map[method]):.4f}"
        )

    # Canonical released combined model: exact release test loader.
    print(
        f"Evaluating {METHODS[6]} using released model "
        f"revision={cfg.vintage}"
    )
    y_eval, pred_eval = evaluate_neural_model_on_loader(
        released_combined,
        release_test_loader,
        release_e_mean,
        release_e_std,
        cfg.device,
    )
    score_map[METHODS[6]] = r2_per_feature(
        y_eval,
        pred_eval,
    )
    print(
        f"  mean R2 = {np.nanmean(score_map[METHODS[6]]):.4f}"
    )

    save_scores(
        cfg.scores_csv,
        score_map,
    )

    # Exact released test PIDs are authoritative for displayed observations.
    test_ids = np.asarray(
        release_split_info["p_te_ids"],
        dtype=int,
    )
    test_ephys = ephys[test_ids]
    test_positions = positions[test_ids]

    coronal_coord_um, coronal_n = _choose_slice_near_center(
        probe_positions=test_positions,
        brain_atlas=brain_atlas,
        view="coronal",
        thickness_um=cfg.observed_slice_thickness_um,
        voxel_size_um=cfg.voxel_size_um,
        min_insertions=cfg.min_insertions_per_slice,
    )

    sagittal_coord_um, sagittal_n = _choose_slice_near_center(
        probe_positions=test_positions,
        brain_atlas=brain_atlas,
        view="sagittal",
        thickness_um=cfg.observed_slice_thickness_um,
        voxel_size_um=cfg.voxel_size_um,
        min_insertions=cfg.min_insertions_per_slice,
    )

    plot_supplementary_view(
        cfg,
        predictors,
        score_map,
        test_ephys,
        test_positions,
        bilateral_ctx_manager,
        brain_atlas,
        view="coronal",
        coord_um=coronal_coord_um,
        n_insertions=coronal_n,
        save_path=cfg.coronal_figure_path,
    )

    plot_supplementary_view(
        cfg,
        predictors,
        score_map,
        test_ephys,
        test_positions,
        bilateral_ctx_manager,
        brain_atlas,
        view="sagittal",
        coord_um=sagittal_coord_um,
        n_insertions=sagittal_n,
        save_path=cfg.sagittal_figure_path,
    )

    print(f"saved {cfg.scores_csv}")
    print(f"saved {cfg.coronal_figure_path}")
    print(f"saved {cfg.sagittal_figure_path}")




if __name__ == "__main__":
    main()
