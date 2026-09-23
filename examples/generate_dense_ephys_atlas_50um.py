from __future__ import annotations

"""
Generate a dense 50 µm channel-level Ephys Atlas volume.

For every non-void voxel in the LEFT hemisphere:
  1. sample the frozen AGEA + MERFISH context used by the released model,
  2. retrieve nearby TRAINING-set electrophysiology channels,
  3. predict all channel-level ephys features,
  4. convert predictions back to the original feature units,
  5. write the left voxel and its mirrored right-hemisphere counterpart.

The output is a chunked/compressed HDF5 file so the full 50 µm atlas does not
need to fit in RAM.

Place this file in:
    ibleatools/examples/generate_dense_ephys_atlas_50um.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from iblatlas.atlas import AllenAtlas
from torch.utils.data import DataLoader
from tqdm import tqdm

from ephysatlas.spatial_encoder.model import NeighborInpaintingModel
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
    split_manifest_to_builder_format,
)
from ephysatlas.spatial_encoder.utils import (
    AtlasPCAConfig,
    ContextAtlasManager,
    FEATURE_LIST,
    GridDS,
    LoadInsertionData,
    NeighborCollate,
    build_training_neighbor_bank_from_release,
    concat_context,
    get_device,
)


@dataclass
class ExportConfig:
    vintage: str = "2026_W26"
    project: str = "ea_active"
    agg: str = "agg_full"

    # Input / released model
    data_dir: Path = Path(".")
    registry_root: Path = DEFAULT_REGISTRY_ROOT
    hf_repo_id: Optional[str] = "AlonSaguy/ephys-atlas-models"
    hf_token: Optional[str] = None

    # Output
    output_dir: Path = Path(__file__).resolve().parent
    output_name: Optional[str] = None

    # Dense atlas
    atlas_resolution_um: int = 50
    batch_size: int = 4096

    # Model execution
    device: torch.device = get_device()
    seed: int = 0

    # HDF5
    compression: str = "gzip"
    compression_opts: int = 4
    h5_chunk_rows: int = 8192


def _apply_saved_architecture(cfg: ExportConfig, release_config: dict) -> dict:
    """Read all model-defining settings from the release."""
    context = release_config.get("context", {})
    channel = release_config.get("channel_level", {})
    architecture = channel.get("architecture", {})
    neighbors = channel.get("neighbors", {})

    return {
        "n_cell_pcs": int(context.get("n_cell_pcs", 50)),
        "n_gene_pcs": int(context.get("n_gene_pcs", 50)),
        "radius_um": int(neighbors.get("radius_um", 500)),
        "m_max": int(neighbors.get("m_max", 8)),
        "d_model": int(architecture.get("d_model", 128)),
        "nhead": int(architecture.get("nhead", 8)),
        "depth": int(architecture.get("depth", 2)),
        "drop": float(architecture.get("drop", 0.15)),
    }


@torch.no_grad()
def _predict_batch_original_units(
    xyz_m: np.ndarray,
    *,
    ctx_manager: ContextAtlasManager,
    model: NeighborInpaintingModel,
    handles: dict,
    radius_um: int,
    m_max: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Predict ephys at arbitrary xyz locations.

    Parameters
    ----------
    xyz_m
        [N, 3] CCF coordinates in meters.

    Returns
    -------
    pred
        [N, F] predictions in ORIGINAL ephys feature units.
    """
    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    n = xyz_m.shape[0]
    f_e = int(model.e_mean.numel())

    # ContextAtlasManager mirrors x to the left internally, matching training.
    pack = ctx_manager.sample_context_numpy_m(xyz_m, mode="clip")
    ctx_raw = concat_context(pack["cell_pc"], pack["gene_pc"]).astype(np.float32)

    ctx_mean = model.ctx_mean.detach().cpu().numpy().astype(np.float32)
    ctx_std = model.ctx_std.detach().cpu().numpy().astype(np.float32)

    # Match training behavior: a truly empty context remains all zero.
    ctx_stdzd = np.zeros_like(ctx_raw, dtype=np.float32)
    has_context = np.any(ctx_raw != 0.0, axis=1)
    ctx_stdzd[has_context] = (
        ctx_raw[has_context] - ctx_mean[None, :]
    ) / (ctx_std[None, :] + 1e-8)

    qds = GridDS(
        torch.from_numpy(ctx_stdzd).float(),
        torch.from_numpy(xyz_m).float(),
        f_e,
    )

    collate = NeighborCollate(
        ctx_manager,
        handles["bank_xyz"],
        handles["bank_feat"],
        handles["bank_pid"],
        handles["nn_bank"],
        e_feat_dim=f_e,
        M_max=m_max,
        radius_um=radius_um,
    )

    loader = DataLoader(
        qds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    model.eval()
    device_type = device.type
    use_autocast = device_type == "cuda"

    out = []
    for batch in loader:
        ctx_q, p_q, e_n, p_n, mask, *_ = batch
        ctx_q = ctx_q.to(device)
        p_q = p_q.to(device)
        e_n = e_n.to(device)
        p_n = p_n.to(device)
        mask = mask.to(device)

        with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
            _, mu_std = model(ctx_q, p_q, e_n, p_n, mask)

        # IMPORTANT: model outputs standardized ephys values.
        mu = (
            mu_std.float() * model.e_std[None, :]
            + model.e_mean[None, :]
        )
        out.append(mu.cpu().numpy().astype(np.float32))

    pred = np.concatenate(out, axis=0)
    if pred.shape != (n, f_e):
        raise RuntimeError(
            f"Prediction shape mismatch: got {pred.shape}, expected {(n, f_e)}"
        )
    return pred


def _make_extendable_dataset(
    h5: h5py.File,
    name: str,
    *,
    trailing_shape: tuple[int, ...],
    dtype,
    cfg: ExportConfig,
):
    chunk_rows = max(1, int(cfg.h5_chunk_rows))
    return h5.create_dataset(
        name,
        shape=(0, *trailing_shape),
        maxshape=(None, *trailing_shape),
        chunks=(chunk_rows, *trailing_shape),
        dtype=dtype,
        compression=cfg.compression,
        compression_opts=cfg.compression_opts,
        shuffle=True,
    )


def _append(ds: h5py.Dataset, values: np.ndarray) -> None:
    values = np.asarray(values)
    n0 = ds.shape[0]
    n1 = n0 + values.shape[0]
    ds.resize((n1, *ds.shape[1:]))
    ds[n0:n1] = values


def _region_ids_for_x_slice(
    atlas: AllenAtlas,
    x_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return y indices, z indices, and Allen region IDs for non-void voxels at x_index.

    AllenAtlas.label is indexed [y, x, z]. The label values are region indices.
    We map those indices to canonical Allen region IDs before saving.
    """
    label_yz = atlas.label[:, x_index, :]
    region_index_yz = atlas._get_mapping(mapping="Allen")[label_yz]
    region_id_yz = atlas.regions.id[region_index_yz]

    keep = region_id_yz != 0
    y_i, z_i = np.where(keep)
    region_id = region_id_yz[y_i, z_i].astype(np.int32)
    return y_i.astype(np.int32), z_i.astype(np.int32), region_id


def _mirror_to_right_indices(
    atlas: AllenAtlas,
    xyz_left_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirror world x around the sagittal midline, then resolve the corresponding
    50 µm atlas indices. Returns (xyz_right_index, xyz_right_m).
    """
    xyz_right_m = xyz_left_m.copy()
    xyz_right_m[:, 0] = np.abs(xyz_right_m[:, 0])

    xyz_right_index = np.rint(
        atlas.bc.xyz2i(xyz_right_m, mode="clip")
    ).astype(np.int32)

    # Snap back to the exact CCF voxel center represented by those indices.
    xyz_right_m = atlas.bc.i2xyz(xyz_right_index).astype(np.float32)
    return xyz_right_index, xyz_right_m


def main(cfg: Optional[ExportConfig] = None) -> Path:
    cfg = cfg or ExportConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    out_name = cfg.output_name or f"ephys_atlas_{cfg.atlas_resolution_um}um_{cfg.vintage}.h5"
    out_path = cfg.output_dir / out_name

    # ------------------------------------------------------------------
    # 1. Resolve exact released model + frozen artifacts
    # ------------------------------------------------------------------
    registry = EphysAtlasReleaseRegistry(cfg.registry_root)
    release_dir = registry.resolve_release(
        cfg.vintage,
        repo_id=cfg.hf_repo_id,
        token=cfg.hf_token,
        require_weights=True,
    )
    registry.verify_checksums(cfg.vintage)

    release_config = registry.load_config(cfg.vintage)
    registry.validate_feature_order(cfg.vintage, FEATURE_LIST)
    model_cfg = _apply_saved_architecture(cfg, release_config)

    split_manifest = split_manifest_to_builder_format(
        registry.load_split(cfg.vintage)
    )
    preprocessing_stats = registry.load_channel_preprocessing_stats(cfg.vintage)

    # ------------------------------------------------------------------
    # 2. Load frozen context volumes
    # ------------------------------------------------------------------
    ctx_manager = ContextAtlasManager(
        AtlasPCAConfig(
            n_cell_pcs=model_cfg["n_cell_pcs"],
            n_gene_pcs=model_cfg["n_gene_pcs"],
        ),
        regenerate_context=False,
        output_dir=release_dir / "context",
    )

    # ------------------------------------------------------------------
    # 3. Build the released TRAIN-ONLY neighbor bank.
    #
    # Do this directly rather than calling
    # build_channels_plus_emptyvoxels_with_neighbors().  This makes the
    # exporter compatible with both old and new utils.py versions and avoids
    # constructing the large 200-um training grid, which is unnecessary for
    # inference.
    # ------------------------------------------------------------------
    pid_names, ephys, probe_positions, _ = LoadInsertionData(
        project=cfg.project,
        agg=cfg.agg,
        VINTAGE=cfg.vintage,
        path_data=cfg.data_dir,
    )
    pid_names = [str(x) for x in pid_names]

    (
        handles,
        e_mean,
        e_std,
        ctx_mean,
        ctx_std,
    ) = build_training_neighbor_bank_from_release(
        pid_names=pid_names,
        ephys=ephys,
        probe_positions=probe_positions,
        split_manifest=split_manifest,
        preprocessing_stats=preprocessing_stats,
        radius_um=model_cfg["radius_um"],
        m_max=model_cfg["m_max"],
    )

    # ------------------------------------------------------------------
    # 4. Instantiate + load released spatial encoder
    # ------------------------------------------------------------------
    f_ctx = int(ctx_mean.numel())
    f_e = int(e_mean.numel())

    model = NeighborInpaintingModel(
        f_ctx=f_ctx,
        f_ephys=f_e,
        f_out=f_e,
        e_mean=e_mean,
        e_std=e_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        depth=model_cfg["depth"],
        drop=model_cfg["drop"],
    ).to(cfg.device)

    checkpoint = torch.load(
        release_dir / "models" / "channel" / "spatial_encoder.pt",
        map_location=cfg.device,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    # ------------------------------------------------------------------
    # 5. Native 50 µm Allen CCF grid
    # ------------------------------------------------------------------
    atlas = AllenAtlas(res_um=cfg.atlas_resolution_um)
    nx = len(atlas.bc.xscale)
    ny = len(atlas.bc.yscale)
    nz = len(atlas.bc.zscale)

    x_indices = np.arange(nx, dtype=np.int32)
    x_xyz = atlas.bc.i2xyz(
        np.column_stack(
            [
                x_indices,
                np.zeros(nx, dtype=np.int32),
                np.zeros(nx, dtype=np.int32),
            ]
        )
    )[:, 0]

    # Predict only the left side. A possible exact midline voxel is written once.
    left_x_indices = x_indices[x_xyz <= 0]

    print(f"Device: {cfg.device}")
    print(f"Release: {release_dir}")
    print(f"50 um atlas shape [x,y,z] = [{nx},{ny},{nz}]")
    print(f"Left x slices: {len(left_x_indices)}")
    print(f"Output: {out_path}")

    # ------------------------------------------------------------------
    # 6. Stream predictions directly to HDF5
    # ------------------------------------------------------------------
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        h5.attrs["format_version"] = 1
        h5.attrs["description"] = (
            "Dense channel-level Ephys Atlas predictions on the 50 um Allen CCF grid."
        )
        h5.attrs["vintage"] = cfg.vintage
        h5.attrs["atlas_resolution_um"] = int(cfg.atlas_resolution_um)
        h5.attrs["coordinate_units"] = "meters"
        h5.attrs["coordinate_order"] = "x,y,z"
        h5.attrs["hemisphere_convention"] = "-1=left, +1=right, 0=midline"
        h5.attrs["prediction_units"] = "original feature units (not standardized)"
        h5.attrs["neighbor_source"] = "training split only"
        h5.attrs["neighbor_radius_um"] = int(model_cfg["radius_um"])
        h5.attrs["max_neighbors"] = int(model_cfg["m_max"])
        h5.attrs["void_rule"] = "Allen region ID == 0 excluded"
        h5.attrs["mirror_rule"] = "predict left hemisphere, copy predictions to mirrored right voxel"

        h5.create_dataset(
            "feature_names",
            data=np.asarray(FEATURE_LIST, dtype=object),
            dtype=string_dtype,
        )

        ds_idx = _make_extendable_dataset(
            h5, "ccf_xyz_index", trailing_shape=(3,), dtype=np.int16, cfg=cfg
        )
        ds_xyz = _make_extendable_dataset(
            h5, "ccf_xyz_m", trailing_shape=(3,), dtype=np.float32, cfg=cfg
        )
        ds_rid = _make_extendable_dataset(
            h5, "region_id", trailing_shape=(), dtype=np.int32, cfg=cfg
        )
        ds_hemi = _make_extendable_dataset(
            h5, "hemisphere", trailing_shape=(), dtype=np.int8, cfg=cfg
        )
        ds_pred = _make_extendable_dataset(
            h5,
            "predicted_features",
            trailing_shape=(len(FEATURE_LIST),),
            dtype=np.float32,
            cfg=cfg,
        )

        n_left = 0
        n_right = 0
        n_midline = 0

        for ix in tqdm(left_x_indices, desc="Predicting left-hemisphere 50 um slices"):
            y_i, z_i, region_id_left = _region_ids_for_x_slice(atlas, int(ix))
            if len(y_i) == 0:
                continue

            idx_left = np.column_stack(
                [
                    np.full(len(y_i), int(ix), dtype=np.int32),
                    y_i,
                    z_i,
                ]
            )
            xyz_left_m = atlas.bc.i2xyz(idx_left).astype(np.float32)

            pred = _predict_batch_original_units(
                xyz_left_m,
                ctx_manager=ctx_manager,
                model=model,
                handles=handles,
                radius_um=model_cfg["radius_um"],
                m_max=model_cfg["m_max"],
                batch_size=cfg.batch_size,
                device=cfg.device,
            )

            # ---------------- left ----------------
            is_midline = np.isclose(xyz_left_m[:, 0], 0.0, atol=1e-12)
            hemi_left = np.where(is_midline, 0, -1).astype(np.int8)

            _append(ds_idx, idx_left.astype(np.int16))
            _append(ds_xyz, xyz_left_m)
            _append(ds_rid, region_id_left)
            _append(ds_hemi, hemi_left)
            _append(ds_pred, pred)

            n_left += int((~is_midline).sum())
            n_midline += int(is_midline.sum())

            # ---------------- mirrored right ----------------
            mirror_mask = ~is_midline
            if not mirror_mask.any():
                continue

            idx_right, xyz_right_m = _mirror_to_right_indices(
                atlas, xyz_left_m[mirror_mask]
            )

            # Get the actual right-hemisphere region IDs rather than assuming
            # perfect anatomical symmetry of the annotation volume.
            right_region_index = atlas._get_mapping(mapping="Allen")[
                atlas.label[
                    idx_right[:, 1],
                    idx_right[:, 0],
                    idx_right[:, 2],
                ]
            ]
            region_id_right = atlas.regions.id[right_region_index].astype(np.int32)

            # If the mirrored location is void in the annotation, do not write it.
            right_keep = region_id_right != 0
            if right_keep.any():
                _append(ds_idx, idx_right[right_keep].astype(np.int16))
                _append(ds_xyz, xyz_right_m[right_keep])
                _append(ds_rid, region_id_right[right_keep])
                _append(
                    ds_hemi,
                    np.ones(int(right_keep.sum()), dtype=np.int8),
                )
                _append(ds_pred, pred[mirror_mask][right_keep])
                n_right += int(right_keep.sum())

        h5.attrs["n_left_voxels"] = n_left
        h5.attrs["n_right_voxels"] = n_right
        h5.attrs["n_midline_voxels"] = n_midline
        h5.attrs["n_rows"] = int(ds_idx.shape[0])
        h5.attrs["n_features"] = len(FEATURE_LIST)

    print("\nDone.")
    print(f"Rows written: {n_left + n_right + n_midline:,}")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
