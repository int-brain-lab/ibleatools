from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

import ephysatlas.data

from ephysatlas.spatial_encoder.model import NeighborInpaintingModel
from ephysatlas.spatial_encoder.model_registry import (
    DEFAULT_REGISTRY_ROOT,
    EphysAtlasReleaseRegistry,
)
from ephysatlas.spatial_encoder.utils import (
    FEATURE_LIST,
    LoadInsertionData,
    get_device,
    mirror_xyz_to_left,
)


# -----------------------------------------------------------------------------
# Hugging Face / local release loading
# -----------------------------------------------------------------------------

def load_channel_interpolation_model_from_registry(
    *,
    vintage: str,
    device: torch.device,
    hf_repo_id: Optional[str],
    registry_root: Path = DEFAULT_REGISTRY_ROOT,
    hf_token: Optional[str] = None,
):
    """
    Load the released channel-level interpolation model for a data vintage.

    Resolution order:
      1. local Ephys Atlas release registry
      2. Hugging Face repo at revision=<vintage>

    Returns
    -------
    model : NeighborInpaintingModel
    release_dir : Path
    release_features : list[str]
    release_config : dict
    """
    registry = EphysAtlasReleaseRegistry(registry_root)

    release_dir = registry.resolve_release(
        vintage,
        repo_id=hf_repo_id,
        token=hf_token,
        require_weights=True,
    )
    registry.verify_checksums(vintage)

    release_features = registry.load_features(vintage)
    registry.validate_feature_order(vintage, FEATURE_LIST)
    release_config = registry.load_config(vintage)
    stats = registry.load_channel_preprocessing_stats(vintage)

    ckpt_path = release_dir / "models" / "channel" / "spatial_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    arch = ckpt.get("architecture", {})
    required_arch = ["f_ctx", "f_ephys", "f_out", "d_model", "nhead", "depth", "drop"]
    missing = [key for key in required_arch if key not in arch]
    if missing:
        raise RuntimeError(
            f"Released spatial encoder is missing architecture fields: {missing}"
        )

    def _stat_tensor(name: str) -> torch.Tensor:
        if name not in stats:
            raise RuntimeError(
                f"Release preprocessing/channel_stats.npz is missing {name!r}"
            )
        return torch.as_tensor(stats[name], dtype=torch.float32)

    model = NeighborInpaintingModel(
        f_ctx=int(arch["f_ctx"]),
        f_ephys=int(arch["f_ephys"]),
        f_out=int(arch["f_out"]),
        e_mean=_stat_tensor("e_mean"),
        e_std=_stat_tensor("e_std"),
        ctx_mean=_stat_tensor("ctx_mean"),
        ctx_std=_stat_tensor("ctx_std"),
        d_model=int(arch["d_model"]),
        nhead=int(arch["nhead"]),
        depth=int(arch["depth"]),
        drop=float(arch["drop"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"[figure 1] loaded channel model vintage={vintage}")
    print(f"[figure 1] release directory: {release_dir}")

    return model, release_dir, release_features, release_config



# -----------------------------------------------------------------------------
# Unit-level metadata download / preparation for panel b
# -----------------------------------------------------------------------------

UNIT_PROJECT_DEFAULT = "ibl_neuropixel_brainwide_01"


def _first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"None of these columns exist: {candidates}. "
            f"Available columns include: {list(df.columns)[:80]}"
        )
    return None


def prepare_figure1_unit_metadata(
    *,
    root_path: Path,
    cache_dir: Path,
    project: str = UNIT_PROJECT_DEFAULT,
    download: bool = True,
    one_base_url: str = "https://alyx.internationalbrainlab.org",
    overwrite_cache: bool = False,
) -> tuple[Path, Path]:
    """
    Download the cell aggregate metadata needed by Figure 1 panel b and create
    local ``cosmos.npy`` and ``pids.npy`` arrays using the same filtering logic
    as the unit-level encoder preparation code.

    Only the small cell metadata are needed here. We intentionally do NOT
    download the large multichannel waveform files or the 3D ACG aggregate,
    because Figure 1 panel b only needs one Cosmos region ID and one PID per
    retained unit.

    Filtering matches the unit encoder preparation:
      1. keep clusters with bitwise_fail == 0
      2. remove known misaligned probe insertions when available
      3. remap Allen region IDs to Cosmos region IDs
      4. retain valid, known Cosmos IDs

    Returns
    -------
    cosmos_path, pids_path : Path
        Paths to cached arrays used by ``plot_ephys_atlas_dataset_summary_figure_ibl_style``.
    """
    root_path = Path(root_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cosmos_path = cache_dir / "cosmos.npy"
    pids_path = cache_dir / "pids.npy"

    if (
        cosmos_path.exists()
        and pids_path.exists()
        and not overwrite_cache
    ):
        cosmos = np.load(cosmos_path, allow_pickle=False)
        pids = np.load(pids_path, allow_pickle=True)
        if len(cosmos) != len(pids):
            raise ValueError(
                "Existing Figure 1 unit cache is inconsistent: "
                f"cosmos.npy has {len(cosmos)} rows but pids.npy has {len(pids)}. "
                "Delete the cache or set overwrite_unit_cache=True."
            )
        print(
            f"[figure 1] using cached unit metadata: {cache_dir} "
            f"({len(cosmos):,} units)"
        )
        return cosmos_path, pids_path

    path_project = root_path / project
    cells_agg_path = path_project / "cells_aggregates"
    clusters_table_path = cells_agg_path / "clusters.table.pqt"

    # Same small-data download pattern used by the unit encoder preparation.
    if download and not clusters_table_path.exists():
        from one.api import ONE

        print(
            "[figure 1] unit metadata not found locally; downloading "
            f"cell aggregates for project={project}"
        )
        one = ONE(base_url=one_base_url)
        ephysatlas.data.download_project_data(
            root_path,
            project=project,
            one=one,
            acg3d=False,
        )

    if not clusters_table_path.exists():
        raise FileNotFoundError(
            "Could not find the unit metadata required for Figure 1 panel b:\n"
            f"  {clusters_table_path}\n\n"
            "Either set download_unit_data=True or point unit_data_root to a "
            "directory containing the downloaded Ephys Atlas cell aggregates."
        )

    data = ephysatlas.data.read_cells_features(path_project)
    if "df_clusters" not in data:
        raise KeyError(
            "read_cells_features() did not return 'df_clusters'. "
            f"Available keys: {sorted(data.keys())}"
        )

    df_clusters = data["df_clusters"]

    if "bitwise_fail" not in df_clusters.columns:
        raise KeyError(
            "Expected df_clusters['bitwise_fail'] so Figure 1 uses the same "
            "good-unit selection as the unit encoder."
        )

    good_mask = df_clusters["bitwise_fail"].to_numpy() == 0
    df_good = df_clusters.iloc[np.flatnonzero(good_mask)].copy()

    pid_col = _first_existing_column(
        df_good,
        ["pid", "probe_insertion", "probe_insertion_id", "insertion_id"],
    )
    atlas_col = _first_existing_column(
        df_good,
        ["atlas_id", "atlas_id_final", "allen_id", "ccf_id"],
    )

    # Match the unit-encoder filtering of known misaligned insertions.
    try:
        from ephysatlas.fixtures import misaligned_pids

        misaligned_set = set(map(str, misaligned_pids))
        aligned_mask = ~df_good[pid_col].astype(str).isin(misaligned_set).to_numpy()
        n_removed = int((~aligned_mask).sum())
        df_good = df_good.iloc[np.flatnonzero(aligned_mask)].copy()
        print(
            f"[figure 1] removed {n_removed:,} good units from known "
            "misaligned probe insertions"
        )
    except Exception as exc:
        print(
            "[figure 1] WARNING: could not apply misaligned_pids filter; "
            f"continuing without it ({exc})"
        )

    br = BrainRegions()

    atlas_ids = df_good[atlas_col].to_numpy(dtype=np.int64)
    cosmos_ids = br.remap(
        atlas_ids,
        source_map="Allen",
        target_map="Cosmos",
    ).astype(np.int64)

    pids = df_good[pid_col].astype(str).to_numpy()

    # Keep only valid Cosmos IDs, using the same convention as the encoder.
    # Remove root by acronym as well as by ID=1, because the anatomical root
    # can have a different ID depending on the mapping representation.
    known_region_ids = set(map(int, br.id))
    br_acronyms = np.asarray(br.acronym, dtype=str)
    br_ids = np.asarray(br.id, dtype=np.int64)
    root_region_ids = set(
        map(
            int,
            br_ids[np.char.lower(br_acronyms) == "root"],
        )
    )
    root_region_ids.add(1)

    valid = cosmos_ids != 0
    valid &= np.asarray(
        [
            int(region_id) in known_region_ids
            and int(region_id) not in root_region_ids
            for region_id in cosmos_ids
        ],
        dtype=bool,
    )

    cosmos_ids = cosmos_ids[valid]
    pids = pids[valid]

    np.save(cosmos_path, cosmos_ids.astype(np.int64), allow_pickle=False)
    np.save(pids_path, pids.astype(object), allow_pickle=True)

    print(
        f"[figure 1] prepared unit metadata: {len(cosmos_ids):,} units, "
        f"{len(np.unique(pids)):,} PIDs, "
        f"{len(np.unique(cosmos_ids))} Cosmos regions"
    )
    print(f"[figure 1] saved {cosmos_path}")
    print(f"[figure 1] saved {pids_path}")

    return cosmos_path, pids_path


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

PANEL_C_FEATURE_GROUPS = {
    "LFP features": [
        "rms_lf",
        "psd_alpha",
        "psd_gamma",
    ],
    "AP features": [
        "rms_ap",
        "alpha_mean",
        "alpha_std",
    ],
    "Spike-detection features": [
        "repolarisation_slope",
        "peak_val",
        "peak_time_secs",
    ],
}

DISPLAY_FEATURE_NAMES = {
    "rms_lf": "RMS LF",
    "psd_alpha": "PSD alpha",
    "psd_gamma": "PSD gamma",
    "rms_ap": "RMS AP",
    "alpha_mean": "Alpha mean",
    "alpha_std": "Alpha std",
    "repolarisation_slope": "Repolarization slope",
    "peak_val": "Peak value",
    "peak_time_secs": "Peak time",
}


# Cosmos "root" is intentionally excluded from every Figure 1 panel.
EXCLUDED_REGION_IDS = {1}


def _root_region_ids(
    brain_atlas: AllenAtlas,
) -> set[int]:
    """Return every atlas ID whose acronym is exactly 'root'."""
    acronyms = np.asarray(
        brain_atlas.regions.acronym,
        dtype=str,
    )
    ids = np.asarray(
        brain_atlas.regions.id,
        dtype=np.int64,
    )
    root_ids = set(
        map(
            int,
            ids[np.char.lower(acronyms) == "root"],
        )
    )
    root_ids.update(EXCLUDED_REGION_IDS)
    return root_ids


def _is_root_region(
    brain_atlas: AllenAtlas,
    rid: int,
) -> bool:
    return int(rid) in _root_region_ids(brain_atlas)


def _cosmos_region_volumes_mm3(
    brain_atlas: AllenAtlas,
    *,
    mapping: str = "Cosmos",
) -> dict[int, float]:
    """
    Compute the physical volume of each Cosmos region from the Allen atlas.

    Region volume is the number of atlas voxels assigned to a Cosmos region
    multiplied by the physical Allen-atlas voxel volume. The returned volumes
    are bilateral because Cosmos region IDs are shared across hemispheres.
    """
    labels = np.asarray(brain_atlas.label)

    # `label` stores Allen region indices. The requested atlas mapping converts
    # those to indices in brain_atlas.regions.
    mapping_indices = np.asarray(
        brain_atlas._get_mapping(mapping=mapping),
        dtype=np.int64,
    )
    mapped_region_indices = mapping_indices[
        labels.astype(np.int64)
    ]

    region_ids = np.asarray(
        brain_atlas.regions.id,
        dtype=np.int64,
    )
    mapped_region_ids = region_ids[
        mapped_region_indices
    ]

    # BrainCoordinates.dxyz is expressed in metres.
    if hasattr(brain_atlas.bc, "dxyz"):
        dxyz_m = np.asarray(
            brain_atlas.bc.dxyz,
            dtype=float,
        )
        voxel_volume_mm3 = float(
            np.prod(np.abs(dxyz_m)) * 1e9
        )
    else:
        # Conservative fallback for standard AllenAtlas instances.
        res_um = float(
            getattr(brain_atlas, "res_um", 25.0)
        )
        voxel_volume_mm3 = (
            res_um * 1e-3
        ) ** 3

    ids, counts = np.unique(
        mapped_region_ids,
        return_counts=True,
    )

    return {
        int(rid): float(count) * voxel_volume_mm3
        for rid, count in zip(ids, counts)
        if int(rid) > 0
        and not _is_root_region(
            brain_atlas,
            int(rid),
        )
    }


def _atlas_region_index(brain_atlas: AllenAtlas, rid: int) -> Optional[int]:
    """
    Convert an Allen/Cosmos region ID to an index in brain_atlas.regions.

    Region IDs are NOT array indices. The previous figure code indexed
    regions.rgb directly by rid, which can produce incorrect colors.
    """
    ids = np.asarray(brain_atlas.regions.id)
    hits = np.flatnonzero(ids == int(rid))
    if len(hits) == 0:
        return None
    return int(hits[0])


def _region_label(brain_atlas: AllenAtlas, rid: int) -> str:
    idx = _atlas_region_index(brain_atlas, rid)
    if idx is None:
        return str(int(rid))
    return str(brain_atlas.regions.acronym[idx])


def _region_color(brain_atlas: AllenAtlas, rid: int):
    """
    Return the canonical Allen color for a region ID.

    Used identically by panels a and b.
    """
    idx = _atlas_region_index(brain_atlas, rid)
    if idx is None:
        return (0.5, 0.5, 0.5)

    rgb = np.asarray(brain_atlas.regions.rgb[idx], dtype=float)
    if np.nanmax(rgb) > 1:
        rgb = rgb / 255.0
    return tuple(rgb[:3])


def _xyz_to_region_ids(
    brain_atlas: AllenAtlas,
    xyz_m: np.ndarray,
    *,
    mapping: str = "Cosmos",
    mode: str = "clip",
) -> np.ndarray:
    """
    Return actual region IDs at xyz positions for the requested mapping.

    Important:
    ``AllenAtlas._get_mapping()`` yields indices into ``brain_atlas.regions``,
    not anatomical region IDs. Panel b stores actual Cosmos region IDs after
    ``BrainRegions.remap``. Converting the mapped indices back through
    ``brain_atlas.regions.id`` makes panels a and b use the same IDs, labels,
    and canonical Allen colors.
    """
    xyz_i = brain_atlas.bc.xyz2i(xyz_m, mode=mode)
    inds = brain_atlas._lookup_inds(xyz_i)

    allen_label_indices = brain_atlas.label.flat[inds]
    mapped_region_indices = np.asarray(
        brain_atlas._get_mapping(mapping=mapping)[allen_label_indices],
        dtype=np.int64,
    )

    region_ids = np.asarray(brain_atlas.regions.id, dtype=np.int64)
    return region_ids[mapped_region_indices].astype(int)


def _count_regions(
    rids: np.ndarray,
    *,
    brain_atlas: AllenAtlas,
) -> dict[int, int]:
    """Count valid regions after removing Cosmos root robustly."""
    rids = np.asarray(rids, dtype=int)

    valid = rids > 0
    root_ids = _root_region_ids(brain_atlas)
    if root_ids:
        valid &= ~np.isin(
            rids,
            np.asarray(sorted(root_ids), dtype=int),
        )

    rids = rids[valid]

    ids, counts = np.unique(
        rids,
        return_counts=True,
    )
    return dict(
        zip(ids.tolist(), counts.tolist())
    )


def _plot_region_sampling_density(
    ax,
    counts: dict[int, int],
    *,
    region_volumes_mm3: dict[int, float],
    brain_atlas: AllenAtlas,
    title: str,
    ylabel: str,
    max_regions: Optional[int] = None,
):
    """
    Plot sampling density = number of observations / Cosmos-region volume.
    """
    if len(counts) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid regions",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rows = []
    for rid, count in counts.items():
        if _is_root_region(
            brain_atlas,
            int(rid),
        ):
            continue

        volume = region_volumes_mm3.get(
            int(rid),
            np.nan,
        )
        if (
            not np.isfinite(volume)
            or volume <= 0
        ):
            continue

        rows.append(
            (
                int(rid),
                float(count) / float(volume),
            )
        )

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No regions with valid volumes",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rids = np.asarray(
        [row[0] for row in rows],
        dtype=int,
    )
    density = np.asarray(
        [row[1] for row in rows],
        dtype=float,
    )

    order = np.argsort(density)[::-1]
    if max_regions is not None:
        order = order[:max_regions]

    rids = rids[order]
    density = density[order]

    labels = [
        _region_label(
            brain_atlas,
            int(rid),
        )
        for rid in rids
    ]
    colors = [
        _region_color(
            brain_atlas,
            int(rid),
        )
        for rid in rids
    ]

    x = np.arange(len(rids))
    ax.bar(
        x,
        density,
        color=colors,
        edgecolor="black",
        linewidth=0.25,
    )

    ax.set_title(
        title,
        pad=4,
    )
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=90,
        ha="center",
    )
    ax.tick_params(
        axis="x",
        labelsize=5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pick_random_probe_for_panel_c(
    *,
    pids: np.ndarray,
    ephys: np.ndarray,
    probe_positions: np.ndarray,
    feature_list: list[str],
    brain_atlas: AllenAtlas,
    mapping: str,
    mirror_fn,
    seed: int,
) -> int:
    """
    Pick a reproducible random probe with enough finite NON-ROOT observations
    for all nine requested panel-c features.
    """
    feature_indices = [
        feature_list.index(name)
        for group in PANEL_C_FEATURE_GROUPS.values()
        for name in group
    ]

    candidates = []

    for p_idx in range(len(pids)):
        xyz = np.asarray(
            probe_positions[p_idx],
            dtype=np.float32,
        )
        values = np.asarray(
            ephys[p_idx]
        )[:, feature_indices]

        valid_xyz = ~np.all(
            xyz == 0.0,
            axis=1,
        )

        xyz_eval = xyz.copy()
        if mirror_fn is not None:
            nonzero = np.flatnonzero(valid_xyz)
            xyz_eval[nonzero] = mirror_fn(
                xyz_eval[nonzero]
            )

        rids = np.full(
            len(xyz),
            -1,
            dtype=int,
        )
        if np.any(valid_xyz):
            rids[valid_xyz] = _xyz_to_region_ids(
                brain_atlas,
                xyz_eval[valid_xyz],
                mapping=mapping,
                mode="clip",
            )

        non_root = valid_xyz.copy()
        for excluded in EXCLUDED_REGION_IDS:
            non_root &= rids != int(excluded)

        finite_all = np.isfinite(
            values
        ).all(axis=1)

        n_good = int(
            np.sum(
                non_root
                & finite_all
            )
        )

        if n_good >= 20:
            candidates.append(p_idx)

    if not candidates:
        raise RuntimeError(
            "Could not find a probe with >=20 valid non-root channels "
            "for all panel-c features."
        )

    rng = np.random.default_rng(seed)
    return int(
        rng.choice(candidates)
    )


def _plot_probe_feature_panel(
    fig,
    subplot_spec,
    *,
    pid: str,
    probe_ephys: np.ndarray,
    probe_xyz: np.ndarray,
    feature_list: list[str],
    brain_atlas: AllenAtlas,
    mapping: str,
    mirror_fn,
):
    """
    Panel c: z-scored electrophysiological feature heatmap for one probe.

    Columns are electrophysiological features and rows are recording channels.
    Each feature is independently z-scored across valid recording channels.

    Visual conventions:
      - thin borders separate every feature column
      - thicker borders separate feature groups
      - y ticks are omitted
      - group labels sit directly above the heatmap
      - the panel title is placed higher to avoid overlap with group labels
    """
    ax = fig.add_subplot(subplot_spec)

    probe_xyz = np.asarray(
        probe_xyz,
        dtype=np.float32,
    )
    valid_xyz = ~np.all(
        probe_xyz == 0.0,
        axis=1,
    )

    xyz_eval = probe_xyz.copy()
    if mirror_fn is not None:
        nonzero = np.flatnonzero(valid_xyz)
        xyz_eval[nonzero] = mirror_fn(
            xyz_eval[nonzero]
        )

    rids = np.full(
        len(probe_xyz),
        -1,
        dtype=int,
    )
    if np.any(valid_xyz):
        rids[valid_xyz] = _xyz_to_region_ids(
            brain_atlas,
            xyz_eval[valid_xyz],
            mapping=mapping,
            mode="clip",
        )

    keep = valid_xyz.copy()
    root_ids = _root_region_ids(brain_atlas)
    if root_ids:
        keep &= ~np.isin(
            rids,
            np.asarray(sorted(root_ids), dtype=int),
        )

    valid_indices = np.flatnonzero(keep)

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"Selected probe {pid} has no valid non-root channels."
        )

    selected_features = [
        feat
        for features in PANEL_C_FEATURE_GROUPS.values()
        for feat in features
    ]
    feature_indices = [feature_list.index(feat) for feat in selected_features]

    values = np.asarray(
        probe_ephys[np.ix_(valid_indices, feature_indices)],
        dtype=float,
    )

    # Z-score each feature independently across channels.
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    z = (values - mean) / std

    # Symmetric robust color range so rare outliers do not dominate.
    finite_abs = np.abs(z[np.isfinite(z)])
    vmax = float(np.nanpercentile(finite_abs, 99)) if finite_abs.size else 1.0
    vmax = max(vmax, 1.0)

    im = ax.imshow(
        z,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )

    display_labels = [
        DISPLAY_FEATURE_NAMES.get(feat, feat)
        for feat in selected_features
    ]
    ax.set_xticks(np.arange(len(selected_features)))
    ax.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=6,
    )

    # No individual channel tick labels in panel c.
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylabel("Recording channels")
    ax.set_xlabel("Electrophysiological feature")

    # Put the title clearly above the feature-group labels.
    ax.set_title(
        f"Z-scored feature profiles along one probe (PID {pid})",
        fontsize=8,
        y=1.145,
        pad=0,
    )

    n_features = len(selected_features)

    # Thin border between EVERY feature column.
    for boundary in np.arange(0.5, n_features - 0.5, 1.0):
        ax.axvline(
            boundary,
            color="black",
            linewidth=0.25,
            alpha=0.8,
            zorder=3,
        )

    # Thicker borders between the three feature groups.
    group_lengths = [len(v) for v in PANEL_C_FEATURE_GROUPS.values()]
    group_boundaries = np.cumsum(group_lengths)[:-1] - 0.5
    for boundary in group_boundaries:
        ax.axvline(
            boundary,
            color="black",
            linewidth=0.8,
            alpha=1.0,
            zorder=4,
        )

    # Group labels between the heatmap and the title.
    left = 0
    for group_name, features in PANEL_C_FEATURE_GROUPS.items():
        right = left + len(features)
        center = (left + right - 1) / 2
        ax.text(
            center,
            1.035,
            group_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )
        left = right

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Z score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    return [ax]


def _rolling_rms(
    signal: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Centered rolling RMS with edge padding, used only for panel-d cartoons."""
    signal = np.asarray(signal, dtype=float)
    window_samples = max(int(window_samples), 1)

    kernel = np.ones(
        window_samples,
        dtype=float,
    ) / float(window_samples)

    squared_mean = np.convolve(
        signal ** 2,
        kernel,
        mode="same",
    )
    return np.sqrt(
        np.maximum(squared_mean, 0.0)
    )


def _simple_psd(signal: np.ndarray, fs_hz: float):
    """Small dependency-free one-sided periodogram for the explanatory cartoon."""
    signal = np.asarray(
        signal,
        dtype=float,
    )
    signal = signal - np.mean(signal)

    window = np.hanning(len(signal))
    xw = signal * window

    spectrum = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(
        len(signal),
        d=1.0 / float(fs_hz),
    )
    psd = (
        np.abs(spectrum) ** 2
        / max(np.sum(window ** 2), 1e-12)
    )
    return freq, psd


def _draw_panel_d_lfp_rms(ax):
    """Top-left: representative LFP and time-resolved RMS computation."""
    fs = 2500.0
    duration = 0.60
    t = np.arange(
        int(fs * duration)
    ) / fs

    rng = np.random.default_rng(4)

    # Slow LFP-like signal with several oscillatory components and weak noise.
    signal = (
        0.72 * np.sin(
            2 * np.pi * 2.7 * t
        )
        + 0.34 * np.sin(
            2 * np.pi * 9.5 * t + 0.4
        )
        + 0.12 * np.sin(
            2 * np.pi * 48.0 * t + 0.8
        )
        + 0.06 * rng.standard_normal(
            len(t)
        )
    )

    # RMS is typically computed over a time window rather than once over the
    # entire recording. A 50-ms window gives an intuitive time-resolved cartoon.
    rms_trace = _rolling_rms(
        signal,
        window_samples=round(
            0.050 * fs
        ),
    )

    time_ms = t * 1000

    ax.plot(
        time_ms,
        signal,
        color="black",
        lw=0.75,
        label="LFP",
    )
    ax.plot(
        time_ms,
        rms_trace,
        color="tab:orange",
        lw=1.15,
        label="RMS LF (50-ms window)",
    )
    ax.axhline(
        0,
        color="0.55",
        lw=0.4,
    )

    ax.set_title(
        "LFP signal (2.5 kHz)",
        pad=2,
    )
    ax.set_ylabel("Amplitude [a.u.]")
    ax.set_xticks([])

    # Add vertical headroom so the legend does not occlude the traces.
    y_min = min(
        float(np.min(signal)),
        float(np.min(rms_trace)),
    )
    y_max = max(
        float(np.max(signal)),
        float(np.max(rms_trace)),
    )
    y_range = y_max - y_min

    ax.set_ylim(
        y_min - 0.08 * y_range,
        y_max + 0.45 * y_range,
    )

    ax.legend(
        frameon=False,
        fontsize=5.7,
        loc="upper right",
        handlelength=1.5,
    )
    ax.spines[["top", "right"]].set_visible(False)


def _draw_panel_d_lfp_psd(ax):
    """
    Bottom-left: realistic schematic LFP spectrum.

    The spectrum has a broadband 1/f-like decay plus an alpha bump and a much
    smaller broad gamma elevation. The shaded regions indicate the bands whose
    integrated PSD values enter the feature table.
    """
    rng = np.random.default_rng(5)

    freq = np.linspace(
        0.5,
        100.0,
        600,
    )

    # Aperiodic background: approximately 1/f^1.35.
    background = 1.0 / (
        freq + 1.5
    ) ** 1.35

    # Oscillatory structure on top of the aperiodic background.
    alpha_bump = 0.055 * np.exp(
        -0.5
        * ((freq - 10.0) / 2.2) ** 2
    )
    gamma_bump = 0.0065 * np.exp(
        -0.5
        * ((freq - 52.0) / 15.0) ** 2
    )

    # Small correlated-looking multiplicative roughness makes the schematic
    # resemble a measured periodogram without overwhelming the main structure.
    rough = rng.normal(
        0.0,
        0.055,
        size=len(freq),
    )
    rough = np.convolve(
        rough,
        np.ones(9) / 9,
        mode="same",
    )

    psd = (
        background
        + alpha_bump
        + gamma_bump
    ) * np.exp(rough)

    ax.axvspan(
        8,
        12,
        color="tab:blue",
        alpha=0.18,
        lw=0,
        label="Alpha band",
    )
    ax.axvspan(
        30,
        90,
        color="tab:orange",
        alpha=0.14,
        lw=0,
        label="Gamma band",
    )

    ax.plot(
        freq,
        psd,
        color="black",
        lw=0.85,
        label="LFP PSD",
    )

    ax.text(
        10,
        np.interp(10, freq, psd) * 1.18,
        "PSD alpha",
        fontsize=6.3,
        ha="center",
        va="bottom",
    )
    ax.text(
        58,
        np.interp(58, freq, psd) * 1.55,
        "PSD gamma",
        fontsize=6.3,
        ha="center",
        va="bottom",
    )

    ax.set_xlim(
        0,
        100,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [a.u.]")
    ax.legend(
        frameon=False,
        fontsize=5.3,
        loc="upper right",
        handlelength=1.4,
    )
    ax.spines[["top", "right"]].set_visible(False)


def _draw_panel_d_ap_rms(ax):
    """Top-middle: AP-band signal and time-resolved RMS computation."""
    fs = 30_000.0
    duration = 0.020
    t = np.arange(
        int(fs * duration)
    ) / fs

    rng = np.random.default_rng(7)

    # Fast AP-band activity with an envelope change so the RMS trace has visible
    # temporal structure.
    envelope = (
        0.72
        + 0.48 * np.exp(
            -0.5
            * ((t - 0.0105) / 0.0022) ** 2
        )
    )

    fast_signal = (
        0.15 * rng.standard_normal(
            len(t)
        )
        + 0.105 * np.sin(
            2 * np.pi * 720.0 * t
        )
        + 0.045 * np.sin(
            2 * np.pi * 1850.0 * t + 0.5
        )
    )
    signal = envelope * fast_signal

    # 1-ms window for a schematic AP RMS time series.
    rms_trace = _rolling_rms(
        signal,
        window_samples=round(
            0.001 * fs
        ),
    )

    time_ms = t * 1000

    ax.plot(
        time_ms,
        signal,
        color="black",
        lw=0.55,
        label="AP",
    )
    ax.plot(
        time_ms,
        rms_trace,
        color="tab:orange",
        lw=1.05,
        label="RMS AP (1-ms window)",
    )
    ax.axhline(
        0,
        color="0.55",
        lw=0.4,
    )

    ax.set_title(
        "AP signal (30 kHz)",
        pad=2,
    )
    ax.set_ylabel("Amplitude [a.u.]")
    ax.set_xticks([])

    # Add vertical headroom so the legend does not occlude the traces.
    y_min = min(
        float(np.min(signal)),
        float(np.min(rms_trace)),
    )
    y_max = max(
        float(np.max(signal)),
        float(np.max(rms_trace)),
    )
    y_range = y_max - y_min

    ax.set_ylim(
        y_min - 0.08 * y_range,
        y_max + 0.45 * y_range,
    )

    ax.legend(
        frameon=False,
        fontsize=5.7,
        loc="upper right",
        handlelength=1.5,
    )
    ax.spines[["top", "right"]].set_visible(False)


def _draw_panel_d_alpha(ax):
    """
    Bottom-middle: alpha_mean / alpha_std.

    alpha is the per-spike localization brightness. The distribution therefore
    represents one alpha value per detected/localized spike on the channel.
    alpha_mean and alpha_std summarize this empirical spike-wise distribution.
    """
    rng = np.random.default_rng(19)

    alpha = rng.normal(
        loc=1.0,
        scale=0.20,
        size=450,
    )
    alpha = alpha[
        alpha > 0
    ]

    mean = float(
        np.mean(alpha)
    )
    std = float(
        np.std(alpha)
    )

    bins = np.linspace(
        max(
            0,
            alpha.min() - 0.05,
        ),
        alpha.max() + 0.05,
        28,
    )

    counts, edges = np.histogram(
        alpha,
        bins=bins,
        density=True,
    )
    centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    ax.plot(
        centers,
        counts,
        color="black",
        lw=0.9,
    )

    ymax = float(
        np.max(counts)
    )

    ax.axvline(
        mean,
        color="black",
        lw=0.9,
        ls="--",
    )

    # Mean label directly above the mean itself.
    ax.text(
        mean,
        ymax * 1.05,
        "alpha mean",
        fontsize=6.2,
        ha="center",
        va="bottom",
    )

    # Two-sided arrow explicitly showing one standard deviation on either side.
    arrow_y = ymax * 0.38
    ax.annotate(
        "",
        xy=(
            mean - std,
            arrow_y,
        ),
        xytext=(
            mean + std,
            arrow_y,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            lw=0.9,
        ),
    )
    ax.text(
        mean,
        arrow_y * 0.83,
        "alpha std",
        fontsize=6.2,
        ha="center",
        va="top",
    )

    ax.set_ylim(
        0,
        ymax * 1.20,
    )
    ax.set_xlabel(
        r"Spike-localization brightness $\alpha$"
    )
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)


def _draw_panel_d_waveform(ax):
    """Right: waveform features with explicit peak and slope annotations."""
    t = np.linspace(-0.75, 1.25, 260)

    waveform = (
        0.18 * np.exp(
            -0.5 * ((t + 0.28) / 0.13) ** 2
        )
        - 1.00 * np.exp(
            -0.5 * ((t - 0.02) / 0.10) ** 2
        )
        + 0.52 * np.exp(
            -0.5 * ((t - 0.42) / 0.17) ** 2
        )
    )

    ax.plot(
        t,
        waveform,
        color="black",
        lw=1.1,
        zorder=2,
    )
    ax.axhline(
        0,
        color="black",
        lw=0.45,
        zorder=1,
    )

    trough_idx = int(np.argmin(waveform))
    post_indices = np.arange(
        trough_idx + 1,
        len(t),
    )
    peak_idx = int(
        post_indices[
            np.argmax(waveform[post_indices])
        ]
    )

    trough_t = float(t[trough_idx])
    peak_t = float(t[peak_idx])
    peak_v = float(waveform[peak_idx])

    # Short local line showing where repolarization slope is measured.
    t1 = trough_t + 0.24 * (peak_t - trough_t)
    t2 = trough_t + 0.50 * (peak_t - trough_t)
    v1 = float(np.interp(t1, t, waveform))
    v2 = float(np.interp(t2, t, waveform))

    ax.plot(
        [t1, t2],
        [v1, v2],
        color="tab:orange",
        lw=2.0,
        zorder=4,
    )
    ax.text(
        t2 + 0.08,
        0.5 * (v1 + v2) - 0.18,
        "repolarization\nslope",
        fontsize=7,
        ha="left",
        va="center",
        color="tab:orange",
    )

    ax.scatter(
        [peak_t],
        [peak_v],
        s=14,
        color="black",
        zorder=5,
    )

    # Left-pointing arrow from text on the right toward the peak.
    ax.annotate(
        "peak value",
        xy=(peak_t + 0.012, peak_v),
        xytext=(peak_t + 0.55, peak_v + 0.03),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.0,
            color="black",
        ),
        fontsize=7,
        ha="left",
        va="center",
        zorder=6,
    )

    # Diagonal peak-time arrow; text sits slightly right of the true peak time.
    ax.annotate(
        "peak time",
        xy=(peak_t, -0.03),
        xytext=(peak_t + 0.34, -0.25),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.0,
            color="black",
        ),
        fontsize=7,
        ha="left",
        va="center",
        zorder=6,
    )

    ax.plot(
        [peak_t, peak_t],
        [-0.01, peak_v - 0.035],
        color="0.55",
        lw=0.7,
        ls=":",
        zorder=1,
    )

    ax.set_xlim(-0.65, 1.18)
    ax.set_ylim(
        min(waveform) - 0.12,
        max(waveform) + 0.22,
    )
    ax.set_title(
        "Average spike waveform",
        pad=2,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Normalized amplitude")
    ax.spines[["top", "right"]].set_visible(False)


def _plot_feature_computation_cartoon(
    fig,
    subplot_spec,
):
    """
    Panel d.

    Left:
      top    = LFP + RMS
      bottom = PSD with alpha/gamma bands

    Middle:
      top    = AP + RMS
      bottom = distribution of spike-localization brightness alpha

    Right:
      average spike waveform with peak time/value and repolarization slope
    """
    outer = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=subplot_spec,
        width_ratios=[1.0, 1.0, 1.08],
        wspace=0.34,
    )

    lfp = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 0],
        height_ratios=[1.0, 1.0],
        hspace=0.42,
    )
    ap = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0, 1],
        height_ratios=[1.0, 1.0],
        hspace=0.42,
    )

    ax_lfp_rms = fig.add_subplot(lfp[0, 0])
    ax_lfp_psd = fig.add_subplot(lfp[1, 0])
    ax_ap_rms = fig.add_subplot(ap[0, 0])
    ax_alpha = fig.add_subplot(ap[1, 0])
    ax_wave = fig.add_subplot(outer[0, 2])

    _draw_panel_d_lfp_rms(ax_lfp_rms)
    _draw_panel_d_lfp_psd(ax_lfp_psd)
    _draw_panel_d_ap_rms(ax_ap_rms)
    _draw_panel_d_alpha(ax_alpha)
    _draw_panel_d_waveform(ax_wave)

    return [
        ax_lfp_rms,
        ax_lfp_psd,
        ax_ap_rms,
        ax_alpha,
        ax_wave,
    ]


def _add_panel_label(ax, label: str):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def plot_ephys_atlas_dataset_summary_figure_ibl_style(
    *,
    brain_atlas: AllenAtlas,
    feature_list: list[str],
    pids: np.ndarray,
    ephys: np.ndarray,
    probe_positions: np.ndarray,
    unit_cosmos_path: Path,
    unit_pids_path: Path,
    save_path: Optional[Path] = None,
    seed: int = 0,
    mapping: str = "Cosmos",
    mirror_fn=mirror_xyz_to_left,
    max_regions: Optional[int] = None,
    dpi: int = 600,
):
    """
    Figure 1:
      a. Recording-channel sampling density per Cosmos region.
      b. Sorted-neuron sampling density per Cosmos region.
      c. Z-scored channel-by-feature heatmap for one random non-root probe.
      d. Cartoon explaining representative LFP, AP, and waveform features.

    Cosmos root (rid=1) is excluded throughout.
    """
    figure_style()

    region_volumes_mm3 = _cosmos_region_volumes_mm3(
        brain_atlas,
        mapping=mapping,
    )

    # ------------------------------------------------------------------
    # Panel a: channel sampling density by region
    # ------------------------------------------------------------------
    channel_rids = []
    for xyz in probe_positions:
        xyz = np.asarray(xyz, dtype=np.float32)
        valid = ~np.all(xyz == 0.0, axis=1)
        xyz_valid = xyz[valid]

        if mirror_fn is not None:
            xyz_valid = mirror_fn(xyz_valid)

        if len(xyz_valid):
            channel_rids.append(
                _xyz_to_region_ids(
                    brain_atlas,
                    xyz_valid,
                    mapping=mapping,
                    mode="clip",
                )
            )

    channel_rids = (
        np.concatenate(channel_rids)
        if channel_rids
        else np.zeros(0, dtype=int)
    )
    channel_counts = _count_regions(
        channel_rids,
        brain_atlas=brain_atlas,
    )

    # ------------------------------------------------------------------
    # Panel b: unit sampling density by region
    # ------------------------------------------------------------------
    unit_rids = np.load(unit_cosmos_path, allow_pickle=True).astype(int)
    # Load the PIDs too, even though splits are no longer displayed. This
    # keeps a simple sanity check that the two unit-level arrays correspond.
    unit_pids = np.load(unit_pids_path, allow_pickle=True).astype(str)
    if len(unit_rids) != len(unit_pids):
        raise ValueError(
            f"unit Cosmos IDs ({len(unit_rids)}) and unit PIDs ({len(unit_pids)}) "
            "have different lengths."
        )
    unit_counts = _count_regions(
        unit_rids,
        brain_atlas=brain_atlas,
    )

    # ------------------------------------------------------------------
    # Panel c: one reproducibly random probe for the z-scored heatmap
    # ------------------------------------------------------------------
    random_probe_idx = _pick_random_probe_for_panel_c(
        pids=pids,
        ephys=ephys,
        probe_positions=probe_positions,
        feature_list=feature_list,
        brain_atlas=brain_atlas,
        mapping=mapping,
        mirror_fn=mirror_fn,
        seed=seed,
    )

    random_pid = str(pids[random_probe_idx])

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig = double_column_fig()
    fig.set_size_inches(
        fig.get_size_inches()[0],
        9.2,
    )

    gs = GridSpec(
        3,
        2,
        height_ratios=[0.82, 0.90, 1.55],
        hspace=0.52,
        wspace=0.18,
        figure=fig,
    )

    ax_ch = fig.add_subplot(
        gs[0, 0]
    )
    ax_unit = fig.add_subplot(
        gs[0, 1]
    )

    _plot_region_sampling_density(
        ax_ch,
        channel_counts,
        region_volumes_mm3=region_volumes_mm3,
        brain_atlas=brain_atlas,
        title="Channel sampling density per Cosmos brain region",
        ylabel=r"Channels / mm$^3$",
        max_regions=max_regions,
    )

    _plot_region_sampling_density(
        ax_unit,
        unit_counts,
        region_volumes_mm3=region_volumes_mm3,
        brain_atlas=brain_atlas,
        title="Neuron sampling density per Cosmos brain region",
        ylabel=r"Neurons / mm$^3$",
        max_regions=max_regions,
    )

    panel_c_frame = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=gs[1, :],
        width_ratios=[0.10, 0.80, 0.10],
        height_ratios=[0.10, 0.80, 0.10],
        wspace=0.0,
        hspace=0.0,
    )

    panel_c_axes = _plot_probe_feature_panel(
        fig,
        panel_c_frame[1, 1],
        pid=random_pid,
        probe_ephys=ephys[random_probe_idx],
        probe_xyz=probe_positions[random_probe_idx],
        feature_list=feature_list,
        brain_atlas=brain_atlas,
        mapping=mapping,
        mirror_fn=mirror_fn,
    )

    panel_d_axes = _plot_feature_computation_cartoon(
        fig,
        gs[2, :],
    )

    _add_panel_label(ax_ch, "a")
    _add_panel_label(ax_unit, "b")
    _add_panel_label(panel_c_axes[0], "c")
    _add_panel_label(panel_d_axes[0], "d")

    fig.align_ylabels([ax_ch, ax_unit])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# -----------------------------------------------------------------------------
# Run config
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    # Data used for panels a-c.
    data_dir: Path = Path("../")
    project: str = "ea_active"
    agg: str = "agg_full"
    vintage: str = "2026_W26"

    # Model registry / Hugging Face.
    registry_root: Path = DEFAULT_REGISTRY_ROOT

    hf_repo_id: Optional[str] = "AlonSaguy/ephys-atlas-models"

    hf_token: Optional[str] = None

    # Unit-level metadata used for panel b.
    #
    # Figure 1 now downloads/reads the same cell aggregates used by the
    # unit-level encoder preparation code and builds its own cosmos.npy/pids.npy
    # cache. Large waveform/ACG files are not needed for this figure.
    unit_data_root: Path = Path("../")
    unit_project: str = UNIT_PROJECT_DEFAULT
    unit_cache_dir: Path = Path("./figure1_unit_metadata")
    download_unit_data: bool = True
    overwrite_unit_cache: bool = False
    one_base_url: str = "https://alyx.internationalbrainlab.org"

    output_path: Path = Path("figure1_dataset_summary_cosmos_features_ibl_style.pdf")

    device: torch.device = get_device()
    seed: int = 0
    max_regions: Optional[int] = None


def main():
    cfg = RunConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Using device: {cfg.device}")

    # ------------------------------------------------------------------
    # Load the interpolation model from the new release registry.
    # This also validates the saved feature ordering against FEATURE_LIST.
    # ------------------------------------------------------------------
    base_model, release_dir, release_features, release_config = (
        load_channel_interpolation_model_from_registry(
            vintage=cfg.vintage,
            device=cfg.device,
            hf_repo_id=cfg.hf_repo_id,
            registry_root=cfg.registry_root,
            hf_token=cfg.hf_token,
        )
    )

    # Keep this variable alive intentionally: Figure 1 does not currently
    # perform interpolation, but loading the released model makes the figure
    # explicitly tied to the same versioned model/data release.
    _ = base_model

    data_cfg = release_config.get("data", {})
    saved_project = str(data_cfg.get("project", cfg.project))
    saved_agg = str(data_cfg.get("agg", cfg.agg))
    saved_vintage = str(data_cfg.get("vintage", cfg.vintage))

    if saved_vintage != cfg.vintage:
        raise RuntimeError(
            f"Requested vintage={cfg.vintage}, but release config says {saved_vintage}."
        )

    # Use the release's data identifiers so the figure cannot accidentally
    # combine a model release with a different data project/aggregation.
    pids, ephys, probe_positions, _ = LoadInsertionData(
        project=saved_project,
        agg=saved_agg,
        VINTAGE=saved_vintage,
        path_data=cfg.data_dir,
    )

    pids = np.asarray(pids).astype(str)

    # ------------------------------------------------------------------
    # Download/prepare the unit-level metadata needed by panel b.
    # This mirrors the good-unit / misaligned-PID / Cosmos-remapping logic
    # used by the unit encoder preparation script.
    # ------------------------------------------------------------------
    unit_cosmos_path, unit_pids_path = prepare_figure1_unit_metadata(
        root_path=cfg.unit_data_root,
        cache_dir=cfg.unit_cache_dir,
        project=cfg.unit_project,
        download=cfg.download_unit_data,
        one_base_url=cfg.one_base_url,
        overwrite_cache=cfg.overwrite_unit_cache,
    )

    plot_ephys_atlas_dataset_summary_figure_ibl_style(
        brain_atlas=AllenAtlas(),
        feature_list=release_features,
        pids=pids,
        ephys=ephys,
        probe_positions=probe_positions,
        unit_cosmos_path=unit_cosmos_path,
        unit_pids_path=unit_pids_path,
        save_path=cfg.output_path,
        seed=cfg.seed,
        mirror_fn=mirror_xyz_to_left,
        max_regions=cfg.max_regions,
    )

    print(f"[figure 1] saved {cfg.output_path}")


if __name__ == "__main__":
    main()
