import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from tqdm.auto import tqdm

import ephysatlas.data


PROJECT_DEFAULT = "ibl_neuropixel_brainwide_01"


def _first_existing_column(
    df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {list(candidates)}")
    return None


def _find_xyz_columns(df: pd.DataFrame) -> tuple[str, str, str, bool]:
    """Return x/y/z columns and whether values are probably in meters."""
    candidates = [
        ("x", "y", "z"),
        ("x_m", "y_m", "z_m"),
        ("x_um", "y_um", "z_um"),
        ("ml", "ap", "dv"),
        ("ml_um", "ap_um", "dv_um"),
        ("atlas_x", "atlas_y", "atlas_z"),
        ("atlas_x_um", "atlas_y_um", "atlas_z_um"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            vals = df.loc[:, list(cols)].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                return (*cols, True)
            probably_meters = np.nanpercentile(np.abs(finite), 95) < 1.0
            return (*cols, probably_meters)
    raise KeyError(
        "Could not infer xyz columns. Add your columns to _find_xyz_columns(). "
        f"Available columns include: {list(df.columns)[:80]}"
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _cluster_ids_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Return cluster ids matching waveforms.table.pqt / spike sorting cluster_id values.

    In most IBL tables, cluster_id is either an explicit column or the dataframe
    index. This helper supports both.
    """
    for col in ("cluster_id", "cluster", "id"):
        if col in df.columns:
            return df[col].to_numpy()
    return df.index.to_numpy()


def _normalize_pid_values(x: np.ndarray | pd.Series) -> np.ndarray:
    """String-normalize pids so matching is robust to object/UUID dtype differences."""
    return pd.Series(x).astype(str).to_numpy()


def _canonical_cluster_id(value) -> str:
    """Normalize cluster IDs so 12, np.int64(12), and 12.0 match."""
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _make_unit_keys(pid_values: np.ndarray, cluster_values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            f"{p}__{_canonical_cluster_id(c)}"
            for p, c in zip(_normalize_pid_values(pid_values), cluster_values)
        ],
        dtype=object,
    )


def _center_crop_or_pad_channels(
    w: np.ndarray, target_channels: int, *, pad_value: float = 0.0
) -> np.ndarray:
    """
    Convert variable-channel waveform [C,T] to fixed [target_channels,T].

    Rows are assumed sorted by abs_channel. If there are too many channels, keep
    the central block. If too few, center-pad with zeros.
    """
    c, t = w.shape
    if c == target_channels:
        return w.astype(np.float32, copy=False)
    if c > target_channels:
        s = (c - target_channels) // 2
        return w[s : s + target_channels].astype(np.float32, copy=False)

    out = np.full((target_channels, t), pad_value, dtype=np.float32)
    s = (target_channels - c) // 2
    out[s : s + c] = w.astype(np.float32, copy=False)
    return out


def _normalize_waveforms_max_abs(
    waveforms: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each waveform by max(abs(waveform)) over channel and time."""
    waveforms = np.asarray(waveforms, dtype=np.float32)
    scale = np.max(np.abs(waveforms), axis=(1, 2), keepdims=True)
    scale = np.maximum(scale, eps).astype(np.float32)
    return (waveforms / scale).astype(np.float32), scale[:, 0, 0].astype(np.float32)


def _build_multichannel_waveform_cache(
    *,
    cells_agg_path: Path,
    df_good: pd.DataFrame,
    pid_col: str,
    target_channels: int,
    cache_path: Path,
    overwrite_cache: bool = False,
    waveforms_voltage_path: Path | None = None,
    waveforms_table_path: Path | None = None,
    normalize_max_abs: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract multi-channel waveforms for df_good rows and cache them.

    Returns:
        waveforms_good: [N_kept, target_channels, 128]
        keep_mask:      [len(df_good)] True for units with waveform rows
        info:           diagnostics for manifest
    """
    if cache_path.exists() and not overwrite_cache:
        print(f"Using cached multichannel waveform file: {cache_path}")
        waveforms_cached = np.load(cache_path, allow_pickle=False)

        keep_mask_path = cache_path.with_name(cache_path.stem + "_keep_mask.npy")
        if not keep_mask_path.exists():
            raise FileNotFoundError(
                f"Found cached waveform file but missing keep mask:\n{keep_mask_path}\n"
                "Delete the waveform cache or rerun with overwrite_multichannel_cache=True."
            )

        keep_mask = np.load(keep_mask_path, allow_pickle=False)

        sidecar = cache_path.with_suffix(".manifest.json")
        info = {"cache_path": str(cache_path), "loaded_existing_cache": True}
        if sidecar.exists():
            with open(sidecar, "r", encoding="utf-8") as f:
                info.update(json.load(f))

        if int(keep_mask.sum()) != int(waveforms_cached.shape[0]):
            raise ValueError(
                f"Waveform cache mismatch: keep_mask.sum()={keep_mask.sum()} "
                f"but waveforms.shape[0]={waveforms_cached.shape[0]}. "
                "Delete the waveform cache or rerun with overwrite_multichannel_cache=True."
            )

        return (
            waveforms_cached.astype(np.float32, copy=False),
            keep_mask.astype(bool),
            info,
        )

    waveform_voltage_path = (
        Path(waveforms_voltage_path).expanduser().resolve()
        if waveforms_voltage_path is not None
        else cells_agg_path / "waveforms.voltage.npy"
    )
    waveform_table_path = (
        Path(waveforms_table_path).expanduser().resolve()
        if waveforms_table_path is not None
        else cells_agg_path / "waveforms.table.pqt"
    )
    missing = [
        str(x) for x in (waveform_voltage_path, waveform_table_path) if not x.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required multi-channel waveform aggregate file(s):\n  "
            + "\n  ".join(missing)
            + "\n\nCall ephysatlas.data.download_cells_features(..., large_files=True), pass "
            "waveforms_voltage_path=... and waveforms_table_path=..., or set allow_peak_fallback=True."
        )

    print(f"Memory-mapping large waveform file: {waveform_voltage_path}")
    all_waveforms = np.load(waveform_voltage_path, mmap_mode="r", allow_pickle=False)
    df_w = pd.read_parquet(waveform_table_path)

    w_pid_col = _first_existing_column(
        df_w, ["pid", "probe_insertion", "probe_insertion_id", "insertion_id"]
    )
    w_cluster_col = _first_existing_column(df_w, ["cluster_id", "cluster", "id"])
    abs_channel_col = _first_existing_column(
        df_w, ["abs_channel", "channel", "channel_id", "ch"], required=False
    )

    print("Building waveform table lookup by pid/cluster_id...")
    w_keys = _make_unit_keys(df_w[w_pid_col].to_numpy(), df_w[w_cluster_col].to_numpy())
    order = np.argsort(w_keys, kind="mergesort")
    sorted_keys = w_keys[order]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    ends = np.r_[starts[1:], len(sorted_keys)]
    key_to_range = {k: (int(s), int(e)) for k, s, e in zip(unique_keys, starts, ends)}

    good_cluster_ids = _cluster_ids_from_df(df_good)
    good_keys = _make_unit_keys(df_good[pid_col].to_numpy(), good_cluster_ids)

    extracted = []
    keep_mask = np.zeros(len(df_good), dtype=bool)
    keep_indices = []
    n_missing = 0
    n_variable_channel_units = 0
    original_channel_counts = []

    for i, key in enumerate(
        tqdm(good_keys, desc="extract good-unit multichannel waveforms")
    ):
        rng = key_to_range.get(key)
        if rng is None:
            n_missing += 1
            continue

        s, e = rng
        rows = order[s:e]
        if abs_channel_col is not None:
            rows = rows[np.argsort(df_w.iloc[rows][abs_channel_col].to_numpy())]
        else:
            rows = np.sort(rows)

        w = np.asarray(all_waveforms[rows], dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(
                f"Expected waveform rows [C,T], got {w.shape} for key={key}"
            )
        original_channel_counts.append(int(w.shape[0]))
        if int(w.shape[0]) != int(target_channels):
            n_variable_channel_units += 1
        extracted.append(_center_crop_or_pad_channels(w, target_channels))
        keep_mask[i] = True
        keep_indices.append(i)

    if not extracted:
        raise RuntimeError("No good units had matching rows in waveforms.table.pqt.")

    waveforms_good = np.stack(extracted, axis=0).astype(np.float32)
    waveform_scales = None
    if normalize_max_abs:
        waveforms_good, waveform_scales = _normalize_waveforms_max_abs(waveforms_good)
        np.save(
            cache_path.with_name(cache_path.stem + "_max_abs_scale.npy"),
            waveform_scales,
            allow_pickle=False,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, waveforms_good, allow_pickle=False)

    keep_mask_path = cache_path.with_name(cache_path.stem + "_keep_mask.npy")
    np.save(keep_mask_path, keep_mask, allow_pickle=False)

    info = {
        "cache_path": str(cache_path),
        "source_waveforms_voltage": str(waveform_voltage_path),
        "source_waveforms_table": str(waveform_table_path),
        "target_channels": int(target_channels),
        "normalize_max_abs": bool(normalize_max_abs),
        "waveform_scale_path": str(
            cache_path.with_name(cache_path.stem + "_max_abs_scale.npy")
        )
        if normalize_max_abs
        else None,
        "n_good_input_units": int(len(df_good)),
        "n_units_with_multichannel_waveforms": int(waveforms_good.shape[0]),
        "n_missing_waveform_rows": int(n_missing),
        "n_units_cropped_or_padded": int(n_variable_channel_units),
        "original_channel_count_min": int(np.min(original_channel_counts)),
        "original_channel_count_median": float(np.median(original_channel_counts)),
        "original_channel_count_max": int(np.max(original_channel_counts)),
        "waveforms_shape": list(waveforms_good.shape),
        "good_keep_indices": [int(i) for i in keep_indices],
    }
    with open(cache_path.with_suffix(".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Saved cached multi-channel waveforms: {cache_path}")
    return waveforms_good, keep_mask, info


def _align_good_cluster_array(
    *,
    df_units: pd.DataFrame,
    unit_pid_col: str,
    df_clusters_good: pd.DataFrame,
    values_good: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    """Align an array indexed by clusters_good.table.pqt to arbitrary unit rows.

    IBL's stPC aggregate is indexed by *good cluster*, not by clusters.table.pqt.
    We therefore join explicitly on (pid, cluster_id) and never use positions from
    the all-cluster table for stPC indexing.
    """
    values_good = np.asarray(values_good)
    if len(values_good) != len(df_clusters_good):
        raise ValueError(
            f"{feature_name} row count ({len(values_good)}) does not match "
            f"df_clusters_good ({len(df_clusters_good)})."
        )

    good_pid_col = _first_existing_column(
        df_clusters_good,
        ["pid", "probe_insertion", "probe_insertion_id", "insertion_id"],
    )
    unit_cluster_ids = _cluster_ids_from_df(df_units)
    good_cluster_ids = _cluster_ids_from_df(df_clusters_good)

    unit_keys = _make_unit_keys(df_units[unit_pid_col].to_numpy(), unit_cluster_ids)
    good_keys = _make_unit_keys(
        df_clusters_good[good_pid_col].to_numpy(),
        good_cluster_ids,
    )

    if len(np.unique(good_keys)) != len(good_keys):
        raise RuntimeError(
            f"{feature_name}: duplicate (pid, cluster_id) keys in clusters_good.table.pqt"
        )

    lookup = {key: i for i, key in enumerate(good_keys)}
    rows = np.asarray([lookup.get(key, -1) for key in unit_keys], dtype=np.int64)
    if np.any(rows < 0):
        missing = unit_keys[rows < 0][:10].tolist()
        raise RuntimeError(
            f"{feature_name}: {int((rows < 0).sum())} selected units were not found "
            f"in clusters_good.table.pqt. Examples: {missing}"
        )
    return np.asarray(values_good[rows], dtype=np.float32)


def _build_context(
    df_good: pd.DataFrame, xyz_m: np.ndarray, cosmos_ids: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """
    Build a conservative anatomy-only context: standardized xyz + Cosmos one-hot.

    This intentionally avoids waveform/ACG/ephys feature columns, because those
    are targets of the unit encoder.
    """
    br = BrainRegions()
    ctx_parts = [_standardize(xyz_m)]
    ctx_names = ["xyz_x_std", "xyz_y_std", "xyz_z_std"]

    valid_cosmos = np.asarray(cosmos_ids) > 0
    unique_cosmos = np.unique(cosmos_ids[valid_cosmos]).astype(np.int64)
    cosmos_to_col = {int(rid): i for i, rid in enumerate(unique_cosmos)}
    onehot = np.zeros((len(df_good), len(unique_cosmos)), dtype=np.float32)
    for rid, col in cosmos_to_col.items():
        onehot[cosmos_ids == rid, col] = 1.0
    ctx_parts.append(onehot)

    id_to_acronym = {}
    for rid in unique_cosmos:
        hit = np.flatnonzero(br.id == int(rid))
        id_to_acronym[int(rid)] = (
            str(br.acronym[hit[0]]) if len(hit) else f"rid_{int(rid)}"
        )
    ctx_names.extend([f"cosmos_{id_to_acronym[int(rid)]}" for rid in unique_cosmos])

    ctx = np.concatenate(ctx_parts, axis=1).astype(np.float32)
    return ctx, ctx_names


def prepare_latest_cells_encoder_data(
    *,
    root_path: Path,
    out_dir: Path,
    project: str = PROJECT_DEFAULT,
    download: bool = True,
    one_base_url: str = "https://alyx.internationalbrainlab.org",
    target_channels: int = 20,
    overwrite_multichannel_cache: bool = False,
    waveforms_voltage_path: Path | str | None = None,
    waveforms_table_path: Path | str | None = None,
    allow_peak_fallback: bool = False,
    normalize_waveforms_max_abs: bool = True,
    use_acg3d: bool = True,
    use_stpc: bool = False,
) -> dict:
    root_path = Path(root_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    path_project = root_path / project
    cells_agg_path = path_project / "cells_aggregates"

    need_small = not (cells_agg_path / "clusters.table.pqt").exists()
    need_acg3d = use_acg3d and not (cells_agg_path / "clusters.acgs_3d.npy").exists()
    need_stpc = use_stpc and not (cells_agg_path / "clusters_good.stpc.npy").exists()
    need_large = (
        not (cells_agg_path / "waveforms.voltage.npy").exists()
        or not (cells_agg_path / "waveforms.table.pqt").exists()
    )

    if download and (need_small or need_acg3d or need_stpc):
        from one.api import ONE

        one = ONE(base_url=one_base_url)
        ephysatlas.data.download_project_data(
            root_path,
            project=project,
            one=one,
            acg3d=use_acg3d,
        )

    # The 3D ACG aggregate is part of download_project_data(acg3d=True), whereas
    # the large multi-channel waveform files may still require the cell-feature
    # downloader, depending on the installed ephysatlas version.
    if download and need_large and not allow_peak_fallback:
        from one.api import ONE

        one = ONE(base_url=one_base_url)
        ephysatlas.data.download_cells_features(
            root_path,
            project=project,
            one=one,
            large_files=True,
        )

    data = ephysatlas.data.read_cells_features(path_project)
    print("Available read_cells_features keys:")
    print(sorted(data.keys()))

    df_clusters = data["df_clusters"]
    df_clusters_good = data.get("df_clusters_good")
    if use_stpc and df_clusters_good is None:
        good_table_path = cells_agg_path / "clusters_good.table.pqt"
        if not good_table_path.exists():
            raise FileNotFoundError(
                "stPC requires clusters_good.table.pqt for safe row alignment, "
                f"but it was not found at {good_table_path}"
            )
        df_clusters_good = pd.read_parquet(good_table_path)
    acgs_log = np.asarray(data["acgs_log"])

    if "bitwise_fail" not in df_clusters.columns:
        raise KeyError("Expected df_clusters['bitwise_fail'] to select good units.")
    good_pos = np.flatnonzero(df_clusters["bitwise_fail"].to_numpy() == 0)
    df_good = df_clusters.iloc[good_pos].copy()

    pid_col = _first_existing_column(
        df_good, ["pid", "probe_insertion", "probe_insertion_id", "insertion_id"]
    )

    # Remove known misaligned probes before waveform / ACG loading
    try:
        from ephysatlas.fixtures import misaligned_pids

        misaligned_set = set(map(str, misaligned_pids))
        aligned_mask = ~df_good[pid_col].astype(str).isin(misaligned_set).to_numpy()

        n_before = len(df_good)
        good_pos = good_pos[aligned_mask]
        df_good = df_good.iloc[np.flatnonzero(aligned_mask)].copy()

        print(
            f"Removed misaligned pids: {n_before - len(df_good):,} units removed; {len(df_good):,} units remain."
        )
    except Exception as exc:
        print(f"WARNING: could not apply misaligned_pids filter: {exc}")

    atlas_col = _first_existing_column(
        df_good, ["atlas_id", "atlas_id_final", "allen_id", "ccf_id"]
    )

    cache_path = out_dir / f"waveforms_good_multichannel_C{int(target_channels)}.npy"
    waveform_source = "multichannel"
    try:
        waveforms, keep_wf_mask, wf_cache_info = _build_multichannel_waveform_cache(
            cells_agg_path=cells_agg_path,
            df_good=df_good,
            pid_col=pid_col,
            target_channels=target_channels,
            cache_path=cache_path,
            overwrite_cache=overwrite_multichannel_cache,
            waveforms_voltage_path=Path(waveforms_voltage_path)
            if waveforms_voltage_path is not None
            else None,
            waveforms_table_path=Path(waveforms_table_path)
            if waveforms_table_path is not None
            else None,
            normalize_max_abs=normalize_waveforms_max_abs,
        )

        if keep_wf_mask.shape[0] == len(df_good):
            good_pos = good_pos[keep_wf_mask]
            df_good = df_good.iloc[np.flatnonzero(keep_wf_mask)].copy()
        else:
            if len(waveforms) != len(df_good):
                raise ValueError(
                    f"Cached waveform count ({len(waveforms)}) does not match current good-unit count ({len(df_good)}). "
                    "Delete the cache or rerun with --overwrite-multichannel-cache."
                )
    except FileNotFoundError as exc:
        if not allow_peak_fallback:
            raise
        print("\nWARNING: multi-channel waveform files are missing.")
        print(str(exc))
        print(
            "\nContinuing with single-channel peak waveforms because allow_peak_fallback=True.\n"
        )
        waveforms_peak = np.asarray(data["waveforms_peak"])[good_pos].astype(np.float32)
        if waveforms_peak.ndim != 2:
            raise ValueError(
                f"Expected waveforms_peak [N,T], got {waveforms_peak.shape}"
            )
        waveforms = waveforms_peak[:, None, :]
        if normalize_waveforms_max_abs:
            waveforms, waveform_scales = _normalize_waveforms_max_abs(waveforms)
            np.save(
                out_dir / "waveforms_peak_max_abs_scale.npy",
                waveform_scales,
                allow_pickle=False,
            )
        wf_cache_info = {
            "cache_path": None,
            "source": "clusters.waveforms_peak.npy",
            "waveform_source": "single_channel_peak_fallback",
            "normalize_max_abs": bool(normalize_waveforms_max_abs),
            "reason": "waveforms.voltage.npy and/or waveforms.table.pqt were not available",
            "waveforms_shape": list(waveforms.shape),
        }
        waveform_source = "single_channel_peak_fallback"

    stpc = None
    stpc_source = None
    if use_stpc:
        stpc_all = data.get("stpc")
        stpc_path = cells_agg_path / "clusters_good.stpc.npy"
        if stpc_all is None:
            if not stpc_path.exists():
                raise FileNotFoundError(
                    "cfg/use_stpc requested stPC, but the aggregate was not returned "
                    "by read_cells_features() and is missing on disk:\n"
                    f"  {stpc_path}\n"
                    "Download the current cell-feature aggregates first."
                )
            print(f"Memory-mapping stPC from:\n  {stpc_path}")
            stpc_all = np.load(stpc_path, mmap_mode="r", allow_pickle=False)

        stpc = _align_good_cluster_array(
            df_units=df_good,
            unit_pid_col=pid_col,
            df_clusters_good=df_clusters_good,
            values_good=stpc_all,
            feature_name="stPC",
        )
        if stpc.ndim != 2 or stpc.shape[1] != 1000:
            raise ValueError(
                f"Expected stPC [N,1000] per IBL aggregate documentation, got {stpc.shape}"
            )
        stpc_source = "clusters_good.stpc.npy"
        print(f"Aligned stPC to selected units: {stpc.shape}")

    if use_acg3d:
        acg_source = "clusters.acgs_3d.npy"
        acgs_3d_all = data.get("acgs_3d")
        acgs_3d_times = data.get("acgs_3d_times")

        # Some versions of ephysatlas download the new aggregate files but do not yet
        # expose them through read_cells_features(). Fall back to direct memory-mapped
        # loading from the cells_aggregates directory.
        acgs_3d_path = cells_agg_path / "clusters.acgs_3d.npy"
        acgs_3d_times_path = cells_agg_path / "acgs_3d.times.npy"

        if acgs_3d_all is None:
            if not acgs_3d_path.exists():
                raise FileNotFoundError(
                    "The 3D ACG aggregate was not returned by read_cells_features() "
                    "and was not found on disk.\n"
                    f"Expected file:\n  {acgs_3d_path}\n\n"
                    "Make sure download_project_data(..., acg3d=True) completed "
                    "successfully."
                )

            print(
                "read_cells_features() does not expose 'acgs_3d'; "
                f"memory-mapping directly from:\n  {acgs_3d_path}"
            )
            acgs_3d_all = np.load(
                acgs_3d_path,
                mmap_mode="r",
                allow_pickle=False,
            )

        if acgs_3d_times is None:
            if not acgs_3d_times_path.exists():
                raise FileNotFoundError(
                    "The 3D ACG time-bin file was not returned by "
                    "read_cells_features() and was not found on disk.\n"
                    f"Expected file:\n  {acgs_3d_times_path}"
                )

            print(
                "read_cells_features() does not expose 'acgs_3d_times'; "
                f"loading directly from:\n  {acgs_3d_times_path}"
            )
            acgs_3d_times = np.load(
                acgs_3d_times_path,
                allow_pickle=False,
            )

        # good_pos always contains row positions in the original clusters table,
        # even after filtering misaligned probes and missing waveforms.
        acgs = np.asarray(acgs_3d_all[good_pos], dtype=np.float32)
        acgs_3d_times = np.asarray(acgs_3d_times, dtype=np.float32)

        if acgs.ndim != 3 or acgs.shape[1:] != (10, 201):
            raise ValueError(
                f"Expected 3D ACGs with shape [N, 10, 201], got {acgs.shape}."
            )
        if acgs_3d_times.shape != (201,):
            raise ValueError(
                f"Expected acgs_3d_times with shape [201], got {acgs_3d_times.shape}."
            )

        np.save(
            out_dir / "acgs_3d_times.npy",
            acgs_3d_times,
            allow_pickle=False,
        )
        acg_cache_info = {
            "source": "clusters.acgs_3d.npy",
            "times_source": "acgs_3d.times.npy",
            "acgs_shape_before_final_valid_filter": list(acgs.shape),
            "acgs_dtype_on_disk": str(np.asarray(acgs_3d_all).dtype),
            "times_shape": list(acgs_3d_times.shape),
            "times_units": "ms",
            "row_alignment": "clusters.table.pqt",
        }
    else:
        acg_source = "clusters.acgs_log.npy"
        acgs = acgs_log[good_pos].astype(np.float32)[:, None, :]
        acgs_3d_times = None
        acg_cache_info = {
            "source": "clusters.acgs_log.npy",
            "acgs_shape": list(acgs.shape),
            "use_stpc": bool(use_stpc),
            "stpc_shape": list(stpc.shape) if use_stpc else None,
            "stpc_source": stpc_source,
            "stpc_alignment": (
                "Explicit (pid, cluster_id) join to clusters_good.table.pqt"
                if use_stpc
                else None
            ),
            "note": "1D ACG fallback; use_acg3d=False",
        }

    atlas_ids = df_good[atlas_col].to_numpy(dtype=np.int64)

    br = BrainRegions()

    # Remap each unit's Allen anatomical ID to its actual Cosmos anatomical ID.
    # These values are region IDs, not row indices into br.id.
    cosmos_ids = br.remap(
        atlas_ids,
        source_map="Allen",
        target_map="Cosmos",
    ).astype(np.int64)

    known_region_ids = set(map(int, br.id))
    is_known_cosmos_id = np.asarray(
        [int(region_id) in known_region_ids for region_id in cosmos_ids],
        dtype=bool,
    )

    x_col, y_col, z_col, xyz_in_meters = _find_xyz_columns(df_good)
    xyz = df_good[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
    xyz_m = xyz if xyz_in_meters else xyz / 1e6
    pids = df_good[pid_col].astype(str).to_numpy()

    valid = np.isfinite(waveforms).all(axis=(1, 2))
    valid &= np.isfinite(acgs).all(axis=tuple(range(1, acgs.ndim)))
    if use_stpc:
        valid &= np.isfinite(stpc).all(axis=1)
    valid &= np.isfinite(xyz_m).all(axis=1)

    # Keep only valid, known Cosmos anatomical IDs.
    valid &= cosmos_ids != 0
    valid &= is_known_cosmos_id

    df_good = df_good.iloc[np.flatnonzero(valid)].copy()
    waveforms = waveforms[valid]
    acgs = acgs[valid]
    if use_stpc:
        stpc = stpc[valid]
    xyz_m = xyz_m[valid]
    pids = pids[valid]
    cosmos_ids = cosmos_ids[valid]
    atlas_ids = atlas_ids[valid]

    ctx, ctx_names = _build_context(
        df_good,
        xyz_m,
        cosmos_ids,
    )

    unique_cosmos_ids, unit_counts = np.unique(
        cosmos_ids,
        return_counts=True,
    )

    id_to_index = {int(region_id): index for index, region_id in enumerate(br.id)}

    region_acronyms = np.asarray(
        [
            str(br.acronym[id_to_index[int(region_id)]])
            for region_id in unique_cosmos_ids
        ],
        dtype=object,
    )

    order = np.argsort(unit_counts)[::-1]

    print(
        "[prepared Cosmos regions] "
        f"n_regions={len(unique_cosmos_ids)} | "
        f"n_units={len(cosmos_ids):,} | "
        f"regions={list(zip(region_acronyms[order].tolist(), unit_counts[order].tolist()))}"
    )

    np.save(out_dir / "waveforms.npy", waveforms.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "acgs.npy", acgs.astype(np.float32), allow_pickle=False)
    if use_stpc:
        np.save(out_dir / "stpc.npy", stpc.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "ctx.npy", ctx.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "xyz.npy", xyz_m.astype(np.float32), allow_pickle=False)
    np.save(out_dir / "pids.npy", pids.astype(object), allow_pickle=True)
    np.save(out_dir / "cosmos.npy", cosmos_ids.astype(np.int64), allow_pickle=False)
    np.save(out_dir / "allen.npy", atlas_ids.astype(np.int64), allow_pickle=False)

    manifest = {
        "project": project,
        "root_path": str(root_path),
        "path_project": str(path_project),
        "cells_agg_path": str(cells_agg_path),
        "out_dir": str(out_dir),
        "good_units_only": True,
        "n_units": int(len(ctx)),
        "n_pids": int(len(np.unique(pids))),
        "waveforms_shape": list(waveforms.shape),
        "acgs_shape": list(acgs.shape),
        "use_stpc": bool(use_stpc),
        "stpc_shape": list(stpc.shape) if use_stpc else None,
        "stpc_source": stpc_source,
        "stpc_alignment": (
            "Explicit (pid, cluster_id) join to clusters_good.table.pqt"
            if use_stpc
            else None
        ),
        "ctx_shape": list(ctx.shape),
        "xyz_shape": list(xyz_m.shape),
        "xyz_columns": [x_col, y_col, z_col],
        "xyz_in_meters_detected": bool(xyz_in_meters),
        "pid_column": pid_col,
        "atlas_column": atlas_col,
        "target_channels": int(target_channels),
        "ctx_names": ctx_names,
        "waveform_source": waveform_source,
        "normalize_waveforms_max_abs": bool(normalize_waveforms_max_abs),
        "multichannel_waveform_cache": wf_cache_info,
        "acg_source": acg_source,
        "acg_cache": acg_cache_info,
        "acgs_3d_times_path": str(out_dir / "acgs_3d_times.npy") if use_acg3d else None,
        "cosmos_representation": (
            "Actual BrainRegions Cosmos anatomical IDs, one ID per unit; "
            "not indices into BrainRegions.id."
        ),
        "n_cosmos_regions": int(len(np.unique(cosmos_ids))),
        "note": (
            "waveforms.npy is [N,C,128]. acgs.npy is [N,10,201] when "
            "use_acg3d=True, otherwise [N,1,128]. stpc.npy is [N,1000] when "
            "use_stpc=True and is explicitly aligned through clusters_good.table.pqt."
        ),
    }
    with open(
        out_dir / "latest_cells_encoder_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "n_units",
                    "n_pids",
                    "waveforms_shape",
                    "acgs_shape",
                    "stpc_shape",
                    "ctx_shape",
                    "xyz_shape",
                    "acg_source",
                ]
            },
            indent=2,
        )
    )
    print(f"saved encoder arrays to: {out_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, default=Path("./examples"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./unit_level_encoder_latest")
    )
    parser.add_argument("--project", type=str, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--one-base-url", type=str, default="https://alyx.internationalbrainlab.org"
    )
    parser.add_argument("--target-channels", type=int, default=20)
    parser.add_argument("--overwrite-multichannel-cache", action="store_true")
    parser.add_argument("--waveforms-voltage-path", type=Path, default=None)
    parser.add_argument("--waveforms-table-path", type=Path, default=None)
    parser.add_argument("--allow-peak-fallback", action="store_true")
    parser.add_argument("--no-waveform-normalization", action="store_true")
    parser.add_argument(
        "--use-1d-acg",
        action="store_true",
        help="Use clusters.acgs_log.npy instead of recomputing 3D ACGs.",
    )
    parser.add_argument(
        "--use-stpc",
        action="store_true",
        help="Load and save clusters_good.stpc.npy as stpc.npy.",
    )
    args = parser.parse_args()

    prepare_latest_cells_encoder_data(
        root_path=args.root_path,
        out_dir=args.out_dir,
        project=args.project,
        one_base_url=args.one_base_url,
        target_channels=args.target_channels,
        overwrite_multichannel_cache=args.overwrite_multichannel_cache,
        waveforms_voltage_path=args.waveforms_voltage_path,
        waveforms_table_path=args.waveforms_table_path,
        allow_peak_fallback=args.allow_peak_fallback,
        normalize_waveforms_max_abs=not args.no_waveform_normalization,
        use_acg3d=not args.use_1d_acg,
        use_stpc=args.use_stpc,
    )
