"""A self-contained copy of a fitted ephys-only localizer, for software outside Python.

:func:`export_localizer_bundle` writes everything :class:`~ephysatlas.alignment.EphysOnlyLocalizer`
needs to score and search trajectories -- no model, no torch -- as ``.npy`` arrays (little-endian,
C order) and a ``manifest.json``:

- ``grid_xyz_um.npy`` ``[V, 3]`` float64: the candidate lattice nodes (µm, left hemisphere);
- ``grid_pred_std.npy`` ``[V, F]`` float32: the channel model's standardised predictions there;
- ``W.npy`` ``[F, F]``, ``mu_r.npy`` ``[F]`` float64: the residual whitening and location;
- ``clip_lo.npy``, ``clip_hi.npy``, ``e_mean.npy``, ``e_std.npy`` ``[F]`` float32: the release
  preprocessing of the recorded features (clip, then standardise);
- ``labels_cosmos.npy``, ``labels_beryl.npy`` ``[nx, ny, nz]`` uint16 and ``regions.json``: region
  of every point of a regular grid (for display), as row indices of the region table;
- ``selftest/``: one probe's inputs with the Python scores of hundreds of trajectories and the
  Python result, so that a port can check itself (see ``manifest["selftest"]``).

The manifest holds the feature names, the fitted scalars (``nu``, ``kappa``, ``T``), the plan
prior, the score and annealing configurations and the grid geometry. The localizer's method is
described in :mod:`ephysatlas.alignment.ephys_only`; a port reproduces it from these files.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

BUNDLE_FORMAT = "ephysatlas-localizer-bundle/1"
# Region label volumes: grid spacing (display only; scores use the full-resolution atlas).
LABEL_STEP_UM = 50.0
# Self-test: trajectories scored (rigid perturbations of the plan, then random parametrised ones).
SELFTEST_N_RIGID = 128
SELFTEST_N_PARAMS = 128


def default_bundle_dir(model_commit: str, feature_set: str) -> Path:
    """``~/ephys-atlas/results/realtime/bundles/<model_commit>_<feature_set>``."""
    from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

    return (
        Path(DEFAULT_RESULTS_DIR)
        / "realtime"
        / "bundles"
        / f"{model_commit}_{feature_set}"
    )


def _save(path: Path, array: np.ndarray, dtype) -> dict:
    a = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    np.save(path, a)
    return dict(file=path.name, shape=list(a.shape), dtype=str(a.dtype))


def _label_volumes(brain_atlas, step_um: float, out_dir: Path) -> dict:
    """Cosmos and Beryl region indices on a regular grid over the atlas, and the region table."""
    from ephysatlas.alignment.geometry import region_ids

    bc = brain_atlas.bc
    lims = [
        np.sort(np.asarray(lim, dtype=np.float64))
        for lim in (bc.xlim, bc.ylim, bc.zlim)
    ]
    step = float(step_um) * 1e-6
    axes = [np.arange(lo, hi + 0.5 * step, step) for lo, hi in lims]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    files, used = {}, set()
    for mapping in ("Cosmos", "Beryl"):
        idx = region_ids(brain_atlas, xyz, mapping).reshape(X.shape)
        used |= set(np.unique(idx).tolist())
        files[mapping] = _save(
            out_dir / f"labels_{mapping.lower()}.npy", idx, np.uint16
        )
    regions = brain_atlas.regions
    table = [
        dict(
            index=int(i),
            id=int(regions.id[i]),
            acronym=str(regions.acronym[i]),
            name=str(regions.name[i]),
            rgb=[int(c) for c in regions.rgb[i]],
        )
        for i in sorted(used)
    ]
    (out_dir / "regions.json").write_text(json.dumps(table, indent=0), encoding="utf-8")
    return dict(
        step_um=float(step_um),
        origin_m=[float(a[0]) for a in axes],
        shape=[len(a) for a in axes],
        axes="[ix, iy, iz] <-> origin_m + step_um * 1e-6 * (ix, iy, iz), x ML, y AP, z DV (m)",
        files=files,
        regions="regions.json (index -> Allen id, acronym, name, rgb; 0 is outside the brain)",
    )


def _selftest(localizer, probe: dict, out_dir: Path) -> dict:
    """Score trajectories of one probe with the Python localizer and save inputs and outputs."""
    from ephysatlas.alignment.ephys_only import (
        ProbeScore,
        TraceParams,
        mad,
        plan_neighbourhood_traces,
        planned_seed,
        stable_seed,
        trace_points,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, acfg = localizer.score_cfg, localizer.anneal_cfg
    features = np.asarray(probe["features"], dtype=np.float64)
    planned = np.asarray(probe["planned_xyz"], dtype=np.float64)
    used = features[:, localizer.feature_idx]
    prep = localizer._prepare(features, "", wg=localizer.wg)
    bins = prep["bins"]
    ref = localizer.prior.reference(planned, bins)
    score = ProbeScore(
        localizer.grid,
        prep["wg"],
        prep["wx"],
        float(localizer.params["nu"]),
        localizer.T,
        localizer.prior,
        ref,
        cfg,
    )
    seed, z_sign = planned_seed(planned, acfg)
    seed_trace, _ = trace_points(seed, bins.centers_um, z_sign, acfg)
    pool = plan_neighbourhood_traces(
        seed_trace,
        cfg.scale_pool_size,
        np.random.default_rng(stable_seed("sa_scale_pool", cfg.scale_pool_version, "")),
    )
    mad_s = mad(score.components(pool)["raw"])
    score.scale = float(cfg.scale_K / mad_s)

    rng = np.random.default_rng(12345)
    rigid = plan_neighbourhood_traces(seed_trace, SELFTEST_N_RIGID, rng)
    n_b = int(acfg.n_sections) - 1
    params = np.column_stack(
        [
            -np.abs(seed.x + rng.normal(0, 500, SELFTEST_N_PARAMS)),
            seed.y + rng.normal(0, 500, SELFTEST_N_PARAMS),
            seed.z + rng.normal(0, 500, SELFTEST_N_PARAMS),
            rng.uniform(acfg.min_angle_deg, acfg.max_angle_deg, SELFTEST_N_PARAMS),
            rng.uniform(0, 360, SELFTEST_N_PARAMS),
            rng.uniform(
                acfg.min_probe_length_um, acfg.max_probe_length_um, SELFTEST_N_PARAMS
            ),
            rng.normal(
                0, 6.0, (SELFTEST_N_PARAMS, 2 * n_b)
            ),  # bends, some beyond the cap
        ]
    )
    param_traces = np.stack(
        [
            trace_points(
                TraceParams(*p[:6], bends=p[6:].reshape(n_b, 2)),
                bins.centers_um,
                z_sign,
                acfg,
            )[0]
            for p in params
        ]
    )
    traces = np.concatenate([rigid, param_traces]).astype(np.float32)
    comp = score.components(traces)
    result = localizer.localize(features, planned, pid="")
    bin_of_channel = np.full(len(features), -1, dtype=np.int32)
    for b, rows in enumerate(bins.channel_idx):
        bin_of_channel[rows] = b

    files = dict(
        features=_save(out_dir / "features.npy", used, np.float64),
        planned_xyz_m=_save(out_dir / "planned_xyz_m.npy", planned, np.float64),
        bin_of_channel=_save(out_dir / "bin_of_channel.npy", bin_of_channel, np.int32),
        bin_centers_um=_save(
            out_dir / "bin_centers_um.npy", bins.centers_um, np.float64
        ),
        wx=_save(out_dir / "wx.npy", prep["wx"], np.float64),
        seed_trace_um=_save(out_dir / "seed_trace_um.npy", seed_trace, np.float64),
        pool_traces_um=_save(out_dir / "pool_traces_um.npy", pool, np.float64),
        trace_params=_save(out_dir / "trace_params.npy", params, np.float64),
        traces_um=_save(out_dir / "traces_um.npy", traces, np.float32),
        lik_sum=_save(out_dir / "lik_sum.npy", comp["lik_sum"], np.float64),
        outside=_save(out_dir / "outside.npy", comp["outside"], np.float64),
        prior=_save(out_dir / "prior.npy", comp["prior"], np.float64),
        raw=_save(out_dir / "raw.npy", comp["raw"], np.float64),
        result_trace_xyz_m=_save(
            out_dir / "result_trace_xyz_m.npy", result.trace_xyz, np.float64
        ),
        result_channel_xyz_m=_save(
            out_dir / "result_channel_xyz_m.npy", result.channel_xyz, np.float64
        ),
    )
    return dict(
        pid=str(probe.get("pid", "")),
        description=(
            "features: [C, F] recorded features of the probe (feature units, row order). "
            "traces_um: the first n_rigid are rigid perturbations of the planned seed, the rest "
            "trace_points(trace_params[i]) (x, y, z, theta, phi, length, bends) at bin_centers_um "
            "with z_sign; lik_sum / outside / prior / raw are their score components (raw = S, "
            "before the scale c_p). result_*: the Python search's result (restarts "
            "seeded from 'seed'; a port with another random generator finds a nearby result)."
        ),
        n_rigid=SELFTEST_N_RIGID,
        z_sign=z_sign,
        seed_params=dict(asdict(seed), bends=np.asarray(seed.bends).tolist()),
        seed_score_raw=float(score.components(seed_trace[None])["raw"][0]),
        mad_S=float(mad_s),
        c_p=float(score.scale),
        prior_reference=dict(
            c_plan_um=np.asarray(ref["c_plan"]).tolist(),
            axis_plan=np.asarray(ref["axis_plan"]).tolist(),
            L_plan_um=float(ref["L_plan"]),
            degenerate=bool(ref["degenerate"]),
            widen=float(ref["widen"]),
            sections=[np.asarray(s).tolist() for s in ref["sections"]],
        ),
        result_score=float(result.diagnostics["score"]),
        files=files,
    )


def export_localizer_bundle(
    localizer,
    out_dir=None,
    *,
    selftest_probe: Optional[dict] = None,
    label_step_um: float = LABEL_STEP_UM,
) -> Path:
    """Write the bundle of a fitted localizer (see the module doc).

    Args:
        localizer: A fitted :class:`~ephysatlas.alignment.EphysOnlyLocalizer`.
        out_dir: Output directory; default :func:`default_bundle_dir`.
        selftest_probe: ``pid``, ``features`` ``[C, 43]`` and ``planned_xyz`` ``[C, 3]`` of a probe
            (e.g. :func:`ephysatlas.alignment.data.load_probe` of a test insertion) for the
            self-test; none written when None.
        label_step_um: Spacing of the region label volumes.

    Returns:
        Path: The bundle directory.
    """
    out_dir = Path(
        out_dir or default_bundle_dir(localizer.model_commit, localizer.feature_set)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    p = localizer.params
    st = localizer._stats
    files = dict(
        grid_xyz_um=_save(
            out_dir / "grid_xyz_um.npy", localizer.grid.xyz_um, np.float64
        ),
        grid_pred_std=_save(
            out_dir / "grid_pred_std.npy", localizer.grid_pred, np.float32
        ),
        W=_save(out_dir / "W.npy", p["W"], np.float64),
        mu_r=_save(out_dir / "mu_r.npy", p["mu_r"], np.float64),
        clip_lo=_save(out_dir / "clip_lo.npy", st["rec_ephys_low_pctl"], np.float32),
        clip_hi=_save(out_dir / "clip_hi.npy", st["rec_ephys_high_pctl"], np.float32),
        e_mean=_save(out_dir / "e_mean.npy", st["e_mean"], np.float32),
        e_std=_save(out_dir / "e_std.npy", st["e_std"], np.float32),
    )
    cm = localizer.channel_model
    manifest = dict(
        format=BUNDLE_FORMAT,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        method="student_t_mahalanobis_plan_posterior (ephysatlas.alignment.ephys_only)",
        channel_model=f"{getattr(cm, 'repo_id', '')}@{getattr(cm, 'revision', '')}",
        model_commit=localizer.model_commit,
        feature_set=localizer.feature_set,
        feature_names=list(localizer.feature_names),
        n_features=len(localizer.feature_names),
        coordinates="IBL-Allen, origin bregma; x ML, y AP, z DV; lattice in µm, left hemisphere",
        grid=dict(
            step_um=float(localizer.grid.step),
            origin_um=localizer.grid.origin.tolist(),
            shape=[int(s) for s in localizer.grid.shape],
            n_voxels=int(localizer.grid.n_voxels),
            node_index="ijk = round((xyz_um - origin_um) / step_um)",
        ),
        params=dict(
            nu=float(p["nu"]),
            kappa=float(p["kappa"]),
            T=float(localizer.T),
            prior=p["prior"],
        ),
        score_config=asdict(localizer.score_cfg),
        anneal_config=asdict(localizer.anneal_cfg),
        n_restarts=int(localizer.n_restarts),
        seed=int(localizer.seed),
        restart_seed="seed + 1000003 * restart",
        files=files,
        labels=_label_volumes(localizer.brain_atlas, label_step_um, out_dir),
    )
    if selftest_probe is not None:
        manifest["selftest"] = _selftest(
            localizer, selftest_probe, out_dir / "selftest"
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=float), encoding="utf-8"
    )
    return out_dir
