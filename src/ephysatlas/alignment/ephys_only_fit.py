"""Fitting the ephys-only localizer (:mod:`.ephys_only`) on the channel model's release split.

Everything is refitted per model release, with the release's own split, so that no probe is
scored by parameters it helped to fit:

1. **Residual model (TRAIN)** -- for every training probe, the bin residuals ``x_b - mu(p_b)`` at
   its reference (human alignment) bin positions, with ``mu`` interpolated on its
   *leave-probe-out* lattice: the lattice voxels within the neighbour radius of the probe's own
   bank channels re-predicted with the probe excluded from the neighbours, so a training probe
   never predicts itself. Ledoit-Wolf covariance ``Sigma`` and location ``mu_r``; Student-t
   degrees of freedom ``nu`` by maximum likelihood over {3, 4, 5, 6, 8, 10, 15, 20, 30}.
2. **Scale ratio kappa (VALIDATION)** -- ``kappa^2`` = mean validation / mean training Mahalanobis
   norm under ``Sigma`` (validation probes are not in the bank: plain lattice);
   ``W = (kappa^2 Sigma)^-1/2``.
3. **Plan prior (TRAIN)** -- reference vs planned bin trajectories: per-axis Student-t of the
   centre offset (``scipy.stats.t.fit``), bivariate-t radial fits of the axis angle and of the
   section bends, median and robust SD of the length ratio (``scoring_components.PlanPrior.fit``).
4. **Temperature T (VALIDATION)** -- the full search on every validation probe with a usable plan,
   for each T of ``temperatures``; T with the lowest mean per-probe channel distance to the human
   alignment wins (``tune_scoring_temperature.py``, SA stage).

TEST probes are never read. Outputs in the localizer's ``fit_dir``: ``fit.npz`` (the parameters;
T is NaN until tuned), ``temperature_tuning.csv`` (one row per probe and T, resumable) and
``fit_summary.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .ephys_only import (
    FIT_FILE,
    SUMMARY_FILE,
    DenseGrid,
    DepthBins,
    PlannedTrajectoryUnavailable,
    _to_metres,
    check_planned_trajectory,
    clip_standardize,
    joint_angles_deg,
    mirror_left,
    plan_axis_um,
    section_bin_indices,
    signal_channels,
)
from .progress import ProgressCallback, report, sub_progress

# Degrees of freedom tried for the residual Student-t (scoring_components.NU_GRID).
NU_GRID = (3, 4, 5, 6, 8, 10, 15, 20, 30)
# Temperatures tried on the validation probes (the source's grid was 3, 10, 20, 40, 80, 160).
TEMPERATURES = (5.0, 10.0, 20.0, 40.0, 80.0)
# Plan-prior fit: probes with fewer bins holding both a reference and a planned position, or with
# a degenerate plan (ScoreConfig.plan_min_span_um), are left out (PlanPrior.fit).
PRIOR_MIN_BINS = 20
# Residuals need at least this many bins with a reference position (residual_sets).
RESIDUAL_MIN_BINS = 5
# Queries per model call when predicting the leave-probe-out lattices.
LOO_CHUNK = 16384
TUNING_FILE = "temperature_tuning.csv"


def robust_sd(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def _inv_sqrt_psd(S: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(S)
    return (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T


def _fit_nu(m: np.ndarray, F: int) -> tuple[float, dict]:
    """ML degrees of freedom of a multivariate t with covariance Sigma from Mahalanobis norms ``m``
    (``m (nu - 2) / (nu F)`` is F(F, nu)-distributed; with the Jacobian)."""
    from scipy import stats

    m = np.asarray(m, dtype=np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    ll = {}
    for nu in NU_GRID:
        c = (nu - 2.0) / (nu * F)
        ll[nu] = float(np.sum(stats.f.logpdf(m * c, F, nu) + np.log(c)))
    return float(max(ll, key=ll.get)), {str(k): v for k, v in ll.items()}


def _fit_radial_t2(r: np.ndarray, nu_grid=(2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)) -> dict:
    """ML isotropic bivariate t (``nu``, scale ``s``) from the norms ``r`` of 2-D vectors."""
    from scipy.optimize import minimize_scalar

    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r) & (r > 1e-6)]
    best = None
    for nu in nu_grid:

        def nll(log_s, nu=nu):
            s2 = np.exp(2.0 * log_s)
            return -float(
                np.sum(np.log(r) - np.log(s2) - 0.5 * (nu + 2.0) * np.log1p(r * r / (nu * s2)))
            )

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(200.0)), method="bounded")
        if best is None or res.fun < best[2]:
            best = (float(nu), float(np.exp(res.x)), float(res.fun))
    return dict(nu=best[0], s=best[1], nll=best[2], n=int(r.size))


def _split_pids(localizer, dataset) -> dict:
    """The release split restricted to the dataset; test pids only for the leakage guard."""
    split = localizer.channel_model.split()
    have = set(dataset.pids.astype(str))
    out = {
        k: sorted(set(map(str, split[f"{k}_pids"])) & have) for k in ("train", "validation", "test")
    }
    out["counts"] = {
        k: dict(split=len(split[f"{k}_pids"]), in_dataset=len(out[k]))
        for k in ("train", "validation", "test")
    }
    return out


def probe_inputs(localizer, dataset, pid: str, test: set) -> dict:
    """Bins, clipped bin means and binned reference / planned positions of a TRAIN or
    VALIDATION probe (µm, left hemisphere)."""
    if pid in test:
        raise RuntimeError(f"{pid} is a test probe; the ephys-only fit never reads test probes")
    p = dataset.probe(pid)
    used = np.asarray(p["features"], dtype=np.float64)[:, localizer.feature_idx]
    bins = DepthBins.from_valid(signal_channels(used), localizer.score_cfg)
    st = localizer._stats
    x = bins.means(
        clip_standardize(
            used, st["rec_ephys_low_pctl"], st["rec_ephys_high_pctl"], st["e_mean"], st["e_std"]
        )
    )
    return dict(
        pid=pid,
        bins=bins,
        x=x,
        planned_xyz=np.asarray(p["planned_xyz"], dtype=np.float64),
        true_bins_um=bins.mean_xyz_um(mirror_left(np.asarray(p["human_xyz"], dtype=np.float64))),
        plan_bins_um=bins.mean_xyz_um(mirror_left(np.asarray(p["planned_xyz"], dtype=np.float64))),
    )


def leave_probe_out_patches(
    localizer, pids: Sequence[str], progress: Optional[ProgressCallback] = None
) -> dict:
    """``{pid: (voxel ids, [n, F] standardised predictions)}``: each training probe's lattice
    voxels re-predicted with that probe excluded from the model's neighbours."""
    vox = {pid: localizer.loo_voxels(pid) for pid in pids}
    all_vox = np.concatenate([vox[p] for p in pids])
    all_pid = np.concatenate([np.full(len(vox[p]), p) for p in pids])
    pred = np.zeros((len(all_vox), len(localizer.feature_idx)))
    for start in range(0, len(all_vox), LOO_CHUNK):
        stop = min(len(all_vox), start + LOO_CHUNK)
        report(
            progress,
            start / max(len(all_vox), 1),
            f"Leave-probe-out lattices: {start:,}/{len(all_vox):,} voxels",
        )
        # One pid per query position: each is predicted without its own probe's bank channels.
        pred[start:stop] = localizer.channel_model.predict_std(
            localizer.grid_xyz_m[all_vox[start:stop]], all_pid[start:stop]
        )[:, localizer.feature_idx]
    out, offset = {}, 0
    for p in pids:
        n = len(vox[p])
        out[p] = (vox[p], pred[offset : offset + n])
        offset += n
    return out


def _residuals(localizer, inputs: list, patches: Optional[dict]) -> dict:
    """Bin residuals ``x_b - mu(p_b)`` at the reference bin positions; ``mu`` on each probe's
    leave-probe-out lattice when ``patches`` are given, else on the cached lattice."""
    grid: DenseGrid = localizer.grid
    G = localizer.grid_pred
    R = []
    for d in inputs:
        T = d["true_bins_um"]
        ok = np.isfinite(T).all(axis=1)
        if ok.sum() < RESIDUAL_MIN_BINS:
            continue
        ids, w, _ = grid.locate(T[ok])
        vals = G[ids]
        if patches is not None and len(patches[d["pid"]][0]):
            vox, pred = patches[d["pid"]]
            lut = np.full(grid.n_voxels, -1, dtype=np.int64)
            lut[vox] = np.arange(len(vox))
            rows = lut[ids]
            vals = np.where((rows >= 0)[..., None], pred[np.maximum(rows, 0)], vals)
        R.append(d["x"][ok] - np.einsum("nk,nkf->nf", w, vals))
    return dict(R=np.concatenate(R), n_probes=len(R))


def fit_residual_model(
    localizer, train_inputs: list, val_inputs: list, patches: dict
) -> tuple[dict, dict]:
    """``mu_r``, ``Sigma`` (Ledoit-Wolf), ``nu`` on TRAIN leave-probe-out residuals; ``kappa`` from
    VALIDATION; ``W = (kappa^2 Sigma)^-1/2`` (``scoring_components.fit_residual_model``)."""
    from sklearn.covariance import LedoitWolf

    tr = _residuals(localizer, train_inputs, patches)
    va = _residuals(localizer, val_inputs, None)
    R, Rv = tr["R"], va["R"]
    F = R.shape[1]
    lw = LedoitWolf().fit(R)
    mu_r = lw.location_.astype(np.float64)
    Sigma = lw.covariance_.astype(np.float64)
    Sinv = np.linalg.inv(Sigma)
    m_tr = np.einsum("ij,jk,ik->i", R - mu_r, Sinv, R - mu_r)
    m_va = np.einsum("ij,jk,ik->i", Rv - mu_r, Sinv, Rv - mu_r)
    # kappa: validation/train ratio of the whitened residual scale, so that held-out residuals
    # have the training Mahalanobis scale under kappa^2 Sigma.
    kappa = float(np.sqrt(np.mean(m_va) / np.mean(m_tr)))
    nu, ll_nu = _fit_nu(m_tr, F)
    W = _inv_sqrt_psd(kappa**2 * Sigma)
    nu_va, _ = _fit_nu(m_va / kappa**2, F)
    evals = np.linalg.eigvalsh(np.corrcoef(R.T))
    params = dict(
        mu_r=mu_r,
        Sigma=Sigma,
        W=W,
        nu=np.float64(nu),
        kappa=np.float64(kappa),
        shrinkage=np.float64(lw.shrinkage_),
    )
    rep = dict(
        n_train_probes=int(tr["n_probes"]),
        n_train_bins=int(len(R)),
        n_validation_probes=int(va["n_probes"]),
        n_validation_bins=int(len(Rv)),
        ledoit_wolf_shrinkage=float(lw.shrinkage_),
        kappa=kappa,
        nu=nu,
        nu_validation=nu_va,
        loglik_nu_train=ll_nu,
        residual_mean_abs_median=float(np.median(np.abs(mu_r))),
        residual_sd_train_median=float(np.median(R.std(axis=0))),
        residual_sd_validation_median=float(np.median(Rv.std(axis=0))),
        mahalanobis_over_F_train_mean=float(np.mean(m_tr) / F),
        mahalanobis_over_F_validation_mean=float(np.mean(m_va) / F),
        effective_features_participation_ratio=float(evals.sum() ** 2 / (evals**2).sum()),
    )
    return params, rep


def fit_plan_prior(localizer, train_inputs: list) -> tuple[dict, dict]:
    """Plan prior parameters from TRAIN reference vs planned bin trajectories
    (``scoring_components.PlanPrior.fit``); degenerate plans are left out."""
    from scipy import stats

    cfg = localizer.score_cfg
    dc, ang, ratio, bends, excluded = [], [], [], [], []
    for d in train_inputs:
        axis_plan, span = plan_axis_um(d["planned_xyz"])
        T, P = d["true_bins_um"], d["plan_bins_um"]
        ok = np.isfinite(T).all(axis=1) & np.isfinite(P).all(axis=1)
        if span < cfg.plan_min_span_um or ok.sum() < PRIOR_MIN_BINS:
            excluded.append(d["pid"])
            continue
        T, P = T[ok], P[ok]
        dc.append(T.mean(axis=0) - P.mean(axis=0))
        at = (T[-1] - T[0]) / max(np.linalg.norm(T[-1] - T[0]), 1e-12)
        ang.append(float(np.degrees(np.arccos(np.clip(np.dot(at, axis_plan), -1.0, 1.0)))))
        ratio.append(np.linalg.norm(T[-1] - T[0]) / max(np.linalg.norm(P[-1] - P[0]), 1.0))
        sections = section_bin_indices(d["bins"].centers_um[ok], cfg.n_sections)
        bends.append(joint_angles_deg(T[None], sections)[0])
    dc, ang, ratio = np.asarray(dc), np.asarray(ang), np.asarray(ratio)
    centre = {}
    for k, name in enumerate("xyz"):
        nu, loc, s = stats.t.fit(dc[:, k])
        centre[name] = (float(nu), float(loc), float(s))
    axis = _fit_radial_t2(ang)
    rr = ratio[(ratio > 0.7) & (ratio < 1.4)]
    bends = np.asarray(bends).ravel()
    bend = _fit_radial_t2(bends)
    params = dict(
        centre=centre,
        axis=(axis["nu"], axis["s"]),
        length=(float(np.median(rr)), max(robust_sd(rr), 1e-3)),
        bend=(bend["nu"], bend["s"]),
    )
    rep = dict(
        n_probes=int(len(dc)),
        n_excluded=int(len(excluded)),
        centre_offset_median_um=np.median(dc, axis=0).round(1).tolist(),
        axis_angle_deg_median=float(np.median(ang)),
        axis_fit=axis,
        length_ratio_n=int(rr.size),
        bend_deg_median=float(np.median(bends)),
        bend_fit=bend,
    )
    return params, rep


def fit_key(localizer) -> str:
    """Digest of everything a validation search depends on (for resuming the T tuning)."""
    h = hashlib.sha256()
    p = localizer.params
    for key in ("mu_r", "W", "nu"):
        h.update(np.ascontiguousarray(np.asarray(p[key], dtype=np.float64)).tobytes())
    score_cfg = {k: v for k, v in asdict(localizer.score_cfg).items() if k != "temperature_T"}
    h.update(
        json.dumps(
            [
                p["prior"],
                score_cfg,
                asdict(localizer.anneal_cfg),
                localizer.n_restarts,
                localizer.seed,
                localizer.model_commit,
            ],
            sort_keys=True,
            default=str,
        ).encode()
    )
    h.update(np.ascontiguousarray(localizer.grid_pred).tobytes())
    return h.hexdigest()[:16]


def tune_temperature(
    localizer,
    dataset,
    pids: Sequence[str],
    temperatures: Sequence[float],
    progress: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Search every VALIDATION probe of ``pids`` at every temperature; one row per (T, probe).

    Rows accumulate in ``fit_dir/temperature_tuning.csv``; rows of the same fit (same
    :func:`fit_key`) are reused, so an interrupted tuning resumes. Probes without a usable plan
    are skipped.
    """
    from .metrics import alignment_metrics

    key = fit_key(localizer)
    path = localizer.fit_dir / TUNING_FILE
    rows = pd.read_csv(path, dtype={"fit_key": str}) if path.exists() else pd.DataFrame()
    if len(rows):
        rows = rows[rows["fit_key"] == key]
    done = set(zip(rows["T"].astype(float), rows["pid"].astype(str))) if len(rows) else set()
    tasks = [
        (float(T), str(pid))
        for T in temperatures
        for pid in pids
        if (float(T), str(pid)) not in done
    ]
    new = []
    for i, (T, pid) in enumerate(tasks):
        report(
            progress,
            i / max(len(tasks), 1),
            f"Tuning T on validation probes: T={T:g}, {i + 1}/{len(tasks)}",
        )
        probe = dataset.probe(pid)
        try:
            check_planned_trajectory(
                probe["planned_xyz"], localizer.score_cfg.min_usable_plan_span_um
            )
        except PlannedTrajectoryUnavailable:
            continue
        t0 = time.time()
        search = localizer._search(probe["features"], probe["planned_xyz"], pid, T, wg=localizer.wg)
        est = _to_metres(search.channel_um, search.right_hemisphere)
        m = alignment_metrics(probe["human_xyz"], est, localizer.brain_atlas, valid=search.valid)
        plan = alignment_metrics(probe["human_xyz"], probe["planned_xyz"], localizer.brain_atlas)
        new.append(
            dict(
                fit_key=key,
                T=T,
                pid=pid,
                **{
                    k: m[k]
                    for k in ("mean_distance_um", "median_distance_um", "cosmos_acc", "beryl_acc")
                },
                **{
                    f"planned_{k}": plan[k] for k in ("mean_distance_um", "cosmos_acc", "beryl_acc")
                },
                score=search.diagnostics["score"],
                c_p=search.diagnostics["c_p"],
                seconds=time.time() - t0,
            )
        )
        if len(new) % 10 == 0 or i == len(tasks) - 1:
            rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True)
            new = []
            rows.to_csv(path, index=False)
    if new:
        rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True)
        rows.to_csv(path, index=False)
    rows = rows[rows["T"].isin([float(t) for t in temperatures])]
    return rows.reset_index(drop=True)


def temperature_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-T summary over the probes searched at every T (paired comparison)."""
    n_t = rows["T"].nunique()
    counts = rows.groupby("pid")["T"].nunique()
    rows = rows[rows["pid"].isin(counts[counts == n_t].index)].copy()
    rows["win_vs_plan"] = rows["mean_distance_um"] < rows["planned_mean_distance_um"]
    table = rows.groupby("T").agg(
        n=("pid", "nunique"),
        mean_distance_um=("mean_distance_um", "mean"),
        sem_distance_um=("mean_distance_um", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
        median_distance_um=("mean_distance_um", "median"),
        cosmos_acc=("cosmos_acc", "mean"),
        beryl_acc=("beryl_acc", "mean"),
        win_vs_plan=("win_vs_plan", "mean"),
        planned_mean_distance_um=("planned_mean_distance_um", "mean"),
        planned_cosmos_acc=("planned_cosmos_acc", "mean"),
        planned_beryl_acc=("planned_beryl_acc", "mean"),
        seconds_per_probe=("seconds", "mean"),
    )
    return table.reset_index()


def _write_fit(localizer, params: dict, T: float) -> None:
    localizer.fit_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        localizer.fit_dir / FIT_FILE,
        model_commit=np.asarray(localizer.model_commit),
        feature_names=np.asarray(localizer.feature_names),
        mu_r=params["mu_r"],
        Sigma=params["Sigma"],
        W=params["W"],
        nu=params["nu"],
        kappa=params["kappa"],
        shrinkage=params["shrinkage"],
        T=np.float64(T),
        scale_K=np.float64(localizer.score_cfg.scale_K),
        prior_json=np.asarray(json.dumps(params["prior"])),
    )


def fit_residual_and_prior(localizer, dataset, progress: Optional[ProgressCallback] = None) -> dict:
    """Steps 1-3: residual model, kappa and plan prior. Sets them on ``localizer`` and writes
    ``fit.npz`` with T = NaN; returns the split, the parameters and the fit reports."""
    t0 = time.time()
    split = _split_pids(localizer, dataset)
    test = set(split["test"])
    if test & (set(split["train"]) | set(split["validation"])):
        raise RuntimeError("the release split overlaps; refusing to fit")
    bank_pids = set(localizer.channel_model.encoder._neighbor_bank()["pid"].astype(str))
    if bank_pids & (set(split["validation"]) | test):
        raise RuntimeError("the neighbour bank holds validation or test probes")

    report(progress, 0.0, "Binning the training and validation probes")
    inputs, no_signal = {}, []
    for which in ("train", "validation"):
        inputs[which] = []
        for pid in split[which]:
            try:
                inputs[which].append(probe_inputs(localizer, dataset, pid, test))
            except ValueError:  # fewer than 2 depth bins with a signal
                no_signal.append(pid)
    t1 = time.time()
    patches = leave_probe_out_patches(localizer, split["train"], sub_progress(progress, 0.05, 0.9))
    t_loo = time.time() - t1
    report(progress, 0.9, "Fitting the residual model")
    residual, residual_report = fit_residual_model(
        localizer, inputs["train"], inputs["validation"], patches
    )
    report(progress, 0.97, "Fitting the plan prior")
    prior, prior_report = fit_plan_prior(localizer, inputs["train"])
    params = dict(residual, prior=prior, T=np.float64(np.nan))
    localizer.set_params(params)
    _write_fit(localizer, params, np.nan)
    residual_report.update(
        W_condition=float(np.linalg.cond(residual["W"])),
        n_leave_probe_out_voxels=int(sum(len(v) for v, _ in patches.values())),
    )
    return dict(
        split=split,
        params=params,
        residual_report=residual_report,
        prior_report=prior_report,
        n_probes_without_signal=len(no_signal),
        timings=dict(leave_probe_out_s=t_loo, residual_prior_s=time.time() - t0),
    )


def fit_localizer(
    localizer,
    dataset=None,
    progress: Optional[ProgressCallback] = None,
    temperatures: Sequence[float] = TEMPERATURES,
) -> dict:
    """Fit ``localizer`` (see the module doc) and write ``fit_dir``; returns the summary."""
    from .data import ChannelDataset
    from .models import VINTAGE

    t0 = time.time()
    if dataset is None:
        report(progress, 0.0, "Loading the channel features")
        dataset = ChannelDataset.load(getattr(localizer.channel_model, "revision", VINTAGE))
    stage = fit_residual_and_prior(localizer, dataset, sub_progress(progress, 0.0, 0.25))
    params = stage["params"]
    t1 = time.time()
    rows = tune_temperature(
        localizer,
        dataset,
        stage["split"]["validation"],
        temperatures,
        sub_progress(progress, 0.25, 1.0),
    )
    table = temperature_table(rows)
    T = float(table.loc[table["mean_distance_um"].idxmin(), "T"])
    params["T"] = np.float64(T)
    localizer.set_params(params)
    _write_fit(localizer, params, T)

    cm = localizer.channel_model
    summary = dict(
        method="student_t_mahalanobis_plan_posterior (research 5.3), ephys-only localization",
        model_commit=localizer.model_commit,
        channel_model=f"{getattr(cm, 'repo_id', '')}@{getattr(cm, 'revision', '')}",
        fitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        fit_key=fit_key(localizer),
        split_counts=stage["split"]["counts"],
        n_probes_without_signal=stage["n_probes_without_signal"],
        n_grid_voxels=int(localizer.grid.n_voxels),
        features=localizer.feature_names,
        residual=stage["residual_report"],
        prior=dict(params=params["prior"], report=stage["prior_report"]),
        scale_K=float(localizer.score_cfg.scale_K),
        temperature=dict(
            chosen_T=T,
            criterion="lowest mean over validation probes of the per-probe mean channel "
            "distance (µm) to the human alignment",
            n_validation_probes_usable=int(rows["pid"].nunique()),
            table=json.loads(table.to_json(orient="records")),
        ),
        n_restarts=localizer.n_restarts,
        seed=localizer.seed,
        score_config=asdict(localizer.score_cfg),
        anneal_config=asdict(localizer.anneal_cfg),
        timings=dict(
            stage["timings"], temperature_tuning_s=time.time() - t1, total_s=time.time() - t0
        ),
    )
    (localizer.fit_dir / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    report(progress, 1.0, f"Ephys-only fit done (T = {T:g})")
    return summary
