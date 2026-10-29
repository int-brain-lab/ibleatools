"""The insertion quality-control pipeline, stage by stage, each stage cached in ``out_dir``.

Every stage reads the recordings and their positions (the human alignment) only: no model, no
auto-alignment and no manual label.

1. :func:`stage_signal` -- signal statistics of every probe (:mod:`.signal`; ``signal.pqt``);
2. :func:`stage_regions` -- the region reference at the human positions (:mod:`.regions`), for
   the region band scores and the figures;
3. :func:`stage_modelfree` and :func:`stage_swap_mf` -- the model-free metrics of every probe
   (:mod:`.modelfree`; ``mf.csv``, ``mf_depth.npz``, ``swap_mf.pqt``): how unusual each band is
   for where the probe is placed, the depth scan, the probe swap and the robust signal
   fingerprints;
4. :func:`stage_fakes` -- problems planted in a random sample of the insertions
   (:func:`fake_sources`), scored the same way (``fakes5.csv``), to measure what each failure
   case and each level catch;
5. :func:`stage_levels` -- the failure cases, their z-scores and the labels at the five levels
   (:mod:`.levels`; ``qc_levels.pqt``, ``fakes_levels.pqt``, ``calibration_levels.json``,
   ``lists/bad_<level>.csv``);
6. :func:`stage_examples` -- one figure per example insertion (:mod:`.cases`; ``examples/``).

Stages 1-4 load every insertion of the release and take hours; they resume where they stopped.
Stage 5 only reads their outputs and takes seconds. :func:`~.fixture.write_fixture` turns the
labels at one level into ``ephysatlas.fixtures.pid_qc_filter``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import levels as lv
from .data import QCDataset, default_results_dir
from .regions import (
    RegionReference,
    RegionSettings,
    component_scores,
    probe_region_values,
)

FAKES = (
    "shift_deeper_500",
    "shift_shallower_500",
    "lateral_1mm",
    "lf_swap_full",
    "lf_swap_half",
    "ap_swap_full",
    "ap_swap_half",
    "lf_noise",
    "ap_noise",
)
# "5": apart from the planted problems of earlier QC versions (``fakes4.csv``: version 4, planted
# in the channel model's held-out test probes and scored with the auto-alignment).
FAKES_CSV = "fakes5.csv"
DONOR_MAX_OVERLAP = 0.25  # a donor shares less than this of its Cosmos composition
NOISE_AMPLITUDE = (0.25, 1.5)  # noise SD, in robust SDs of each feature
NOISE_SMOOTH_CHANNELS = 10.0  # depth correlation of the noise (about 100 µm)


@dataclass
class QCSettings:
    vintage: str = "2026_W42"
    out_dir: Optional[Path] = None
    region: RegionSettings = field(default_factory=RegionSettings)
    round1_exclude_fraction: float = (
        0.10  # most unusual probes left out of the round-2 region reference
    )
    family_shift_limit: float = (
        0.5  # 3A vs NP1 LF shift (median |z|) above which families split
    )
    fake_n_sources: int = 150  # insertions drawn at random to plant problems in
    fake_seed: int = 0  # of that draw and of the planted problems
    fake_min_distance_um: float = 2000.0  # a donor probe is at least this far away

    def __post_init__(self):
        self.out_dir = Path(self.out_dir or default_results_dir(self.vintage))

    def dump(self) -> dict:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


def _append(csv: Path, rows: list) -> None:
    """Add rows to ``csv``; rows may carry columns the file does not have yet."""
    new = pd.DataFrame(rows)
    if csv.exists():
        new = pd.concat([pd.read_csv(csv), new], ignore_index=True)
    new.to_csv(csv, index=False)


# -- stage 1 -----------------------------------------------------------------------------------
def stage_signal(ds: QCDataset, s: QCSettings, overwrite: bool = False) -> pd.DataFrame:
    from .signal import signal_table

    path = s.out_dir / "signal.pqt"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    table = signal_table(ds)
    table.to_parquet(path)
    return table


# -- stage 2 -----------------------------------------------------------------------------------
def human_region_values(ds: QCDataset, s: QCSettings, brain_atlas) -> pd.DataFrame:
    path = s.out_dir / "region_values_human.pqt"
    if path.exists():
        return pd.read_parquet(path)
    values = pd.concat(
        [
            probe_region_values(
                ds.features[i],
                ds.human_xyz[i],
                ds.channel_labels[i],
                brain_atlas,
                s.region,
                probe=i,
            )
            for i in range(len(ds.pids))
        ],
        ignore_index=True,
    )
    values.to_parquet(path)
    return values


def stage_regions(
    ds: QCDataset, s: QCSettings, brain_atlas, overwrite: bool = False
) -> dict:
    """The region reference at the human positions, in two rounds: round 1 against every probe,
    round 2 without the ``round1_exclude_fraction`` most unusual probes of round 1. The 3A probes
    are compared with their own family when their LF differs from the NP1 probes' by more than
    ``family_shift_limit`` (median ``|z|`` over the LF features).

    Returns:
        dict: ``reference`` (the :class:`~.regions.RegionReference` of round 2) and ``info``
        (round sizes, family decision; cached in ``region_reference.json``).
    """
    values = human_region_values(ds, s, brain_atlas)
    n = len(ds.pids)
    info_path = s.out_dir / "region_reference.json"
    if info_path.exists() and not overwrite:
        info = json.loads(info_path.read_text())
    else:
        ref1 = RegionReference(values, range(n), ds.feature_names, s.region)
        sc1 = component_scores(ref1.zscores(values), ds.feature_names)
        worst = sc1.filter(regex=r"^(beryl|cosmos)_(lf|ap|waveform)$").max(axis=1)
        cut = worst.quantile(1.0 - s.round1_exclude_fraction)
        keep = sorted(set(range(n)) - set(worst.index[worst > cut]))
        # Do 3A probes differ from NP1 probes in LF? Median signed Cosmos z of the 3A probes.
        z1 = ref1.zscores(values)
        z1 = z1[z1["level"] == "Cosmos"]
        fam = ds.probe_family
        lf = list(ds.component_index("lf"))
        is3a = fam[z1["probe"].to_numpy()] == "3A"
        shift = np.nanmedian(
            z1.loc[is3a, [f"z{i}" for i in lf]].to_numpy(float), axis=0
        )
        family_shift = float(np.nanmedian(np.abs(shift)))
        info = dict(
            n_probes=n,
            round1_cut=float(cut),
            round2_reference=[int(i) for i in keep],
            family_shift_lf=family_shift,
            stratify_by_family=bool(family_shift > s.family_shift_limit),
            family_shift_per_feature={
                ds.feature_names[i]: float(v) for i, v in zip(lf, shift)
            },
        )
        info_path.write_text(json.dumps(info, indent=1))
    family = dict(enumerate(ds.probe_family)) if info["stratify_by_family"] else None
    reference = RegionReference(
        values, info["round2_reference"], ds.feature_names, s.region, family=family
    )
    return dict(reference=reference, info=info)


# -- stage 3 -----------------------------------------------------------------------------------
def feature_sd(ds: QCDataset) -> np.ndarray:
    """``[F]`` robust SD of every feature over the signal channels of all probes."""
    from .regions import signal_rows

    rows = [
        ds.features[i][signal_rows(ds.features[i], ds.channel_labels[i])]
        for i in range(len(ds.pids))
    ]
    x = np.concatenate(rows)
    med = np.nanmedian(x, axis=0)
    sd = 1.4826 * np.nanmedian(np.abs(x - med), axis=0)
    return np.where(sd > 0, sd, np.nanstd(x, axis=0) + 1e-9)


def local_reference(ds: QCDataset, s: QCSettings):
    """The :class:`~.modelfree.LocalReference` of the dataset, its feature noise fitted on every
    probe (cached in ``local_noise.npy``)."""
    from .modelfree import LocalReference

    local = LocalReference(ds, feature_sd(ds))
    path = s.out_dir / "local_noise.npy"
    if path.exists():
        local.noise = np.load(path)
    else:
        np.save(path, local.fit_noise(ds))
    return local


def _mf_metrics(
    ds,
    s,
    brain_atlas,
    reference,
    local,
    features,
    human,
    labels,
    probe,
    fp_stats,
):
    """Model-free metrics of one probe (real or planted) at its positions ``human``: local band
    scores, the local depth scan, region band scores (for the figures) and the robust
    fingerprints."""
    from .modelfree import depth_scan, region_band_scores
    from .regions import BAD_CHANNEL_LABELS
    from .signal import probe_fingerprints, robust_fingerprint_scores

    rec = {
        f"h_lc_{k}": v
        for k, v in local.band_scores(features, human, labels, [probe]).items()
    }
    rb = region_band_scores(
        features, human, labels, brain_atlas, reference, s.region, probe
    )
    rec.update({f"h_rb_{k}": v for k, v in rb.items()})
    summary, profile = depth_scan(local, brain_atlas, features, human, labels, probe)
    rec.update(summary)
    fp = probe_fingerprints(np.asarray(features)[None], np.asarray(labels)[None])
    for comp, v in robust_fingerprint_scores(fp, fp_stats, ds.feature_names).items():
        rec[f"sig_fpr_{comp}"] = float(v[0])
    r2 = ds.feature_names.index("decay_fit_r_squared")
    rec["sig_lf_fit_r2"] = float(fp[0, r2, 1])
    rec["sig_bad_channel_frac"] = float(np.isin(labels, BAD_CHANNEL_LABELS).mean())
    return rec, profile


def stage_modelfree(
    ds: QCDataset, s: QCSettings, brain_atlas, reference, pids=None
) -> pd.DataFrame:
    """Model-free metrics of every probe at its human positions (``mf.csv``, depth-scan profiles
    in ``mf_depth.npz``; resumable)."""
    from .signal import fingerprint_stats, probe_fingerprints

    csv = s.out_dir / "mf.csv"
    done = set(pd.read_csv(csv)["pid"].astype(str)) if csv.exists() else set()
    todo = [p for p in (pids if pids is not None else ds.pids) if str(p) not in done]
    prof_path = s.out_dir / "mf_depth.npz"
    profiles = dict(np.load(prof_path)) if prof_path.exists() else {}
    fp_stats = fingerprint_stats(probe_fingerprints(ds.features, ds.channel_labels))
    local = local_reference(ds, s)
    t0, rows = time.time(), []
    for k, pid in enumerate(todo, start=1):
        i = ds.index(pid)
        try:
            rec, prof = _mf_metrics(
                ds,
                s,
                brain_atlas,
                reference,
                local,
                ds.features[i],
                ds.human_xyz[i],
                ds.channel_labels[i],
                i,
                fp_stats,
            )
            profiles[str(pid)] = np.stack(
                [prof[c] for c in ("all", "lf", "ap", "waveform")]
            )
            rows.append({"pid": str(pid), **rec})
        except Exception as exc:
            rows.append({"pid": str(pid), "mf_error": f"{type(exc).__name__}: {exc}"})
        if k % 50 == 0 or k == len(todo):
            _append(csv, rows)
            rows = []
            np.savez_compressed(prof_path, **profiles)
            el = time.time() - t0
            print(
                f"[mf] {k}/{len(todo)} ({el / k:.1f} s/probe, {el / 60:.0f} min)",
                flush=True,
            )
    return pd.read_csv(csv)


def stage_swap_mf(
    ds: QCDataset,
    s: QCSettings,
    eid_of: dict,
    min_distance_um: float = 1000.0,
) -> pd.DataFrame:
    """Model-free probe-swap gain of every pair of probes of one session at least
    ``min_distance_um`` apart (``swap_mf.pqt``; closer tracks cannot be told apart)."""
    from .modelfree import swap_gain

    local = local_reference(ds, s)
    by_eid = {}
    for pid in ds.pids:
        if eid_of.get(pid):
            by_eid.setdefault(eid_of[pid], []).append(pid)

    def probe(pid):
        i = ds.index(pid)
        return dict(
            features=ds.features[i],
            xyz=ds.human_xyz[i],
            labels=ds.channel_labels[i],
            probe=i,
        )

    rows = []
    for eid, pids in by_eid.items():
        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                pa, pb = probe(pids[a]), probe(pids[b])
                d = float(
                    np.nanmedian(np.linalg.norm(pa["xyz"] - pb["xyz"], axis=1)) * 1e6
                )
                if d < min_distance_um:
                    continue
                rows.append(
                    dict(
                        eid=eid,
                        pid_a=pids[a],
                        pid_b=pids[b],
                        swap_gain_mf=swap_gain(local, pa, pb),
                        distance_um=d,
                    )
                )
    table = pd.DataFrame(rows)
    table.to_parquet(s.out_dir / "swap_mf.pqt")
    return table


# -- stage 4 -----------------------------------------------------------------------------------
def _cosmos_composition(ds, brain_atlas, pids) -> dict:
    from .regions import row_regions

    out = {}
    for p in pids:
        lab = row_regions(ds.human_xyz[ds.index(p)], brain_atlas)["Cosmos"]
        lab = lab[lab > 0]
        u, n = np.unique(lab, return_counts=True)
        out[p] = dict(zip(u.tolist(), (n / max(n.sum(), 1)).tolist()))
    return out


def _smooth_noise(n_rows: int, n_feat: int, rng) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    x = gaussian_filter1d(
        rng.normal(size=(n_rows, n_feat)), NOISE_SMOOTH_CHANNELS, axis=0
    )
    return x / x.std(axis=0, keepdims=True)


def fake_sources(pids, n: int = 150, seed: int = 0) -> list:
    """``n`` insertions drawn at random (without replacement, with ``seed``) from ``pids``, all
    the insertions of the release, sorted: the sources of the planted problems. The draw ignores
    every label, so the sources are a random sample of the release."""
    pids = np.asarray([str(p) for p in pids])
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pids, size=min(int(n), len(pids)), replace=False).tolist())


def make_fakes(
    ds: QCDataset, s: QCSettings, brain_atlas, pids, sd: np.ndarray, seed: int = 0
) -> list:
    """Problems planted in the probes ``pids`` (:func:`fake_sources`), one of each of
    :data:`FAKES` per probe.

    - ``shift_<deeper|shallower>_500``: the human positions moved 500 µm along the probe;
    - ``lateral_1mm``: the positions moved 1 mm laterally (a probe traced on a neighbouring
      track);
    - ``<lf|ap>_swap_<full|half>``: the LF band (or the AP and waveform bands) replaced by those of
      another of the probes ``pids``, on all channels or on the deeper or the shallower half of
      the probe. The donor crosses different brain divisions (it shares less than a quarter of the
      source's Cosmos composition) and is more than ``fake_min_distance_um`` away, so a swap is
      not missed only because the donor looks alike;
    - ``<lf|ap>_noise``: depth-correlated Gaussian noise added to the band, with an SD drawn
      uniformly from ``NOISE_AMPLITUDE`` robust SDs of each feature.
    """
    from .modelfree import probe_axis
    from .regions import signal_rows

    rng = np.random.default_rng(seed)
    comp = _cosmos_composition(ds, brain_atlas, pids)
    centre = {p: np.nanmean(ds.human_xyz[ds.index(p)], axis=0) for p in pids}
    bands = {
        "lf": ds.component_index("lf"),
        "ap": np.r_[ds.component_index("ap"), ds.component_index("waveform")],
    }
    fakes = []
    for pid in pids:
        i = ds.index(pid)
        overlap = {
            q: sum(min(v, comp[q].get(r, 0.0)) for r, v in comp[pid].items())
            for q in pids
            if q != pid
        }
        far = [
            q
            for q in overlap
            if np.linalg.norm(centre[q] - centre[pid]) * 1e6 > s.fake_min_distance_um
        ]
        good = [q for q in far if overlap[q] < DONOR_MAX_OVERLAP] or sorted(
            far, key=overlap.get
        )[:10]
        donor = str(rng.choice(good))
        j = ds.index(donor)
        human = ds.human_xyz[i]
        rows = signal_rows(ds.features[i], ds.channel_labels[i])
        n = len(human)
        half = np.arange(n) < n // 2 if rng.integers(2) else np.arange(n) >= n // 2
        for kind in FAKES:
            f = ds.features[i].copy()
            xyz = human
            labels = ds.channel_labels[i]
            meta = dict(
                fake=kind, source_pid=pid, donor_pid=donor, donor_overlap=overlap[donor]
            )
            if kind.startswith("shift_"):
                sign = 1.0 if "deeper" in kind else -1.0
                xyz = human + sign * probe_axis(human) * 500e-6
            elif kind == "lateral_1mm":
                lateral = np.array([np.sign(np.nanmean(human[:, 0])) or 1.0, 0.0, 0.0])
                xyz = human + lateral * 1e-3
            elif "_swap_" in kind:
                band = kind.split("_")[0]
                sel = np.ones(n, bool) if kind.endswith("full") else half
                cols = bands[band]
                f[np.ix_(sel, cols)] = ds.features[j][np.ix_(sel, cols)]
                if band == "ap":  # the bad-channel labels come from the AP band
                    labels = np.where(sel, ds.channel_labels[j], labels)
                meta["fraction"] = float(sel.mean())
            elif kind.endswith("_noise"):
                cols = bands[kind.split("_")[0]]
                amp = float(rng.uniform(*NOISE_AMPLITUDE))
                noise = _smooth_noise(n, len(cols), rng) * amp * sd[cols]
                f[np.ix_(rows, cols)] += noise[rows]
                meta["noise_sd"] = amp
            fakes.append(dict(probe=i, features=f, human=xyz, labels=labels, meta=meta))
    return fakes


def stage_fakes(
    ds: QCDataset, s: QCSettings, brain_atlas, reference, pids=None
) -> pd.DataFrame:
    """Score the planted problems of :func:`make_fakes` like the real probes (``fakes5.csv``;
    resumable), with ``harm``, how large each problem is (:func:`~.modelfree.band_harm`,
    :func:`~.modelfree.position_harm`).

    Args:
        pids: The source insertions; by default :func:`fake_sources` of every insertion of
            ``ds`` (``fake_n_sources`` of them, drawn with ``fake_seed``).
    """
    from .modelfree import band_harm, position_harm
    from .regions import signal_rows
    from .signal import fingerprint_stats, probe_fingerprints

    if pids is None:
        pids = fake_sources(ds.pids, s.fake_n_sources, s.fake_seed)
    pids = [str(p) for p in pids]
    csv = s.out_dir / FAKES_CSV
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = set(zip(prev["source_pid"].astype(str), prev["fake"]))
    sd = feature_sd(ds)
    fakes = [
        f
        for f in make_fakes(ds, s, brain_atlas, pids, sd, seed=s.fake_seed)
        if (f["meta"]["source_pid"], f["meta"]["fake"]) not in done
    ]
    fp_stats = fingerprint_stats(probe_fingerprints(ds.features, ds.channel_labels))
    local = local_reference(ds, s)
    bands = {
        "lf": ds.component_index("lf"),
        "ap": np.r_[ds.component_index("ap"), ds.component_index("waveform")],
    }
    t0, rows = time.time(), []
    for k, fk in enumerate(fakes, start=1):
        meta = dict(fk["meta"])
        pid, kind, i = meta["source_pid"], meta["fake"], fk["probe"]
        rec = {"pid": pid, **meta}
        try:
            m, _ = _mf_metrics(
                ds,
                s,
                brain_atlas,
                reference,
                local,
                fk["features"],
                fk["human"],
                fk["labels"],
                i,
                fp_stats,
            )
            rec.update(m)
            rows_i = signal_rows(ds.features[i], ds.channel_labels[i])
            if kind.startswith("shift_") or kind == "lateral_1mm":
                rec["harm"] = position_harm(
                    local, ds.human_xyz[i], fk["human"], i, rows_i
                )
            else:
                rec["harm"] = band_harm(
                    ds.features[i],
                    fk["features"],
                    bands[kind.split("_")[0]],
                    sd,
                    rows_i,
                )
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        if k % 50 == 0 or k == len(fakes):
            _append(csv, rows)
            rows = []
            el = time.time() - t0
            print(
                f"[fakes] {k}/{len(fakes)} ({el / k:.1f} s/fake, {el / 60:.0f} min)",
                flush=True,
            )
    return pd.read_csv(csv)


# -- stage 5 -----------------------------------------------------------------------------------
def stage_levels(s: QCSettings) -> dict:
    """The failure cases, their z-scores and the labels at every level (:mod:`.levels`), from the
    outputs of :func:`stage_modelfree`, :func:`stage_swap_mf` and :func:`stage_fakes`.

    The z-scores are calibrated on every insertion, without any label or split
    (``calibration_levels.json``), and applied unchanged to the planted problems. Columns that
    ``mf.csv`` may still carry from QC version 4 (:data:`~.levels.LEGACY_COLUMNS`) are ignored.

    Returns:
        dict: ``table`` (one row per insertion, ``qc_levels.pqt``), ``fakes`` (one row per planted
        problem, ``fakes_levels.pqt``; None without ``fakes5.csv``) and ``calibration``. Also
        writes ``lists/bad_<level>.csv``.
    """
    swap_path = s.out_dir / "swap_mf.pqt"
    df = lv.insertion_table(
        pd.read_csv(s.out_dir / "mf.csv"),
        pd.read_parquet(swap_path) if swap_path.exists() else None,
    )
    cal = lv.calibrate(df)
    table = lv.level_table(df, cal)
    table.to_parquet(s.out_dir / "qc_levels.pqt")
    fakes = None
    if (s.out_dir / FAKES_CSV).exists():
        fakes = lv.level_table(lv.fakes_table(pd.read_csv(s.out_dir / FAKES_CSV)), cal)
        fakes.to_parquet(s.out_dir / "fakes_levels.pqt")
    (s.out_dir / "calibration_levels.json").write_text(
        json.dumps(cal, indent=1, default=float)
    )
    (s.out_dir / "lists").mkdir(exist_ok=True)
    for level in lv.LEVELS:
        lv.bad_list(table, level.name).to_csv(
            s.out_dir / "lists" / f"bad_{level.name}.csv"
        )
    return dict(table=table, fakes=fakes, calibration=cal)


# -- stage 6 -----------------------------------------------------------------------------------
def stage_examples(
    ds: QCDataset,
    s: QCSettings,
    brain_atlas,
    reference,
    level: str = "L3",
    overwrite: bool = False,
    auto_alignment_dir: Optional[Path] = None,
) -> dict:
    """One figure per example insertion of ``qc_levels.pqt`` (:func:`~.cases.render_cases`), in
    ``examples/``. ``auto_alignment_dir``, optional, only adds the auto-aligned regions to the
    figures as a diagnostic."""
    from .cases import render_cases

    table = pd.read_parquet(s.out_dir / "qc_levels.pqt")
    prof = np.load(s.out_dir / "mf_depth.npz")
    return render_cases(
        ds,
        brain_atlas,
        reference,
        table,
        {k: prof[k] for k in prof.files},
        s.out_dir / "examples",
        overwrite=overwrite,
        local=local_reference(ds, s),
        level=level,
        auto_alignment_dir=auto_alignment_dir,
    )
