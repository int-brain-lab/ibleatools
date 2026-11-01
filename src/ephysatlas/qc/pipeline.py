"""The insertion quality-control pipeline, stage by stage, each stage cached in ``out_dir``.

Every stage reads the recordings and their positions (the human alignment) only: no model, no
auto-alignment and no manual label.

1. :func:`stage_measures` -- what the QC measures on every probe (:mod:`.measures`, against the
   region references of :mod:`.regions`; ``measures.csv``, ``scans.npz``);
2. :func:`stage_planted` -- problems planted in a random sample of the insertions
   (:func:`fake_sources`), measured the same way (``planted.csv``), to see what each level catches
   compared with the same insertions as recorded;
3. :func:`stage_levels` -- the z-scores and the labels at the five levels (:mod:`.levels`;
   ``qc_levels.pqt``, ``planted_levels.pqt``, ``calibration_levels.json``,
   ``lists/bad_<level>.csv``);
4. :func:`stage_examples` -- one figure per example insertion (:mod:`.cases`; ``examples/``).

Stages 1 and 2 load every insertion of the release and take minutes; stage 3 only reads their
outputs and takes seconds. :func:`~.fixture.write_fixture` turns the labels at one level into
``ephysatlas.fixtures.pid_qc_filter``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import levels as lv
from .data import QCDataset, default_results_dir
from .measures import dead_fraction, measure_probe, swap_gain
from .regions import (
    BinnedDataset,
    RegionReference,
    bin_probe,
    parcel_labels,
    region_labels,
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
DONOR_MAX_OVERLAP = 0.25  # a donor shares less than this of its Cosmos composition
NOISE_AMPLITUDE = (0.25, 1.5)  # noise SD, in robust SDs of each feature
NOISE_SMOOTH_CHANNELS = 10.0  # depth correlation of the noise (about 100 µm)


@dataclass
class QCSettings:
    vintage: str = "2026_W42"
    out_dir: Optional[Path] = None
    swap_min_distance_um: float = 1000.0  # closer tracks cannot be told apart
    fake_n_sources: int = 150  # insertions drawn at random to plant problems in
    fake_seed: int = 0  # of that draw and of the planted problems
    fake_min_distance_um: float = 2000.0  # a donor probe is at least this far away

    def __post_init__(self):
        self.out_dir = Path(self.out_dir or default_results_dir(self.vintage))

    def dump(self) -> dict:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


def bands_of(ds: QCDataset) -> dict:
    """``{band: feature columns}`` of the band tests."""
    return {
        "lf": ds.component_index("lf"),
        "ap": ds.component_index("ap"),
        "wf": ds.component_index("waveform"),
    }


def build_reference(bins: BinnedDataset, brain_atlas) -> RegionReference:
    """The region references of a release, from every probe at its recorded positions."""
    return RegionReference(bins.z, parcel_labels(bins.xyz, brain_atlas))


def session_pairs(ds: QCDataset, s: QCSettings, eid_of: dict) -> dict:
    """``{probe: [partners]}`` of the probes recorded in one session at least
    ``swap_min_distance_um`` apart (indices in ``ds``)."""
    by_eid = {}
    for i, pid in enumerate(ds.pids):
        if eid_of.get(str(pid)):
            by_eid.setdefault(eid_of[str(pid)], []).append(i)
    pairs = {}
    for probes in by_eid.values():
        for a in probes:
            for b in probes:
                d = ds.human_xyz[a] - ds.human_xyz[b]
                apart = np.nanmedian(np.linalg.norm(d, axis=1)) * 1e6
                if a != b and apart >= s.swap_min_distance_um:
                    pairs.setdefault(a, []).append(b)
    return pairs


def probe_swap_gain(
    bins: BinnedDataset, brain_atlas, reference, probe: int, partners, z=None, xyz=None
) -> float:
    """The largest :func:`~.measures.swap_gain` of a probe over its session's other probes (NaN
    without any); ``z``, ``xyz``: its binned features and positions, by default those of ``bins``."""
    a = (bins.z[probe] if z is None else z, bins.xyz[probe] if xyz is None else xyz)
    gains = [
        swap_gain(reference, brain_atlas, a, (bins.z[b], bins.xyz[b]), (probe, b))
        for b in partners
    ]
    return max(gains) if gains else np.nan


# -- stage 1 -----------------------------------------------------------------------------------
def stage_measures(
    ds: QCDataset,
    bins: BinnedDataset,
    s: QCSettings,
    brain_atlas,
    reference: RegionReference,
    pairs: dict,
) -> pd.DataFrame:
    """What the QC measures on every probe (``measures.csv``; the gain of every move of the scans in
    ``scans.npz``, for the figures)."""
    bands = bands_of(ds)
    rows, depth, sideways = [], [], []
    t0 = time.time()
    for i, pid in enumerate(ds.pids):
        rec, profile = measure_probe(
            reference, brain_atlas, bins.z[i], bins.xyz[i], i, bands
        )
        rec["swap_gain"] = probe_swap_gain(
            bins, brain_atlas, reference, i, pairs.get(i, [])
        )
        rec["dead_fraction"] = dead_fraction(ds.channel_labels[i])
        rows.append({"pid": str(pid), **rec})
        depth.append([g for _, g in profile["depth"]])
        sideways.append([g for _, g in profile["sideways"]])
        if (i + 1) % 200 == 0:
            print(
                f"[measures] {i + 1}/{len(ds.pids)} {time.time() - t0:.0f} s",
                flush=True,
            )
    table = pd.DataFrame(rows).set_index("pid").sort_index()
    table.to_csv(s.out_dir / "measures.csv")
    np.savez(
        s.out_dir / "scans.npz",
        pids=ds.pids.astype(str),
        depth=np.array(depth),
        sideways=np.array(sideways),
    )
    return table


# -- stage 2 -----------------------------------------------------------------------------------
def _cosmos_composition(ds, brain_atlas, pids) -> dict:
    out = {}
    for p in pids:
        lab = region_labels(ds.human_xyz[ds.index(p)], brain_atlas, "Cosmos")
        lab = lab[lab > 0]
        u, n = np.unique(lab, return_counts=True)
        out[p] = dict(zip(u.tolist(), (n / max(n.sum(), 1)).tolist()))
    return out


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
    from .measures import probe_axis
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


def stage_planted(
    ds: QCDataset,
    bins: BinnedDataset,
    s: QCSettings,
    brain_atlas,
    reference: RegionReference,
    pairs: dict,
    pids=None,
) -> pd.DataFrame:
    """Measure the problems planted by :func:`make_fakes` like the real probes (``planted.csv``).
    A planted copy is compared with the other probes, never with its source, and with its
    session's other probe as recorded. ``pids``: the source insertions, by default
    :func:`fake_sources` of every insertion of ``ds``."""
    if pids is None:
        pids = fake_sources(ds.pids, s.fake_n_sources, s.fake_seed)
    fakes = make_fakes(ds, s, brain_atlas, pids, feature_sd(ds), seed=s.fake_seed)
    bands = bands_of(ds)
    rows = []
    t0 = time.time()
    for k, fk in enumerate(fakes, start=1):
        i = fk["probe"]
        values, xyz = bin_probe(fk["features"], fk["labels"], fk["human"])
        z = bins.standardize(values)
        rec, _ = measure_probe(reference, brain_atlas, z, xyz, i, bands)
        rec["swap_gain"] = probe_swap_gain(
            bins, brain_atlas, reference, i, pairs.get(i, []), z, xyz
        )
        rec["dead_fraction"] = dead_fraction(fk["labels"])
        rows.append({**fk["meta"], **rec})
        if k % 200 == 0:
            print(f"[planted] {k}/{len(fakes)} {time.time() - t0:.0f} s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(s.out_dir / "planted.csv", index=False)
    return table


# -- stage 3 -----------------------------------------------------------------------------------
def stage_levels(s: QCSettings) -> dict:
    """The z-scores and the labels at every level (:mod:`.levels`), from the outputs of
    :func:`stage_measures` and :func:`stage_planted`. The z-scores are calibrated on every insertion,
    without any label or split (``calibration_levels.json``), and applied unchanged to the planted
    problems.

    Returns:
        dict: ``table`` (one row per insertion, ``qc_levels.pqt``), ``planted`` (one row per planted
        problem, ``planted_levels.pqt``; None without ``planted.csv``) and ``calibration``. Also
        writes ``lists/bad_<level>.csv``.
    """
    df = pd.read_csv(s.out_dir / "measures.csv", index_col="pid")
    cal = lv.calibrate(df)
    table = lv.level_table(df, cal)
    table.to_parquet(s.out_dir / "qc_levels.pqt")
    planted = None
    if (s.out_dir / "planted.csv").exists():
        planted = lv.level_table(pd.read_csv(s.out_dir / "planted.csv"), cal)
        planted.to_parquet(s.out_dir / "planted_levels.pqt")
    (s.out_dir / "calibration_levels.json").write_text(
        json.dumps(cal, indent=1, default=float)
    )
    (s.out_dir / "lists").mkdir(exist_ok=True)
    for level in lv.LEVELS:
        lv.bad_list(table, level.name).to_csv(
            s.out_dir / "lists" / f"bad_{level.name}.csv"
        )
    return dict(table=table, planted=planted, calibration=cal)


# -- stage 4 -----------------------------------------------------------------------------------
def stage_examples(
    ds: QCDataset,
    bins: BinnedDataset,
    s: QCSettings,
    brain_atlas,
    reference: RegionReference,
    pids=None,
    out_dir: Optional[Path] = None,
) -> dict:
    """One figure per insertion ``pids`` (:func:`~.cases.render_examples`), by default those of
    :func:`~.cases.pick_examples`, in ``out_dir`` (by default ``examples/``)."""
    from .cases import pick_examples, render_examples

    table = pd.read_parquet(s.out_dir / "qc_levels.pqt")
    scans = np.load(s.out_dir / "scans.npz")
    if pids is None:
        pids = pick_examples(table)
    return render_examples(
        pids,
        bins,
        table,
        {k: scans[k] for k in scans.files},
        reference,
        brain_atlas,
        bands_of(ds),
        out_dir or s.out_dir / "examples",
    )
