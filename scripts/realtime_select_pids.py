"""Choose the insertions the real-time localization is tested on.

Candidates are the held-out (test split) insertions of the channel model release whose
histology alignment is trustworthy and whose raw data can be replayed by SpikeGLX and Open
Ephys:

- the offline ephys-only localization ran (both feature sets);
- Alyx ``alignment_resolved`` with at least ``MIN_ALIGNMENT_COUNT`` alignments, resolved by one of
  ``RESOLVED_BY`` ("qc": independent alignments agreed), insertion QC not in ``EXCLUDED_QC``;
- a Neuropixels 1.0 probe of ``PROBE_MODELS`` (an LF band, and SpikeGLX can simulate it);
- a micro-manipulator or planned trajectory (so the plan is not a histology track);
- the raw AP and LF ``.cbin`` / ``.ch`` / ``.meta`` datasets.

``N_PIDS`` are drawn from the candidates (``FORCED_PIDS`` first), at most one per session,
stratified in quartiles of the offline 27-feature localization error so the subset spans easy
and hard probes. Writes ``candidates.csv`` (every test insertion, with the criteria) and
``selected.csv`` to ``OUT_DIR``.

Usage::

    python scripts/realtime_select_pids.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.alignment.models import VINTAGE
from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings ------------------------------------------------------------------------------
N_PIDS = 20
MIN_ALIGNMENT_COUNT = 2
RESOLVED_BY = (
    "qc",
)  # add "experimenter" to also accept experimenter-resolved alignments
EXCLUDED_QC = ("CRITICAL",)
PROBE_MODELS = ("3B1", "3B2")
# Already-downloaded test insertions kept in the set (if they pass the criteria).
FORCED_PIDS = (
    "143dd7cf-6a47-47a1-906d-927ad7fe9117",
    "3eb6e6e0-8a57-49d6-b7c9-f39d5834e682",
)
SEED = 0
RESULTS = DEFAULT_RESULTS_DIR / "alignment"
OFFLINE_30 = RESULTS / f"heldout_{VINTAGE}" / "alignment_metrics.csv"
OFFLINE_27 = RESULTS / f"heldout_{VINTAGE}_lf_ap_no_detection" / "alignment_metrics.csv"
OUT_DIR = DEFAULT_RESULTS_DIR / "realtime" / "pids"
RAW_SUFFIXES = ("ap.cbin", "ap.ch", "ap.meta", "lf.cbin", "lf.ch", "lf.meta")
METRICS = ("cosmos_acc", "beryl_acc", "mean_distance_um")


def alyx_info(one, pid: str) -> dict:
    """Alignment, QC, probe and data facts of one insertion."""
    ins = one.alyx.rest("insertions", "read", id=pid)
    j = ins.get("json") or {}
    eq = j.get("extended_qc") or {}
    provenances = {
        t["provenance"]
        for t in one.alyx.rest("trajectories", "list", probe_insertion=pid)
    }
    datasets = one.list_datasets(
        ins["session"], collection=f"raw_ephys_data/{ins['name']}"
    )
    raw_ok = all(any(str(d).endswith(s) for d in datasets) for s in RAW_SUFFIXES)
    info = ins.get("session_info") or {}
    return dict(
        pid=pid,
        eid=ins["session"],
        probe=ins["name"],
        lab=info.get("lab"),
        subject=info.get("subject"),
        date=str(info.get("start_time", ""))[:10],
        model=ins.get("model"),
        insertion_qc=j.get("qc"),
        alignment_resolved=bool(eq.get("alignment_resolved")),
        alignment_resolved_by=eq.get("alignment_resolved_by"),
        alignment_count=eq.get("alignment_count") or 0,
        alignment_qc=eq.get("alignment_qc"),
        micro_manipulator="Micro-manipulator" in provenances,
        planned="Planned" in provenances,
        raw_data=raw_ok,
    )


def cosmos_path(pid: str, brain_atlas) -> str:
    """Cosmos regions along the human alignment, top to tip (consecutive duplicates merged)."""
    from ephysatlas.alignment.data import load_probe
    from ephysatlas.alignment.geometry import region_ids

    ids = region_ids(brain_atlas, load_probe(pid)["human_xyz"], "Cosmos")
    acr = brain_atlas.regions.acronym[ids]
    return "-".join(a for k, a in enumerate(acr) if k == 0 or a != acr[k - 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from iblatlas.atlas import AllenAtlas
    from one.api import ONE

    from ephysatlas.alignment.models import ChannelModel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = sorted(map(str, ChannelModel().split()["test_pids"]))

    cache = out_dir / "alyx_info.csv"
    known = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["pid"])
    todo = [p for p in test if p not in set(known["pid"])]
    if todo:
        one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="remote")
        rows = []
        for i, pid in enumerate(todo, 1):
            print(f"[select] Alyx {i}/{len(todo)} {pid}", flush=True)
            rows.append(alyx_info(one, pid))
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known.to_csv(cache, index=False)
    df = known[known["pid"].isin(test)].copy()

    for label, path in (("30", OFFLINE_30), ("27", OFFLINE_27)):
        res = pd.read_csv(path)
        res = res[(res["method"] == "ephys_only") & (res["error"].fillna("") == "")]
        cols = {m: f"{m}_{label}" for m in METRICS}
        if label == "30":
            cols.update({f"planned_{m}": f"planned_{m}" for m in METRICS})
        df = df.merge(res[["pid", *cols]].rename(columns=cols), on="pid", how="left")

    criteria = {
        "offline_ok": df["mean_distance_um_27"].notna()
        & df["mean_distance_um_30"].notna(),
        "resolved": df["alignment_resolved"].astype(bool),
        "count_ok": df["alignment_count"].astype(float) >= MIN_ALIGNMENT_COUNT,
        "resolved_by_ok": df["alignment_resolved_by"].isin(RESOLVED_BY),
        "qc_ok": ~df["insertion_qc"].isin(EXCLUDED_QC),
        "np1": df["model"].isin(PROBE_MODELS),
        "plan_ok": df["micro_manipulator"].astype(bool) | df["planned"].astype(bool),
        "raw_ok": df["raw_data"].astype(bool),
    }
    for name, ok in criteria.items():
        df[name] = ok
    df["eligible"] = np.logical_and.reduce(list(criteria.values()))
    print("[select] test insertions:", len(df))
    for name in criteria:
        print(f"    failing {name}: {int((~df[name]).sum())}")
    pool = df[df["eligible"]].copy()
    print(f"[select] eligible: {len(pool)}")

    # Quartiles of the offline 27-feature error over the eligible insertions.
    pool["error_quartile"] = pd.qcut(pool["mean_distance_um_27"], 4, labels=False)
    rng = np.random.default_rng(SEED)
    chosen, sessions = [], set()
    for pid in FORCED_PIDS:
        if pid in set(pool["pid"]):
            chosen.append(pid)
            sessions.add(pool.loc[pool["pid"] == pid, "eid"].iloc[0])
        else:
            print(f"[select] forced {pid} is not eligible; left out")
    per_quartile = N_PIDS // 4
    for q in range(4):
        have = pool[pool["pid"].isin(chosen) & (pool["error_quartile"] == q)]
        need = per_quartile - len(have)
        rest = pool[(pool["error_quartile"] == q) & ~pool["pid"].isin(chosen)]
        for idx in rng.permutation(len(rest)):
            if need <= 0:
                break
            row = rest.iloc[idx]
            if row["eid"] in sessions:
                continue
            chosen.append(row["pid"])
            sessions.add(row["eid"])
            need -= 1
    sel = pool[pool["pid"].isin(chosen)].copy()
    brain_atlas = AllenAtlas()
    sel["cosmos_path"] = [cosmos_path(p, brain_atlas) for p in sel["pid"]]
    sel = sel.sort_values(["error_quartile", "mean_distance_um_27"])
    df.to_csv(out_dir / "candidates.csv", index=False)
    sel.to_csv(out_dir / "selected.csv", index=False)

    cols = [
        *(f"{m}_27" for m in METRICS),
        *(f"{m}_30" for m in METRICS),
        *(f"planned_{m}" for m in METRICS),
    ]
    summary = pd.DataFrame(
        {
            "selected": sel[cols].mean(),
            "eligible": pool[cols].mean(),
            "all test": df[cols].mean(),
        }
    )
    print(summary.round(3).to_string())
    print(f"[select] {len(sel)} insertions -> {out_dir / 'selected.csv'}")


if __name__ == "__main__":
    main()
