"""Score replay runs of the online localization against the human alignment.

For every insertion of a run (realtime_replay.py) and every update: Cosmos / Beryl accuracy
and channel distance to the human alignment, as a function of the recording time used. Baselines
per insertion: the planned trajectory, and the same real-time localizer run offline on the
release features (5 snippets over 25 min, ``pid=''`` like the online runs).

The localizer's annealing makes single results seed-dependent (estimates from different seeds
on the same features are ~100 µm apart), so with ``--seeds N`` the last update's features and
the release features are also localized with seeds 0..N-1 and the mean scores reported
(``*_seedmean_*``). Writes, in the run directory, ``update_metrics.csv`` (one row per update),
``final_metrics.csv`` (last update and baselines per insertion) and ``timings.csv``, and prints
the summaries.

Usage::

    python scripts/realtime_evaluate.py speedmax_cadence10
"""

import argparse

import numpy as np
import pandas as pd

from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

# %% User settings ------------------------------------------------------------------------------
REPLAY_DIR = DEFAULT_RESULTS_DIR / "realtime" / "replay"
TRUTH_DIR = DEFAULT_RESULTS_DIR / "realtime" / "pids" / "truth"
FEATURE_SET = "lf_ap_no_detection"
METRICS = ("cosmos_acc", "beryl_acc", "mean_distance_um")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="run name under REPLAY_DIR")
    parser.add_argument(
        "--no-offline", action="store_true", help="skip the offline baseline"
    )
    parser.add_argument(
        "--seeds", type=int, default=1, help="seeds for the seed-averaged scores"
    )
    args = parser.parse_args()

    from iblatlas.atlas import AllenAtlas

    from ephysatlas.alignment import alignment_metrics

    run_dir = REPLAY_DIR / args.run
    pids = sorted(p.name for p in run_dir.iterdir() if (p / "updates.npz").exists())
    ba = AllenAtlas()
    localizer, seeded = None, []
    if not args.no_offline:
        from ephysatlas.alignment import ChannelModel, EphysOnlyLocalizer

        cm = ChannelModel()
        localizer = EphysOnlyLocalizer(cm, ba, feature_set=FEATURE_SET)
        if args.seeds > 1:
            seeded = [
                EphysOnlyLocalizer(cm, ba, feature_set=FEATURE_SET, seed=s)
                for s in range(args.seeds)
            ]

    def seed_mean(features, truth, human) -> dict:
        scores = []
        for loc in seeded:
            res = loc.localize(features, truth["planned_xyz"], pid="")
            scores.append(
                alignment_metrics(human, res.channel_xyz, ba, valid=res.valid)
            )
        return {x: float(np.mean([s[x] for s in scores])) for x in METRICS}

    rows, finals, timings = [], [], []
    for pid in pids:
        d = run_dir / pid
        truth = np.load(TRUTH_DIR / f"{pid}.npz")
        human = truth["human_xyz"]
        upd = pd.read_csv(d / "updates.csv")
        arr = np.load(d / "updates.npz")
        for k, u in upd.iterrows():
            m = alignment_metrics(
                human, arr["channel_xyz"][k], ba, valid=arr["valid"][k]
            )
            rows.append(dict(pid=pid, **u.to_dict(), **{x: m[x] for x in METRICS}))
        plan = alignment_metrics(human, truth["planned_xyz"], ba)
        final = dict(pid=pid, **{f"online_{x}": rows[-1][x] for x in METRICS})
        final.update({f"planned_{x}": plan[x] for x in METRICS})
        if localizer is not None:
            res = localizer.localize(
                truth["release_features"], truth["planned_xyz"], pid=""
            )
            off = alignment_metrics(human, res.channel_xyz, ba, valid=res.valid)
            final.update({f"offline_release_{x}": off[x] for x in METRICS})
        if seeded:
            on = seed_mean(arr["features"][-1], truth, human)
            off = seed_mean(truth["release_features"], truth, human)
            final.update({f"online_seedmean_{x}": on[x] for x in METRICS})
            final.update({f"offline_release_seedmean_{x}": off[x] for x in METRICS})
        finals.append(final)
        snips = pd.read_csv(d / "snippets.csv")
        timings.append(
            dict(
                pid=pid,
                n_snippets=len(snips),
                dropped=int(snips["dropped"].sum()),
                compute_s_mean=snips["compute_s"].mean(),
                latency_s_median=upd["latency_s"].median(),
                latency_s_max=upd["latency_s"].max(),
                localize_s_mean=upd["localize_s"].mean(),
            )
        )
    per_update = pd.DataFrame(rows)
    final = pd.DataFrame(finals)
    timing = pd.DataFrame(timings)
    per_update.to_csv(run_dir / "update_metrics.csv", index=False)
    final.to_csv(run_dir / "final_metrics.csv", index=False)
    timing.to_csv(run_dir / "timings.csv", index=False)

    print(f"[evaluate] {args.run}: {len(pids)} insertions")
    by_time = per_update.groupby("n_snippets")[["acquired_s", *METRICS]].mean()
    print("\nMean over insertions, by snippets used (online):")
    print(by_time.round(3).to_string())
    print("\nLast update vs baselines (mean / median over insertions):")
    cols = [c for c in final.columns if c != "pid"]
    print(final[cols].agg(["mean", "median"]).T.round(3).to_string())
    print("\nTimings:")
    print(timing.drop(columns="pid").agg(["mean", "max"]).T.round(2).to_string())


if __name__ == "__main__":
    main()
