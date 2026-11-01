"""Build the insertion quality-control filter (``ephysatlas.fixtures.pid_qc_filter``).

Runs the stages of :mod:`ephysatlas.qc.pipeline` on every insertion of the feature release: the
measurements against the region references (``measures``), the problems planted in a random sample
of the insertions (``planted``), the z-scores and the five levels (``levels``) and the example
figures (``examples``). Stages are cached in ``~/ephys-atlas/results/qc/<vintage>/``; ``measures``
and ``planted`` take minutes, ``levels`` only reads them and takes seconds.

Needs the release's features only (``ea_active/<vintage>/agg_full``), and Alyx once for the mouse
of each insertion (cached in ``mice.csv``). No model and no auto-alignment.

Usage::

    python scripts/build_pid_qc_filter.py                                # every stage
    python scripts/build_pid_qc_filter.py --stages levels --write-fixture  # from the cached stages
    python scripts/build_pid_qc_filter.py --stages levels --level L3 --write-fixture
"""

import argparse
import json

import pandas as pd

# %% User settings (each can also be given on the command line; see --help) ----------------------
VINTAGE = "2026_W42"
STAGES = ("measures", "planted", "levels", "examples")
LEVEL = "L3"  # the level written to the fixture
WRITE_FIXTURE = False


def main():
    from ephysatlas.qc import levels as lv
    from ephysatlas.qc import pipeline as qp

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vintage", default=VINTAGE)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--level", default=LEVEL, choices=[x.name for x in lv.LEVELS])
    parser.add_argument("--write-fixture", action="store_true", default=WRITE_FIXTURE)
    args = parser.parse_args()
    stages = set(args.stages)

    s = qp.QCSettings(vintage=args.vintage)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[qc] {s.vintage} -> {s.out_dir}")

    if stages & {"measures", "planted", "examples"}:
        from iblatlas.atlas import AllenAtlas

        from ephysatlas.qc.data import QCDataset, insertion_mice
        from ephysatlas.qc.regions import bin_dataset

        (s.out_dir / "settings.json").write_text(json.dumps(s.dump(), indent=1))
        ba = AllenAtlas()
        ds = QCDataset.load(s.vintage)
        print(f"[qc] {len(ds.pids)} insertions")
        bins = bin_dataset(ds)
        reference = qp.build_reference(bins, ba)
    if stages & {"measures", "planted"}:
        mice = insertion_mice(ds.pids, cache=s.out_dir / "mice.csv")
        pairs = qp.mouse_pairs(ds, s, mice)
    if "measures" in stages:
        qp.stage_measures(ds, bins, s, ba, reference, pairs)
    if "planted" in stages:
        qp.stage_planted(ds, bins, s, ba, reference, pairs)

    if "levels" in stages:
        out = qp.stage_levels(s)
        pd.set_option("display.width", 250)
        print(lv.summary(out["table"], out["planted"]).T.to_string())
    if args.write_fixture:
        from ephysatlas.qc.fixture import write_fixture

        table = pd.read_parquet(s.out_dir / "qc_levels.pqt")
        print(f"[qc] wrote {write_fixture(table, s.vintage, level=args.level)}")
    if "examples" in stages:
        qp.stage_examples(ds, bins, s, ba, reference)


if __name__ == "__main__":
    main()
