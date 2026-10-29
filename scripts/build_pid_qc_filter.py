"""Build the insertion quality-control filter (``ephysatlas.fixtures.pid_qc_filter``).

Runs the stages of :mod:`ephysatlas.qc.pipeline` on every insertion of the feature release: the
model-free measurements (``signal``, ``regions``, ``mf``), the problems planted in a random sample
of the insertions (``fakes``), the failure cases and the five levels (``levels``) and the example
figures (``examples``). Stages are cached in ``~/ephys-atlas/results/qc/<vintage>/``; the
measurements take hours and resume where they stopped, ``levels`` only reads them and takes
seconds.

Needs the release's features only (``ea_active/<vintage>/agg_full``), and Alyx once for the session
of each insertion (cached in ``sessions.csv``). No model and no auto-alignment: a histology
auto-alignment folder (``--auto-alignment-dir``) only adds the auto-aligned regions to the example
figures, as a diagnostic.

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
STAGES = ("signal", "regions", "mf", "fakes", "levels", "examples")
LEVEL = "L3"  # the level written to the fixture and labelled in the example figures
WRITE_FIXTURE = False


def main():
    from ephysatlas.qc import levels as lv
    from ephysatlas.qc import pipeline as qp

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vintage", default=VINTAGE)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--level", default=LEVEL, choices=[x.name for x in lv.LEVELS])
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N insertions"
    )
    parser.add_argument("--write-fixture", action="store_true", default=WRITE_FIXTURE)
    parser.add_argument(
        "--auto-alignment-dir",
        default=None,
        help="optional: draw these auto-aligned regions in the example figures (diagnostic)",
    )
    args = parser.parse_args()
    stages = set(args.stages)

    s = qp.QCSettings(vintage=args.vintage)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[qc] {s.vintage} -> {s.out_dir}")

    measure = stages & {"signal", "regions", "mf", "fakes", "examples"}
    if measure:
        from iblatlas.atlas import AllenAtlas

        from ephysatlas.qc.data import QCDataset

        if stages & {"signal", "regions", "mf", "fakes"}:
            (s.out_dir / "settings.json").write_text(json.dumps(s.dump(), indent=1))
        ba = AllenAtlas()
        ds = QCDataset.load(s.vintage)
        pids = ds.pids[: args.limit] if args.limit else ds.pids
        print(f"[qc] {len(ds.pids)} insertions")
    if "signal" in stages:
        qp.stage_signal(ds, s)
    if measure - {"signal"}:
        regions = qp.stage_regions(ds, s, ba)
        print(
            f"[qc] round-2 region reference: {len(regions['info']['round2_reference'])} "
            f"probes; 3A LF shift {regions['info']['family_shift_lf']:.2f} "
            f"(split by family: {regions['info']['stratify_by_family']})"
        )
    if "mf" in stages:
        from ephysatlas.qc.data import insertion_sessions

        qp.stage_modelfree(ds, s, ba, regions["reference"], pids)
        sessions = insertion_sessions(ds.pids, cache=s.out_dir / "sessions.csv")
        qp.stage_swap_mf(ds, s, sessions)
    if "fakes" in stages:
        sources = qp.fake_sources(ds.pids, s.fake_n_sources, s.fake_seed)
        qp.stage_fakes(
            ds,
            s,
            ba,
            regions["reference"],
            sources[: args.limit] if args.limit else sources,
        )

    if "levels" in stages:
        out = qp.stage_levels(s)
        pd.set_option("display.width", 250)
        print(lv.summary(out["table"], out["fakes"]).T.to_string())
    if args.write_fixture:
        from ephysatlas.qc.fixture import write_fixture

        table = pd.read_parquet(s.out_dir / "qc_levels.pqt")
        print(f"[qc] wrote {write_fixture(table, s.vintage, level=args.level)}")
    if "examples" in stages:
        qp.stage_examples(
            ds,
            s,
            ba,
            regions["reference"],
            level=args.level,
            auto_alignment_dir=args.auto_alignment_dir,
        )


if __name__ == "__main__":
    main()
