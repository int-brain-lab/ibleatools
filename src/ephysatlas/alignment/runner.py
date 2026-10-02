"""Run the alignment methods over many insertions, offline, with resumable per-probe results.

For each insertion and method this writes one row of scores (against the human alignment) to
``<out_dir>/alignment_metrics.csv``, the full result to ``<out_dir>/results/<method>/<pid>.npz``
and, optionally, the result figure to ``<out_dir>/figures/<method>/<pid>.png``. Rows already in
the CSV are skipped unless ``overwrite``, so an interrupted run resumes where it stopped.

Used by ``scripts/run_alignment_heldout.py`` and ``scripts/run_alignment_dataset.py``.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import alignment_metrics
from .models import CHANNEL_MODEL_REPO_ID, UNIT_MODEL_REPO_ID, VINTAGE
from .progress import print_progress

METHODS = ("histology", "histology_unit", "ephys_only")
# Histology traces: the reconstructed track (Alyx picks; what an experimenter has) or the human
# alignment's channel positions (the setting of the paper's Figure 5, optimistic: the trace is
# built from the reference it is scored against).
TRACES = ("picks", "human")


@dataclass
class RunSettings:
    """Settings of an offline alignment run.

    Attributes:
        out_dir: Output directory.
        methods: Any of :data:`METHODS`.
        trace: Histology trace for the histology methods, one of :data:`TRACES`.
        vintage: Feature release and model tag.
        channel_repo_id, unit_repo_id: Released models (Hub repo ids or local release dirs).
        save_results: Write each result's ``.npz``.
        save_figures: Draw each result's figure (slower).
        overwrite: Recompute rows already in the CSV.
        limit: Process at most this many insertions (for a quick check).
        ephys_only_kwargs: Keyword arguments for :func:`.ephys_only.localize_ephys_only`.
    """

    out_dir: Path
    methods: Sequence[str] = ("histology", "ephys_only")
    trace: str = "picks"
    vintage: str = VINTAGE
    channel_repo_id: str = CHANNEL_MODEL_REPO_ID
    unit_repo_id: str = UNIT_MODEL_REPO_ID
    save_results: bool = True
    save_figures: bool = False
    overwrite: bool = False
    limit: Optional[int] = None
    ephys_only_kwargs: dict = field(default_factory=dict)


def _done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["pid"].astype(str), df["method"].astype(str)))


def _append(csv_path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~((old["pid"].astype(str) == row["pid"]) & (old["method"] == row["method"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def run_alignments(pids: Sequence[str], settings: RunSettings, split_of: Optional[dict] = None) -> Path:
    """Align every insertion of ``pids`` with every method of ``settings``.

    Args:
        pids: Insertion ids (must be in the feature release).
        settings: :class:`RunSettings`.
        split_of: Optional pid -> split name, recorded in the CSV.

    Returns:
        Path: The metrics CSV.
    """
    from iblatlas.atlas import AllenAtlas

    from .data import ChannelDataset, HistologyTraces
    from .histology import align_histology
    from .models import ChannelModel, UnitModel

    unknown = [m for m in settings.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {METHODS}")
    if settings.trace not in TRACES:
        raise ValueError(f"trace must be one of {TRACES}")
    out_dir = Path(settings.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "alignment_metrics.csv"
    done = set() if settings.overwrite else _done(csv_path)

    print(f"[alignment] loading models {settings.channel_repo_id}@{settings.vintage} ...")
    channel_model = ChannelModel(settings.channel_repo_id, settings.vintage)
    unit_model = None
    if "histology_unit" in settings.methods:
        print(f"[alignment] loading {settings.unit_repo_id}@{settings.vintage} and its unit data ...")
        unit_model = UnitModel.load(settings.unit_repo_id, settings.vintage)
    dataset = ChannelDataset.load(settings.vintage)
    brain_atlas = AllenAtlas()
    traces = HistologyTraces()
    if "ephys_only" in settings.methods:
        from .ephys_only import EphysOnlyLocalizer

        localizer = EphysOnlyLocalizer(channel_model, brain_atlas, **settings.ephys_only_kwargs)

    pids = [str(p) for p in pids]
    if settings.limit:
        pids = pids[: int(settings.limit)]
    n = len(pids)
    for i, pid in enumerate(pids, start=1):
        todo = [m for m in settings.methods if (pid, m) not in done]
        if not todo:
            continue
        print(f"[alignment] {i}/{n} {pid}: {', '.join(todo)}", flush=True)
        probe = dataset.probe(pid)
        base = dict(pid=pid, split=(split_of or {}).get(pid, ""), trace=settings.trace,
                    model_commit=channel_model.model_commit)
        planned = alignment_metrics(probe["human_xyz"], probe["planned_xyz"], brain_atlas)
        base.update({f"planned_{k}": v for k, v in planned.items()})
        channel_result = None
        for method in todo:
            row = dict(base, method=method)
            t0 = time.time()
            try:
                if method in ("histology", "histology_unit"):
                    if settings.trace == "picks":
                        trace = traces.trace(pid, brain_atlas)
                        if trace is None:
                            raise RuntimeError("no histology picks for this insertion")
                    else:
                        from .geometry import extend_trace_to_brain

                        trace = extend_trace_to_brain(probe["human_xyz"], brain_atlas)
                    if channel_result is None:
                        channel_result = align_histology(
                            channel_model, probe["features"], trace, pid=pid
                        )
                    if method == "histology":
                        result = channel_result
                    else:
                        from .unit_cost import align_histology_with_units

                        _, result = align_histology_with_units(
                            channel_model, unit_model, probe["features"], trace, pid=pid,
                            channel_result=channel_result,
                        )
                        if result is None:
                            raise RuntimeError("no unit evidence on this probe")
                else:
                    result = localizer.localize(
                        probe["features"], probe["planned_xyz"], pid=pid,
                        progress=print_progress("    "),
                    )
                row.update(alignment_metrics(
                    probe["human_xyz"], result.channel_xyz, brain_atlas,
                    p_good=result.p_good, valid=result.valid,
                ))
                row.update({k: v for k, v in result.diagnostics.items()
                            if isinstance(v, (int, float, str, bool, np.floating, np.integer))})
                row["error"] = ""
                if settings.save_results:
                    result.save(out_dir / "results" / method / f"{pid}.npz")
                if settings.save_figures:
                    import matplotlib.pyplot as plt

                    from .plotting import plot_alignment_result

                    fig = plot_alignment_result(
                        result, brain_atlas, human_xyz=probe["human_xyz"],
                        planned_xyz=probe["planned_xyz"],
                    )
                    fig_path = out_dir / "figures" / method / f"{pid}.png"
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(fig_path, dpi=110)
                    plt.close(fig)
            except Exception as exc:  # noqa: BLE001 - one bad probe must not stop the run
                row["error"] = repr(exc)
                print(f"[alignment] {pid} {method} failed: {exc}")
                traceback.print_exc()
            row["seconds"] = time.time() - t0
            _append(csv_path, row)
            print(
                f"    {method}: cosmos={row.get('cosmos_acc', np.nan):.3f} "
                f"beryl={row.get('beryl_acc', np.nan):.3f} "
                f"distance={row.get('mean_distance_um', np.nan):.0f} um "
                f"({row['seconds']:.1f} s)", flush=True,
            )
    return csv_path


def summarize(csv_path: Path) -> pd.DataFrame:
    """Mean and median of the main scores per method (successful rows only)."""
    df = pd.read_csv(csv_path)
    df = df[df["error"].fillna("") == ""]
    cols = [c for c in ("cosmos_acc", "beryl_acc", "mean_distance_um", "cosmos_acc_high_conf",
                        "beryl_acc_high_conf", "mean_distance_um_high_conf", "high_conf_fraction")
            if c in df.columns]
    return df.groupby("method")[cols].agg(["mean", "median", "count"])
