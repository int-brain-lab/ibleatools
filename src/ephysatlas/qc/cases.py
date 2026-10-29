"""Example insertions of the QC: one figure per insertion, and which insertions to show.

Each figure shows, along the probe, six recorded features against what the nearby probes record
(the local reference of the tests) and against what the other probes record in the same regions
(median and 1 MAD), with the regions at the human and the best-depth positions; every feature,
recorded, as the nearby probes record it and as expected from the regions; the tracks; the z-score
of every failure case against the limits of the levels; and the model-free depth scan.

Everything drawn comes from the recordings and the cached measurements of the pipeline. A
histology auto-alignment, when given, only adds its regions and its track as a diagnostic; no
QC decision uses it.

:func:`render_examples` draws chosen insertions with custom titles; :func:`render_cases` draws the
examples :func:`select_cases` picks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .levels import LABELS, LEVELS, TESTS  # noqa: E402
from .modelfree import SHIFTS_UM, probe_axis  # noqa: E402
from .regions import row_regions, signal_rows  # noqa: E402

# Categorical slots in fixed order (one per QC component), ink and grid greys.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABEL_COLOR = dict(zip(LABELS, SERIES))
LABEL_NAMES = {"alignment": "Alignment", "lf": "LF", "ap": "AP", "waveform": "Waveform"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"]
)
DIVERGING.set_bad("#ffffff")

LINE_FEATURES = {
    "lf": ("rms_lf", "aperiodic_exponent"),
    "ap": ("rms_ap", "spike_count"),
    "waveform": ("spike_width_secs", "peak_to_trough_ratio_log"),
}
TEST_NAMES = {k: v[2] for k, v in TESTS.items()}
LAST = LEVELS[-1].name  # the most sensitive level


def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def feature_norm(reference) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature median and robust spread of all Beryl probe-region values of the reference
    (the scale the expected / recorded images share)."""
    v = np.concatenate(
        [arr for (lvl, _), (_, arr) in reference.groups.items() if lvl == "Beryl"]
    )
    med = np.nanmedian(v, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(v - med), axis=0)
    return med, np.where(mad > 0, mad, np.nanstd(v, axis=0) + 1e-9)


def expected_rows(
    xyz, probe: int, reference, brain_atlas
) -> tuple[np.ndarray, np.ndarray]:
    """``(median [C, F], scale [C, F])`` of the reference of each row's region (Beryl, Cosmos when
    the Beryl region has too few probes; NaN outside the brain), the probe left out."""
    regs = row_regions(xyz, brain_atlas)
    n, f = len(regs["Beryl"]), reference.n_features
    med, scale = np.full((n, f), np.nan), np.full((n, f), np.nan)
    for b, c in set(zip(regs["Beryl"], regs["Cosmos"])):
        if b <= 0:
            continue
        st = reference.stats("Beryl", b, probe) or reference.stats("Cosmos", c, probe)
        if st is None:
            continue
        rows = (regs["Beryl"] == b) & (regs["Cosmos"] == c)
        med[rows], scale[rows] = st[0], st[1]
    return med, scale


def _region_strip(ax, xyz, axial, brain_atlas, title):
    from ephysatlas.alignment.geometry import region_ids

    xyz = np.asarray(xyz, float)
    ok = np.isfinite(xyz).all(axis=1)
    lab = np.zeros(len(xyz), dtype=int)
    lab[ok] = region_ids(brain_atlas, xyz[ok], "Beryl")
    order = np.argsort(axial)
    ax.imshow(
        (brain_atlas.regions.rgb[lab] / 255.0)[order][:, None, :],
        aspect="auto",
        origin="lower",
        extent=[0, 1, axial.min(), axial.max()],
    )
    change = np.r_[0, np.flatnonzero(np.diff(lab[order]) != 0) + 1, len(order)]
    for a, b in zip(change[:-1], change[1:]):
        if b - a >= 12:
            ax.text(
                0.5,
                axial[order][(a + b - 1) // 2],
                brain_atlas.regions.acronym[lab[order][a]],
                ha="center",
                va="center",
                fontsize=6,
                color=INK,
            )
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=8, color=INK)


def _heatmap(ax, values, axial, title, groups):
    order = np.argsort(axial)
    im = ax.imshow(
        values[order],
        aspect="auto",
        origin="lower",
        cmap=DIVERGING,
        vmin=-3,
        vmax=3,
        extent=[-0.5, values.shape[1] - 0.5, axial.min(), axial.max()],
        interpolation="nearest",
    )
    edge = -0.5
    for name, n in groups:
        if edge > -0.5:
            ax.axvline(edge, color=INK, linewidth=0.8)
        ax.text(
            edge + n / 2,
            axial.max(),
            name,
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK2,
        )
        edge += n
    ax.set_xticks([])
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=12)
    ax.tick_params(labelsize=6, colors=INK2)
    return im


def _first_level(row) -> str:
    for lv in LEVELS:
        if bool(row.get(f"{lv.name}_bad", False)):
            return lv.name
    return "none"


def case_figure(
    pid,
    row,
    ds,
    brain_atlas,
    reference,
    norm,
    profile,
    out_path,
    title="",
    local=None,
    level="L3",
    auto_xyz=None,
):
    """Draw one insertion to ``out_path``: ``row``, its row of the :func:`~.levels.level_table`
    (component labels at ``level``); ``profile``, its ``[4, S]`` depth-scan gains (mean of the
    bands, LF, AP, waveform); ``norm``, :func:`feature_norm`; ``local``, the
    :class:`~.modelfree.LocalReference`; ``auto_xyz``, optional, the ``[C, 3]`` channel positions
    of a histology auto-alignment, drawn as a diagnostic only (its regions and its track)."""
    from ephysatlas.alignment.geometry import np1_axial_um_in_row_order

    i = ds.index(pid)
    feats, human = ds.features[i], ds.human_xyz[i]
    axial = np1_axial_um_in_row_order(len(feats))
    ok_rows = signal_rows(feats, ds.channel_labels[i])
    names = ds.feature_names
    comp_idx = {c: ds.component_index(c) for c in ("lf", "ap", "waveform")}
    order_f = np.r_[comp_idx["lf"], comp_idx["ap"], comp_idx["waveform"]]
    groups = [
        ("LF", len(comp_idx["lf"])),
        ("AP", len(comp_idx["ap"])),
        ("WF", len(comp_idx["waveform"])),
    ]
    med_h, sc_h = expected_rows(human, i, reference, brain_atlas)
    loc_h = None
    if local is not None:  # what the nearby probes record (the reference of the tests)
        loc_h = local.reference(human, [i], ok_rows) * local.sd + local.med
    best = row.get("depth_best", np.nan)
    depth_xyz = human + best * 1e-6 * probe_axis(human) if np.isfinite(best) else None

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(
        3,
        10,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.38,
        wspace=0.55,
        top=0.885,
        bottom=0.04,
        left=0.05,
        right=0.98,
    )
    labels = "   ".join(
        f"{LABEL_NAMES[k]}: {row.get(f'{level}_{k}_label', '-')}" for k in LABELS
    )
    head = (
        f"{pid}   {title}\nfirst bad at: {_first_level(row)}   |   at {level}: {labels}"
    )
    fig.text(0.01, 0.995, head, fontsize=10.5, color=INK, va="top")
    failed = row.get(f"{LAST}_failed", "") or ""
    fig.text(
        0.01,
        0.955,
        f"Tests failed at {LAST} (the most sensitive level): "
        + (failed.replace(";", ", ") or "none"),
        fontsize=8.5,
        color=INK2,
        va="top",
    )

    # Row 1: features along the probe against the region reference (median and 1 MAD).
    col = 0
    order = np.argsort(axial)
    for comp, fnames in LINE_FEATURES.items():
        for fname in fnames:
            ax = fig.add_subplot(gs[0, col])
            j = names.index(fname)
            lo, hi = med_h[:, j] - sc_h[:, j], med_h[:, j] + sc_h[:, j]
            ax.fill_betweenx(
                axial[order], lo[order], hi[order], color="#cde2fb", linewidth=0
            )
            ax.plot(
                med_h[order, j], axial[order], color=SERIES[0], linewidth=0.8, alpha=0.8
            )
            if loc_h is not None:
                ax.plot(
                    loc_h[order, j], axial[order], color=INK, linewidth=0.9, ls="--"
                )
            v = np.where(ok_rows, feats[:, j], np.nan)
            ax.plot(v[order], axial[order], color=LABEL_COLOR[comp], linewidth=1.1)
            ax.set_title(
                f"{LABEL_NAMES[comp]}\n{fname}", fontsize=7, color=INK, loc="left"
            )
            ax.set_ylim(axial.min(), axial.max())
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("µm from tip", fontsize=8, color=INK2)
            _style(ax, None)
            ax.tick_params(labelsize=6)
            col += 1
    for name, xyz in (
        ("human", human),
        (f"best depth ({best:+.0f})" if np.isfinite(best) else "best depth", depth_xyz),
        ("auto (diagnostic)", auto_xyz),
    ):
        ax = fig.add_subplot(gs[0, col])
        col += 1
        if xyz is None:
            ax.axis("off")
            continue
        _region_strip(ax, xyz, axial, brain_atlas, name)
        ax.set_ylim(axial.min(), axial.max())

    # Row 2: recorded vs expected, every feature, on one scale.
    nmed, nsc = norm
    rec = np.where(ok_rows[:, None], (feats - nmed) / nsc, np.nan)[:, order_f]
    panels = [("Recorded", rec)]
    if loc_h is not None:
        panels.append(
            (
                "What the nearby probes record (the tests' reference)",
                ((loc_h - nmed) / nsc)[:, order_f],
            )
        )
    panels.append(
        (
            "Expected from the regions, human positions",
            ((med_h - nmed) / nsc)[:, order_f],
        )
    )
    im = None
    for k, (name, values) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 3 * k : 3 * k + 3])
        im = _heatmap(ax, np.clip(values, -3, 3), axial, name, groups)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("µm from tip", fontsize=8, color=INK2)
    cax = fig.add_subplot(gs[1, 9])
    cax.set_position(
        [cax.get_position().x0, cax.get_position().y0, 0.012, cax.get_position().height]
    )
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.set_label(
        "feature value, robust z over all probes and regions", fontsize=7, color=INK2
    )

    # Row 3: tracks, test z-scores, depth scan.
    ax = fig.add_subplot(gs[2, 0:3])
    ml = float(np.nanmean(human[:, 0]))
    brain_atlas.plot_sslice(
        ml, volume="annotation", mapping="Cosmos", ax=ax, alpha=0.55
    )
    for name, xyz, c, ls, lw in (
        ("human", human, SERIES[0], "-", 2.4),
        ("best depth", depth_xyz, SERIES[3], "--", 1.6),
        ("planned", ds.planned_xyz[i], MUTED, ":", 1.4),
        ("auto-alignment (diagnostic)", auto_xyz, SERIES[1], "-", 1.4),
    ):
        if xyz is not None:
            xyz = np.asarray(xyz, float) * 1e6
            ax.plot(xyz[:, 1], xyz[:, 2], ls, color=c, linewidth=lw, label=name)
    ax.legend(
        fontsize=7, loc="lower left", facecolor="white", framealpha=0.9, edgecolor=GRID
    )
    ax.set_title(
        f"Tracks on the sagittal slice at ML {ml * 1e6:.0f} µm",
        fontsize=8,
        color=INK,
        loc="left",
    )
    ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[2, 4:6])
    tests = list(TESTS)
    zs = np.array([row.get(f"z_{t}", np.nan) for t in tests], dtype=float)
    y = np.arange(len(tests))[::-1]
    colors = [
        LABEL_COLOR["lf"]
        if TESTS[t][0].startswith("lf")
        else LABEL_COLOR["ap"]
        if TESTS[t][0] in ("spike", "ap_signal", "wf_signal")
        else SERIES[0]
        for t in tests
    ]
    ax.barh(y, np.nan_to_num(np.clip(zs, -4, 8)), color=colors, height=0.6)
    for yi, v in zip(y, zs):
        ax.text(
            min(np.nan_to_num(v), 8),
            yi,
            f" {v:.1f}" if np.isfinite(v) else " n/a",
            va="center",
            fontsize=7,
            color=INK2,
        )
    limits = sorted({lv.k for lv in LEVELS} | {lv.k2 for lv in LEVELS if lv.k2})[::-1]
    for kx, ls in zip(limits, ("-", "-", "--", ":", ":")):
        ax.axvline(kx, color=MUTED, linestyle=ls, linewidth=0.8)
    ax.set_yticks(y, [t.replace("_", " ") for t in tests], fontsize=7)
    ax.set_title(
        f"Test z-scores (lines: {', '.join(f'{k:g}' for k in limits)} SD); bad at\n"
        + "; ".join(
            f"{lv.name} {lv.m} > {lv.k:g}"
            if lv.k2 is None
            else f"{lv.name} or {lv.m2} > {lv.k2:g}"
            for lv in LEVELS
        ),
        fontsize=7,
        color=INK,
        loc="left",
    )
    ax.set_xlim(-4, 9)
    _style(ax)

    ax = fig.add_subplot(gs[2, 7:10])
    if profile is not None:
        for k, (lab, c) in enumerate(
            zip(
                ["mean of the bands", "LF", "AP", "waveform"],
                [INK] + [LABEL_COLOR[x] for x in ("lf", "ap", "waveform")],
            )
        ):
            ax.plot(
                SHIFTS_UM,
                profile[k],
                "o-" if k == 0 else "-",
                color=c,
                lw=2 if k == 0 else 1.1,
                ms=3,
                label=lab,
            )
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(-200, 200, color=GRID, alpha=0.6, lw=0)
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("probe moved along its axis (µm, + deeper)", fontsize=8, color=INK2)
    ax.set_title(
        "Depth scan: how much more typical of what nearby probes record\n"
        "(above 0 = better than at the human positions)",
        fontsize=8,
        color=INK,
        loc="left",
    )
    _style(ax, "y")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, facecolor="white", pil_kwargs={"quality": 80})
    plt.close(fig)


def select_cases(
    t: pd.DataFrame, n_good=10, n_worst=10, n_each=5, n_level=5, seed=0
) -> dict:
    """Which insertions of the level table ``t`` to draw.

    - ``good``: random insertions bad at no level;
    - ``worst``: the most failed tests at the most sensitive level, then the largest z;
    - ``by_test``: per failure case, among insertions bad at the most sensitive level with that
      test beyond 2 SD, the ``n_each`` largest and the ``n_each`` smallest z;
    - ``by_level``: per level, the ``n_level`` worst insertions it adds (bad there, not at the level
      before), by the number of failed tests and then the largest z.
    """
    rng = np.random.default_rng(seed)
    zcols = [f"z_{k}" for k in TESTS]
    zmax = t[zcols].max(axis=1)
    good = t.index[~t[f"{LAST}_bad"]]
    bad = t[t[f"{LAST}_bad"]].assign(_zmax=zmax)
    worst = bad.sort_values([f"{LAST}_n_fail", "_zmax"], ascending=False).index[
        :n_worst
    ]
    by_test = {}
    for k in TESTS:
        g = bad[bad[f"z_{k}"] > 2].sort_values(f"z_{k}", ascending=False)
        if len(g):
            top = list(g.index[:n_each])
            low = [p for p in g.index[::-1][:n_each] if p not in top][::-1]
            by_test[k] = dict(n=len(g), worst=top, least=low)
    by_level, prev = {}, pd.Series(False, index=t.index)
    for lv in LEVELS:
        new = t[t[f"{lv.name}_bad"] & ~prev].assign(_zmax=zmax)
        by_level[lv.name] = dict(
            n=len(new),
            worst=list(
                new.sort_values([f"{lv.name}_n_fail", "_zmax"], ascending=False).index[
                    :n_level
                ]
            ),
        )
        prev = t[f"{lv.name}_bad"]
    return dict(
        good=sorted(
            rng.choice(good, size=min(n_good, len(good)), replace=False).tolist()
        ),
        worst=list(worst),
        by_test=by_test,
        by_level=by_level,
    )


def _auto_xyz(auto_alignment_dir, pid):
    """``[C, 3]`` channel positions of the histology auto-alignment of ``pid`` in
    ``auto_alignment_dir`` (a diagnostic for the figures only), or None."""
    if auto_alignment_dir is None:
        return None
    from .data import load_auto_alignment

    auto = load_auto_alignment(auto_alignment_dir, pid)
    return None if auto is None else auto.channel_xyz


def render_cases(
    ds,
    brain_atlas,
    reference,
    t,
    profiles: dict,
    out: Path,
    overwrite=False,
    local=None,
    level="L3",
    auto_alignment_dir=None,
) -> dict:
    """Draw every insertion :func:`select_cases` picks from the level table ``t`` once
    (``out/cases/<pid>.jpg``; ``profiles``: ``{pid: [4, S]}`` depth-scan gains, as in
    ``mf_depth.npz``) and write the selection to ``out/cases.json``. ``auto_alignment_dir``,
    optional, adds the auto-aligned regions and track as a diagnostic."""
    sel = select_cases(t)
    pids = set(sel["good"]) | set(sel["worst"])
    for v in sel["by_test"].values():
        pids |= set(v["worst"]) | set(v["least"])
    for v in sel["by_level"].values():
        pids |= set(v["worst"])
    norm = feature_norm(reference)
    for pid in sorted(pids):
        path = out / "cases" / f"{pid}.jpg"
        if path.exists() and not overwrite:
            continue
        row = t.loc[pid]
        title = (
            "good at every level"
            if pid in sel["good"]
            else f"bad at {LAST}: {row.get(f'{LAST}_bad_type', '')}"
        )
        case_figure(
            pid,
            row,
            ds,
            brain_atlas,
            reference,
            norm,
            profiles.get(pid),
            path,
            title,
            local=local,
            level=level,
            auto_xyz=_auto_xyz(auto_alignment_dir, pid),
        )
    (out / "cases.json").write_text(json.dumps(sel, indent=1))
    print(f"[cases] {len(pids)} example figures")
    return sel


_CONTEXTS: dict = {}  # example_context, per out_dir


def example_context(settings=None) -> dict:
    """What :func:`render_examples` draws from, read from the cached stages of the pipeline once
    per ``settings.out_dir`` and kept for the next calls.

    Returns:
        dict: ``settings`` (:class:`~.pipeline.QCSettings`, by default the default one), ``ds``
        (:class:`~.data.QCDataset`), ``brain_atlas`` (the Allen atlas), ``reference`` (the region
        reference, :func:`~.pipeline.stage_regions`), ``local``
        (:func:`~.pipeline.local_reference`), ``norm`` (:func:`feature_norm`) and ``profiles``
        (the depth scans of ``mf_depth.npz``).
    """
    from iblatlas.atlas import AllenAtlas

    from . import pipeline as qp
    from .data import QCDataset

    s = settings or qp.QCSettings()
    key = str(Path(s.out_dir).resolve())
    if key not in _CONTEXTS:
        ds = QCDataset.load(s.vintage)
        ba = AllenAtlas()
        reference = qp.stage_regions(ds, s, ba)["reference"]
        prof = np.load(s.out_dir / "mf_depth.npz")
        _CONTEXTS[key] = dict(
            settings=s,
            ds=ds,
            brain_atlas=ba,
            reference=reference,
            local=qp.local_reference(ds, s),
            norm=feature_norm(reference),
            profiles={k: prof[k] for k in prof.files},
        )
    return _CONTEXTS[key]


def _default_title(row, level: str) -> str:
    if bool(row.get(f"{level}_bad", False)):
        return f"bad at {level}: {row.get(f'{level}_bad_type', '')}"
    if bool(row.get(f"{level}_review", False)):
        return f"to review at {level}"
    return f"good at {level}"


def render_examples(
    pids,
    out_dir,
    titles: Optional[dict] = None,
    level: str = "L3",
    auto_alignment_dir=None,
    settings=None,
    context: Optional[dict] = None,
) -> dict:
    """Draw one figure per insertion of ``pids`` to ``out_dir/<pid>.jpg``, from the cached
    measurements and labels of the pipeline (``qc_levels.pqt``, ``mf_depth.npz``, the region and
    local references); no model and no auto-alignment needed.

    Args:
        pids: Insertions of the release (rows of ``qc_levels.pqt``).
        out_dir: Folder of the figures (created); existing figures are redrawn.
        titles: ``{pid: title}``, the text after the pid in each figure's header; by default
            what is bad at ``level`` (or ``good`` / ``to review``).
        level: The level whose component labels the header shows.
        auto_alignment_dir: Optional histology auto-alignment output folder
            (``<dir>/results/histology/<pid>.npz``), whose regions and track are drawn as a
            diagnostic.
        settings: :class:`~.pipeline.QCSettings` of the cached stages (default: the default).
        context: :func:`example_context`; when omitted it is loaded (about a minute) on the first
            call and reused on the next ones.

    Returns:
        dict: ``{pid: path}`` of the figures drawn.
    """
    ctx = context or example_context(settings)
    table = pd.read_parquet(ctx["settings"].out_dir / "qc_levels.pqt")
    titles = titles or {}
    out_dir = Path(out_dir)
    out, t0 = {}, time.time()
    for pid in [str(p) for p in pids]:
        row = table.loc[pid]
        path = out_dir / f"{pid}.jpg"
        case_figure(
            pid,
            row,
            ctx["ds"],
            ctx["brain_atlas"],
            ctx["reference"],
            ctx["norm"],
            ctx["profiles"].get(pid),
            path,
            titles.get(pid, _default_title(row, level)),
            local=ctx["local"],
            level=level,
            auto_xyz=_auto_xyz(auto_alignment_dir, pid),
        )
        out[pid] = path
    if out:
        el = time.time() - t0
        print(
            f"[examples] {len(out)} figures in {out_dir} ({el / len(out):.1f} s each)"
        )
    return out
