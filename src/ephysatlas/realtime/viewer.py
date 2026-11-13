"""A live window for the online localization (PyQt5 + pyqtgraph).

The window runs an :class:`~ephysatlas.realtime.online.OnlinePipeline` in a background thread
and redraws at every update:

- **Features** -- the localizer's features along the probe (top first), standardised with the
  release statistics, as the median over the snippets so far;
- **Regions** -- Cosmos regions along the channels: planned trajectory, current estimate, and the
  reference alignment when one is given (test recordings);
- **Trajectory** -- planned and estimated channel positions on tilted coronal and sagittal slices
  of the Allen atlas (the planes containing the probe), cut through the current estimate or
  through the plan (a toggle above the views), with the earlier estimates fading;
- **Convergence** -- distance of the estimate from the plan (and from the reference, if given)
  and the update latency, against acquisition time.

Without a plan (whole-brain search) there is no plan to draw or compare with: the slices are cut
through the estimate, the convergence panel shows how far each estimate moved from the previous
one, and every update prints the search's candidates (score, position, regions) on the console.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np


def _region_rgb(brain_atlas, xyz_m: np.ndarray, mapping: str = "Cosmos") -> tuple:
    """``([C, 3] uint8 colours, [C] acronyms)`` of the regions at ``xyz_m``."""
    from ephysatlas.alignment.geometry import region_ids

    idx = region_ids(
        brain_atlas, np.nan_to_num(np.asarray(xyz_m, dtype=float)), mapping
    )
    return brain_atlas.regions.rgb[idx].astype(np.uint8), brain_atlas.regions.acronym[
        idx
    ]


class LocalizationViewer:
    """The live window.

    Args:
        make_pipeline: ``make_pipeline(on_update) -> OnlinePipeline``.
        localizer: The pipeline's :class:`~ephysatlas.alignment.EphysOnlyLocalizer` (feature
            names and release statistics for the feature panel).
        planned_xyz: ``[C, 3]`` planned channel positions (m, row order), or None (whole-brain
            search).
        brain_atlas: ``iblatlas.atlas.AllenAtlas``.
        reference_xyz: Optional ``[C, 3]`` reference (histology) alignment, drawn for comparison.
        title: Window title.
        n_channels: Channels of the probe (only needed without a plan).
    """

    def __init__(
        self,
        make_pipeline: Callable,
        localizer,
        planned_xyz: Optional[np.ndarray],
        brain_atlas,
        reference_xyz: Optional[np.ndarray] = None,
        title: str = "Ephys-only localization",
        n_channels: int = 384,
    ):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.pg, self.QtCore, self.QtWidgets = pg, QtCore, QtWidgets
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(
            antialias=True, background="w", foreground="k", imageAxisOrder="row-major"
        )
        self.make_pipeline = make_pipeline
        self.localizer = localizer
        self.ba = brain_atlas
        self.planned = (
            None if planned_xyz is None else np.asarray(planned_xyz, dtype=float)
        )
        self.n_channels = len(self.planned) if self.planned is not None else int(n_channels)
        self.reference = (
            None if reference_xyz is None else np.asarray(reference_xyz, dtype=float)
        )
        # (acquired_s, latency_s, distance to the plan -- or to the previous estimate without
        # one --, distance to the reference)
        self.history = []

        class _Bridge(QtCore.QObject):
            update = QtCore.pyqtSignal(object)
            finished = QtCore.pyqtSignal()

        self.bridge = _Bridge()
        self.bridge.update.connect(self._on_update)
        self.bridge.finished.connect(self._on_finished)
        self._build(title)

    # -- layout -----------------------------------------------------------------------------

    def _build(self, title: str) -> None:
        pg, QtWidgets = self.pg, self.QtWidgets
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(title)
        self.win.resize(1750, 850)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        bar = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Waiting for the first snippet ...")
        self.status.setStyleSheet("font-size: 13px; padding: 4px;")
        bar.addWidget(self.status, stretch=1)
        bar.addWidget(QtWidgets.QLabel("Slices through:"))
        self.slice_choice = QtWidgets.QComboBox()
        self.slice_choice.addItems(
            ["current estimate", "plan"] if self.planned is not None else ["current estimate"]
        )
        self.slice_choice.currentIndexChanged.connect(lambda _: self._draw_slices())
        bar.addWidget(self.slice_choice)
        bar.addWidget(QtWidgets.QLabel("  Slice regions:"))
        self.mapping_choice = QtWidgets.QComboBox()
        self.mapping_choice.addItems(["Cosmos", "Beryl", "Allen"])
        self.mapping_choice.currentIndexChanged.connect(lambda _: self._draw_slices())
        bar.addWidget(self.mapping_choice)
        layout.addLayout(bar)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)
        self.win.setCentralWidget(central)
        n_c = self.n_channels

        # Features along the probe: depth on y (top of the probe at the top).
        self.p_feat = glw.addPlot(
            row=0, col=0, rowspan=2, title="Features (z-score, median of snippets)"
        )
        self.img_feat = pg.ImageItem()
        self.p_feat.addItem(self.img_feat)
        self.img_feat.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(nPts=256)
        )
        self.img_feat.setLevels((-3, 3))
        names = list(self.localizer.feature_names)
        abbrev = {
            "aperiodic_offset": "ap_off",
            "aperiodic_exponent": "ap_exp",
            "decay_fit_error": "fit_err",
            "decay_fit_r_squared": "fit_r2",
            "decay_n_peaks": "n_peaks",
        }
        short = [
            abbrev.get(
                n,
                n.replace("psd_residual_", "res_")
                .replace("psd_", "")
                .replace("_csd_diff1", "_csd"),
            )
            for n in names
        ]
        self.p_feat.getAxis("bottom").setTicks(
            [[(k + 0.5, s) for k, s in enumerate(short)][::3]]
        )
        self.p_feat.setLabel("left", "channel row (top = 0)")
        self.p_feat.invertY(True)
        self.p_feat.setXRange(0, len(names), padding=0)

        # Region strips.
        self.p_reg = glw.addPlot(row=0, col=1, rowspan=2, title="Cosmos regions")
        self.p_reg.invertY(True)
        self.p_reg.setYLink(self.p_feat)
        self.p_reg.hideAxis("left")
        labels = (
            (["plan"] if self.planned is not None else [])
            + ["estimate"]
            + (["reference"] if self.reference is not None else [])
        )
        self.p_reg.getAxis("bottom").setTicks(
            [[(k + 0.5, s) for k, s in enumerate(labels)]]
        )
        self.img_reg = pg.ImageItem()
        self.p_reg.addItem(self.img_reg)
        self.p_reg.setXRange(0, len(labels))
        # Linking the depth axes resets them: fix both to the channel rows afterwards.
        for plot in (self.p_feat, self.p_reg):
            plot.vb.disableAutoRange()
            plot.setYRange(0, n_c, padding=0)
        self._reg_text = []
        glw.ci.layout.setColumnStretchFactor(0, 3)
        glw.ci.layout.setColumnStretchFactor(1, 1)

        # Trajectory on a coronal slice.
        # Trajectory: tilted coronal and sagittal slices of the Allen atlas through the probe
        # (the planes containing the trajectory, see _draw_slices), with the 3-D positions
        # projected: ML-DV on the coronal view, AP-DV on the sagittal one.
        self.views = {}
        for col, (name, axis, dim, label) in enumerate(
            (
                ("coronal", 1, 0, "ML (mm)"),
                ("sagittal", 0, 1, "AP (mm, anterior to the right)"),
            )
        ):
            plot = glw.addPlot(row=0, col=2 + col, title=f"Tilted {name} slice")
            plot.setAspectLocked(True)
            plot.setLabel("bottom", label)
            plot.setLabel("left", "DV (mm)")
            img = pg.ImageItem()
            img.setOpacity(0.6)
            plot.addItem(img)
            if self.planned is not None:
                plot.plot(
                    self.planned[:, dim] * 1e3,
                    self.planned[:, 2] * 1e3,
                    pen=pg.mkPen("k", width=2, style=self.QtCore.Qt.DashLine),
                )
            if self.reference is not None:
                plot.plot(
                    self.reference[:, dim] * 1e3,
                    self.reference[:, 2] * 1e3,
                    pen=pg.mkPen((0, 140, 0), width=2),
                )
            current = plot.plot([], [], pen=pg.mkPen((220, 30, 30), width=3))
            # Centred on the plan; without one on the reference, or on the first estimate.
            around = self.planned if self.planned is not None else self.reference
            if around is not None:
                c = np.nanmean(around[:, [dim, 2]], axis=0) * 1e3
                plot.setXRange(c[0] - 3, c[0] + 3)
                plot.setYRange(c[1] - 3, c[1] + 3)
            self.views[name] = dict(
                plot=plot, img=img, axis=axis, dim=dim, current=current, old=[]
            )
        self._estimate = None
        self._draw_slices()

        # Convergence.
        self.p_conv = glw.addPlot(row=1, col=2, colspan=2, title="Convergence")
        self.p_conv.setLabel("bottom", "acquisition time (s)")
        self.p_conv.setLabel("left", "mean channel distance (µm)")
        self.p_conv.addLegend(offset=(5, 5))
        self.c_plan = self.p_conv.plot(
            [],
            [],
            pen=pg.mkPen((220, 30, 30), width=2),
            symbol="o",
            symbolSize=5,
            name="estimate vs plan"
            if self.planned is not None
            else "estimate vs the previous one",
        )
        self.c_ref = None
        if self.reference is not None:
            self.c_ref = self.p_conv.plot(
                [],
                [],
                pen=pg.mkPen((0, 140, 0), width=2),
                symbol="o",
                symbolSize=5,
                name="estimate vs reference",
            )
            if self.planned is not None:
                plan_err = (
                    np.nanmean(np.linalg.norm(self.planned - self.reference, axis=1))
                    * 1e6
                )
                self.p_conv.addItem(
                    pg.InfiniteLine(
                        plan_err,
                        angle=0,
                        pen=pg.mkPen((0, 140, 0), style=self.QtCore.Qt.DotLine),
                        label="plan vs reference",
                        labelOpts=dict(position=0.9, color=(0, 140, 0)),
                    )
                )
        glw.ci.layout.setColumnStretchFactor(2, 2)
        glw.ci.layout.setColumnStretchFactor(3, 2)
        self._draw_regions(None)

    # -- updates ----------------------------------------------------------------------------

    def _draw_slices(self) -> None:
        """Atlas slices through the current estimate (once there is one) or the plan."""
        through_estimate = self._estimate is not None and (
            self.slice_choice.currentIndex() == 0 or self.planned is None
        )
        xyz = self._estimate if through_estimate else self.planned
        if xyz is None:  # no plan, no estimate yet
            for name, v in self.views.items():
                v["plot"].setTitle(f"Tilted {name} slice (waiting for the first estimate)")
            return
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        mapping = self.mapping_choice.currentText()
        for name, v in self.views.items():
            labels, width, height, _ = self.ba.tilted_slice(
                xyz, v["axis"], volume=self.ba.label
            )
            rgb = self.ba._label2rgb(self.ba._get_mapping(mapping=mapping)[labels])
            img = rgb[
                ::-1
            ]  # rows run top-down in the atlas; pyqtgraph draws row 0 at the bottom
            if width[0] > width[1]:
                img = img[:, ::-1]
            (w0, w1), (h0, h1) = sorted(width * 1e3), sorted(height * 1e3)
            v["img"].setImage(
                np.ascontiguousarray(img), autoLevels=False, levels=(0, 255)
            )
            v["img"].setRect(w0, h0, w1 - w0, h1 - h0)
            v["plot"].setTitle(
                f"Tilted {name} slice through the {'estimate' if through_estimate else 'plan'}"
            )

    def _draw_regions(self, estimate_xyz) -> None:
        n_c = self.n_channels
        cols = ([self.planned] if self.planned is not None else []) + [
            estimate_xyz if estimate_xyz is not None else np.full((n_c, 3), np.nan)
        ]
        if self.reference is not None:
            cols.append(self.reference)
        strips, acronyms = [], []
        for xyz in cols:
            if np.isfinite(xyz).all():
                rgb, acr = _region_rgb(self.ba, xyz)
            else:
                rgb, acr = (
                    np.full((n_c, 3), 230, np.uint8),
                    np.array([""] * n_c),
                )
            strips.append(rgb)
            acronyms.append(acr)
        self.img_reg.setImage(
            np.stack(strips, axis=1), autoLevels=False, levels=(0, 255)
        )
        for t in self._reg_text:
            self.p_reg.removeItem(t)
        self._reg_text = []
        for k, acr in enumerate(acronyms):
            starts = (
                [0]
                + [i for i in range(1, len(acr)) if acr[i] != acr[i - 1]]
                + [len(acr)]
            )
            for a, b in zip(starts[:-1], starts[1:]):
                if acr[a] and b - a >= 15:
                    text = self.pg.TextItem(str(acr[a]), color="k", anchor=(0.5, 0.5))
                    text.setPos(k + 0.5, 0.5 * (a + b))
                    self.p_reg.addItem(text)
                    self._reg_text.append(text)

    def _on_update(self, u) -> None:
        loc = self.localizer
        st = loc._stats
        x = np.asarray(u.features, dtype=float)[:, loc.feature_idx]
        z = (
            np.clip(x, st["rec_ephys_low_pctl"], st["rec_ephys_high_pctl"])
            - st["e_mean"]
        ) / np.maximum(st["e_std"], 1e-6)
        self.img_feat.setImage(z, autoLevels=False)
        self.img_feat.setRect(0, 0, z.shape[1], z.shape[0])
        est = np.asarray(u.result.channel_xyz, dtype=float)
        self._draw_regions(est)
        first_without_plan = self._estimate is None and self.planned is None
        for v in self.views.values():
            if first_without_plan and self.reference is None:
                c = np.nanmean(est[:, [v["dim"], 2]], axis=0) * 1e3
                v["plot"].setXRange(c[0] - 3, c[0] + 3)
                v["plot"].setYRange(c[1] - 3, c[1] + 3)
            for i, item in enumerate(v["old"]):
                item.setOpacity(max(0.05, 0.5 * 0.7 ** (len(v["old"]) - i)))
            v["old"].append(
                v["plot"].plot(
                    est[:, v["dim"]] * 1e3,
                    est[:, 2] * 1e3,
                    pen=self.pg.mkPen((220, 30, 30), width=1),
                )
            )
            v["current"].setData(est[:, v["dim"]] * 1e3, est[:, 2] * 1e3)
        previous = self._estimate
        self._estimate = est
        if self.slice_choice.currentIndex() == 0 or self.planned is None:
            self._draw_slices()
        # Distance to the plan, or without one to the previous estimate (NaN at the first).
        versus = self.planned if self.planned is not None else previous
        d_plan = (
            float(np.nanmean(np.linalg.norm(est - versus, axis=1)) * 1e6)
            if versus is not None
            else np.nan
        )
        d_ref = (
            float(np.nanmean(np.linalg.norm(est - self.reference, axis=1)) * 1e6)
            if self.reference is not None
            else np.nan
        )
        self.history.append((u.acquired_s, u.latency_s, d_plan, d_ref))
        h = np.array(self.history)
        ok = np.isfinite(h[:, 2])
        self.c_plan.setData(h[ok, 0], h[ok, 2])
        if self.c_ref is not None:
            self.c_ref.setData(h[:, 0], h[:, 3])
        ref = (
            f" | distance to reference {d_ref:.0f} µm"
            if self.reference is not None
            else ""
        )
        if self.planned is not None:
            versus_text = f"estimate {d_plan:.0f} µm from the plan"
        else:
            dg = u.result.diagnostics
            versus_text = (
                f"no plan: {dg['n_scanned']:,} trajectories scanned, best of "
                f"{len(dg['seed_score'])} candidates #{dg['selected_seed'] + 1} "
                f"(score {dg['score']:.3f})"
                + (f", moved {d_plan:.0f} µm" if np.isfinite(d_plan) else "")
            )
            self._print_candidates(u)
        self.status.setText(
            f"Update {u.index + 1}: {u.n_snippets} snippets, data up to {u.acquired_s:.0f} s | "
            f"latency {u.latency_s:.1f} s (localization {u.localize_s:.1f} s) | "
            f"{versus_text}{ref}"
        )

    def _print_candidates(self, u) -> None:
        """The whole-brain search's candidates of an update, on the console."""
        dg = u.result.diagnostics
        print(
            f"[viewer] update {u.index + 1} ({u.n_snippets} snippets): whole-brain search, "
            f"{dg['n_scanned']:,} straight trajectories scanned in {dg['scan_s']:.1f} s; "
            f"{len(dg['seed_score'])} candidates at least 1 mm apart, refined:",
            flush=True,
        )
        for k, (xyz, s0, s1) in enumerate(
            zip(dg["seed_xyz"], dg["seed_initial_score"], dg["seed_score"])
        ):
            _, acr = _region_rgb(self.ba, xyz)
            path = [a for i, a in enumerate(acr) if a and (i == 0 or a != acr[i - 1])]
            c = np.mean(xyz, axis=0) * 1e3
            print(
                f"    #{k + 1}  score {s1:.3f} (straight start {s0:.3f})  "
                f"ML {c[0]:.1f} AP {c[1]:.1f} DV {c[2]:.1f} mm  {' > '.join(path[:7])}"
                + ("   <- best" if k == dg["selected_seed"] else ""),
                flush=True,
            )

    def _on_finished(self) -> None:
        self.status.setText(self.status.text() + "  —  recording finished")
        if self.snapshot is not None:
            self.app.processEvents()
            self.win.grab().save(str(self.snapshot))
            self.app.quit()

    # -- run --------------------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        try:
            self.pipeline = self.make_pipeline(self.bridge.update.emit)
            self.pipeline.run()
        finally:
            self.bridge.finished.emit()

    def run(self, snapshot=None) -> None:
        """Show the window, start the pipeline thread and run the Qt event loop.

        Args:
            snapshot: If given, save the window as an image there when the recording ends and
                close (e.g. for figures, or with ``QT_QPA_PLATFORM=offscreen``).
        """
        self.snapshot = snapshot
        self.win.show()
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.app.aboutToQuit.connect(self._stop_pipeline)
        thread.start()
        self.app.exec_()
        thread.join(timeout=90)  # a live run finishes the snippets in progress

    def _stop_pipeline(self) -> None:
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.stop()
