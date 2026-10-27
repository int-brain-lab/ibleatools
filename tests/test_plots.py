"""Tests of the figure functions of ephysatlas.plots, on small synthetic data."""

import unittest

import matplotlib

matplotlib.use("Agg")  # headless: the figures are only inspected, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iblatlas.atlas import BrainAtlas  # noqa: E402

import ephysatlas.plots  # noqa: E402
from ephysatlas.anatomy import NEW_VOID, ClassifierRegions  # noqa: E402

REGIONS = ClassifierRegions()
REGIONS.add_new_region(NEW_VOID)

# Cosmos ids of the regions used below, and Allen ids that map onto them
ISOCORTEX, HPF, TH, VOID_FLUID = 315, 1089, 549, 2000
VISP, CA1, LGD = 385, 382, 170  # Allen: Isocortex, HPF and TH at the Cosmos level


def _rgb(atlas_id):
    """The colour of a region in [0, 1], as the region bars draw it."""
    return REGIONS.rgb[np.flatnonzero(REGIONS.id == atlas_id)[0]] / 255


def _probe_tables(features, pids=("p0", "p1"), n_channels=96, seed=0):
    """A feature table and its channel table, indexed by (pid, channel), shuffled in depth.

    Along each probe, from the tip up: TH, then HPF, then Isocortex (as Allen ids).
    """
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [list(pids), np.arange(n_channels)], names=["pid", "channel"]
    )
    df = pd.DataFrame(
        rng.normal(size=(len(index), len(features))), index=index, columns=features
    )
    # Channel k sits at a shuffled depth, so the tables are not in depth order
    depths = np.concatenate([rng.permutation(n_channels) * 40.0 for _ in pids])
    atlas_id = np.select([depths < 1280, depths < 2560], [LGD, CA1], default=VISP)
    channels = pd.DataFrame({"axial_um": depths, "atlas_id": atlas_id}, index=index)
    return df, channels


class TestPlotRegionBar(unittest.TestCase):
    def setUp(self):
        # 96 channels: TH at the tip, HPF above, and 4 void_fluid channels at the top
        rng = np.random.default_rng(1)
        self.depths = np.arange(96) * 40.0
        self.ids = np.where(self.depths < 1800, TH, HPF)
        self.ids[-4:] = VOID_FLUID
        shuffle = rng.permutation(96)
        self.depths, self.ids = self.depths[shuffle], self.ids[shuffle]
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_one_span_per_channel_in_region_colour(self):
        ephysatlas.plots.plot_region_bar(
            self.ax, self.depths, self.ids, regions=REGIONS
        )
        self.assertEqual(len(self.ax.patches), 96)
        # The spans are drawn from the tip up, whatever the input order
        np.testing.assert_allclose(self.ax.patches[0].get_facecolor()[:3], _rgb(TH))
        np.testing.assert_allclose(
            self.ax.patches[-1].get_facecolor()[:3], _rgb(VOID_FLUID)
        )

    def test_labels_only_on_tall_runs(self):
        ephysatlas.plots.plot_region_bar(
            self.ax, self.depths, self.ids, regions=REGIONS
        )
        # void_fluid spans 4 of 96 channels, below the default 6 % needed for a label
        self.assertEqual({t.get_text() for t in self.ax.texts}, {"TH", "HPF"})

    def test_hide_labels(self):
        ephysatlas.plots.plot_region_bar(
            self.ax, self.depths, self.ids, regions=REGIONS, show_labels=False
        )
        self.assertEqual(len(self.ax.texts), 0)

    def test_default_regions_know_void_fluid(self):
        ephysatlas.plots.plot_region_bar(self.ax, self.depths, self.ids)
        self.assertEqual(len(self.ax.patches), 96)

    def test_unknown_id_raises(self):
        ids = self.ids.copy()
        ids[0] = 123456789
        with self.assertRaises(ValueError):
            ephysatlas.plots.plot_region_bar(self.ax, self.depths, ids, regions=REGIONS)


class TestPlotFeaturesHeatmap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.features = ["rms_lf", "psd_delta", "rms_ap"]
        self.values = rng.normal(size=(50, 3))
        self.depths = rng.permutation(50) * 20.0
        self.groups = {"LF": ["rms_lf", "psd_delta"], "AP": ["rms_ap"]}
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_rows_sorted_by_depth(self):
        im = ephysatlas.plots.plot_features_heatmap(
            self.values, self.depths, self.features, self.ax
        )
        order = np.argsort(self.depths)
        np.testing.assert_array_equal(np.asarray(im.get_array()), self.values[order])
        self.assertEqual(im.get_clim(), (-2, 2))

    def test_groups_labelled_and_separated(self):
        ephysatlas.plots.plot_features_heatmap(
            self.values, self.depths, self.features, self.ax, feature_groups=self.groups
        )
        self.assertEqual({t.get_text() for t in self.ax.texts}, {"LF", "AP"})
        self.assertEqual(len(self.ax.lines), 1)  # one separator, between the two groups

    def test_hide_groups_and_names(self):
        ephysatlas.plots.plot_features_heatmap(
            self.values,
            self.depths,
            self.features,
            self.ax,
            feature_groups=self.groups,
            show_group_labels=False,
            show_group_separators=False,
            show_feature_names=False,
        )
        self.assertEqual(len(self.ax.texts), 0)
        self.assertEqual(len(self.ax.lines), 0)
        self.assertEqual(len(self.ax.get_xticks()), 0)

    def test_groups_must_match_features(self):
        with self.assertRaises(ValueError):
            ephysatlas.plots.plot_features_heatmap(
                self.values,
                self.depths,
                self.features,
                self.ax,
                feature_groups={"LF": ["rms_lf"], "AP": ["rms_ap"]},
            )


class TestFigureFeaturesHeatmap(unittest.TestCase):
    def setUp(self):
        self.groups = {"LF": ["rms_lf", "psd_delta"], "AP": ["rms_ap", "cor_ratio"]}
        self.features = [f for group in self.groups.values() for f in group]
        self.df, self.channels = _probe_tables(self.features)

    def tearDown(self):
        plt.close("all")

    def _expected(self, table, pid, mean, std):
        """The z-scored features of one probe, sorted by depth."""
        z = ((table.loc[pid, self.features] - mean) / std).to_numpy()
        return z[
            np.argsort(self.channels.loc[pid, "axial_um"].to_numpy(), kind="stable")
        ]

    def test_zscored_over_every_probe(self):
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            self.df, "p0", channels=self.channels, feature_groups=self.groups
        )
        self.assertEqual(set(axs), {"histology", "features", "colorbar"})
        mean, std = self.df[self.features].mean(), self.df[self.features].std()
        np.testing.assert_allclose(
            np.asarray(axs["features"].images[0].get_array()),
            self._expected(self.df, "p0", mean, std),
        )

    def test_two_tables_share_the_first_tables_scale(self):
        df_raw = self.df * 3 + 1
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            {"raw": df_raw, "denoised": self.df},
            "p1",
            channels=self.channels,
            feature_groups=self.groups,
        )
        mean, std = df_raw[self.features].mean(), df_raw[self.features].std()
        np.testing.assert_allclose(
            np.asarray(axs["denoised"].images[0].get_array()),
            self._expected(self.df, "p1", mean, std),
        )
        self.assertEqual(
            axs["raw"].images[0].get_clim(), axs["denoised"].images[0].get_clim()
        )

    def test_hide_histology_and_colorbar(self):
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            self.df,
            "p0",
            channels=self.channels,
            feature_groups=self.groups,
            show_histology=False,
            show_colorbar=False,
        )
        self.assertEqual(set(axs), {"features"})
        self.assertEqual(len(fig.axes), 1)

    def test_feature_name_size(self):
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            self.df,
            "p0",
            channels=self.channels,
            feature_groups=self.groups,
            fontsize=5,
        )
        sizes = {t.get_fontsize() for t in axs["features"].get_xticklabels()}
        self.assertEqual(sizes, {5})

    def test_channels_read_from_the_table(self):
        merged = self.df.join(self.channels)
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            merged, "p0", feature_groups=self.groups
        )
        self.assertIn("histology", axs)

    def test_missing_feature_skipped_with_a_warning(self):
        groups = {"LF": ["rms_lf", "psd_delta", "not_a_feature"], "AP": ["rms_ap"]}
        with self.assertLogs("ephysatlas.plots", level="WARNING"):
            fig, axs = ephysatlas.plots.figure_features_heatmap(
                self.df, "p0", channels=self.channels, feature_groups=groups
            )
        self.assertEqual(axs["features"].images[0].get_array().shape, (96, 3))

    def test_default_groups_from_the_schemas(self):
        fig, axs = ephysatlas.plots.figure_features_heatmap(
            self.df, "p0", channels=self.channels
        )
        # Only the families the table has: LF (rms_lf, psd_delta) and AP (rms_ap, cor_ratio)
        self.assertEqual({t.get_text() for t in axs["features"].texts}, {"LF", "AP"})
        self.assertEqual(axs["features"].images[0].get_array().shape, (96, 4))


class TestFigurePredictionsAlongProbe(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.classes = [0, ISOCORTEX, HPF, VOID_FLUID]
        acronyms = ["void", "Isocortex", "HPF", "void_fluid"]
        probas = rng.dirichlet(np.ones(4), size=64)
        self.out = pd.DataFrame(probas, columns=[f"p_{a}" for a in acronyms])
        self.out.insert(
            0, "predicted_atlas_id", np.asarray(self.classes)[probas.argmax(1)]
        )
        self.depths = rng.permutation(64) * 60.0
        self.histology = {
            "axial_um": np.arange(64) * 60.0,
            "atlas_id": np.full(64, CA1),
        }

    def tearDown(self):
        plt.close("all")

    def test_three_panels(self):
        fig, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out, self.depths, self.classes, histology=self.histology
        )
        self.assertEqual(set(axs), {"histology", "prediction", "probabilities"})
        # The prediction bar starts at the tip with the channel predicted there
        tip = self.out["predicted_atlas_id"].to_numpy()[np.argmin(self.depths)]
        np.testing.assert_allclose(
            axs["prediction"].patches[0].get_facecolor()[:3], _rgb(tip)
        )
        # The histology is remapped to Cosmos: CA1 is drawn as HPF
        np.testing.assert_allclose(
            axs["histology"].patches[0].get_facecolor()[:3], _rgb(HPF)
        )

    def test_panels_can_be_hidden(self):
        fig, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out,
            self.depths,
            self.classes,
            histology=self.histology,
            show_histology=False,
            show_prediction=False,
        )
        self.assertEqual(set(axs), {"probabilities"})
        self.assertEqual(len(fig.axes), 1)

    def test_legend_outside_the_probabilities(self):
        fig, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out, self.depths, self.classes, show_legend=True
        )
        ax = axs["probabilities"]
        legend = ax.get_legend()
        self.assertEqual(
            [t.get_text() for t in legend.get_texts()],
            ["void", "Isocortex", "HPF", "void_fluid"],
        )
        # Anchored right of the panel, in axes coordinates, so it covers no band
        anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        self.assertGreaterEqual(anchor.x0, 1)

    def test_no_legend_by_default(self):
        fig, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out, self.depths, self.classes
        )
        self.assertIsNone(axs["probabilities"].get_legend())

    def test_no_histology_panel_without_histology(self):
        fig, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out, self.depths, self.classes
        )
        self.assertEqual(set(axs), {"prediction", "probabilities"})

    def test_given_axes_are_used(self):
        fig, given = plt.subplots(1, 3)
        _, axs = ephysatlas.plots.figure_predictions_along_probe(
            self.out, self.depths, self.classes, histology=self.histology, axs=given
        )
        self.assertIs(axs["probabilities"], given[2])

    def test_classes_must_match_probability_columns(self):
        with self.assertRaises(ValueError):
            ephysatlas.plots.figure_predictions_along_probe(
                self.out, self.depths, self.classes[:3]
            )


class TestFigureVoltageSnippets(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.ap = rng.normal(0, 10e-6, size=(32, 300))  # volts
        self.lf = rng.normal(0, 50e-6, size=(32, 100))

    def tearDown(self):
        plt.close("all")

    def test_default_range_on_the_typical_channel(self):
        fig, axs = ephysatlas.plots.figure_voltage_snippets(
            {"AP": (self.ap, 30_000), "LF": (self.lf, 2_500)}
        )
        for name, data in [("AP", self.ap), ("LF", self.lf)]:
            vrange = 4 * np.median(np.std(data * 1e6, axis=1))
            np.testing.assert_allclose(
                axs[name].images[0].get_clim(), (-vrange, vrange)
            )
        self.assertEqual(len(fig.axes), 4)  # two panels and their colorbars

    def test_float16_voltage_does_not_overflow(self):
        # Snippets are often stored as float16, whose largest value is 65504: in µV, a
        # 10 µV channel would overflow if scaled before casting
        fig, axs = ephysatlas.plots.figure_voltage_snippets(
            {"AP": (self.ap.astype(np.float16), 30_000)}
        )
        self.assertTrue(np.isfinite(axs["AP"].images[0].get_clim()).all())
        self.assertTrue(np.isfinite(axs["AP"].images[0].get_array()).all())

    def test_fixed_range_and_hidden_parts(self):
        fig, axs = ephysatlas.plots.figure_voltage_snippets(
            {"AP": (self.ap, 30_000)}, vrange=50, show_colorbar=False, show_titles=False
        )
        self.assertEqual(axs["AP"].images[0].get_clim(), (-50, 50))
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(axs["AP"].get_title(), "")


class StubEncoder:
    """Two features that depend only on position, so the expected image is known."""

    features = ["rms_ap", "psd_delta"]

    def predict_xyz(self, xyz):
        return np.c_[xyz[:, 0] * 1e3, xyz[:, 2] * 1e3]


class TestFigureEncoderCoronal(unittest.TestCase):
    def setUp(self):
        # A small atlas: AP x ML x DV voxels of 100 µm, with Isocortex over TH in the middle
        br = REGIONS
        label = np.zeros((10, 12, 8), dtype=np.int16)
        label[:, 2:10, 1:4] = np.flatnonzero(br.id == ISOCORTEX)[0]
        label[:, 2:10, 4:7] = np.flatnonzero(br.id == TH)[0]
        self.ba = BrainAtlas(
            np.ones(label.shape, dtype=np.float32),
            label,
            dxyz=np.array([100e-6, -100e-6, -100e-6]),
            regions=br,
            iorigin=[6, 5, 0],
            dims2xyz=[1, 0, 2],
            xyz2dims=[1, 0, 2],
        )

    def tearDown(self):
        plt.close("all")

    def test_image_holds_the_prediction_inside_the_brain(self):
        fig, axs = ephysatlas.plots.figure_encoder_coronal(
            StubEncoder(), ["psd_delta"], ap_m=0.0, brain_atlas=self.ba, step=1
        )
        image = np.asarray(axs["psd_delta"].images[0].get_array(), dtype=float).T
        label = self.ba.label[self.ba.bc.y2i(0.0)]
        inside = label > 0
        expected = np.broadcast_to(self.ba.bc.zscale * 1e3, label.shape)
        np.testing.assert_allclose(image[inside], expected[inside])
        self.assertTrue(np.isnan(image[~inside]).all())
        self.assertEqual(len(axs["psd_delta"].images), 2)  # the map and the outlines

    def test_hide_boundaries_colorbar_and_titles(self):
        fig, axs = ephysatlas.plots.figure_encoder_coronal(
            StubEncoder(),
            ["rms_ap", "psd_delta"],
            ap_m=0.0,
            brain_atlas=self.ba,
            step=1,
            show_boundaries=False,
            show_colorbar=False,
            show_titles=False,
        )
        self.assertEqual(set(axs), {"rms_ap", "psd_delta"})
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(len(axs["rms_ap"].images), 1)
        self.assertEqual(axs["rms_ap"].get_title(), "")

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            ephysatlas.plots.figure_encoder_coronal(
                StubEncoder(), ["spike_amplitude"], ap_m=0.0, brain_atlas=self.ba
            )


if __name__ == "__main__":
    unittest.main()
