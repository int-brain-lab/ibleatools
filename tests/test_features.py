import torch
from pathlib import Path
import tempfile
import unittest

import numpy as np

import neuropixel
import ibldsp.utils
import ibldsp.waveforms
import ephysatlas.features
import ephysatlas.data

FIXTURE_PATH = (
    Path(ephysatlas.features.__file__).parents[2].joinpath("tests", "fixtures")
)

print(f"Torch number of threads = {torch.get_num_threads()}")


class TestFeatureSets(unittest.TestCase):
    def test_sets(self):
        # 53/36 + spatial_spread_um/slowness_s_per_m (#123) = 55/38
        self.assertEqual(len(ephysatlas.features.voltage_features_set("all")), 55)
        self.assertEqual(len(ephysatlas.features.voltage_features_set(["raw_ap"])), 3)
        self.assertEqual(len(ephysatlas.features.voltage_features_set()), 38)


class TestLFPFeatures(unittest.TestCase):
    def setUp(self):
        self.data_lf = np.load(FIXTURE_PATH / "lf_destriped.npy").astype(np.float32)

    def test_csd(self):
        df = ephysatlas.features.csd(
            self.data_lf, fs=2500, geometry=neuropixel.trace_header(version=1)
        )
        self.assertTrue(df.shape[0] == self.data_lf.shape[0])

    def test_lf(self):
        df = ephysatlas.features.lf(self.data_lf, fs=2500)
        self.assertTrue(df.shape[0] == self.data_lf.shape[0])

    def test_get_psd_decay_features(self):
        """Test PSD decay features extraction using spectral parameterization."""
        import scipy.signal

        # Test with a subset of channels and samples to keep test fast
        test_data = self.data_lf[:10, :5000]  # 10 channels, 5000 samples
        fs = 2500

        # Create required parameters as done in the lf() function
        fscale, period = scipy.signal.periodogram(test_data, fs)
        bands = ephysatlas.features.BANDS

        # Test basic functionality
        df = ephysatlas.features.get_psd_decay_features(
            test_data, fs, fscale, period, bands
        )

        # Check output shape
        self.assertEqual(df.shape[0], test_data.shape[0])

        # Check expected aperiodic component columns are present
        expected_aperiodic_columns = [
            "aperiodic_offset",
            "aperiodic_exponent",
            "decay_fit_error",
            "decay_fit_r_squared",
            "decay_n_peaks",
        ]
        for col in expected_aperiodic_columns:
            self.assertIn(col, df.columns)

        # Check expected residual power columns are present
        expected_residual_columns = [
            "psd_residual_delta",
            "psd_residual_theta",
            "psd_residual_alpha",
            "psd_residual_beta",
            "psd_residual_gamma",
            "psd_residual_lfp",
        ]
        for col in expected_residual_columns:
            self.assertIn(col, df.columns)

        # Check data types and ranges for aperiodic features
        self.assertTrue(df["aperiodic_offset"].dtype in [np.float64, np.float32])
        self.assertTrue(df["aperiodic_exponent"].dtype in [np.float64, np.float32])
        self.assertTrue(df["decay_fit_error"].dtype in [np.float64, np.float32])
        self.assertTrue(df["decay_fit_r_squared"].dtype in [np.float64, np.float32])
        self.assertTrue(df["decay_n_peaks"].dtype in [np.int64, np.int32])

        # Check data types for residual features
        for col in expected_residual_columns:
            self.assertTrue(df[col].dtype in [np.float64, np.float32])

        # Check reasonable value ranges
        self.assertTrue(all(df["decay_fit_r_squared"] >= 0))
        self.assertTrue(all(df["decay_fit_r_squared"] <= 1))
        self.assertTrue(all(df["decay_fit_error"] >= 0))
        self.assertTrue(all(df["decay_n_peaks"] >= 0))
        self.assertTrue(all(df["decay_n_peaks"] <= 4))  # max_n_peaks=4 in function

        # Test with custom parameters
        df_custom = ephysatlas.features.get_psd_decay_features(
            test_data, fs, fscale, period, bands, nperseg=256, PSD_range=[1, 50]
        )
        self.assertEqual(df_custom.shape[0], test_data.shape[0])
        # Should have 5 aperiodic + 6 residual = 11 columns total
        self.assertEqual(df_custom.shape[1], 11)


class TestAPFeatures(unittest.TestCase):
    def setUp(self):
        self.data_ap = np.load(FIXTURE_PATH / "ap_destriped.npy").astype(np.float32)

    def test_ap(self):
        df = ephysatlas.features.ap(
            self.data_ap[:, 10_000:11_000],
            geometry=neuropixel.trace_header(version=1),
            channel_labels=np.ones(self.data_ap.shape[0]),
        )
        self.assertTrue(df.shape[0] == self.data_ap.shape[0])


class TestWaveformFeatures(unittest.TestCase):
    def setUp(self):
        """Load a short, eight-channel slice for the DARTsort CPU smoke test."""
        self.data_ap = np.load(FIXTURE_PATH / "ap_destriped.npy").astype(np.float32)[
            :8, :30_000
        ]
        trace_header = neuropixel.trace_header(version=1)
        self.geometry = {
            "x": np.asarray(trace_header["x"])[:8],
            "y": np.asarray(trace_header["y"])[:8],
        }

    def test_ap(self):
        """Run the upgraded DARTsort stack on one second of eight-channel data."""
        with tempfile.TemporaryDirectory(prefix="dartsort_smoke_") as scratch_dir:
            df, waveforms = ephysatlas.features.spikes(
                self.data_ap,
                fs=30_000,
                geometry=self.geometry,
                return_waveforms=True,
                scratch_dir=scratch_dir,
                chunk_length_samples=5_000,
                n_jobs=0,
            )
        self.assertTrue(df.shape[0] == waveforms["df_spikes"]["channel"].nunique())
        self.assertEqual(5, len(waveforms.keys()))
        self.assertEqual(waveforms["raw"].shape[1], 121)
        self.assertEqual(waveforms["denoised"].shape[1], 121)
        # Exact count for this fixture is 81, so to catch any deviations
        self.assertTrue(60 < df["spike_count"].sum() < 110)

        # Amplitudes must be un-z-scored back to Volts: on the order of the
        # fixture's real per-channel AP-band RMS, not O(1)-O(10) z-score units.
        rms_channel = ibldsp.utils.rms(self.data_ap, axis=-1)
        rms_scale = np.median(rms_channel)
        ptp = waveforms["df_spikes"]["ptp"]
        self.assertTrue((ptp.abs() > 0.1 * rms_scale).all())
        self.assertTrue((ptp.abs() < 100 * rms_scale).all())
        self.assertTrue((df["peak_val"].abs().median() < 100 * rms_scale))
        self.assertTrue((df["trough_val"].abs().median() < 100 * rms_scale))


class TestRemapWaveformShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.df_features = ephysatlas.data.read_features_from_disk(
            FIXTURE_PATH.joinpath("features", "2025_W28"), load_denoised=False
        )

    def test_columns_and_index(self):
        df_shape = ephysatlas.features.remap_waveform_shape_features(self.df_features)
        self.assertEqual(
            set(df_shape.columns),
            set(ephysatlas.features.ModelSpikeShapeFeatures.to_schema().columns.keys()),
        )
        np.testing.assert_array_equal(df_shape.index, self.df_features.index)
        # dropped columns must not leak through
        raw_cols = set(
            ephysatlas.features.ModelSpikeFeatures.to_schema().columns.keys()
        )
        shape_cols = set(
            ephysatlas.features.ModelSpikeShapeFeatures.to_schema().columns.keys()
        )
        for col in raw_cols - shape_cols:
            self.assertNotIn(col, df_shape.columns)

    def test_derived_values(self):
        df_shape = ephysatlas.features.remap_waveform_shape_features(self.df_features)
        np.testing.assert_allclose(
            df_shape["spike_width_secs"],
            self.df_features["trough_time_secs"] - self.df_features["peak_time_secs"],
        )
        np.testing.assert_allclose(
            df_shape["predepolarisation_width_secs"],
            self.df_features["peak_time_secs"] - self.df_features["tip_time_secs"],
        )
        np.testing.assert_allclose(
            df_shape["spike_amplitude"],
            self.df_features["trough_val"] - self.df_features["peak_val"],
        )
        np.testing.assert_allclose(
            df_shape["peak_to_trough_ratio_log"],
            np.log(
                np.abs(self.df_features["peak_val"] / self.df_features["trough_val"])
            ),
        )
        # pass-through columns are untouched
        for col in [
            "tip_val",
            "alpha_mean",
            "alpha_std",
            "polarity",
            "spike_count",
            "depolarisation_slope",
            "repolarisation_slope",
            "recovery_slope",
        ]:
            np.testing.assert_array_equal(df_shape[col], self.df_features[col])

    def test_missing_column_raises(self):
        df_missing = self.df_features.drop(columns=["trough_val"])
        with self.assertRaises(KeyError):
            ephysatlas.features.remap_waveform_shape_features(df_missing)

    def test_volume_matches_dataframe(self):
        """The volume entry point must reproduce the dataframe one exactly."""
        raw_cols = list(
            ephysatlas.features.ModelSpikeFeatures.to_schema().columns.keys()
        )
        # slowness_s_per_m/spatial_spread_um (#123, nullable) postdate this fixture:
        # add them as NaN rather than re-recording real waveform-derived data for it.
        # They pass through both the dataframe and volume remap paths unchanged either
        # way, so also exclude them from the "must have real data" dropna gate below.
        nullable_cols = {"slowness_s_per_m", "spatial_spread_um"}
        df_source = self.df_features.assign(
            **{c: np.nan for c in nullable_cols if c not in self.df_features.columns}
        )
        df = df_source[raw_cols].dropna(
            subset=[c for c in raw_cols if c not in nullable_cols]
        )
        n = len(df) - (len(df) % 2)  # even, so it reshapes cleanly into (2, n // 2)
        df = df.iloc[:n]
        # Shuffle the feature order to exercise name-based (not positional) lookup,
        # and use a non-trivial volume shape (nx, ny) rather than a flat vector.
        shuffled_cols = raw_cols[::-1]
        vol = np.stack([df[c].to_numpy() for c in shuffled_cols], axis=-1).reshape(
            2, n // 2, len(shuffled_cols)
        )
        feature_names = np.array(shuffled_cols)

        remapped_vol, remapped_names = (
            ephysatlas.features.remap_waveform_shape_features_volume(vol, feature_names)
        )
        df_shape = ephysatlas.features.remap_waveform_shape_features(df)

        expected_names = list(
            ephysatlas.features.ModelSpikeShapeFeatures.to_schema().columns.keys()
        )
        self.assertEqual(remapped_vol.shape, (2, n // 2, len(expected_names)))
        self.assertEqual(remapped_vol.dtype, np.float32)
        np.testing.assert_array_equal(remapped_names, expected_names)
        for i, name in enumerate(remapped_names):
            np.testing.assert_allclose(
                remapped_vol[..., i].reshape(-1),
                df_shape[name].to_numpy(),
                rtol=1e-4,
                atol=1e-6,
            )

    def test_volume_missing_feature_raises(self):
        raw_cols = [
            c
            for c in ephysatlas.features.ModelSpikeFeatures.to_schema().columns.keys()
            if c != "trough_val"
        ]
        vol = np.zeros((2, 2, 2, len(raw_cols)), dtype=np.float32)
        with self.assertRaises(KeyError):
            ephysatlas.features.remap_waveform_shape_features_volume(vol, raw_cols)


class TestTransformDenoiseFeatures(unittest.TestCase):
    def setUp(self):
        self.df_features = ephysatlas.data.read_features_from_disk(
            FIXTURE_PATH.joinpath("features", "2025_W28"), load_denoised=False
        )

    def test_transform_features(self):
        et = ephysatlas.features.EphysTransformer()
        et.fit(self.df_features)
        df_orig = self.df_features.copy()
        dft = et.transform(df_orig)
        np.testing.assert_array_equal(
            self.df_features["spike_count"], df_orig["spike_count"]
        )
        self.assertTrue(
            np.any(
                np.not_equal(
                    self.df_features["spike_count"].to_numpy(),
                    dft["spike_count"].to_numpy(),
                )
            )
        )

    def test_denoise_features(self):
        pid = self.df_features.index.get_level_values(0).unique()[0]
        df_pid = self.df_features.loc[pid, :]
        dfcopy = df_pid.copy()
        df_denoised = ephysatlas.features.denoise_dataframe(df_pid, fac=1)
        expected = np.array(
            [
                5.20833333e-03,
                4.26839660e-01,
                -9.63778296e01,
                -1.01903195e02,
                -1.07077043e02,
                -9.46285888e01,
                -1.09517576e02,
                -1.05195293e02,
                -9.80637846e01,
                -8.40091202e01,
                8.65057920e01,
                3.65743265e01,
                -1.72646966e04,
                3.42320737e-06,
                -3.98955222e00,
                -7.70363475e-01,
                -1.93382630e03,
                5.94736050e-04,
                1.37698524e04,
                5.67310266e00,
                -4.22213179e-04,
                5.62216732e-01,
                4.28069032e-04,
                1.35432377e00,
            ]
        )
        np.testing.assert_array_equal(self.df_features.loc[pid, :], dfcopy)
        columns_set = [
            "channel_labels",
            "cor_ratio",
            "rms_ap",
            "psd_alpha",
            "psd_beta",
            "psd_delta",
            "psd_gamma",
            "psd_lfp",
            "psd_theta",
            "rms_lf",
            "alpha_mean",
            "alpha_std",
            "depolarisation_slope",
            "peak_time_secs",
            "peak_val",
            "polarity",
            "recovery_slope",
            "recovery_time_secs",
            "repolarisation_slope",
            "spike_count",
            "tip_time_secs",
            "tip_val",
            "trough_time_secs",
            "trough_val",
        ]
        np.testing.assert_allclose(
            expected,
            df_denoised.loc[:, columns_set].mean().values,
            atol=1e-5,
        )
