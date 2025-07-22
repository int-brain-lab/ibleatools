import torch
from pathlib import Path
import unittest

import numpy as np

import neuropixel
import ephysatlas.features
import ephysatlas.data

FIXTURE_PATH = (
    Path(ephysatlas.features.__file__).parents[2].joinpath("tests", "fixtures")
)

print(f"Torch number of threads = {torch.get_num_threads()}")


class TestFeatureSets(unittest.TestCase):
    def test_sets(self):
        self.assertEqual(len(ephysatlas.features.voltage_features_set("all")), 34)
        self.assertEqual(len(ephysatlas.features.voltage_features_set(["raw_ap"])), 3)
        self.assertEqual(len(ephysatlas.features.voltage_features_set()), 24)


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
        self.data_ap = np.load(FIXTURE_PATH / "ap_destriped.npy").astype(np.float32)

    def test_ap(self):
        df, waveforms = ephysatlas.features.spikes(
            self.data_ap[:, 10_000:11_000],
            fs=30_000,
            geometry=neuropixel.trace_header(version=1),
            return_waveforms=True,
        )
        self.assertTrue(df.shape[0] == waveforms["df_spikes"]["channel"].nunique())
        self.assertEqual(4, len(waveforms.keys()))


class TestDenoiseFeatures(unittest.TestCase):
    def setUp(self):
        self.df_features = ephysatlas.data.read_features_from_disk(
            FIXTURE_PATH.joinpath("features", "2025_W28"), load_denoised=False
        )

    def test_denoise_features(self):
        pid = self.df_features.index.get_level_values(0).unique()[0]
        df_pid = self.df_features.loc[pid, :]
        df_denoised = ephysatlas.features.denoise_dataframe(df_pid)
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
        np.testing.assert_allclose(
            expected,
            df_denoised.loc[:, ephysatlas.features.voltage_features_set()]
            .mean()
            .values,
            atol=1e-5,
        )
