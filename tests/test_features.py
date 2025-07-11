from pathlib import Path
import unittest

import numpy as np

import neuropixel
import ephysatlas.features

FIXTURE_PATH = (
    Path(ephysatlas.features.__file__).parents[2].joinpath("tests", "fixtures")
)


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
