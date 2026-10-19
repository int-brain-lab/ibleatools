"""Tests for ephysatlas.alignment helpers that need neither the released model nor the atlas.

``ChannelModel.predict_std`` is checked against a fake encoder: it must run the model through the
release's public ``predict(df)`` (with the probe's pid on the index, so its own channels are
excluded from the neighbours) and invert the release's ``X * std + mean`` exactly.
"""

import unittest

import numpy as np
import pandas as pd

from ephysatlas.alignment.models import ChannelModel
from ephysatlas.alignment.plotting import trace_depth_mm, values_on_trace


class _FakeEncoder:
    """Stands in for SpatialEncoder: ``predict(df)`` returns ``pred_<f>`` columns, feature units."""

    inputs = {"columns": ["x", "y", "z"]}

    def __init__(self, features, mean, std):
        self.features = features
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.calls = []

    def standardized(self, xyz):
        return np.column_stack([xyz.sum(axis=1) * (k + 1) for k in range(len(self.features))])

    def predict(self, df, batch_size=1024):
        self.calls.append(df)
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        out = self.standardized(xyz) * self.std + self.mean
        return pd.DataFrame(out, index=df.index, columns=[f"pred_{f}" for f in self.features])


class TestChannelModelPredict(unittest.TestCase):
    def setUp(self):
        self.features = ["rms_lf", "rms_ap", "spike_width_secs"]
        self.mean = np.array([1.0, -2.0, 3e-4])
        self.std = np.array([0.5, 4.0, 1e-4])
        self.model = object.__new__(ChannelModel)  # no Hub download: set what predict needs
        self.model.encoder = _FakeEncoder(self.features, self.mean, self.std)
        self.model.features = list(self.features)
        self.model.e_mean, self.model.e_std = self.mean, self.std
        self.xyz = np.random.default_rng(0).uniform(-5e-3, 0, size=(7, 3)).astype(np.float32)

    def test_predict_std_goes_through_public_predict(self):
        out = self.model.predict_std(self.xyz, "probe-1")
        np.testing.assert_allclose(
            out, self.model.encoder.standardized(self.xyz.astype(np.float64)), rtol=1e-6, atol=1e-9)
        df = self.model.encoder.calls[-1]
        self.assertEqual(list(df.index.names), ["pid", "channel"])
        self.assertTrue((df.index.get_level_values("pid") == "probe-1").all())
        np.testing.assert_array_equal(df[["x", "y", "z"]].to_numpy(), self.xyz)

    def test_predict_returns_feature_units_in_release_order(self):
        out = self.model.predict(self.xyz)
        expected = self.model.encoder.standardized(self.xyz.astype(np.float64)) * self.std + self.mean
        np.testing.assert_allclose(out, expected, rtol=1e-6)


class TestTraceStripes(unittest.TestCase):
    def test_values_land_on_their_aligned_samples(self):
        # 4 channels on a 10-sample trace: two share sample 3, sample 5 is skipped by the warp.
        values = np.array([1.0, 3.0, 4.0, 6.0])
        j = np.array([3, 3, 4, 6])
        valid = np.array([True, True, True, True])
        out = values_on_trace(values, j, valid, 10)
        self.assertTrue(np.isnan(out[:3]).all())
        self.assertTrue(np.isnan(out[7:]).all())
        self.assertEqual(out[3], 2.0)  # mean of the two channels on sample 3
        self.assertEqual(out[4], 4.0)
        self.assertEqual(out[6], 6.0)
        self.assertIn(out[5], (4.0, 6.0))  # skipped sample takes its nearest aligned neighbour

    def test_invalid_channels_are_not_drawn(self):
        out = values_on_trace(np.array([1.0, 2.0]), np.array([0, 1]), np.array([True, False]), 4)
        self.assertEqual(out[0], 1.0)
        self.assertTrue(np.isnan(out[1:]).all())

    def test_trace_depth_is_arc_length_in_mm(self):
        trace = np.column_stack([np.zeros(11), np.zeros(11), -np.arange(11) * 1e-5])  # 10 µm steps
        np.testing.assert_allclose(trace_depth_mm(trace), np.arange(11) * 0.01)


if __name__ == "__main__":
    unittest.main()
