import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.cells import (
    compute_burstiness_and_memory,
    compute_log_acg,
    get_neighbours_members,
    select_good_units_relaxed_rp,
    spike_triggered_population_coupling,
    spike_triggered_population_coupling_df,
)

try:
    from ephysatlas.cells import (
        ACG3D_N_LOG_BINS,
        ACG3D_NUM_FIRING_RATE_QUANTILES,
        compute_3d_acgs,
    )

    _HAS_ACG3D_DEPS = True
except ImportError:
    _HAS_ACG3D_DEPS = False


class TestComputeBurstinessAndMemory(unittest.TestCase):
    def test_too_few_spikes_returns_nan(self):
        for n in range(6):
            b, m = compute_burstiness_and_memory(np.arange(n, dtype=float))
            self.assertTrue(np.isnan(b) and np.isnan(m))

    def test_regular_train(self):
        """Constant ISIs: B = -1, M = NaN (zero std → Pearson undefined)."""
        spikes = np.arange(20, dtype=float)
        b, m = compute_burstiness_and_memory(spikes)
        self.assertAlmostEqual(b, -1.0)
        self.assertTrue(np.isnan(m))

    def test_output_range(self):
        """B and M are in [-1, 1] for a Poisson-like spike train."""
        rng = np.random.default_rng(0)
        spikes = np.sort(rng.uniform(0, 100, 200))
        b, m = compute_burstiness_and_memory(spikes)
        self.assertGreaterEqual(b, -1.0)
        self.assertLessEqual(b, 1.0)
        self.assertGreaterEqual(m, -1.0)
        self.assertLessEqual(m, 1.0)

    def test_bursty_train_burstiness_positive(self):
        """Short bursts separated by long silences → B > 0."""
        bursts = np.concatenate([np.linspace(i, i + 0.05, 6) for i in range(5)])
        b, _ = compute_burstiness_and_memory(np.sort(bursts))
        self.assertGreater(b, 0)


class TestComputeLogAcg(unittest.TestCase):
    FS = 30_000  # synthetic sampling rate, Hz

    def _poisson_spikes(self, seed=0, duration=60.0, rate=20.0):
        rng = np.random.default_rng(seed)
        return np.sort(rng.uniform(0, duration, int(duration * rate)))

    def test_too_few_spikes_returns_zeros(self):
        acg, t_log = compute_log_acg(np.array([0.5]), fs=self.FS)
        np.testing.assert_array_equal(acg, 0.0)
        self.assertGreater(t_log.size, 0)

    def test_empty_spike_train_returns_zeros(self):
        acg, t_log = compute_log_acg(np.array([]), fs=self.FS)
        np.testing.assert_array_equal(acg, 0.0)

    def test_output_shapes_consistent(self):
        acg, t_log = compute_log_acg(self._poisson_spikes(), fs=self.FS)
        self.assertEqual(acg.shape, t_log.shape)

    def test_n_log_bins_exact(self):
        """n_log_bins controls the exact output length."""
        for n in (64, 256, 512):
            acg, t_log = compute_log_acg(
                self._poisson_spikes(), fs=self.FS, n_log_bins=n
            )
            self.assertEqual(acg.size, n)
            self.assertEqual(t_log.size, n)

    def test_t_log_monotone_and_above_trim(self):
        _, t_log = compute_log_acg(self._poisson_spikes(), fs=self.FS)
        self.assertTrue(np.all(np.diff(t_log) > 0), "t_log must be strictly increasing")
        self.assertGreaterEqual(t_log[0], 1e-3)

    def test_acg_nonnegative(self):
        acg, _ = compute_log_acg(self._poisson_spikes(), fs=self.FS)
        self.assertTrue(np.all(acg >= 0))

    def test_custom_log_trim(self):
        trim = 2e-3
        _, t_log = compute_log_acg(self._poisson_spikes(), fs=self.FS, log_trim=trim)
        self.assertGreaterEqual(t_log[0], trim)

    def test_poisson_acg_nonzero(self):
        """A sufficiently long Poisson train should produce a non-trivial ACG."""
        acg, _ = compute_log_acg(self._poisson_spikes(duration=60.0), fs=self.FS)
        self.assertGreater(acg.sum(), 0)

    def test_multi_cluster_shape(self):
        """spike_clusters path returns 2-D array with one row per unique cluster."""
        rng = np.random.default_rng(3)
        n_spikes = 600
        spike_times = np.sort(rng.uniform(0, 30, n_spikes))
        spike_clusters = rng.integers(0, 3, n_spikes)
        acg, t_log = compute_log_acg(
            spike_times, fs=self.FS, spike_clusters=spike_clusters
        )
        self.assertEqual(acg.ndim, 2)
        self.assertEqual(acg.shape[0], 3)
        self.assertEqual(acg.shape[1], t_log.size)

    def test_multi_cluster_matches_single(self):
        """Each row of the multi-cluster output equals the single-cluster call."""
        rng = np.random.default_rng(5)
        spike_times = np.sort(rng.uniform(0, 60, 1000))
        spike_clusters = np.zeros(1000, dtype=int)
        spike_clusters[500:] = 1
        acg_multi, t_log = compute_log_acg(
            spike_times, self.FS, spike_clusters=spike_clusters
        )
        for i, cid in enumerate([0, 1]):
            acg_single, _ = compute_log_acg(spike_times[spike_clusters == cid], self.FS)
            np.testing.assert_array_equal(acg_multi[i], acg_single)


@unittest.skipUnless(_HAS_ACG3D_DEPS, "requires ibleatools[full] (spikeinterface)")
class TestCompute3DAcgs(unittest.TestCase):
    """Wiring smoke test: spikeinterface (linear ACG) -> npyx (log resampling)."""

    FS = 30_000  # synthetic AP-band sampling rate, Hz
    N_BINS = 2 * ACG3D_N_LOG_BINS + 1

    def _dummy_spike_train(self, seed=0, duration=60.0, rate=20.0, refractory=0.003):
        """Poisson-ish spike train (as cumulative ISIs) with an enforced refractory period."""
        rng = np.random.default_rng(seed)
        isis = rng.exponential(1 / rate, int(duration * rate * 2))
        isis = isis[isis > refractory][: int(duration * rate)]
        return np.cumsum(isis)

    def test_output_shape(self):
        spike_times = self._dummy_spike_train()
        spike_clusters = np.zeros(spike_times.size, dtype=int)
        acgs_3d, t_log = compute_3d_acgs(
            spike_times, spike_clusters, np.array([0]), self.FS
        )
        self.assertEqual(
            acgs_3d.shape, (1, ACG3D_NUM_FIRING_RATE_QUANTILES, self.N_BINS)
        )
        self.assertEqual(t_log.shape, (self.N_BINS,))

    def test_t_log_symmetric_and_monotone(self):
        spike_times = self._dummy_spike_train()
        spike_clusters = np.zeros(spike_times.size, dtype=int)
        _, t_log = compute_3d_acgs(spike_times, spike_clusters, np.array([0]), self.FS)
        self.assertTrue(np.all(np.diff(t_log) > 0), "t_log must be strictly increasing")
        mid = self.N_BINS // 2
        self.assertAlmostEqual(t_log[mid], 0.0)
        np.testing.assert_allclose(t_log[:mid], -t_log[mid + 1 :][::-1])

    def test_values_finite_and_nonnegative(self):
        spike_times = self._dummy_spike_train()
        spike_clusters = np.zeros(spike_times.size, dtype=int)
        acgs_3d, _ = compute_3d_acgs(
            spike_times, spike_clusters, np.array([0]), self.FS
        )
        self.assertTrue(np.all(np.isfinite(acgs_3d)))
        self.assertTrue(np.all(acgs_3d >= 0))

    def test_multi_cluster_row_order(self):
        """Two clusters at different rates should produce two independent rows."""
        st0 = self._dummy_spike_train(seed=1, rate=10.0)
        st1 = self._dummy_spike_train(seed=2, rate=30.0)
        spike_times = np.concatenate([st0, st1])
        spike_clusters = np.concatenate(
            [np.zeros(st0.size, dtype=int), np.ones(st1.size, dtype=int)]
        )
        order = np.argsort(spike_times)
        spike_times, spike_clusters = spike_times[order], spike_clusters[order]
        acgs_3d, _ = compute_3d_acgs(
            spike_times, spike_clusters, np.array([0, 1]), self.FS
        )
        self.assertEqual(acgs_3d.shape[0], 2)
        self.assertFalse(np.array_equal(acgs_3d[0], acgs_3d[1]))

    def test_refractory_period_visible_near_zero_lag(self):
        """A hard refractory period should show lower density near zero lag than far from it."""
        spike_times = self._dummy_spike_train(
            duration=120.0, rate=30.0, refractory=0.01
        )
        spike_clusters = np.zeros(spike_times.size, dtype=int)
        acgs_3d, t_log = compute_3d_acgs(
            spike_times, spike_clusters, np.array([0]), self.FS
        )
        near_zero = np.abs(t_log) < 5  # ms
        far = np.abs(t_log) > 200  # ms
        self.assertLess(acgs_3d[0][:, near_zero].mean(), acgs_3d[0][:, far].mean())


class TestSelectGoodUnitsRelaxedRp(unittest.TestCase):
    def _df(self, bitwise_fail, max_confidence):
        return pd.DataFrame(
            {
                "bitwise_fail": bitwise_fail,
                "max_confidence": max_confidence,
            }
        )

    def test_missing_columns_raises(self):
        with self.assertRaises(AssertionError):
            select_good_units_relaxed_rp(pd.DataFrame({"bitwise_fail": [0]}))

    def test_relaxed_threshold_admits_more_units_than_bitwise_fail(self):
        # bit 0 (RP) fails on rows 1 and 2 under the strict 90 threshold, but their
        # max_confidence sits between the relaxed (70) and standard (90) thresholds ->
        # only the strict bitwise_fail==0 selection excludes them.
        df = self._df(
            bitwise_fail=[0, 1, 1, 0b010, 0b100],
            max_confidence=[95.0, 80.0, 71.0, 95.0, 95.0],
        )
        relaxed = select_good_units_relaxed_rp(df, rp_confidence_threshold=70.0)
        strict = (df["bitwise_fail"] == 0).to_numpy()
        np.testing.assert_array_equal(relaxed, [True, True, True, False, False])
        self.assertGreater(relaxed.sum(), strict.sum())
        # since both thresholds apply to the same max_confidence metric, relaxing
        # can only add units, never drop one that was already strict-good.
        self.assertTrue(np.all(relaxed[strict]))

    def test_noise_and_amp_vetoes_still_apply(self):
        # bit 1 (noise_cutoff) and bit 2 (amp_median) failures are never overridden,
        # regardless of a high max_confidence.
        df = self._df(
            bitwise_fail=[0b010, 0b100, 0b110],
            max_confidence=[100.0, 100.0, 100.0],
        )
        relaxed = select_good_units_relaxed_rp(df)
        np.testing.assert_array_equal(relaxed, [False, False, False])

    def test_nan_confidence_treated_as_fail(self):
        df = self._df(bitwise_fail=[0], max_confidence=[np.nan])
        relaxed = select_good_units_relaxed_rp(df)
        self.assertFalse(relaxed[0])

    def test_default_threshold_is_70(self):
        df = self._df(bitwise_fail=[0, 0], max_confidence=[69.9, 70.0])
        relaxed = select_good_units_relaxed_rp(df)
        np.testing.assert_array_equal(relaxed, [False, True])


class TestGetNeighboursMembers(unittest.TestCase):
    def test_radius_inclusion_and_self_excluded(self):
        lateral_um = np.array([0.0, 10.0, 1000.0])
        axial_um = np.array([0.0, 0.0, 0.0])
        neighbours = get_neighbours_members(lateral_um, axial_um, radius_um=50)
        expected = np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        )
        np.testing.assert_array_equal(neighbours, expected)

    def test_diagonal_always_false(self):
        neighbours = get_neighbours_members(np.zeros(4), np.zeros(4), radius_um=1e6)
        self.assertFalse(neighbours.diagonal().any())


class TestSpikeTriggeredPopulationCoupling(unittest.TestCase):
    """
    Synthetic ground truth: cluster B fires `LAG_S` after every spike of cluster A (a
    shared, jittered common drive), so their coupling should peak at +/- LAG_S with the
    expected sign; an isolated cluster C, too far away to be anyone's neighbour, should
    get an all-nan row.
    """

    LAG_S = 0.005  # 5 ms
    RADIUS_UM = 50.0
    LATERAL_UM = np.array([0.0, 10.0, 5000.0])  # A & B close, C far
    AXIAL_UM = np.zeros(3)

    def _synthetic_spike_trains(
        self, seed=0, duration=200.0, common_rate=5.0, far_rate=3.0
    ):
        rng = np.random.default_rng(seed)
        common_times = np.sort(rng.uniform(0, duration, int(common_rate * duration)))
        jitter = 0.001
        t_a = np.sort(common_times + rng.normal(0, jitter, common_times.size))
        t_b = np.sort(
            common_times + self.LAG_S + rng.normal(0, jitter, common_times.size)
        )
        t_c = np.sort(rng.uniform(0, duration, int(far_rate * duration)))
        spike_times = np.concatenate([t_a, t_b, t_c])
        spike_clusters = np.concatenate(
            [np.zeros(t_a.size), np.ones(t_b.size), np.full(t_c.size, 2)]
        ).astype(int)
        order = np.argsort(spike_times)
        return spike_times[order], spike_clusters[order]

    def _coupling(self, **kwargs):
        spike_times, spike_clusters = self._synthetic_spike_trains()
        return spike_triggered_population_coupling(
            spike_times,
            spike_clusters,
            np.array([0, 1, 2]),
            self.LATERAL_UM,
            self.AXIAL_UM,
            self.RADIUS_UM,
            **kwargs,
        )

    def test_leader_follower_peak_lag(self):
        """A leads (B follows 5 ms later) -> A's coupling peaks at +5ms, B's at -5ms."""
        result = self._coupling()
        peak_a = result["tscale"][np.nanargmax(result["stpc"][0])]
        peak_b = result["tscale"][np.nanargmax(result["stpc"][1])]
        self.assertAlmostEqual(peak_a, self.LAG_S)
        self.assertAlmostEqual(peak_b, -self.LAG_S)

    def test_isolated_cluster_has_no_neighbours_and_is_nan(self):
        result = self._coupling()
        self.assertEqual(result["n_neighbours"][2], 0)
        self.assertTrue(np.all(np.isnan(result["stpc"][2])))
        self.assertTrue(np.isnan(result["coupling_strength"][2]))
        self.assertTrue(np.isnan(result["coupling_delay"][2]))

    def test_coupled_clusters_have_one_neighbour_each(self):
        result = self._coupling()
        np.testing.assert_array_equal(result["n_neighbours"][:2], [1, 1])

    def test_coupling_strength_is_value_at_zero_lag(self):
        result = self._coupling()
        i0 = np.searchsorted(result["tscale"], 0)
        np.testing.assert_array_equal(
            result["coupling_strength"], result["stpc"][:, i0]
        )

    def test_lags_and_binsize_control_output_shape(self):
        result = self._coupling(lags=0.02, binsize=0.002)
        # [-20, 20] ms window at 2 ms bins -> 21 samples
        self.assertEqual(result["tscale"].size, 21)
        self.assertAlmostEqual(result["tscale"][0], -0.02)
        self.assertAlmostEqual(result["tscale"][-1], 0.02)
        self.assertEqual(result["stpc"].shape, (3, 21))

    def test_asymmetric_lags_tuple(self):
        result = self._coupling(lags=(-0.01, 0.03))
        self.assertAlmostEqual(result["tscale"][0], -0.01)
        self.assertAlmostEqual(result["tscale"][-1], 0.03)


class TestSpikeTriggeredPopulationCouplingDf(unittest.TestCase):
    def _synthetic(self, n_clusters=6, seed=1, duration=60.0):
        rng = np.random.default_rng(seed)
        spike_times, spike_clusters = [], []
        for c in range(n_clusters):
            rate = rng.uniform(2, 8)
            t = np.sort(rng.uniform(0, duration, int(rate * duration)))
            spike_times.append(t)
            spike_clusters.append(np.full(t.size, c))
        spike_times = np.concatenate(spike_times)
        spike_clusters = np.concatenate(spike_clusters)
        order = np.argsort(spike_times)
        spikes = {"times": spike_times[order], "clusters": spike_clusters[order]}
        df_clusters = pd.DataFrame(
            {
                "lateral_um": rng.uniform(0, 2000, n_clusters),
                "axial_um": rng.uniform(0, 2000, n_clusters),
                "bitwise_fail": [0, 0, 0, 1, 0, 1][:n_clusters],
            }
        )
        return spikes, df_clusters

    def test_good_units_only_default_restricts_rows(self):
        spikes, df_clusters = self._synthetic()
        result = spike_triggered_population_coupling_df(
            spikes, df_clusters, radius_um=1000
        )
        n_good = int((df_clusters["bitwise_fail"] == 0).sum())
        self.assertEqual(result["stpc"].shape[0], n_good)
        np.testing.assert_array_equal(
            result["i_reported"], np.where(df_clusters["bitwise_fail"] == 0)[0]
        )

    def test_good_units_only_false_reports_every_cluster(self):
        spikes, df_clusters = self._synthetic()
        result = spike_triggered_population_coupling_df(
            spikes, df_clusters, radius_um=1000, good_units_only=False
        )
        self.assertEqual(result["stpc"].shape[0], df_clusters.shape[0])

    def test_df_clusters_columns_written_back(self):
        spikes, df_clusters = self._synthetic()
        result = spike_triggered_population_coupling_df(
            spikes, df_clusters, radius_um=1000
        )
        out = result["df_clusters"]
        self.assertIn("coupling_strength", out.columns)
        self.assertIn("coupling_delay", out.columns)
        # units not reported (bitwise_fail != 0, with the default good_units_only) are nan
        bad = out.index[out["bitwise_fail"] != 0]
        self.assertTrue(out.loc[bad, "coupling_strength"].isna().all())

    def test_file_stpc_cache_round_trip(self):
        spikes, df_clusters = self._synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            file_stpc = Path(tmp).joinpath("stpc.npy")
            result_fresh = spike_triggered_population_coupling_df(
                spikes, df_clusters, radius_um=1000, file_stpc=file_stpc
            )
            self.assertTrue(file_stpc.exists())
            result_cached = spike_triggered_population_coupling_df(
                spikes, df_clusters, radius_um=1000, file_stpc=file_stpc
            )
        np.testing.assert_allclose(result_fresh["stpc"], result_cached["stpc"])
        np.testing.assert_allclose(
            result_fresh["coupling_strength"], result_cached["coupling_strength"]
        )
        np.testing.assert_allclose(
            result_fresh["firing_rate"], result_cached["firing_rate"]
        )


if __name__ == "__main__":
    unittest.main()
