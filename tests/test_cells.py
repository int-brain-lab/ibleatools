import unittest

import numpy as np

from ephysatlas.cells import compute_burstiness_and_memory, compute_log_acg


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
            acg, t_log = compute_log_acg(self._poisson_spikes(), fs=self.FS, n_log_bins=n)
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
        acg, t_log = compute_log_acg(spike_times, fs=self.FS, spike_clusters=spike_clusters)
        self.assertEqual(acg.ndim, 2)
        self.assertEqual(acg.shape[0], 3)
        self.assertEqual(acg.shape[1], t_log.size)

    def test_multi_cluster_matches_single(self):
        """Each row of the multi-cluster output equals the single-cluster call."""
        rng = np.random.default_rng(5)
        spike_times = np.sort(rng.uniform(0, 60, 1000))
        spike_clusters = np.zeros(1000, dtype=int)
        spike_clusters[500:] = 1
        acg_multi, t_log = compute_log_acg(spike_times, self.FS, spike_clusters=spike_clusters)
        for i, cid in enumerate([0, 1]):
            acg_single, _ = compute_log_acg(spike_times[spike_clusters == cid], self.FS)
            np.testing.assert_array_equal(acg_multi[i], acg_single)


if __name__ == "__main__":
    unittest.main()
