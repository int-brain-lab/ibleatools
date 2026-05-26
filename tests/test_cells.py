import unittest

import numpy as np

from ephysatlas.cells import compute_burstiness_and_memory


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


if __name__ == "__main__":
    unittest.main()
