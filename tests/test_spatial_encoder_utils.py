"""Released preprocessing statistics as accepted by the channel-level data loaders.

``build_channels_plus_emptyvoxels_with_neighbors`` freezes the released ephys preprocessing and may
recompute the context normalisation (for an ablation sampling the context differently); the
inference-time neighbour bank needs the full set.
"""

import unittest

import numpy as np

from ephysatlas.spatial_encoder.utils import (
    CONTEXT_STATS_KEYS,
    EPHYS_STATS_KEYS,
    ChannelNN,
    _coerce_preprocessing_stats,
)


def _stats(keys=EPHYS_STATS_KEYS + CONTEXT_STATS_KEYS):
    return {k: np.full(3, 2.0, np.float32) for k in keys}


class TestCoercePreprocessingStats(unittest.TestCase):
    def test_no_stats_means_first_training(self):
        self.assertIsNone(_coerce_preprocessing_stats(None))

    def test_full_stats_are_converted(self):
        stats = _coerce_preprocessing_stats(_stats())
        self.assertEqual(set(stats), set(EPHYS_STATS_KEYS + CONTEXT_STATS_KEYS))

    def test_context_normalisation_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "ctx_mean"):
            _coerce_preprocessing_stats(_stats(EPHYS_STATS_KEYS))

    def test_context_normalisation_may_be_left_out_to_be_recomputed(self):
        stats = _coerce_preprocessing_stats(
            _stats(EPHYS_STATS_KEYS), require_context=False
        )
        self.assertEqual(set(stats), set(EPHYS_STATS_KEYS))

    def test_context_normalisation_is_all_or_nothing(self):
        with self.assertRaisesRegex(ValueError, "ctx_std"):
            _coerce_preprocessing_stats(
                _stats(EPHYS_STATS_KEYS + ("ctx_mean",)), require_context=False
            )

    def test_ephys_preprocessing_is_always_required(self):
        with self.assertRaisesRegex(ValueError, "e_std"):
            _coerce_preprocessing_stats(
                _stats(("rec_ephys_low_pctl", "rec_ephys_high_pctl", "e_mean")),
                require_context=False,
            )


class TestChannelNNDistances(unittest.TestCase):
    """``ChannelNN.query_radius`` returns the distances of its hits when asked, for both searches.

    ``SpatialEncoder._neighbour_draws`` orders the hits by these distances, so it does not compute
    them again for every query.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        xyz = rng.uniform(0, 1e-3, size=(60, 3))
        # The two channels of a probe row share their position, so many hits tie exactly.
        self.bank = np.r_[xyz, xyz[:20]]
        self.queries = rng.uniform(3e-4, 7e-4, size=(5, 3))
        self.r_m = 4e-4

    def _searches(self):
        """The KD-tree search, and the brute-force search used when scikit-learn is missing."""
        brute = ChannelNN(self.bank)
        brute.tree = None
        return {"kd-tree": ChannelNN(self.bank), "brute force": brute}

    def test_query_radius_returns_the_distances_of_its_hits(self):
        # k_cap=8 is below the number of hits, so the brute force keeps its 8 nearest by partition.
        for name, nn in self._searches().items():
            for k_cap in (len(self.bank), 8):
                with self.subTest(search=name, k_cap=k_cap):
                    hits = nn.query_radius(self.queries, self.r_m, k_cap=k_cap)
                    inds, dists = nn.query_radius(
                        self.queries, self.r_m, k_cap=k_cap, return_distance=True
                    )
                    for i, q in enumerate(self.queries):
                        self.assertEqual(inds[i].size, min(k_cap, len(hits[i])))
                        self.assertEqual(set(inds[i]), set(hits[i]))
                        np.testing.assert_allclose(
                            dists[i], np.linalg.norm(self.bank[inds[i]] - q, axis=1)
                        )

    def test_query_radius_without_hits_returns_empty_arrays(self):
        far = np.array([[1.0, 1.0, 1.0]])
        for name, nn in self._searches().items():
            with self.subTest(search=name):
                inds, dists = nn.query_radius(far, self.r_m, return_distance=True)
                self.assertEqual(inds[0].size, 0)
                self.assertEqual(dists[0].size, 0)


if __name__ == "__main__":
    unittest.main()
