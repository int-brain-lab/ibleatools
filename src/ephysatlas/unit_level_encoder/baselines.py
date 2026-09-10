from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp


LOG2PI = float(np.log(2.0 * np.pi))


class RegionalGaussianBaseline:
    """Diagonal Gaussian per anatomical region, with TRAIN global fallback."""

    def __init__(self, z, region_ids, train_mask, variance_floor=1e-3):
        self.z = np.asarray(z, np.float64)
        self.region_ids = np.asarray(region_ids, np.int64)
        self.global_mean = self.z[train_mask].mean(axis=0)
        self.global_var = np.maximum(self.z[train_mask].var(axis=0), variance_floor)
        self.tables = {}
        train_regions = self.region_ids[train_mask]
        for rid in np.unique(train_regions):
            x = self.z[train_mask][train_regions == rid]
            if len(x) < 2:
                continue
            self.tables[int(rid)] = (
                x.mean(axis=0),
                np.maximum(x.var(axis=0), variance_floor),
            )

    def _params(self, rid):
        return self.tables.get(int(rid), (self.global_mean, self.global_var))

    def log_prob(self, indices):
        out = np.empty(len(indices), np.float64)
        for j, idx in enumerate(np.asarray(indices, int)):
            mu, var = self._params(self.region_ids[idx])
            d = self.z[idx] - mu
            out[j] = -0.5 * np.sum(LOG2PI + np.log(var) + d * d / var)
        return out

    def sample(self, indices, n_per_index, rng):
        out = []
        for idx in np.asarray(indices, int):
            mu, var = self._params(self.region_ids[idx])
            out.append((mu + rng.normal(size=(n_per_index, len(mu))) * np.sqrt(var)).astype(np.float32))
        return out

    def sample_for_regions(self, region_ids, n_per_index, rng):
        """Sample at arbitrary atlas voxels specified by anatomical region ID."""
        out = []
        for rid in np.asarray(region_ids, int):
            mu, var = self._params(rid)
            out.append((mu + rng.normal(size=(n_per_index, len(mu))) * np.sqrt(var)).astype(np.float32))
        return out

    def mean_for_regions(self, region_ids):
        return np.stack([self._params(rid)[0] for rid in np.asarray(region_ids, int)]).astype(np.float32)


class SpatialKDEBaseline:
    """Spatially conditioned latent KDE using TRAIN units only."""

    def __init__(self, z, xyz_m, train_mask, cfg):
        self.z_train = np.asarray(z[train_mask], np.float64)
        self.xyz_train = np.asarray(xyz_m[train_mask], np.float64)
        self.tree = cKDTree(self.xyz_train)
        self.k = min(int(cfg.kde_neighbors), len(self.z_train))
        self.spatial_bw = float(cfg.kde_spatial_bandwidth_um) * 1e-6
        self.latent_bw = float(cfg.kde_latent_bandwidth)
        self.dim = self.z_train.shape[1]

    def _neighbors(self, xyz):
        dist, ind = self.tree.query(np.asarray(xyz, np.float64), k=self.k)
        if np.ndim(dist) == 1:
            dist = dist[:, None]
            ind = ind[:, None]
        w = np.exp(-0.5 * (dist / max(self.spatial_bw, 1e-12)) ** 2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        return ind, w

    def log_prob(self, z_query, xyz_query):
        z_query = np.asarray(z_query, np.float64)
        ind, spatial_w = self._neighbors(xyz_query)
        out = np.empty(len(z_query), np.float64)
        bw2 = self.latent_bw ** 2
        norm = -0.5 * self.dim * np.log(2.0 * np.pi * bw2)
        for i in range(len(z_query)):
            delta = self.z_train[ind[i]] - z_query[i][None, :]
            lp = norm - 0.5 * np.sum(delta * delta, axis=1) / bw2
            out[i] = logsumexp(lp + np.log(np.maximum(spatial_w[i], 1e-12)))
        return out

    def sample(self, xyz_query, n_per_index, rng):
        ind, spatial_w = self._neighbors(xyz_query)
        out = []
        for i in range(len(ind)):
            chosen = rng.choice(len(ind[i]), size=int(n_per_index), p=spatial_w[i])
            centers = self.z_train[ind[i][chosen]]
            out.append((centers + rng.normal(size=centers.shape) * self.latent_bw).astype(np.float32))
        return out

    def mean_for_xyz(self, xyz_query):
        ind, spatial_w = self._neighbors(xyz_query)
        return np.sum(self.z_train[ind] * spatial_w[:, :, None], axis=1).astype(np.float32)
