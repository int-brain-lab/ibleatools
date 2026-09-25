from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class KNNQuery:
    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray


class EmpiricalKNNDecoder:
    """Distance-weighted empirical phenotype projection in standardized latent space.

    The released kNN stage contains TRAIN exemplars only.  A query latent is
    projected to a categorical distribution over the k nearest TRAIN units using
    an adaptive Gaussian kernel whose scale is the median non-zero neighbor
    distance for that query.

    A bank fitted with :meth:`set_context_readout` also holds each exemplar's GMM component
    (``labels_train``) and a low-dimensional key of its molecular context (``key_train``), for
    the context-local member readout (:meth:`context_local_means`, :meth:`sample_context_local`):
    a component's phenotype at a position is represented by its TRAIN members whose context key
    is near the position's, shrunk toward all of its members -- the more so the farther the key
    lies from the TRAIN keys, and entirely where the position has no molecular context.
    """

    # Arrays of the context-local member readout, saved with the bank when fitted.
    READOUT_ARRAYS = (
        "labels_train",
        "key_train",
        "key_coef",
        "key_intercept",
        "key_alpha",
    )
    # Optional readout array: the key of an all-zero (absent) molecular context.
    OPTIONAL_READOUT_ARRAYS = ("key_void",)

    def __init__(
        self,
        z_scaled: np.ndarray,
        train_mask: np.ndarray,
        waveform_features: np.ndarray,
        *,
        k: int = 20,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ):
        z_scaled = np.asarray(z_scaled, np.float32)
        train_mask = np.asarray(train_mask, bool)
        self.train_indices = np.flatnonzero(train_mask).astype(np.int64)
        self.z_train = z_scaled[self.train_indices]
        self.feature_train = np.asarray(waveform_features, np.float32)[
            self.train_indices
        ]
        self.feature_names = tuple(
            feature_names
            or [f"feature_{i}" for i in range(self.feature_train.shape[1])]
        )
        self.k = min(int(k), len(self.train_indices))
        self._clear_readout()
        self._fit_index()

    @classmethod
    def from_bank(
        cls,
        z_train: np.ndarray,
        feature_train: np.ndarray,
        *,
        k: int,
        train_indices: np.ndarray | None = None,
        feature_names: list[str] | tuple[str, ...] | None = None,
        readout: dict | None = None,
    ) -> "EmpiricalKNNDecoder":
        obj = cls.__new__(cls)
        obj._clear_readout()
        for name, value in (readout or {}).items():
            setattr(obj, name, value)
        obj.z_train = np.asarray(z_train, np.float32)
        obj.feature_train = np.asarray(feature_train, np.float32)
        obj.train_indices = (
            np.arange(len(obj.z_train), dtype=np.int64)
            if train_indices is None
            else np.asarray(train_indices, np.int64)
        )
        obj.feature_names = tuple(
            feature_names or [f"feature_{i}" for i in range(obj.feature_train.shape[1])]
        )
        obj.k = min(int(k), len(obj.z_train))
        obj._fit_index()
        return obj

    def _fit_index(self) -> None:
        if self.k < 1:
            raise ValueError("EmpiricalKNNDecoder requires at least one TRAIN unit")
        if len(self.z_train) != len(self.feature_train):
            raise ValueError(
                "z_train and feature_train must have the same number of rows"
            )
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto", n_jobs=-1)
        self.nn.fit(self.z_train)

    def save_bank(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        readout = {}
        if self.has_context_readout:
            readout = {
                name: np.asarray(getattr(self, name))
                for name in self.READOUT_ARRAYS + self.OPTIONAL_READOUT_ARRAYS
                if getattr(self, name) is not None
            }
        np.savez_compressed(
            path,
            z_train=self.z_train.astype(np.float32),
            feature_train=self.feature_train.astype(np.float32),
            train_indices=self.train_indices.astype(np.int64),
            feature_names=np.asarray(self.feature_names, dtype="U"),
            k=np.asarray(self.k, dtype=np.int64),
            **readout,
        )
        return path

    @classmethod
    def load_bank(
        cls, path: Path | str, *, k: int | None = None
    ) -> "EmpiricalKNNDecoder":
        with np.load(Path(path), allow_pickle=False) as bank:
            saved_k = int(np.asarray(bank["k"]).item()) if "k" in bank else 20
            names = (
                bank["feature_names"].astype(str).tolist()
                if "feature_names" in bank
                else None
            )
            # Banks written before the context-local readout have none of its arrays.
            readout = (
                {
                    name: bank[name]
                    for name in cls.READOUT_ARRAYS + cls.OPTIONAL_READOUT_ARRAYS
                    if name in bank
                }
                if all(name in bank for name in cls.READOUT_ARRAYS)
                else None
            )
            return cls.from_bank(
                bank["z_train"],
                bank["feature_train"],
                k=saved_k if k is None else int(k),
                train_indices=bank["train_indices"]
                if "train_indices" in bank
                else None,
                feature_names=names,
                readout=readout,
            )

    def query(self, z_query: np.ndarray) -> KNNQuery:
        z_query = np.asarray(z_query, np.float32)
        distances, local_indices = self.nn.kneighbors(z_query, return_distance=True)
        distances = np.asarray(distances, np.float64)
        positive = np.where(distances > 1e-12, distances, np.nan)
        scale = np.nanmedian(positive, axis=1)
        fallback = np.maximum(distances[:, -1], 1e-6)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
        weights = np.exp(-0.5 * (distances / scale[:, None]) ** 2)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return KNNQuery(
            indices=np.asarray(local_indices, np.int64),
            distances=distances.astype(np.float32),
            weights=weights.astype(np.float32),
        )

    def expected_features(self, z_query: np.ndarray) -> np.ndarray:
        q = self.query(z_query)
        values = self.feature_train[q.indices]
        return np.sum(values * q.weights[:, :, None], axis=1).astype(np.float32)

    def sample_rows(self, z_query: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One bank row per query latent, drawn with the distance-weighted kNN probabilities."""
        q = self.query(z_query)
        chosen = np.asarray(
            [rng.choice(self.k, p=q.weights[i]) for i in range(len(q.indices))],
            dtype=np.int64,
        )
        return q.indices[np.arange(len(q.indices)), chosen]

    def sample_features(
        self, z_query: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return self.feature_train[self.sample_rows(z_query, rng)].astype(np.float32)

    def sample_training_indices(
        self, z_query: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return self.train_indices[self.sample_rows(z_query, rng)]

    def distance_summary(self, z_query: np.ndarray) -> dict:
        q = self.query(z_query)
        d1 = q.distances[:, 0]
        dk = q.distances[:, -1]
        effective_n = 1.0 / np.maximum(np.sum(q.weights**2, axis=1), 1e-12)
        return {
            "k": int(self.k),
            "nearest_distance_mean": float(np.mean(d1)),
            "nearest_distance_median": float(np.median(d1)),
            "kth_distance_mean": float(np.mean(dk)),
            "kth_distance_median": float(np.median(dk)),
            "effective_neighbors_mean": float(np.mean(effective_n)),
            "effective_neighbors_median": float(np.median(effective_n)),
        }

    # -- context-local member readout -------------------------------------------------------

    def _clear_readout(self) -> None:
        for name in self.READOUT_ARRAYS + self.OPTIONAL_READOUT_ARRAYS:
            setattr(self, name, None)
        self._key_index = None
        self._train_scales = {}

    @property
    def has_context_readout(self) -> bool:
        """True when the bank carries the context-local member readout (see the class doc)."""
        return self.labels_train is not None and self.key_train is not None

    def _require_readout(self) -> None:
        if not self.has_context_readout:
            raise RuntimeError(
                "this kNN bank has no context-local member readout (it was saved before the "
                "readout existed); fit one with pipeline.fit_context_readout"
            )

    def set_context_readout(
        self,
        labels_train,
        key_coef,
        key_intercept,
        context_pc_train,
        *,
        key_alpha=np.nan,
        void_context_pc=None,
    ) -> None:
        """Attach the context-local member readout to the bank.

        Args:
            labels_train: ``[n_train]`` GMM component (hard assignment) of each bank exemplar.
            key_coef: ``[key_dim, n_context]`` and ``key_intercept``: ``[key_dim]``, the linear
                map ``key = context_pc @ key_coef.T + key_intercept`` from the standardized
                molecular context to the readout key.
            context_pc_train: ``[n_train, n_context]`` standardized context of the bank exemplars.
            key_alpha: Ridge penalty the key was fitted with (recorded only).
            void_context_pc: ``[n_context]`` standardized form of an all-zero (absent) molecular
                context; queries with its key get the global member means.
        """
        self._clear_readout()
        self.key_coef = np.asarray(key_coef, np.float32)
        self.key_intercept = np.asarray(key_intercept, np.float32)
        self.key_alpha = np.asarray(float(key_alpha), np.float64)
        self.labels_train = np.asarray(labels_train, np.int64)
        self.key_train = self.context_key(context_pc_train)
        if void_context_pc is not None:
            self.key_void = self.context_key(np.reshape(void_context_pc, (1, -1)))[0]
        if not len(self.labels_train) == len(self.key_train) == len(self.z_train):
            raise ValueError(
                "labels_train and context_pc_train need one row per bank exemplar"
            )

    def context_key(self, context_pc: np.ndarray) -> np.ndarray:
        """``[n, key_dim]`` readout key of standardized molecular contexts."""
        if self.key_coef is None:
            self._require_readout()
        context_pc = np.asarray(context_pc, np.float32)
        return (context_pc @ self.key_coef.T + self.key_intercept).astype(np.float32)

    def member_means(self, n_components: int) -> np.ndarray:
        """``[n_components, n_features]`` mean features of each component's bank members.

        A component without members falls back to the mean of the whole bank.
        """
        self._require_readout()
        counts = np.bincount(self.labels_train, minlength=n_components).astype(
            np.float64
        )
        sums = np.zeros((n_components, self.feature_train.shape[1]))
        np.add.at(sums, self.labels_train, self.feature_train.astype(np.float64))
        means = sums / np.maximum(counts, 1.0)[:, None]
        means[counts == 0] = self.feature_train.mean(axis=0)
        return means.astype(np.float32)

    def _key_neighbours(self, key, m):
        """Distances ``[n, m]`` (float64) and bank rows of the ``m`` nearest TRAIN keys."""
        if self._key_index is None or self._key_index.n_neighbors != m:
            self._key_index = NearestNeighbors(n_neighbors=m, n_jobs=-1).fit(
                self.key_train
            )
        dist, ind = self._key_index.kneighbors(
            np.asarray(key, np.float32), return_distance=True
        )
        return np.asarray(dist, np.float64), ind

    def train_kernel_scales(self, neighbours: int) -> np.ndarray:
        """The readout's kernel scale at up to 5000 evenly spaced TRAIN exemplars' own keys."""
        m = min(int(neighbours), len(self.key_train))
        if m not in self._train_scales:
            rows = np.unique(np.linspace(0, len(self.key_train) - 1, 5000).astype(int))
            dist, _ = self._key_neighbours(self.key_train[rows], m)
            self._train_scales[m] = np.median(dist, axis=1)
        return self._train_scales[m]

    def _local_member_stats(
        self, key, n_components, neighbours, *, off_data_quantile=None, features=True
    ):
        """Key neighbours with adaptive Gaussian kernel weights, per-component statistics.

        Returns the neighbour rows ``[n, m]``, their weights (rows sum to one), their component
        labels, the effective member count ``n_k(x)`` of each component ``[n, K]``, when
        ``features`` the kernel-weighted feature sums of each component's local members
        ``[n, K, F]`` scaled by the effective neighbour count (so ``sums / n_k`` is the local mean),
        else None, and the query's in-data weight ``r(x)`` ``[n]``.

        ``r(x) = min(1, (h_ref / h(x)) ** key_dim)`` is the density of TRAIN keys around the query
        relative to the edge of the TRAIN data: ``h(x)`` is the query's kernel scale (the median
        distance to its ``m`` nearest keys) and ``h_ref`` the ``off_data_quantile`` of that scale
        over TRAIN exemplars (``r = 1`` when None). A query with the key of an absent context
        (``key_void``) has ``r = 0``.
        """
        m = min(int(neighbours), len(self.key_train))
        dist, ind = self._key_neighbours(key, m)
        scale = np.median(dist, axis=1, keepdims=True)
        scale = np.where(scale > 1e-12, scale, np.maximum(dist[:, -1:], 1e-6))
        w = np.exp(-0.5 * (dist / scale) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        n_total = 1.0 / np.sum(w**2, axis=1)
        in_data = np.ones(len(ind))
        if off_data_quantile is not None:
            h_ref = np.quantile(self.train_kernel_scales(m), float(off_data_quantile))
            in_data = np.minimum(1.0, (h_ref / scale[:, 0]) ** self.key_train.shape[1])
        if self.key_void is not None:
            void = np.abs(np.asarray(key) - self.key_void) <= 1e-5 * (
                1.0 + np.abs(self.key_void)
            )
            in_data[np.all(void, axis=1)] = 0.0
        lab = self.labels_train[ind]
        n, size = len(ind), len(ind) * n_components
        flat = (np.arange(n)[:, None] * n_components + lab).ravel()
        weight_k = np.bincount(flat, weights=w.ravel(), minlength=size).reshape(n, -1)
        neff = weight_k * n_total[:, None]
        sums = None
        if features:
            sums = (
                np.stack(
                    [
                        np.bincount(
                            flat,
                            weights=(w * self.feature_train[ind, j]).ravel(),
                            minlength=size,
                        )
                        for j in range(self.feature_train.shape[1])
                    ],
                    axis=1,
                ).reshape(n, n_components, -1)
                * n_total[:, None, None]
            )
        return ind, w, lab, neff, sums, in_data

    def context_local_means(
        self,
        context_pc,
        weights,
        *,
        neighbours: int,
        shrinkage: float,
        off_data_quantile: float | None = None,
        chunk: int = 4096,
    ) -> np.ndarray:
        """Expected features ``sum_k weights_k mu_k(x)`` under the context-local member readout.

        ``mu_k(x) = (n_k(x) m_k(x) + shrinkage m_k) / (n_k(x) + shrinkage)``: the kernel-weighted
        mean ``m_k(x)`` of component k's members among the ``neighbours`` exemplars nearest to
        the query's context key (Gaussian kernel, bandwidth the median neighbour distance), with
        effective count ``n_k(x)``, shrunk toward the mean ``m_k`` of all of k's members. Off the
        TRAIN data it is blended further toward ``m_k``, ``r(x) mu_k(x) + (1 - r(x)) m_k``, with
        the in-data weight ``r(x)`` of :meth:`_local_member_stats` (1 within the data, 0 without
        context).

        Args:
            context_pc: ``[n, n_context]`` standardized molecular contexts.
            weights: ``[n, K]`` mixture weights at those contexts.
            neighbours: Exemplars considered per query.
            shrinkage: Pseudo-count of the shrinkage toward the global member means (> 0).
            off_data_quantile: Quantile of the TRAIN kernel scale beyond which a query counts as
                off the data; None disables the off-data blending.
            chunk: Queries per batch (bounds memory).

        Returns:
            np.ndarray: ``[n, n_features]`` float32.
        """
        self._require_readout()
        if not float(shrinkage) > 0:
            raise ValueError("shrinkage must be positive")
        weights = np.asarray(weights, np.float64)
        n_components = weights.shape[1]
        key = self.context_key(context_pc)
        global_means = self.member_means(n_components).astype(np.float64)
        out = np.empty((len(key), self.feature_train.shape[1]), np.float32)
        for start in range(0, len(key), int(chunk)):
            stop = start + int(chunk)
            _, _, _, neff, sums, in_data = self._local_member_stats(
                key[start:stop],
                n_components,
                neighbours,
                off_data_quantile=off_data_quantile,
            )
            local = (sums + shrinkage * global_means[None]) / (neff + shrinkage)[
                ..., None
            ]
            in_data = in_data[:, None, None]
            local = in_data * local + (1.0 - in_data) * global_means[None]
            out[start:stop] = np.einsum("nk,nkf->nf", weights[start:stop], local)
        return out

    def sample_context_local(
        self,
        context_pc,
        weights,
        n_samples: int,
        rng: np.random.Generator,
        *,
        neighbours: int,
        shrinkage: float,
        off_data_quantile: float | None = None,
    ) -> np.ndarray:
        """``[n, n_samples]`` bank rows drawn from the context-local member readout.

        Each draw picks a component ``k ~ weights``, then with probability
        ``r(x) n_k(x) / (n_k(x) + shrinkage)`` one of k's local members with probability
        proportional to its kernel weight, and otherwise any member of k uniformly -- so the draws
        average to :meth:`context_local_means`. The features, latents and dataset indices of the drawn
        exemplars are ``feature_train[rows]``, ``z_train[rows]`` and ``train_indices[rows]``.
        """
        self._require_readout()
        if not float(shrinkage) > 0:
            raise ValueError("shrinkage must be positive")
        weights = np.asarray(weights, np.float64)
        weights = weights / weights.sum(axis=1, keepdims=True)
        n_components, n_samples = weights.shape[1], int(n_samples)
        key = self.context_key(context_pc)
        order = np.argsort(self.labels_train, kind="stable")
        counts = np.bincount(self.labels_train, minlength=n_components)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        m = min(int(neighbours), len(self.key_train))
        chunk = max(1, (1 << 22) // max(n_samples * m, 1))
        rows = np.empty((len(key), n_samples), np.int64)
        for start in range(0, len(key), chunk):
            stop = start + chunk
            ind, w, lab, neff, _, in_data = self._local_member_stats(
                key[start:stop],
                n_components,
                neighbours,
                off_data_quantile=off_data_quantile,
                features=False,
            )
            n = len(ind)
            r = np.arange(n)[:, None]
            cum_w = np.cumsum(weights[start:stop], axis=1)
            comp = np.minimum(
                (rng.uniform(size=(n, n_samples))[:, :, None] > cum_w[:, None, :]).sum(
                    -1
                ),
                n_components - 1,
            )
            n_k = neff[r, comp]
            use_local = rng.uniform(size=(n, n_samples)) < in_data[:, None] * n_k / (
                n_k + shrinkage
            )
            member_w = np.where(lab[:, None, :] == comp[:, :, None], w[:, None, :], 0.0)
            cum = np.cumsum(member_w, axis=2)
            u = rng.uniform(size=(n, n_samples)) * cum[:, :, -1]
            local_rows = ind[r, np.minimum((u[:, :, None] >= cum).sum(-1), m - 1)]
            count = counts[comp]
            pick = starts[comp] + np.minimum(
                (rng.uniform(size=(n, n_samples)) * count).astype(np.int64),
                np.maximum(count - 1, 0),
            )
            member_rows = np.where(
                count > 0,
                order[np.minimum(pick, len(order) - 1)],
                rng.integers(0, len(order), size=(n, n_samples)),
            )
            rows[start:stop] = np.where(use_local, local_rows, member_rows)
        return rows


# Seed offset (added to ``Config.feature_slice_seed``) used for the component feature
# expectations. The published atlas figures and ``UnitEncoder.predict`` share it, so a released
# ``component_feature_expectations.npz`` reproduces the figure maps exactly.
COMPONENT_FEATURE_SEED_OFFSET = 3200


def component_feature_expectations(
    gmm, decoder: EmpiricalKNNDecoder, *, n_samples: int = 128, seed: int = 0
):
    """Stable E[TRAIN phenotype feature | GMM component] under the kNN projection.

    Each component is sampled once with a fixed Monte Carlo bank of ``n_samples`` standardized
    latents, every draw is projected onto its kNN-weighted TRAIN exemplars, and the projections are
    averaged. A location's expected phenotype is then ``weights(x) @ expectations``: smooth in
    space and deterministic, unlike independent per-location sampling.

    Args:
        gmm: Fitted sklearn ``GaussianMixture`` (``full`` or ``diag`` covariance).
        decoder: The released :class:`EmpiricalKNNDecoder`.
        n_samples: Monte Carlo draws per component.
        seed: Seed of the draws.

    Returns:
        np.ndarray: ``[n_components, n_features]`` float32.
    """
    n = max(1, int(n_samples))
    rng = np.random.default_rng(int(seed))
    draws = []
    for k in range(gmm.n_components):
        if gmm.covariance_type == "full":
            z = rng.multivariate_normal(gmm.means_[k], gmm.covariances_[k], size=n)
        elif gmm.covariance_type == "diag":
            eps = rng.normal(size=(n, gmm.means_.shape[1]))
            z = gmm.means_[k][None, :] + eps * np.sqrt(gmm.covariances_[k])[None, :]
        else:
            raise ValueError(gmm.covariance_type)
        draws.append(np.asarray(z, np.float32))
    z = np.concatenate(draws, axis=0)
    feat = decoder.expected_features(z)
    return feat.reshape(gmm.n_components, n, -1).mean(axis=1).astype(np.float32)
