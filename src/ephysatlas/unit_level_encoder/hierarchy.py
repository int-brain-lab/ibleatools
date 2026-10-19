"""The density model of the unit-level atlas: putative cell types, their sub-types and their members.

The standardized latents z of the TRAIN units are modeled by a two-level Gaussian mixture:

* types: a full-covariance GMM with ``cfg.n_types`` components;
* sub-types: inside each type k (the TRAIN units whose most likely type is k), a full-covariance
  GMM with up to ``cfg.max_subtypes`` components and at least ``cfg.min_units_per_subtype`` units
  per component. The sub-types of all types, pooled, form one density
  ``sum_kj pi_k pi_j|k N(z; mu_kj, Sigma_kj)``.

Assignment is top-down, ``P(k, j | z) = P(k | z) P(j | k, z)``, with ``P(j | k, z)`` computed
among the sub-types of type k only, so a unit's sub-type always lies inside its type.

At a position x the sub-type weights are ``w_kj(x) = gamma_k(x) gamma_j|k(x)``: two context
networks (:mod:`.context_weights`) predict them from the molecular context c(x), one trained on
the type posteriors and one on the top-down sub-type posteriors, whose within-type shares give
``gamma_j|k(x)``.

Phenotypes are read out through the TRAIN units (:class:`MemberReadout`): a sub-type is
represented by its members from anywhere in the brain, weighted by their top-down posterior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, softmax
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from ephysatlas import model_registry

LOG2PI = float(np.log(2.0 * np.pi))
# Units per chunk when evaluating all Gaussians, to bound the [n, n_subtypes] float64 arrays.
CHUNK = 16384

TYPES_FILE = model_registry.UNIT_TYPES_FILE
SUBTYPES_FILE = model_registry.UNIT_SUBTYPES_FILE
PARENT_FILE = model_registry.UNIT_SUBTYPE_PARENT_FILE
SUMMARY_FILE = "hierarchy_summary.json"


def gaussian_log_prob(means, covariances, z, dims=None) -> np.ndarray:
    """``[n, L]`` log N(z; mu_l, Sigma_l) of full-covariance Gaussians.

    Args:
        means: ``[L, d]`` means.
        covariances: ``[L, d, d]`` covariances.
        z: ``[n, d]`` points.
        dims: Optional subset of dimensions; the Gaussians are then marginalized onto them.
    """
    z = np.asarray(z, np.float64)
    means = np.asarray(means, np.float64)
    covariances = np.asarray(covariances, np.float64)
    if dims is not None:
        dims = np.asarray(dims, int)
        z, means = z[:, dims], means[:, dims]
        covariances = covariances[:, dims][:, :, dims]
    d = z.shape[1]
    out = np.empty((len(z), len(means)), np.float64)
    for k, (mu, cov) in enumerate(zip(means, covariances)):
        chol = np.linalg.cholesky(cov)
        y = solve_triangular(chol, (z - mu).T, lower=True)
        out[:, k] = (
            -0.5 * (d * LOG2PI + np.sum(y * y, axis=0)) - np.log(np.diag(chol)).sum()
        )
    return out


def _chunked(fn, z, *args):
    """Apply ``fn(z_chunk, *args)`` over row chunks of ``z`` and stack the results."""
    z = np.asarray(z)
    if len(z) <= CHUNK:
        return fn(z, *args)
    parts = [fn(z[i : i + CHUNK], *args) for i in range(0, len(z), CHUNK)]
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate(p, axis=0) for p in zip(*parts))
    return np.concatenate(parts, axis=0)


@dataclass
class HierarchicalMixture:
    """Putative cell types and their sub-types, global across the brain.

    Attributes:
        types: ``GaussianMixture`` with one full-covariance component per type.
        subtypes: ``GaussianMixture`` with one component per sub-type, the sub-types of all types
            pooled; its weights are ``pi_k pi_j|k``.
        parent: ``[n_subtypes]`` type of each sub-type; the sub-types of a type are contiguous.
    """

    types: GaussianMixture
    subtypes: GaussianMixture
    parent: np.ndarray

    def __post_init__(self):
        self.parent = np.asarray(self.parent, np.int64)

    @property
    def n_types(self) -> int:
        return int(self.types.n_components)

    @property
    def n_subtypes(self) -> int:
        return int(len(self.parent))

    @property
    def within_weights(self) -> np.ndarray:
        """``[n_subtypes]`` pi_j|k: each sub-type's share of its type."""
        w = np.asarray(self.subtypes.weights_, np.float64)
        return (
            w / np.bincount(self.parent, weights=w, minlength=self.n_types)[self.parent]
        )

    def subtype_index(self) -> np.ndarray:
        """``[n_subtypes]`` index j of each sub-type within its type."""
        start = np.searchsorted(self.parent, self.parent, side="left")
        return np.arange(self.n_subtypes) - start

    def type_log_prob(self, z, dims=None) -> np.ndarray:
        """``[n, n_types]`` log N(z; mu_k, Sigma_k), optionally marginal on ``dims``."""
        return gaussian_log_prob(self.types.means_, self.types.covariances_, z, dims)

    def subtype_log_prob(self, z, dims=None) -> np.ndarray:
        """``[n, n_subtypes]`` log N(z; mu_kj, Sigma_kj), optionally marginal on ``dims``."""
        return gaussian_log_prob(
            self.subtypes.means_, self.subtypes.covariances_, z, dims
        )

    def _posterior(self, z, dims):
        lp_type = np.log(self.types.weights_)[None] + self.type_log_prob(z, dims)
        p_type = softmax(lp_type, axis=1)
        lp = np.log(self.within_weights)[None] + self.subtype_log_prob(z, dims)
        within = np.empty_like(lp)
        for k in range(self.n_types):
            s = self.parent == k
            within[:, s] = softmax(lp[:, s], axis=1)
        return p_type, within

    def posterior(self, z, dims=None) -> tuple[np.ndarray, np.ndarray]:
        """Top-down posteriors of standardized latents.

        Args:
            z: ``[n, latent_dim]`` standardized latents.
            dims: Optional latent dimensions to marginalize the mixture onto (e.g. the waveform and
                ACG dimensions of units without population coupling).

        Returns:
            tuple: ``(P(k | z) [n, n_types], P(k, j | z) [n, n_subtypes])``; the second is
            ``P(k | z) P(j | k, z)`` and sums, over a type's sub-types, to that type's posterior.
        """
        p_type, within = _chunked(self._posterior, z, dims)
        return p_type, p_type[:, self.parent] * within

    def assign(self, z, dims=None) -> tuple[np.ndarray, np.ndarray]:
        """Hard top-down assignment: the most likely type, then its most likely sub-type.

        Returns:
            tuple: ``(type [n], subtype [n])``; ``subtype`` indexes the pooled sub-types and always
            belongs to ``type``.
        """
        p_type, within = _chunked(self._posterior, z, dims)
        types = p_type.argmax(axis=1)
        inside = self.parent[None, :] == types[:, None]
        return types, np.where(inside, within, -1.0).argmax(axis=1)

    def log_prob(self, z, weights, dims=None) -> np.ndarray:
        """``[n]`` log sum_kj w_kj N(z; mu_kj, Sigma_kj) with per-point sub-type weights ``[n, L]``."""
        log_w = np.log(np.maximum(np.asarray(weights, np.float64), 1e-12))
        return logsumexp(log_w + _chunked(self.subtype_log_prob, z, dims), axis=1)

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.types, directory / TYPES_FILE)
        joblib.dump(self.subtypes, directory / SUBTYPES_FILE)
        np.save(directory / PARENT_FILE, self.parent, allow_pickle=False)
        return directory

    @classmethod
    def load(
        cls, directory=None, *, types=None, subtypes=None, parent=None
    ) -> "HierarchicalMixture":
        """Load from a directory written by :meth:`save`, or from the three explicit files."""
        if directory is not None:
            directory = Path(directory)
            types, subtypes, parent = (
                directory / TYPES_FILE,
                directory / SUBTYPES_FILE,
                directory / PARENT_FILE,
            )
        return cls(
            joblib.load(types),
            joblib.load(subtypes),
            np.load(parent, allow_pickle=False),
        )


def _gaussian_mixture(n_components, cfg, n_init):
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type="full",
        reg_covar=float(cfg.gmm_reg_covar),
        n_init=int(n_init),
        max_iter=int(cfg.gmm_max_iter),
        random_state=int(cfg.seed),
        init_params="kmeans",
    )


def fit_hierarchical_mixture(z_train, cfg, out_dir=None) -> HierarchicalMixture:
    """Fit the types, then the sub-types inside each type, on the TRAIN latents.

    Args:
        z_train: ``[n_train, latent_dim]`` standardized latents of the TRAIN units.
        cfg: Unit-model configuration (``n_types``, ``max_subtypes``, ``min_units_per_subtype``,
            ``type_n_init``, ``subtype_n_init``, ``gmm_reg_covar``, ``gmm_max_iter``, ``seed``).
        out_dir: Optional directory for the fitted mixture and its summary.
    """
    z = np.asarray(z_train, np.float64)
    types = _gaussian_mixture(cfg.n_types, cfg, cfg.type_n_init).fit(z)
    if not types.converged_:
        raise RuntimeError("The type GMM did not converge")
    top = types.predict(z)
    weights, means, covariances, parent = [], [], [], []
    for k in range(int(cfg.n_types)):
        zk = z[top == k]
        if len(zk) < 2:
            # A type that wins no TRAIN unit keeps its own Gaussian as its only sub-type.
            sub_w, sub_mu, sub_cov = (
                np.ones(1),
                types.means_[k : k + 1],
                types.covariances_[k : k + 1],
            )
        else:
            n_sub = int(
                min(cfg.max_subtypes, max(1, len(zk) // int(cfg.min_units_per_subtype)))
            )
            child = _gaussian_mixture(n_sub, cfg, cfg.subtype_n_init).fit(zk)
            sub_w, sub_mu, sub_cov = child.weights_, child.means_, child.covariances_
        weights.append(types.weights_[k] * sub_w)
        means.append(sub_mu)
        covariances.append(sub_cov)
        parent += [k] * len(sub_w)
    weights = np.concatenate(weights)
    subtypes = GaussianMixture(len(parent), covariance_type="full")
    subtypes.weights_ = weights / weights.sum()
    subtypes.means_ = np.concatenate(means)
    subtypes.covariances_ = np.concatenate(covariances)
    subtypes.precisions_cholesky_ = _compute_precision_cholesky(
        subtypes.covariances_, "full"
    )
    subtypes.converged_, subtypes.n_iter_, subtypes.lower_bound_ = True, 0, np.nan
    subtypes.n_features_in_ = z.shape[1]
    mixture = HierarchicalMixture(types, subtypes, np.asarray(parent))
    if out_dir is not None:
        mixture.save(out_dir)
        counts = np.bincount(mixture.parent, minlength=mixture.n_types)
        (Path(out_dir) / SUMMARY_FILE).write_text(
            json.dumps(
                {
                    "n_types": mixture.n_types,
                    "n_subtypes": mixture.n_subtypes,
                    "subtypes_per_type": counts.tolist(),
                    "type_weights": types.weights_.tolist(),
                    "type_converged": bool(types.converged_),
                    "type_n_iter": int(types.n_iter_),
                    "train_units_per_type": np.bincount(
                        top, minlength=mixture.n_types
                    ).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return mixture


class MemberReadout:
    """Phenotypes of the sub-types, read out through their TRAIN members.

    A sub-type is represented by the TRAIN units weighted by their top-down posterior of it, from
    anywhere in the brain: its expected phenotype is their weighted mean feature vector, and a draw
    from it is one of them, with probability proportional to that posterior.

    Attributes:
        features: ``[n_train, n_features]`` phenotype features of the TRAIN units.
        feature_names: Their names.
        means: ``[n_subtypes, n_features]`` expected phenotype of each sub-type.
        offsets, rows, cdf: Per sub-type ``l``, its members ``rows[offsets[l]:offsets[l + 1]]``
            (rows of ``features``) and their cumulative draw probabilities.
    """

    # Posteriors below this are not stored for drawing (the means use every unit).
    MIN_RESPONSIBILITY = 1e-6

    def __init__(self, features, feature_names, means, offsets, rows, cdf):
        self.features = np.asarray(features, np.float32)
        self.feature_names = [str(f) for f in feature_names]
        self.means = np.asarray(means, np.float32)
        self.offsets = np.asarray(offsets, np.int64)
        self.rows = np.asarray(rows, np.int64)
        self.cdf = np.asarray(cdf, np.float64)

    @property
    def n_subtypes(self) -> int:
        return int(len(self.offsets) - 1)

    @classmethod
    def fit(cls, resp_train, features_train, feature_names) -> "MemberReadout":
        """Build the readout from the top-down posteriors ``[n_train, n_subtypes]`` of the TRAIN units."""
        r = np.asarray(resp_train, np.float64)
        x = np.asarray(features_train, np.float64)
        mass = np.maximum(r.sum(axis=0), 1e-12)
        means = (r.T @ x) / mass[:, None]
        offsets, rows, cdf = [0], [], []
        for col in r.T:
            keep = np.flatnonzero(col >= cls.MIN_RESPONSIBILITY)
            if len(keep) == 0:
                keep = np.array([int(col.argmax())])
            c = np.cumsum(col[keep])
            rows.append(keep)
            cdf.append(c / c[-1])
            offsets.append(offsets[-1] + len(keep))
        return cls(
            x, feature_names, means, offsets, np.concatenate(rows), np.concatenate(cdf)
        )

    def expected(self, weights) -> np.ndarray:
        """``[n, n_features]`` expected phenotype under sub-type weights ``[n, n_subtypes]``."""
        return (np.asarray(weights, np.float64) @ self.means.astype(np.float64)).astype(
            np.float32
        )

    def draw_members(self, subtypes, rng) -> np.ndarray:
        """One TRAIN member (a row of ``features``) per entry of ``subtypes``."""
        subtypes = np.asarray(subtypes, np.int64)
        u = rng.random(subtypes.shape)
        out = np.empty(subtypes.shape, np.int64)
        for s in np.unique(subtypes):
            at = subtypes == s
            lo, hi = self.offsets[s], self.offsets[s + 1]
            pos = np.minimum(np.searchsorted(self.cdf[lo:hi], u[at]), hi - lo - 1)
            out[at] = self.rows[lo + pos]
        return out

    def draw(self, weights, n_samples, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw sub-types from per-point weights, then one member of each.

        Args:
            weights: ``[n, n_subtypes]`` sub-type weights.
            n_samples: Draws per point.
            rng: ``np.random.Generator``.

        Returns:
            tuple: ``(subtypes [n, n_samples], member rows [n, n_samples])``.
        """
        w = np.asarray(weights, np.float64)
        n, n_sub = w.shape
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        cum[:, -1] = 1.0
        # Inverse-CDF draws of all rows at once: row i's CDF is shifted into [i, i + 1].
        shift = np.arange(n, dtype=np.float64)[:, None]
        u = rng.random((n, int(n_samples)))
        flat = np.searchsorted((cum + shift).ravel(), (u + shift).ravel(), side="left")
        subtypes = np.clip(
            flat.reshape(n, -1) - np.arange(n)[:, None] * n_sub, 0, n_sub - 1
        )
        return subtypes, self.draw_members(subtypes, rng)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            feature_names=np.asarray(self.feature_names, dtype="U"),
            means=self.means,
            offsets=self.offsets,
            rows=self.rows.astype(np.int32),
            cdf=self.cdf,
        )
        return path

    @classmethod
    def load(cls, path) -> "MemberReadout":
        with np.load(Path(path), allow_pickle=False) as f:
            return cls(
                f["features"],
                f["feature_names"].astype(str).tolist(),
                f["means"],
                f["offsets"],
                f["rows"],
                f["cdf"],
            )
