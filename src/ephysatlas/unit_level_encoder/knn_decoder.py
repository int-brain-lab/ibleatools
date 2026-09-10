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
    """

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
        self.feature_train = np.asarray(waveform_features, np.float32)[self.train_indices]
        self.feature_names = tuple(feature_names or [f"feature_{i}" for i in range(self.feature_train.shape[1])])
        self.k = min(int(k), len(self.train_indices))
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
    ) -> "EmpiricalKNNDecoder":
        obj = cls.__new__(cls)
        obj.z_train = np.asarray(z_train, np.float32)
        obj.feature_train = np.asarray(feature_train, np.float32)
        obj.train_indices = (
            np.arange(len(obj.z_train), dtype=np.int64)
            if train_indices is None
            else np.asarray(train_indices, np.int64)
        )
        obj.feature_names = tuple(feature_names or [f"feature_{i}" for i in range(obj.feature_train.shape[1])])
        obj.k = min(int(k), len(obj.z_train))
        obj._fit_index()
        return obj

    def _fit_index(self) -> None:
        if self.k < 1:
            raise ValueError("EmpiricalKNNDecoder requires at least one TRAIN unit")
        if len(self.z_train) != len(self.feature_train):
            raise ValueError("z_train and feature_train must have the same number of rows")
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto", n_jobs=-1)
        self.nn.fit(self.z_train)

    def save_bank(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_train=self.z_train.astype(np.float32),
            feature_train=self.feature_train.astype(np.float32),
            train_indices=self.train_indices.astype(np.int64),
            feature_names=np.asarray(self.feature_names, dtype="U"),
            k=np.asarray(self.k, dtype=np.int64),
        )
        return path

    @classmethod
    def load_bank(cls, path: Path | str, *, k: int | None = None) -> "EmpiricalKNNDecoder":
        with np.load(Path(path), allow_pickle=False) as bank:
            saved_k = int(np.asarray(bank["k"]).item()) if "k" in bank else 20
            names = bank["feature_names"].astype(str).tolist() if "feature_names" in bank else None
            return cls.from_bank(
                bank["z_train"],
                bank["feature_train"],
                k=saved_k if k is None else int(k),
                train_indices=bank["train_indices"] if "train_indices" in bank else None,
                feature_names=names,
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

    def sample_features(self, z_query: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = self.query(z_query)
        chosen = np.asarray([rng.choice(self.k, p=q.weights[i]) for i in range(len(q.indices))], dtype=np.int64)
        return self.feature_train[q.indices[np.arange(len(q.indices)), chosen]].astype(np.float32)

    def sample_training_indices(self, z_query: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = self.query(z_query)
        chosen = np.asarray([rng.choice(self.k, p=q.weights[i]) for i in range(len(q.indices))], dtype=np.int64)
        local = q.indices[np.arange(len(q.indices)), chosen]
        return self.train_indices[local]

    def distance_summary(self, z_query: np.ndarray) -> dict:
        q = self.query(z_query)
        d1 = q.distances[:, 0]
        dk = q.distances[:, -1]
        effective_n = 1.0 / np.maximum(np.sum(q.weights ** 2, axis=1), 1e-12)
        return {
            "k": int(self.k),
            "nearest_distance_mean": float(np.mean(d1)),
            "nearest_distance_median": float(np.median(d1)),
            "kth_distance_mean": float(np.mean(dk)),
            "kth_distance_median": float(np.median(dk)),
            "effective_neighbors_mean": float(np.mean(effective_n)),
            "effective_neighbors_median": float(np.median(effective_n)),
        }
