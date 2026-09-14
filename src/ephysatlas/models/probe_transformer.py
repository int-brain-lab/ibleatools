"""The ProbeTransformer channel-region classifier, as a published model family.

This model predicts a brain region for every channel of a probe at once: a transformer with
rotary positional embeddings attends across the whole probe, using each channel's physical depth
(``axial_um``) as the positional signal. A release ships one trained transformer per random seed
(``artifacts.seeds``); :meth:`ProbeTransformerClassifier.predict` averages their per-class
probabilities, so the ensemble is the default and a single seed is the cheaper ``"global"`` mode.

The architecture (:class:`ProbeTransformer` and its RoPE stack) is the single source of truth for
both inference here and training in ``paper-ephys-atlas``, which imports it from this module.

``torch`` is imported at module scope, which is safe because this module is only reached through
the lazy dispatch builder in :mod:`ephysatlas.models`: ``regionclassifier`` imports xgboost at
module scope, and the two runtimes segfault together on macOS arm64.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ephysatlas import model_registry

logger = logging.getLogger(__name__)


# -- architecture (ported verbatim from paper's spike_ephys.localizer.end_to_end) --------------


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) using physical channel positions.

    Encodes the depth (z) position of each electrode into rotation
    matrices applied to query/key vectors in self-attention.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        """
        positions: (B, L) physical positions (e.g. depth in µm)
        Returns: cos, sin each of shape (B, L, dim//2)
        """
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    """Apply rotary embedding to tensor x. Splits last dim into pairs and rotates."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE on Q and K."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, positions, mask=None):
        """
        x: (B, L, D), positions: (B, L), mask: (B, L) bool True=padded
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, head_dim)

        cos, sin = self.rope(positions)  # (B, L, head_dim//2)
        cos = cos.unsqueeze(2)  # (B, L, 1, head_dim//2)
        sin = sin.unsqueeze(2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with RoPE attention."""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, positions, mask=None):
        x = x + self.attn(self.norm1(x), positions, mask)
        x = x + self.ff(self.norm2(x))
        return x


class ProbeTransformer(nn.Module):
    """Transformer classifier over full probes with RoPE positional encoding.

    Uses physical channel depth positions (from probe geometry) as the
    positional signal via Rotary Positional Embeddings, so the model
    learns attention patterns based on actual spatial distances.

    Input: (B, L, F) features + (B, L) depth positions
    Output: (B, L, C) region logits, (B, L, n_merfish) MERFISH logits or None
    """

    def __init__(
        self,
        n_features,
        n_classes,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        n_merfish=0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.merfish_head = nn.Linear(d_model, n_merfish) if n_merfish > 0 else None

    def forward(self, x, positions, mask=None):
        """
        x: (B, L, F) probe features
        positions: (B, L) channel depth in µm
        mask: (B, L) bool, True = padded/invalid
        Returns: (B, L, C) region logits, (B, L, n_merfish) MERFISH logits or None
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, positions, mask)
        x = self.norm(x)
        region_logits = self.head(x)
        merfish_logits = self.merfish_head(x) if self.merfish_head is not None else None
        return region_logits, merfish_logits


# -- serving wrapper ---------------------------------------------------------------------------


class ProbeTransformerClassifier:
    """A published ProbeTransformer ensemble, ready to predict a region per channel.

    A release ships one transformer per random seed under ``artifacts.seeds`` (weights +
    per-seed feature scaler); the single ``config.model_config`` and ``inputs.features`` describe
    them all. ``predict`` standardises with each seed's own scaler, runs one probe at a time, and
    averages the per-class probabilities.

    Attributes:
        path_model (Path): Local model directory.
        index (dict): Contents of ``ephysatlas_model.json`` (the publication manifest).
        config (dict): Manifest ``config`` block -- class ids, acronyms, region map, model config.
        inputs (dict): Manifest ``inputs`` block -- feature list and the position column name.
    """

    def __init__(self, path_model, index: dict = None):
        self.path_model = Path(path_model)
        self.index = (
            index
            if index is not None
            else model_registry.read_manifest(self.path_model)
        )
        if self.index is None:
            raise FileNotFoundError(
                f"{self.path_model} has no {model_registry.MODEL_MANIFEST_FILE}"
            )
        self.config = self.index["config"]
        self.inputs = self.index["inputs"]
        self.artifacts = self.index.get("artifacts") or {}

    def _seed_dirs(self, estimator: str) -> list:
        """Directories to average over: every seed for ``ensemble``, the first for ``global``."""
        seeds = self.artifacts.get("seeds") or []
        dirs = [self.path_model.joinpath(s) for s in seeds] or [self.path_model]
        if estimator == "global":
            return dirs[:1]
        if estimator == "ensemble":
            return dirs
        raise ValueError(
            f"unknown estimator {estimator!r}; expected 'ensemble' or 'global'"
        )

    def _load_seed(self, seed_dir: Path):
        """Build the transformer for one seed and return it with its feature scaler mean/std."""
        mc = self.config["model_config"]
        model = ProbeTransformer(
            n_features=mc["n_features"],
            n_classes=mc["n_classes"],
            d_model=mc["d_model"],
            n_heads=mc["n_heads"],
            n_layers=mc["n_transformer_layers"],
            dropout=mc["dropout"],
            n_merfish=mc.get("n_merfish", 0),
        )
        weights = self.artifacts.get("weights", "best_model.pt")
        model.load_state_dict(
            torch.load(seed_dir.joinpath(weights), map_location="cpu")
        )
        model.eval()
        sc = np.load(
            seed_dir.joinpath(self.artifacts.get("scaler", "feature_scaler.npz"))
        )
        return model, sc["mean"], sc["std"]

    def _seed_probas(self, seed_dir, x, positions, pids) -> np.ndarray:
        """Per-channel class probabilities from one seed: standardise, then one probe at a time."""
        model, mean, std = self._load_seed(seed_dir)
        temperature = self.config["model_config"].get("temperature", 1.0)
        xs = (x - mean) / std
        probs = np.zeros((x.shape[0], len(self.config["classes"])), dtype=np.float32)
        with torch.no_grad():
            for pid in np.unique(pids):
                m = pids == pid
                # B=1, whole probe at once: nothing is padded, so no mask is needed.
                logits, _ = model(
                    torch.tensor(xs[m], dtype=torch.float32).unsqueeze(0),
                    torch.tensor(positions[m], dtype=torch.float32).unsqueeze(0),
                )
                probs[m] = torch.softmax(logits[0] / temperature, dim=-1).numpy()
        return probs

    def predict(self, df, estimator: str = "ensemble") -> pd.DataFrame:
        """Predict a brain region per channel.

        Args:
            df (pd.DataFrame): Features, indexed by ``(pid, channel)``. Must contain every column
                in the manifest's ``features`` plus the position column (``axial_um``). Channels
                with any non-finite feature are dropped, so the result may be a subset of ``df``.
            estimator (str, optional): ``"ensemble"`` (default) averages every seed and reports
                ``seed_agreement``; ``"global"`` uses the first seed alone, at a fraction of the
                cost, with ``seed_agreement`` returned as NaN.

        Returns:
            pd.DataFrame: Indexed like the surviving rows of ``df``, with ``predicted_acronym``,
            ``predicted_atlas_id``, ``prediction_probability`` (probability of the winning class),
            ``seed_agreement`` (fraction of seeds voting for the winner), and one ``p_<acronym>``
            column per class. Prediction columns are namespaced so ``df.join(out)`` cannot collide
            with histology-derived ``acronym``/``atlas_id`` columns.

        Raises:
            KeyError: If a required feature or the position column is absent, naming it.
            ValueError: If the manifest's feature list no longer matches its recorded digest.
        """
        feats = list(self.inputs["features"])
        position_column = self.inputs.get("position_column", "axial_um")
        missing = [c for c in feats + [position_column] if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} column(s) required by this model are missing from the input "
                f"DataFrame: {missing}. Expected the features plus the {position_column!r} "
                f"position column."
            )
        model_registry.validate_feature_order(
            feats, self.inputs.get("feature_order_sha256")
        )

        x = df.loc[:, feats].to_numpy(dtype=float)
        finite = np.isfinite(x).all(axis=1)
        if not finite.all():
            logger.warning(
                "dropping %d of %d channels with non-finite features",
                (~finite).sum(),
                len(df),
            )
            df, x = df[finite], x[finite]
        positions = df[position_column].to_numpy(dtype=float)
        pids = df.index.get_level_values(0).to_numpy()

        probas = np.stack(
            [
                self._seed_probas(d, x, positions, pids)
                for d in self._seed_dirs(estimator)
            ]
        )
        mean_probas = probas.mean(axis=0)
        winner = np.argmax(mean_probas, axis=1)
        if probas.shape[0] > 1:
            agreement = (np.argmax(probas, axis=2) == winner[np.newaxis, :]).mean(
                axis=0
            )
        else:
            # A single model has nothing to agree with. NaN rather than a false unanimity.
            agreement = np.full(winner.size, np.nan)

        ids = np.asarray(self.config["classes"])
        acronyms = np.asarray(self.config["class_acronyms"])
        out = pd.DataFrame(index=df.index)
        out["predicted_acronym"] = acronyms[winner]
        out["predicted_atlas_id"] = ids[winner]
        out["prediction_probability"] = mean_probas[np.arange(winner.size), winner]
        out["seed_agreement"] = agreement
        for j, acronym in enumerate(acronyms):
            out[f"p_{acronym}"] = mean_probas[:, j]
        return out

    def selftest(self, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
        """Reproduce the shipped golden predictions, if the model ships an example.

        Turns silent numerical drift (a changed scaler, an incompatible torch, a reordered class
        vector) into one explicit failure. The tolerances are loose because these are float32
        transformer outputs that drift ~1e-4 across environments -- a genuine reproduction.

        Raises:
            FileNotFoundError: If the model does not ship ``example/`` files.
            AssertionError: If the predictions differ.
        """
        example = self.path_model.joinpath("example")
        sample_file = example.joinpath("features_sample.parquet")
        expected_file = example.joinpath("expected_predictions.parquet")
        if not (sample_file.exists() and expected_file.exists()):
            raise FileNotFoundError(f"no example/golden files under {example}")
        got = self.predict(pd.read_parquet(sample_file), estimator="ensemble")
        expected = pd.read_parquet(expected_file)
        mismatched = (
            got["predicted_acronym"].values != expected["predicted_acronym"].values
        ).sum()
        assert mismatched == 0, f"{mismatched} of {len(got)} predicted acronyms differ"
        np.testing.assert_allclose(
            got["prediction_probability"].values,
            expected["prediction_probability"].values,
            rtol=rtol,
            atol=atol,
        )
        logger.info(f"selftest passed on {len(got)} channels")
        return True
