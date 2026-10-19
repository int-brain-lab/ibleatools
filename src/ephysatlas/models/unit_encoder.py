"""The serving wrapper for the unit-level encoder family.

The unit-level model describes the spike-sorted units (neurons) found at a brain location:

1. a multimodal autoencoder (:class:`UnitAutoencoder`) embeds a unit's multi-channel waveform,
   3D autocorrelogram and population coupling into a joint latent -- the unit's
   phenotype -- which a train-only scaler standardizes;
2. a hierarchical Gaussian mixture over the standardized latents: putative cell types (a
   full-covariance GMM) and, inside each type, its sub-types (a full-covariance GMM fitted on the
   type's training units). Assignment is top-down: a unit's sub-type always lies inside its type;
3. at a position x, the sub-type weights ``w_kj(x) = gamma_k(x) gamma_j|k(x)`` come from two
   networks of the molecular context (MERFISH + AGEA PCA) at x: one for the types, one for the
   shares of the sub-types within each type;
4. each sub-type is read out through its training members, from anywhere in the brain: their
   posterior-weighted mean waveform features, or one of them for a draw.

So, like the spatial encoder, this family's ``predict`` takes positions and returns the phenotype
expected there: the local sub-type weights mixed with each sub-type's expected features.
``sample`` draws from the same predictive distribution: real training units.

``import torch`` stays inside the methods: the region classifier imports xgboost at module scope
and the two segfault together on macOS arm64, so nothing here may pull torch at import time.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas import model_registry

logger = logging.getLogger(__name__)

ROLE_AUTOENCODER = "autoencoder"
ROLE_CONFIG = "config"
ROLE_SCALER = "scaler"
ROLE_TYPES = "types"
ROLE_SUBTYPES = "subtypes"
ROLE_SUBTYPE_PARENT = "subtype_parent"
ROLE_CONTEXT_TRANSFORM = "context_transform"
ROLE_TYPE_WEIGHTS = "type_weights"
ROLE_SUBTYPE_WEIGHTS = "subtype_weights"
ROLE_MEMBERS = "members"
ROLE_CONTEXT = "context"
ROLE_SPLIT = "split"
ROLE_STATS = "stats"

# The canonical release filenames, used when the manifest omits a role.
DEFAULT_ARTIFACTS = {
    ROLE_AUTOENCODER: model_registry.UNIT_AE_FILE,
    ROLE_CONFIG: model_registry.UNIT_CONFIG_FILE,
    ROLE_SCALER: model_registry.UNIT_SCALER_FILE,
    ROLE_TYPES: model_registry.UNIT_TYPES_FILE,
    ROLE_SUBTYPES: model_registry.UNIT_SUBTYPES_FILE,
    ROLE_SUBTYPE_PARENT: model_registry.UNIT_SUBTYPE_PARENT_FILE,
    ROLE_CONTEXT_TRANSFORM: model_registry.UNIT_CONTEXT_TRANSFORM_FILE,
    ROLE_TYPE_WEIGHTS: model_registry.UNIT_TYPE_WEIGHTS_FILE,
    ROLE_SUBTYPE_WEIGHTS: model_registry.UNIT_SUBTYPE_WEIGHTS_FILE,
    ROLE_MEMBERS: model_registry.UNIT_MEMBERS_FILE,
    ROLE_CONTEXT: list(model_registry.ENCODER_CONTEXT_FILES),
    ROLE_SPLIT: "split.json",
}

# Golden files written at publication and re-checked by selftest().
EXAMPLE_POSITIONS = "example/positions_sample.parquet"
EXAMPLE_PREDICTIONS = "example/expected_predictions.parquet"
EXAMPLE_UNITS = "example/units_sample.npz"
EXAMPLE_LATENTS = "example/expected_latents.npy"


class UnitEncoder:
    """A published unit-level model, ready to predict unit phenotypes at brain positions.

    Attributes:
        path_model (Path): Local model directory.
        index (dict): The publication manifest.
        artifacts (dict): Manifest ``artifacts`` block, completed with the canonical names.
        inputs (dict): Manifest ``inputs`` block.
        outputs (dict): Manifest ``outputs`` block -- the ordered phenotype features predicted.
    """

    def __init__(self, path_model, index: dict = None, device=None):
        self.path_model = Path(path_model)
        self.index = (
            index
            if index is not None
            else model_registry.read_manifest(self.path_model)
        )
        if self.index is None:
            raise FileNotFoundError(
                f"{self.path_model} has no {model_registry.MODEL_MANIFEST_FILE}; the unit encoder "
                f"needs its manifest to locate its checkpoints."
            )
        self.artifacts = {**DEFAULT_ARTIFACTS, **(self.index.get("artifacts") or {})}
        self.inputs = self.index.get("inputs") or {}
        self.outputs = self.index.get("outputs") or {}
        self._device = device
        self._cfg = None
        self._ae = None
        self._scaler = None
        self._mixture = None
        self._context_model = None
        self._readout = None
        self._ctx_manager = None

    # -- lazily loaded pieces --------------------------------------------------------------

    def _artifact_path(self, role: str) -> Path:
        name = self.artifacts.get(role)
        if not name:
            raise FileNotFoundError(
                f"{self.path_model.name} manifest names no {role!r} artifact"
            )
        return self.path_model.joinpath(name)

    @property
    def cfg(self):
        """The released ``Config``, rebuilt from ``config.json`` (paths are runtime defaults)."""
        if self._cfg is None:
            from ephysatlas.unit_level_encoder.config import Config

            device = self._device
            if device is None:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cfg = Config.from_json(
                self._artifact_path(ROLE_CONFIG), device=str(device)
            )
        return self._cfg

    def _load(self):
        """Load every fitted stage on first use."""
        if self._ae is not None:
            return
        import joblib

        from ephysatlas.unit_level_encoder.context_weights import (
            HierarchicalWeightModel,
        )
        from ephysatlas.unit_level_encoder.hierarchy import (
            HierarchicalMixture,
            MemberReadout,
        )
        from ephysatlas.unit_level_encoder.train import load_autoencoder_file

        cfg = self.cfg
        self._ae, _, _ = load_autoencoder_file(
            self._artifact_path(ROLE_AUTOENCODER), cfg
        )
        self._scaler = joblib.load(self._artifact_path(ROLE_SCALER))
        self._mixture = HierarchicalMixture.load(
            types=self._artifact_path(ROLE_TYPES),
            subtypes=self._artifact_path(ROLE_SUBTYPES),
            parent=self._artifact_path(ROLE_SUBTYPE_PARENT),
        )
        self._context_model = HierarchicalWeightModel.load(
            self._mixture.parent,
            cfg,
            transform=self._artifact_path(ROLE_CONTEXT_TRANSFORM),
            types=self._artifact_path(ROLE_TYPE_WEIGHTS),
            subtypes=self._artifact_path(ROLE_SUBTYPE_WEIGHTS),
        )
        self._readout = MemberReadout.load(self._artifact_path(ROLE_MEMBERS))
        logger.info(
            f"loaded unit encoder from {self.path_model.name}: latent_dim={cfg.latent_dim()} "
            f"types={self._mixture.n_types} subtypes={self._mixture.n_subtypes}"
        )

    # -- read-only accessors ---------------------------------------------------------------

    @property
    def autoencoder(self):
        self._load()
        return self._ae

    @property
    def latent_scaler(self):
        self._load()
        return self._scaler

    @property
    def mixture(self):
        """The putative cell types and their sub-types (``HierarchicalMixture``)."""
        self._load()
        return self._mixture

    @property
    def context_model(self):
        """The sub-type weights at raw contexts (``HierarchicalWeightModel``)."""
        self._load()
        return self._context_model

    @property
    def readout(self):
        """The sub-type phenotypes (``MemberReadout``)."""
        self._load()
        return self._readout

    @property
    def features(self) -> list:
        """Ordered names of the phenotype features ``predict`` returns."""
        columns = self.outputs.get("columns")
        if columns:
            return list(columns)
        return list(self.readout.feature_names)

    @property
    def subtype_labels(self) -> list:
        """``"<type>.<sub-type>"`` label of each sub-type, e.g. ``"7.2"``."""
        mixture = self.mixture
        return [f"{k}.{j}" for k, j in zip(mixture.parent, mixture.subtype_index())]

    @property
    def context_dir(self) -> Path:
        """Directory holding the released context volumes (``agea_vol_pca.npy`` etc.)."""
        names = self.artifacts.get(ROLE_CONTEXT) or []
        if not names:
            return self.path_model
        return (self.path_model / names[0]).parent

    def split(self) -> dict:
        """The probe split the model was trained with (``train``/``validation``/``test`` pids)."""
        path = self._artifact_path(ROLE_SPLIT)
        if not path.exists():
            raise FileNotFoundError(f"{self.path_model.name} publishes no {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def preprocessing_stats(self) -> dict:
        """The training-data statistics released with the model (``preprocessing/unit_stats.npz``)."""
        name = self.artifacts.get(ROLE_STATS, model_registry.UNIT_STATS_FILE)
        path = self.path_model.joinpath(name)
        if not path.exists():
            raise FileNotFoundError(f"{self.path_model.name} publishes no {name}")
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    # -- positions -> phenotype -------------------------------------------------------------

    def _context_manager(self):
        """Context sampler over the released PCA volumes (first use downloads the Allen atlas)."""
        if self._ctx_manager is None:
            from ephysatlas.spatial_encoder.utils import (
                AtlasPCAConfig,
                ContextAtlasManager,
            )

            cfg = self.cfg
            self._ctx_manager = ContextAtlasManager(
                AtlasPCAConfig(
                    n_cell_pcs=int(cfg.n_cell_pcs), n_gene_pcs=int(cfg.n_gene_pcs)
                ),
                regenerate_context=False,
                output_dir=self.context_dir,
            )
        return self._ctx_manager

    def _raw_context(self, xyz_m: np.ndarray) -> np.ndarray:
        """``[N, n_cell_pcs + n_gene_pcs]`` molecular context, MERFISH PCs first, as in training."""
        from ephysatlas.unit_level_encoder.data import mirror_xyz_to_hemisphere

        cfg = self.cfg
        xyz = np.asarray(xyz_m, np.float32)
        if bool(cfg.mirror_x_to_single_hemisphere):
            xyz = mirror_xyz_to_hemisphere(xyz, float(cfg.mirror_x_sign))
        pack = self._context_manager().sample_context_numpy_m(xyz, mode="clip")
        cell = np.asarray(pack["cell_pc"], np.float32)[:, : int(cfg.n_cell_pcs)]
        gene = np.asarray(pack["gene_pc"], np.float32)[:, : int(cfg.n_gene_pcs)]
        return np.concatenate([cell, gene], axis=1).astype(np.float32)

    def _coordinates(self, df) -> np.ndarray:
        columns = list(self.inputs.get("columns") or ["x", "y", "z"])
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} coordinate column(s) required by this model are missing from the "
                f"input DataFrame: {missing}. The unit model predicts phenotypes *from* position, "
                f"so it needs {columns} (Allen/IBL frame, metres)."
            )
        return df.loc[:, columns].to_numpy(dtype=np.float32)

    def _context_pc(self, xyz: np.ndarray, batch_size: int):
        """Yield the standardized context of positions, one batch at a time."""
        transform = self._context_model.transform
        for i in range(0, len(xyz), batch_size):
            yield transform.transform(self._raw_context(xyz[i : i + batch_size]))

    def mixture_weights(
        self, df, level: str = "subtype", batch_size: int = 65536
    ) -> pd.DataFrame:
        """Putative cell-type composition at each position.

        Args:
            df (pd.DataFrame): Carries ``x, y, z`` in metres; any index.
            level (str, optional): ``"type"`` for the type weights gamma_k(x), ``"subtype"`` for
                the sub-type weights gamma_k(x) gamma_j|k(x).
            batch_size (int, optional): Positions per forward pass.

        Returns:
            pd.DataFrame: Indexed like ``df``; one ``type_<k>`` column per type, or one
            ``subtype_<k>.<j>`` column per sub-type. Each row sums to one.
        """
        if level not in {"type", "subtype"}:
            raise ValueError("level must be 'type' or 'subtype'")
        self._load()
        xyz = self._coordinates(df)
        model = self._context_model
        fn = model.types.weights_from_pc if level == "type" else model.weights_from_pc
        chunks = [fn(pc) for pc in self._context_pc(xyz, batch_size)]
        if level == "type":
            columns = [f"type_{k}" for k in range(self._mixture.n_types)]
        else:
            columns = [f"subtype_{label}" for label in self.subtype_labels]
        weights = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, len(columns)), np.float32)
        )
        return pd.DataFrame(weights, index=df.index, columns=columns)

    def predict(self, df, batch_size: int = 65536) -> pd.DataFrame:
        """Predict the expected unit phenotype at each position.

        The expected phenotype is the local sub-type weights times each sub-type's expected
        features: the mean features of its training members, weighted by their posterior.
        Deterministic, and exactly what the published unit-level atlas figures map.

        Args:
            df (pd.DataFrame): Carries the coordinate columns the manifest names in
                ``inputs.columns`` (``x, y, z``, metres). Any index; it is preserved.
            batch_size (int, optional): Positions per forward pass.

        Returns:
            pd.DataFrame: Indexed exactly like ``df``, one ``pred_<feature>`` column per phenotype
            feature. The prefix keeps ``df.join(out)`` from colliding with observed features.

        Raises:
            KeyError: If a coordinate column is absent, naming it.
            ValueError: If the manifest's feature list no longer matches its recorded digest.
        """
        features = self.features
        model_registry.validate_feature_order(
            features, self.outputs.get("feature_order_sha256")
        )
        self._load()
        xyz = self._coordinates(df)
        chunks = [
            self._readout.expected(self._context_model.weights_from_pc(pc))
            for pc in self._context_pc(xyz, batch_size)
        ]
        predictions = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, len(features)), np.float32)
        )
        return pd.DataFrame(
            predictions, index=df.index, columns=[f"pred_{f}" for f in features]
        )

    def sample(
        self, df, n_samples: int = 16, seed: int = 0, batch_size: int = 65536
    ) -> pd.DataFrame:
        """Draw unit phenotypes at each position from the model's predictive distribution.

        Each draw picks a sub-type from the local sub-type weights, then one of its training
        members with probability proportional to its posterior of that sub-type: a real training
        unit. The draws average to :meth:`predict`.

        Args:
            df (pd.DataFrame): Carries ``x, y, z`` in metres; any index.
            n_samples (int, optional): Draws per position.
            seed (int, optional): Seed of the draws.
            batch_size (int, optional): Positions per forward pass.

        Returns:
            pd.DataFrame: ``n_samples`` rows per position, indexed like ``df`` with each label
            repeated (``df.index.repeat(n_samples)``): ``sample`` (draw number), ``type`` and
            ``subtype`` (the drawn sub-type, an index into :attr:`subtype_labels`), and one column
            per phenotype feature.
        """
        self._load()
        xyz = self._coordinates(df)
        rng = np.random.default_rng(seed)
        subtypes, rows = [], []
        for pc in self._context_pc(xyz, batch_size):
            s, r = self._readout.draw(
                self._context_model.weights_from_pc(pc), n_samples, rng
            )
            subtypes.append(s.ravel())
            rows.append(r.ravel())
        subtypes = np.concatenate(subtypes) if subtypes else np.zeros(0, np.int64)
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        out = pd.DataFrame(
            self._readout.features[rows],
            index=df.index.repeat(int(n_samples)),
            columns=self.features,
        )
        out.insert(0, "subtype", subtypes)
        out.insert(0, "type", self._mixture.parent[subtypes])
        out.insert(0, "sample", np.tile(np.arange(int(n_samples)), len(xyz)))
        return out

    # -- units -> latent phenotype ----------------------------------------------------------

    def encode(
        self,
        waveform,
        acg=None,
        stpc=None,
        standardize: bool = True,
        batch_size: int = 4096,
    ):
        """Embed units into the joint latent phenotype.

        Args:
            waveform (np.ndarray): ``[N, C, T]`` max-abs normalized multi-channel waveforms, as
                prepared for training (``waveform_shape`` in the config).
            acg (np.ndarray, optional): ``[N, n_bins, n_lags]`` 3D autocorrelograms; required
                when the model uses the ACG modality.
            stpc (np.ndarray, optional): ``[N, n_lags]`` population coupling;
                required when the model uses the population-coupling (stPC) modality.
            standardize (bool, optional): Apply the released latent scaler (the space the mixture
                lives in). Defaults to True.
            batch_size (int, optional): Units per forward pass.

        Returns:
            np.ndarray: ``[N, latent_dim]`` latents.
        """
        import torch

        self._load()
        cfg = self.cfg
        device = torch.device(cfg.device)

        def _tensor(x, name):
            if x is None:
                raise ValueError(
                    f"this model uses the {name} modality; pass {name}=..."
                )
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

        w = _tensor(waveform, "waveform")
        a = _tensor(acg, "acg") if cfg.use_acg else None
        s = _tensor(stpc, "stpc") if cfg.use_stpc else None
        chunks = []
        with torch.no_grad():
            for start in range(0, len(w), batch_size):
                stop = start + batch_size
                lat = self._ae.encode(
                    w[start:stop].to(device),
                    None if a is None else a[start:stop].to(device),
                    None if s is None else s[start:stop].to(device),
                )
                chunks.append(
                    torch.cat([lat[name] for name in cfg.active_modalities()], dim=1)
                    .cpu()
                    .numpy()
                )
        z = np.concatenate(chunks, axis=0).astype(np.float32)
        if standardize:
            z = self._scaler.transform(z).astype(np.float32)
        return z

    def latent_dims(self, modalities) -> np.ndarray:
        """Indices of the latent dimensions of ``modalities`` (e.g. ``("waveform", "acg")``)."""
        cfg = self.cfg
        d = int(cfg.modality_latent_dim)
        active = cfg.active_modalities()
        return np.concatenate(
            [
                np.arange(i * d, (i + 1) * d)
                for i, m in enumerate(active)
                if m in modalities
            ]
        )

    def components(self) -> dict:
        """The types and sub-types: weights, means and covariances, and each sub-type's type."""
        mixture = self.mixture

        def block(gmm):
            return {
                "weights": np.asarray(gmm.weights_, np.float32),
                "means": np.asarray(gmm.means_, np.float32),
                "covariances": np.asarray(gmm.covariances_, np.float32),
            }

        return {
            "types": block(mixture.types),
            "subtypes": block(mixture.subtypes),
            "parent": mixture.parent.copy(),
        }

    def posterior(self, standardized_latents, dims=None):
        """Top-down posteriors ``(P(type | z) [N, K], P(type, sub-type | z) [N, L])``.

        Args:
            standardized_latents: ``[N, latent_dim]`` latents from :meth:`encode`.
            dims: Optional latent dimensions to marginalize onto, e.g.
                ``latent_dims(("waveform", "acg"))`` for units without population coupling.
        """
        return self.mixture.posterior(standardized_latents, dims)

    def assign(self, standardized_latents, dims=None):
        """Top-down hard assignment ``(type [N], sub-type [N])``: the most likely type, then its
        most likely sub-type (an index into :attr:`subtype_labels`)."""
        return self.mixture.assign(standardized_latents, dims)

    # -- the dataset-level view used by training diagnostics and figures --------------------

    def bundle(self, data):
        """A :class:`~ephysatlas.unit_level_encoder.pipeline.UnitModelBundle` over a unit dataset.

        Args:
            data: Prepared unit dataset (``UnitData``), e.g. from ``prepare_unit_data``.
        """
        from ephysatlas.unit_level_encoder.pipeline import UnitModelBundle
        from ephysatlas.unit_level_encoder.train import encode_all

        self._load()
        cfg = self.cfg
        for name, shape in (
            ("waveforms", cfg.waveform_shape),
            ("acgs", cfg.acg_shape),
            ("stpc", cfg.stpc_shape),
        ):
            array = getattr(data, name)
            if array is not None and tuple(array.shape[1:]) != tuple(shape):
                raise ValueError(
                    f"prepared {name} have shape {tuple(array.shape[1:])}, the released model "
                    f"expects {tuple(shape)}"
                )
        # Figures index the model's features and the data's by the same positions.
        if list(data.waveform_feature_names) != list(self._readout.feature_names):
            raise ValueError(
                f"the prepared unit data describes units by {list(data.waveform_feature_names)}, "
                f"the released model predicts {list(self._readout.feature_names)}; use a model "
                "trained with the current unit waveform features"
            )
        latents = encode_all(self._ae, data, cfg)
        z_scaled = self._scaler.transform(latents["joint"]).astype(np.float32)
        return UnitModelBundle(
            cfg=cfg,
            autoencoder=self._ae,
            latent_scaler=self._scaler,
            mixture=self._mixture,
            context_model=self._context_model.over(data.context),
            readout=self._readout,
            data=data,
            latents=latents,
            z_scaled=z_scaled,
            model_root=self.path_model,
        )

    # -- integrity --------------------------------------------------------------------------

    def selftest(self, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
        """Reproduce the shipped golden outputs.

        Checks both halves of the model: phenotype predictions at a sample of positions (context
        sampling, sub-type weights and the members readout), and the latents of a small sample of
        synthetic units (the autoencoder and the scaler). Neither needs IBL data access.
        Tolerances allow the ~1e-4 relative float32 drift between BLAS/library versions; real
        corruption or the wrong model differ by orders of magnitude more.

        Args:
            rtol (float, optional): Relative tolerance.
            atol (float, optional): Absolute tolerance, for near-zero outputs.

        Returns:
            bool: True when every shipped golden output is reproduced.

        Raises:
            FileNotFoundError: If the model ships no golden example.
        """
        positions = self.path_model.joinpath(EXAMPLE_POSITIONS)
        expected = self.path_model.joinpath(EXAMPLE_PREDICTIONS)
        if not (positions.exists() and expected.exists()):
            raise FileNotFoundError(
                f"no example/golden files under {self.path_model / 'example'}"
            )
        got = self.predict(pd.read_parquet(positions))
        golden = pd.read_parquet(expected).loc[:, got.columns].to_numpy(np.float64)
        # Features span many orders of magnitude (seconds vs slopes), so compare each one in
        # units of its own scale: atol then means "a fraction of that feature's range".
        scale = np.maximum(
            np.abs(golden).max(axis=0, keepdims=True), np.finfo(np.float32).tiny
        )
        np.testing.assert_allclose(
            got.to_numpy(np.float64) / scale, golden / scale, rtol=rtol, atol=atol
        )
        units = self.path_model.joinpath(EXAMPLE_UNITS)
        latents = self.path_model.joinpath(EXAMPLE_LATENTS)
        n_units = 0
        if units.exists() and latents.exists():
            with np.load(units, allow_pickle=False) as sample:
                z = self.encode(
                    sample["waveform"],
                    sample["acg"] if "acg" in sample.files else None,
                    sample["stpc"] if "stpc" in sample.files else None,
                )
            np.testing.assert_allclose(z, np.load(latents), rtol=rtol, atol=atol)
            n_units = len(z)
        logger.info(f"selftest passed on {len(got)} positions and {n_units} units")
        return True
