"""The serving wrapper for the unit-level encoder family.

The unit-level encoder is a two-stage model over spike-sorted units, not channels:

1. a :class:`MultimodalAutoencoder` embeds a unit's multi-channel waveform and its
   autocorrelogram into a shared 32-d latent -- the unit's phenotype;
2. a :class:`PointTransformerGMM` is a Gaussian mixture over those latents, whose components read
   as *putative cell types*.

This wrapper serves, from published files, the four operations the family supports: encode units
to their latent, reconstruct waveform/ACG, expose the GMM components, and assign each unit to its
nearest component.

``import torch`` stays inside the methods: the region classifier imports xgboost at module scope
and the two segfault together on macOS arm64, so nothing here may pull torch at import time.
"""

import logging
from dataclasses import fields
from pathlib import Path

import numpy as np

from ephysatlas import model_registry

logger = logging.getLogger(__name__)

ROLE_AUTOENCODER = "autoencoder"
ROLE_PT_GMM = "pt_gmm"
ROLE_SCALER = "scaler"

# The canonical checkpoint filenames, used when the manifest omits a role.
DEFAULT_ARTIFACTS = {
    ROLE_AUTOENCODER: "autoencoder.pt",
    ROLE_PT_GMM: "point_transformer_gmm.pt",
    ROLE_SCALER: "shared_latent_scaler.joblib",
}


def _config_from_payload(payload: dict):
    """Rebuild a training ``Config`` from a checkpoint's saved ``config`` dict.

    The checkpoint is self-describing -- ``train_autoencoder`` saves ``asdict(cfg)`` -- so the
    architecture is restored from the weights file itself, not guessed. Runtime-only fields
    (``device``/``output_dir``) are left at their defaults.
    """
    from ephysatlas.unit_level_encoder.config import Config

    cfg = Config()
    valid = {f.name for f in fields(Config)}
    tuple_fields = {"waveform_shape", "acg_shape", "cosmos_region_names"}
    for key, value in (payload or {}).items():
        if key in valid and key not in {"device", "output_dir"}:
            setattr(cfg, key, tuple(value) if key in tuple_fields else value)
    return cfg


class UnitEncoder:
    """A published unit-level encoder, ready to encode, reconstruct and cluster units.

    Attributes:
        path_model (Path): Local model directory.
        index (dict): The publication manifest.
        artifacts (dict): Manifest ``artifacts`` block -- the two checkpoints and the scaler.
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
                f"needs its manifest to locate the autoencoder, the GMM and the latent scaler."
            )
        self.artifacts = {**DEFAULT_ARTIFACTS, **(self.index.get("artifacts") or {})}
        self._device = device
        self._cfg = None
        self._model_ae = None
        self._model_gmm = None
        self._scaler = None
        self._atlas = None

    # -- lazily built pieces ---------------------------------------------------------------

    def _artifact_path(self, role: str) -> Path:
        name = self.artifacts.get(role)
        if not name:
            raise FileNotFoundError(
                f"{self.path_model.name} manifest names no {role!r} artifact"
            )
        return self.path_model.joinpath(name)

    def _load(self):
        """Load the autoencoder, the PT-GMM and the scaler on first use."""
        if self._model_ae is not None:
            return
        import joblib
        import torch

        from ephysatlas.unit_level_encoder.gmm_models import PointTransformerGMM
        from ephysatlas.unit_level_encoder.model import MultimodalAutoencoder

        device = self._device or "cpu"

        ae_payload = torch.load(
            self._artifact_path(ROLE_AUTOENCODER),
            map_location=device,
            weights_only=False,
        )
        self._cfg = _config_from_payload(ae_payload.get("config"))
        self._cfg.device = device
        model_ae = MultimodalAutoencoder(self._cfg).to(device)
        model_ae.load_state_dict(ae_payload["model_state_dict"], strict=True)
        model_ae.eval()
        self._model_ae = model_ae

        gmm_payload = torch.load(
            self._artifact_path(ROLE_PT_GMM), map_location=device, weights_only=False
        )
        model_gmm = PointTransformerGMM(
            int(gmm_payload["latent_dim"]),
            int(gmm_payload["context_dim"]),
            int(gmm_payload["n_components"]),
            self._cfg,
        ).to(device)
        model_gmm.load_state_dict(gmm_payload["model_state_dict"], strict=True)
        model_gmm.eval()
        self._model_gmm = model_gmm

        self._scaler = joblib.load(self._artifact_path(ROLE_SCALER))
        logger.info(
            f"loaded unit encoder from {self.path_model.name}: latent_dim={gmm_payload['latent_dim']} "
            f"components={gmm_payload['n_components']}"
        )

    def _normalise(self, waveform, acg):
        """Apply the training-time preprocessing so inputs match the encoder's distribution."""
        from ephysatlas.unit_level_encoder.data import (
            normalize_acgs,
            normalize_waveforms,
        )

        w = normalize_waveforms(np.asarray(waveform, dtype=np.float32))
        a = normalize_acgs(np.asarray(acg, dtype=np.float32))
        return w, a

    # -- read-only accessors, for callers that need the loaded pieces directly --------------

    @property
    def cfg(self):
        """The training ``Config`` restored from the checkpoint (loads on first use)."""
        self._load()
        return self._cfg

    @property
    def model_ae(self):
        """The loaded :class:`MultimodalAutoencoder`."""
        self._load()
        return self._model_ae

    @property
    def model_gmm(self):
        """The loaded :class:`PointTransformerGMM`."""
        self._load()
        return self._model_gmm

    @property
    def scaler(self):
        """The loaded latent ``StandardScaler``."""
        self._load()
        return self._scaler

    def atlas_arrays(self, cache_dir=None):
        """Public accessor for the cached atlas arrays. See :meth:`_atlas_arrays`."""
        return self._atlas_arrays(cache_dir)

    # -- the operations this family supports -----------------------------------------------

    def encode(self, waveform, acg, standardize: bool = False, batch_size: int = 4096):
        """Embed each unit's waveform + ACG into its 32-d latent phenotype.

        Args:
            waveform (np.ndarray): ``[N, C, T]`` multi-channel waveforms.
            acg (np.ndarray): ``[N, n_bins, n_lags]`` autocorrelograms.
            standardize (bool, optional): If True, apply the published latent scaler, giving the
                standardised latent the GMM operates on (what :meth:`assign` expects).
            batch_size (int, optional): Units per forward pass.

        Returns:
            np.ndarray: ``[N, latent_dim]`` latents, one per unit.
        """
        import torch

        self._load()
        w, a = self._normalise(waveform, acg)
        device = torch.device(self._cfg.device)
        chunks = []
        with torch.no_grad():
            for start in range(0, w.shape[0], batch_size):
                stop = start + batch_size
                enc = self._model_ae.encode(
                    torch.from_numpy(w[start:stop]).to(device),
                    torch.from_numpy(a[start:stop]).to(device),
                )
                chunks.append(enc["z_unit_shared"].cpu().numpy())
        z = np.concatenate(chunks, axis=0)
        if standardize:
            z = self._scaler.transform(z).astype(np.float32)
        return z

    def reconstruct(self, waveform, acg):
        """Reconstruct each unit's waveform and ACG through the autoencoder.

        Returns:
            dict: ``{"waveform": [N, C, T], "acg": [N, n_bins, n_lags]}`` reconstructions.
        """
        import torch

        self._load()
        w, a = self._normalise(waveform, acg)
        device = torch.device(self._cfg.device)
        with torch.no_grad():
            enc = self._model_ae.encode(
                torch.from_numpy(w).to(device), torch.from_numpy(a).to(device)
            )
            rec = self._model_ae.reconstruct(enc)
        return {
            "waveform": rec["waveform_reconstruction"].cpu().numpy(),
            "acg": rec["acg_reconstruction"].cpu().numpy(),
        }

    def components(self):
        """Return the GMM's putative-cell-type components.

        Returns:
            tuple: ``(means, log_var)``, each ``[n_components, latent_dim]``.
        """
        self._load()
        return (
            self._model_gmm.means.detach().cpu().numpy(),
            self._model_gmm.log_var.detach().cpu().numpy(),
        )

    def assign(self, standardized_latents):
        """Hard-assign each standardised latent to its nearest GMM component.

        Args:
            standardized_latents (np.ndarray): ``[N, latent_dim]`` latents, as returned by
                :meth:`encode` with ``standardize=True``.

        Returns:
            np.ndarray: ``[N]`` component index per unit.
        """
        import torch

        from ephysatlas.unit_level_encoder.gmm_models import diag_log_prob

        self._load()
        z = torch.from_numpy(np.asarray(standardized_latents, dtype=np.float32))
        log_prob = diag_log_prob(
            z,
            self._model_gmm.means.detach().cpu(),
            self._model_gmm.log_var.detach().cpu(),
        )
        return log_prob.argmax(dim=1).cpu().numpy()

    # -- the atlas dataset, read from a local cache (never republished on the Hub) ----------

    _ATLAS_ARRAYS = ("waveforms", "acgs", "ctx", "xyz", "pids")

    def _atlas_arrays(self, cache_dir=None):
        """Load the published atlas's per-unit arrays from a local cache.

        Unlike the classifier and spatial encoder, this family's recorded dataset is **not**
        shipped on the Hub -- it stays under IBL's existing S3 access controls, so the model
        download carries only weights. *Preparing* those arrays is training machinery that lives
        outside this package; here the prepared cache is only read, never written or fetched.

        Only the atlas-wide operations (:meth:`latents`) need this. ``encode``/``reconstruct``/
        ``components``/``assign`` run on the caller's own units and never trigger it.

        Args:
            cache_dir (Path, optional): Where the prepared arrays live. Defaults to
                ``~/.ephysatlas/unit_data``.

        Returns:
            dict: ``{name: np.ndarray}`` for each of ``waveforms, acgs, ctx, xyz, pids``.

        Raises:
            FileNotFoundError: If the arrays are not cached, with the command that builds them.
        """
        if self._atlas is not None:
            return self._atlas
        cache = (
            Path(cache_dir)
            if cache_dir
            else Path.home().joinpath(".ephysatlas", "unit_data")
        )
        arrays_dir = cache.joinpath("arrays")
        if not arrays_dir.joinpath("waveforms.npy").exists():
            raise FileNotFoundError(
                f"{self.path_model.name}: atlas dataset not cached under {arrays_dir}. Prepare it "
                f"from paper-ephys-atlas with `python -m "
                f"ephys_atlas.unit_level_encoder.prepare_latest_cells_encoder_data "
                f"--out-dir {arrays_dir}` (needs ONE/S3, multi-GB)."
            )
        self._atlas = {
            name: np.load(
                arrays_dir.joinpath(f"{name}.npy"), allow_pickle=(name == "pids")
            )
            for name in self._ATLAS_ARRAYS
        }
        return self._atlas

    def latents(self, cache_dir=None):
        """Encode every atlas unit to its standardised latent.

        Reads the cached atlas dataset via :meth:`_atlas_arrays`, then applies the same
        encode + scaler as :meth:`encode`.

        Returns:
            np.ndarray: ``[n_units, latent_dim]`` standardised latents for the published units.
        """
        atlas = self._atlas_arrays(cache_dir)
        return self.encode(atlas["waveforms"], atlas["acgs"], standardize=True)

    def selftest(self, rtol: float = 1e-4) -> bool:
        """Reproduce the shipped golden latents, if the model ships an example.

        Encode-based, deliberately: encoding is the family's primary operation and, unlike the
        atlas-wide ``latents``, needs only the weights -- so the self-test needs no cached atlas.

        Args:
            rtol (float, optional): Relative tolerance on the comparison.

        Returns:
            bool: True when the recomputed latents match the shipped ones.

        Raises:
            FileNotFoundError: If the model does not ship ``example/`` files.
        """
        example = self.path_model.joinpath("example")
        sample_file = example.joinpath("units_sample.npz")
        expected_file = example.joinpath("expected_latents.npy")
        if not (sample_file.exists() and expected_file.exists()):
            raise FileNotFoundError(f"no example/golden files under {example}")
        sample = np.load(sample_file)
        got = self.encode(sample["waveform"], sample["acg"])
        np.testing.assert_allclose(got, np.load(expected_file), rtol=rtol)
        logger.info(f"selftest passed on {got.shape[0]} units")
        return True
