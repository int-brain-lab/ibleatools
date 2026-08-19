"""The neighbour-inpainting spatial encoder, as a published model family.

This model predicts a channel's electrophysiological features from where it sits in the brain:
its anatomical context (MERFISH + AGEA PCA volumes sampled at its coordinates) plus the recorded
features of nearby channels on *other* probes. So unlike the region classifier, its input is a
position and its output is the feature vector -- the opposite direction.

That has a consequence for what must be published. The weights alone are not a usable model: it
also needs the frozen PCA context volumes and a bank of training-channel features to draw
neighbours from, neither of which can be recomputed downstream. Both ship as artifacts.

``torch`` is imported inside the functions that need it. ``regionclassifier`` imports xgboost at
module scope, so putting a torch loader there would make every ``[lite]`` install pay for torch.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas import model_registry

logger = logging.getLogger(__name__)

# Roles this family reads out of the manifest's `artifacts` block.
ROLE_WEIGHTS = "weights"
ROLE_CONTEXT = "context"
ROLE_BANK = "neighbor_bank"
ROLE_CONFIDENCE = "confidence"


def _architecture(path_model: Path, manifest: dict, state: dict) -> dict:
    """Resolve the constructor arguments, preferring the manifest but tolerating its absence.

    Three sources, in order: the manifest's ``config.architecture``; an ``architecture`` key
    inside the checkpoint itself (which ``run_spatial_encoder.py`` writes); and the shapes of the
    normalisation buffers, which pin ``f_ctx`` and ``f_ephys`` even when nothing else says so.
    """
    arch = dict(((manifest or {}).get("config") or {}).get("architecture") or {})
    # A checkpoint saved as {"model_state": ..., "architecture": {...}} carries its own copy.
    arch = {**arch}
    # Buffer shapes are authoritative for the two dimensions they determine: a manifest that
    # disagrees with the weights it ships beside would produce a load error at best.
    if "ctx_mean" in state:
        arch["f_ctx"] = int(state["ctx_mean"].shape[0])
    if "e_mean" in state:
        arch["f_ephys"] = int(state["e_mean"].shape[0])
        arch.setdefault("f_out", int(state["e_mean"].shape[0]))
    missing = [k for k in ("f_ctx", "f_ephys", "f_out") if not arch.get(k)]
    if missing:
        raise ValueError(
            f"{path_model.name}: cannot determine {missing} for the encoder. The manifest's "
            f"config.architecture does not say and the checkpoint has no normalisation buffers."
        )
    return {
        "f_ctx": int(arch["f_ctx"]),
        "f_ephys": int(arch["f_ephys"]),
        "f_out": int(arch["f_out"]),
        "d_model": int(arch.get("d_model", 256)),
        "nhead": int(arch.get("nhead", 8)),
        "depth": int(arch.get("depth", 2)),
        "drop": float(arch.get("drop", 0.1)),
    }


def _load_inpainting_encoder(path_model: Path, manifest: dict = None):
    """Load a ``NeighborInpaintingModel`` from the weights the manifest names.

    Handles both checkpoint shapes in circulation: a bare ``state_dict`` with a sidecar
    ``meta.yaml`` for the architecture, and the wrapped
    ``{"model_state": ..., "architecture": {...}}`` dict.

    The normalisation statistics (``e_mean``/``e_std``/``ctx_mean``/``ctx_std``) are registered
    buffers, so they live *inside* the state dict -- but ``NeighborInpaintingModel.__init__``
    only registers them when it is handed all four. The model is therefore constructed with
    correctly-shaped zeros so the buffers exist, then ``load_state_dict`` overwrites them with
    the real values.

    Args:
        path_model (Path): Model directory.
        manifest (dict, optional): Parsed manifest.

    Returns:
        NeighborInpaintingModel: In ``eval()`` mode, with normalisation buffers populated.

    Raises:
        ValueError: If the architecture cannot be determined, or if the buffers are still zero
            after loading -- which would silently make every prediction wrong.
    """
    import torch

    from ephysatlas.spatial_encoder.model import NeighborInpaintingModel

    path_model = Path(path_model)
    weights = ((manifest or {}).get("artifacts") or {}).get(
        ROLE_WEIGHTS, model_registry.ENCODER_WEIGHTS_FILE
    )
    checkpoint = torch.load(path_model.joinpath(weights), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
        manifest = manifest or {}
        if checkpoint.get("architecture"):
            # Fold the checkpoint's own record in, without letting it override the manifest.
            config = dict(manifest.get("config") or {})
            config["architecture"] = {**checkpoint["architecture"], **(config.get("architecture") or {})}
            manifest = {**manifest, "config": config}
    else:
        state = checkpoint

    arch = _architecture(path_model, manifest, state)
    zeros_e = torch.zeros(arch["f_ephys"])
    zeros_c = torch.zeros(arch["f_ctx"])
    model = NeighborInpaintingModel(
        f_ctx=arch["f_ctx"],
        f_ephys=arch["f_ephys"],
        f_out=arch["f_out"],
        e_mean=zeros_e,
        e_std=zeros_e.clone(),
        ctx_mean=zeros_c,
        ctx_std=zeros_c.clone(),
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        depth=arch["depth"],
        drop=arch["drop"],
    )
    model.load_state_dict(state, strict=True)
    # A silent strict=False, or a checkpoint that never held the buffers, would leave these zero
    # and every unstandardized prediction would collapse to the mean. Fail loudly instead.
    for name in ("e_std", "ctx_std"):
        buffer = getattr(model, name, None)
        if buffer is None or not torch.any(buffer != 0):
            raise ValueError(
                f"{path_model.name}: {name} is missing or all-zero after loading {weights}. "
                f"The checkpoint does not carry its normalisation statistics, so predictions "
                f"cannot be returned in feature units."
            )
    model.eval()
    logger.info(
        f"loaded encoder from {weights}: f_ctx={arch['f_ctx']} f_ephys={arch['f_ephys']} "
        f"d_model={arch['d_model']} depth={arch['depth']}"
    )
    return model


def build_neighbor_bank(path_model: Path, df, manifest: dict, model=None) -> Path:
    """Write the neighbour bank a published encoder needs in order to run at all.

    The bank holds, for every training channel: its position, its **standardised** feature
    vector, and its insertion id. Standardising at build time rather than load time is what
    keeps the bank self-consistent with the weights shipping beside it -- the statistics used are
    the checkpoint's own buffers, not anything recomputed from the caller's data.

    Args:
        path_model (Path): Model directory to write into.
        df (pd.DataFrame): Feature table indexed by ``(pid, channel)``, carrying ``x, y, z`` and
            every column in the manifest's ``outputs.columns``.
        manifest (dict): Parsed manifest, read for the feature list.
        model (optional): A loaded encoder, to take ``e_mean``/``e_std`` from. Loaded from
            ``path_model`` when omitted.

    Returns:
        Path: The written ``neighbor_bank.npz``.
    """
    import torch

    path_model = Path(path_model)
    features = list((manifest.get("outputs") or {}).get("columns") or [])
    missing = [c for c in features + ["x", "y", "z"] if c not in df.columns]
    if missing:
        raise KeyError(f"the bank source table is missing {len(missing)} column(s): {missing}")
    if model is None:
        model = _load_inpainting_encoder(path_model, manifest)

    xyz = df.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
    raw = df.loc[:, features].to_numpy(dtype=np.float32)
    e_mean = model.e_mean.detach().cpu().numpy()
    e_std = model.e_std.detach().cpu().numpy()
    feat = ((raw - e_mean) / e_std).astype(np.float32)
    # Insertion ids are strings; store them as such so a caller can trace a neighbour back.
    pid = df.index.get_level_values(0).to_numpy().astype(str)

    out = path_model.joinpath(model_registry.ENCODER_BANK_FILE)
    np.savez_compressed(out, xyz=xyz, feat=feat, pid=pid)
    logger.info(f"wrote {out} with {xyz.shape[0]} channels from {len(set(pid))} insertions")
    return out


class SpatialEncoder:
    """A published spatial encoder, ready to predict features from position.

    Attributes:
        path_model (Path): Local model directory.
        index (dict): The publication manifest.
        inputs (dict): Manifest ``inputs`` block -- row identity and the required coordinates.
        outputs (dict): Manifest ``outputs`` block -- the ordered feature list this predicts.
        config (dict): Architecture, neighbourhood and context settings.
    """

    def __init__(self, path_model, index: dict = None, device=None):
        self.path_model = Path(path_model)
        self.index = index if index is not None else model_registry.read_manifest(self.path_model)
        if self.index is None:
            raise FileNotFoundError(
                f"{self.path_model} has no {model_registry.MODEL_MANIFEST_FILE}; the encoder has "
                f"no meta.yaml fallback because its manifest is the only record of its "
                f"neighbourhood and context settings."
            )
        self.inputs = self.index.get("inputs") or {}
        self.outputs = self.index.get("outputs") or {}
        self.config = self.index.get("config") or {}
        self._device = device
        self._model = None
        self._ctx_manager = None
        self._bank = None

    # -- lazily built pieces ---------------------------------------------------------------

    @property
    def model(self):
        """The loaded encoder, built on first use."""
        if self._model is None:
            self._model = _load_inpainting_encoder(self.path_model, self.index)
        return self._model

    def preprocessing_stats(self) -> dict:
        """Return the standardisation statistics baked into the checkpoint.

        These are the register-buffers the model ships with -- the feature and context means and
        stds -- exposed for figure code that reconstructs his ``load_channel_preprocessing_stats``
        without going through his release registry.

        Returns:
            dict: ``{"e_mean", "e_std", "ctx_mean", "ctx_std"}`` as numpy arrays.
        """
        model = self.model
        return {
            "e_mean": model.e_mean.detach().cpu().numpy(),
            "e_std": model.e_std.detach().cpu().numpy(),
            "ctx_mean": model.ctx_mean.detach().cpu().numpy(),
            "ctx_std": model.ctx_std.detach().cpu().numpy(),
        }

    def confidence_model(self):
        """Load the probe-confidence model published alongside, if there is one.

        Returns:
            The loaded checkpoint dict, or None when the release does not ship one. Returned raw
            rather than instantiated: it is a separate architecture with its own config, and
            nothing in ``predict`` consumes it.
        """
        import torch

        name = (self.index.get("artifacts") or {}).get(ROLE_CONFIDENCE)
        if not name:
            logger.info(f"{self.path_model.name} publishes no confidence model")
            return None
        return torch.load(
            self.path_model.joinpath(name), map_location="cpu", weights_only=False
        )

    def _context_manager(self):
        """Build the atlas context manager over the *published* PCA volumes.

        Note this triggers a one-off download of the Allen volume from
        ``download.alleninstitute.org`` (public, no account, a few hundred MB) the first time it
        runs on a machine -- ``ContextAtlasManager`` constructs an ``AllenAtlas`` unconditionally.
        """
        if self._ctx_manager is None:
            from ephysatlas.spatial_encoder.utils import AtlasPCAConfig, ContextAtlasManager

            context = self.config.get("context") or {}
            names = (self.index.get("artifacts") or {}).get(ROLE_CONTEXT) or []
            if not names:
                raise FileNotFoundError(
                    f"{self.path_model.name} publishes no context volumes "
                    f"(artifacts.{ROLE_CONTEXT}); the encoder cannot sample anatomy without them"
                )
            cfg = AtlasPCAConfig(
                n_cell_pcs=int(context.get("n_cell_pcs", 50)),
                n_gene_pcs=int(context.get("n_gene_pcs", 50)),
            )
            # regenerate_context=False makes it *load* agea_vol_pca.npy / merfish_vol_pca.npy
            # from output_dir rather than refitting the PCA, which is the whole point of
            # publishing them.
            self._ctx_manager = ContextAtlasManager(
                cfg, regenerate_context=False, output_dir=self.path_model
            )
        return self._ctx_manager

    def _neighbor_bank(self):
        """Load the published neighbour bank."""
        if self._bank is None:
            name = (self.index.get("artifacts") or {}).get(ROLE_BANK)
            if not name:
                raise FileNotFoundError(
                    f"{self.path_model.name} publishes no {ROLE_BANK}; this model predicts from "
                    f"the features of nearby channels, so it cannot run without one"
                )
            with np.load(self.path_model.joinpath(name), allow_pickle=False) as data:
                self._bank = {k: data[k].copy() for k in ("xyz", "feat", "pid")}
        return self._bank

    # -- the pipeline ----------------------------------------------------------------------

    def _standardised_context(self, xyz_m: np.ndarray) -> np.ndarray:
        """Sample anatomical context at each position and standardise it.

        Mirrors the recorded-channel path of ``build_channels_plus_emptyvoxels_with_neighbors``
        exactly, including the detail that channels whose context is all zero -- outside the
        atlas -- are left at zero rather than standardised into a large negative number.
        """
        from ephysatlas.spatial_encoder.utils import mirror_xyz_to_left

        manager = self._context_manager()
        pack = manager.sample_context_numpy_m(mirror_xyz_to_left(xyz_m.copy()), mode="clip")
        ctx = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1).astype(np.float32)

        ctx_mean = self.model.ctx_mean.detach().cpu().numpy()
        ctx_std = self.model.ctx_std.detach().cpu().numpy()
        out = ctx.copy()
        valid = ctx.sum(axis=1) != 0
        out[valid] = (ctx[valid] - ctx_mean) / ctx_std
        return out

    def _neighbours(self, xyz_m: np.ndarray, pids: np.ndarray):
        """Gather the nearest published neighbours for each query position.

        Selection is deterministic nearest-M. The training-time collate drew a *random* subset
        whenever more candidates than ``m_max`` fell inside the radius, which makes predictions
        vary run to run with the global numpy seed -- unacceptable in a published inference API.

        Returns:
            tuple: ``(e_n, p_n, mask)`` as float32/bool arrays shaped ``[N, M, F]``, ``[N, M, 3]``
            and ``[N, M]``.
        """
        from ephysatlas.spatial_encoder.utils import ChannelNN

        bank = self._neighbor_bank()
        neighbourhood = self.config.get("neighbourhood") or {}
        m_max = int(neighbourhood.get("m_max", 64))
        radius_m = float(neighbourhood.get("radius_um", 600.0)) * 1e-6
        allow_same_probe = bool(neighbourhood.get("allow_same_probe", False))

        n = xyz_m.shape[0]
        f_e = bank["feat"].shape[1]
        e_n = np.zeros((n, m_max, f_e), dtype=np.float32)
        p_n = np.zeros((n, m_max, 3), dtype=np.float32)
        mask = np.zeros((n, m_max), dtype=bool)

        nn = ChannelNN(bank["xyz"])
        # k_cap wider than m_max so same-probe exclusion cannot starve the neighbourhood.
        candidates = nn.query_radius(xyz_m.astype(np.float64), r_m=radius_m, k_cap=8 * m_max)
        bank_pid = bank["pid"]
        for i, cand in enumerate(candidates):
            cand = np.asarray(cand, dtype=int)
            if not allow_same_probe and cand.size:
                cand = cand[bank_pid[cand] != str(pids[i])]
            # query_radius returns KD-tree hits already distance-sorted, so the nearest M is a
            # slice. The brute-force fallback is NOT sorted, hence the explicit sort below.
            if cand.size > m_max:
                order = np.argsort(
                    np.sum((bank["xyz"][cand] - xyz_m[i][None, :]) ** 2, axis=1), kind="stable"
                )
                cand = cand[order[:m_max]]
            taken = cand.size
            if taken:
                e_n[i, :taken] = bank["feat"][cand]
                p_n[i, :taken] = bank["xyz"][cand]
                mask[i, :taken] = True
        return e_n, p_n, mask

    def predict(self, df, batch_size: int = 1024) -> pd.DataFrame:
        """Predict electrophysiological features for each channel position.

        Args:
            df (pd.DataFrame): Indexed by ``(pid, channel)``, carrying the coordinate columns the
                manifest names in ``inputs.columns`` (``x, y, z``, in metres). The feature columns
                are *not* read -- they are what this predicts.
            batch_size (int, optional): Rows per forward pass.

        Returns:
            pd.DataFrame: Indexed exactly like ``df``, one ``pred_<feature>`` column per entry in
            ``outputs.columns``, in feature units. The prefix keeps ``df.join(out)`` from
            colliding with the ground-truth columns of the same names.

        Raises:
            KeyError: If a required coordinate column is absent, naming it.
            ValueError: If the manifest's feature list no longer matches its recorded digest.
        """
        import torch

        from ephysatlas.spatial_encoder.model import unstandardize

        columns = list(self.inputs.get("columns") or ["x", "y", "z"])
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} coordinate column(s) required by this model are missing from "
                f"the input DataFrame: {missing}. This model predicts features *from* position, "
                f"so it needs {columns}, not the feature columns themselves."
            )
        features = list(self.outputs.get("columns") or [])
        model_registry.validate_feature_order(
            features, self.outputs.get("feature_order_sha256")
        )

        xyz = df.loc[:, columns].to_numpy(dtype=np.float32)
        pids = df.index.get_level_values(0).to_numpy().astype(str)
        ctx = self._standardised_context(xyz)
        e_n, p_n, mask = self._neighbours(xyz, pids)

        model = self.model
        device = torch.device(self._device) if self._device else next(model.parameters()).device
        model = model.to(device)
        chunks = []
        with torch.no_grad():
            for start in range(0, xyz.shape[0], batch_size):
                stop = start + batch_size
                _, mu = model(
                    torch.from_numpy(ctx[start:stop]).to(device),
                    torch.from_numpy(xyz[start:stop]).to(device),
                    torch.from_numpy(e_n[start:stop]).to(device),
                    torch.from_numpy(p_n[start:stop]).to(device),
                    torch.from_numpy(mask[start:stop]).to(device),
                )
                # Back into feature units with the statistics the checkpoint shipped.
                chunks.append(
                    unstandardize(mu.float(), model.e_mean, model.e_std).cpu().numpy()
                )
        predictions = np.concatenate(chunks, axis=0)
        return pd.DataFrame(
            predictions, index=df.index, columns=[f"pred_{f}" for f in features]
        )

    def selftest(self, rtol: float = 1e-5) -> bool:
        """Reproduce the shipped golden predictions, if the model ships an example.

        Args:
            rtol (float, optional): Relative tolerance on the comparison.

        Returns:
            bool: True when the recomputed predictions match the shipped ones.

        Raises:
            FileNotFoundError: If the model does not ship ``example/`` files.
        """
        example = self.path_model.joinpath("example")
        sample_file = example.joinpath("features_sample.parquet")
        expected_file = example.joinpath("expected_predictions.parquet")
        if not (sample_file.exists() and expected_file.exists()):
            raise FileNotFoundError(f"no example/golden files under {example}")
        got = self.predict(pd.read_parquet(sample_file))
        expected = pd.read_parquet(expected_file)
        np.testing.assert_allclose(
            got.to_numpy(), expected.loc[:, got.columns].to_numpy(), rtol=rtol
        )
        logger.info(f"selftest passed on {len(got)} channels")
        return True
