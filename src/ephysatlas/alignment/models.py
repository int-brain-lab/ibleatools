"""The released models the alignment methods run on, loaded from the Hugging Face Hub.

:class:`ChannelModel` wraps the published channel-level interpolation model
(``int-brain-lab/ea-encoder-channel``) with its probe-confidence model. It only needs the release
itself -- context volumes, neighbour bank and statistics ship with it -- so the alignment GUI can
use it without any recording data. :class:`UnitModel` wraps the unit-level model and the prepared
unit dataset, for the channel + unit cost of the offline runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np

CHANNEL_MODEL_REPO_ID = "int-brain-lab/ea-encoder-channel"
UNIT_MODEL_REPO_ID = "int-brain-lab/ea-encoder-unit"
VINTAGE = "2026_W39"


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ChannelModel:
    """The published channel-level model, with array-level helpers used by every alignment method.

    Args:
        repo_id (str, optional): Hugging Face repo id or a local release directory.
        revision (str, optional): Release tag.
        device (str, optional): Torch device; auto-detected when None.
    """

    def __init__(
        self,
        repo_id: str = CHANNEL_MODEL_REPO_ID,
        revision: str = VINTAGE,
        device: Optional[str] = None,
    ):
        from ephysatlas import load_pretrained

        self.repo_id = str(repo_id)
        self.revision = str(revision)
        self.device = device or default_device()
        self.encoder = load_pretrained(self.repo_id, revision=self.revision, device=self.device)
        self.model = self.encoder.model.to(self.device).eval()
        self.features = list(self.encoder.features)
        self.e_mean = self.model.e_mean.detach().cpu().numpy().astype(np.float64)
        self.e_std = self.model.e_std.detach().cpu().numpy().astype(np.float64)

    @cached_property
    def conf_model(self):
        """The probe-confidence model shipped with the release (None when it ships none)."""
        conf = self.encoder.load_confidence_model()
        return None if conf is None else conf.to(self.device).eval()

    @property
    def model_commit(self) -> str:
        """The Hub snapshot the model was loaded from (its directory name), or ''."""
        return Path(self.encoder.path_model).name

    def split(self) -> dict:
        """The release's probe split (``train_pids``, ``validation_pids``, ``test_pids``)."""
        return self.encoder.split()

    def standardize(self, features: np.ndarray) -> np.ndarray:
        """Recorded features in model units (NaN/inf set to 0), float64."""
        x = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return (x - self.e_mean) / (self.e_std + 1e-8)

    def unstandardize(self, features_std: np.ndarray) -> np.ndarray:
        """Model units back to feature units."""
        return np.asarray(features_std, dtype=np.float64) * (self.e_std + 1e-8) + self.e_mean

    def predict_std(self, xyz_m: np.ndarray, pid: str = "", batch_size: int = 2048) -> np.ndarray:
        """``[N, F]`` standardised predictions at ``xyz_m``, excluding ``pid``'s own neighbours."""
        return self.encoder.predict_xyz(
            xyz_m, pid, batch_size=batch_size, standardized=True
        ).astype(np.float64)

    def context_std(self, xyz_m: np.ndarray) -> np.ndarray:
        """``[N, F_ctx]`` standardised molecular context (all zero outside the atlas)."""
        from ephysatlas.spatial_encoder.utils import mirror_xyz_to_left

        xyz = mirror_xyz_to_left(np.asarray(xyz_m, dtype=np.float32).reshape(-1, 3).copy())
        return self.encoder._standardised_context(xyz)

    def context_raw(self, xyz_m: np.ndarray) -> np.ndarray:
        """``[N, F_ctx]`` raw molecular context (MERFISH PCs then AGEA PCs), as the unit model uses."""
        from ephysatlas.spatial_encoder.utils import mirror_xyz_to_left

        xyz = mirror_xyz_to_left(np.asarray(xyz_m, dtype=np.float32).reshape(-1, 3).copy())
        pack = self.encoder._context_manager().sample_context_numpy_m(xyz, mode="clip")
        return np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1).astype(np.float32)

    def confidence(
        self, recorded: np.ndarray, channel_xyz: np.ndarray, predicted_std: np.ndarray
    ) -> np.ndarray:
        """``[C]`` p(good) of each channel of a probe placed at ``channel_xyz``.

        The released probe-confidence transformer reads, along the probe, the recorded and
        predicted (standardised) features and the context at each channel; class 0 is "good".
        Invalid channels (no signal, non-finite position or prediction) are NaN. All NaN when the
        release ships no confidence model.
        """
        import torch

        recorded = np.asarray(recorded, dtype=np.float64)
        xyz = np.asarray(channel_xyz, dtype=np.float64)
        pred = np.asarray(predicted_std, dtype=np.float64)
        p_good = np.full(len(recorded), np.nan)
        if self.conf_model is None:
            return p_good
        valid = (
            np.isfinite(recorded).all(axis=1)
            & ~np.all(np.nan_to_num(recorded) == 0.0, axis=1)
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(pred).all(axis=1)
        )
        if not valid.any():
            return p_good
        rec_std = self.standardize(recorded)
        rec_std[~valid] = 0.0
        pred = np.nan_to_num(pred).copy()
        pred[~valid] = 0.0
        ctx = self.context_std(np.nan_to_num(xyz))
        ctx[~valid] = 0.0
        with torch.no_grad():
            logits = self.conf_model(
                rec=torch.from_numpy(rec_std[None].astype(np.float32)).to(self.device),
                pred=torch.from_numpy(pred[None].astype(np.float32)).to(self.device),
                ctx=torch.from_numpy(ctx[None].astype(np.float32)).to(self.device),
                valid=torch.from_numpy(valid[None]).to(self.device),
            )[0]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        p_good[valid] = probs[valid, 0]
        return p_good


@dataclass
class UnitModel:
    """The released unit-level model over the prepared unit dataset (offline runs only).

    ``z_scaled`` are the standardised 60-d latents of every prepared unit, ``pids`` their
    insertion and ``axial_um`` their physical NP1 axial position (µm from the tip) from the spike
    sorter -- never the human alignment.
    """

    bundle: object
    z_scaled: np.ndarray
    pids: np.ndarray
    axial_um: np.ndarray

    @classmethod
    def load(
        cls,
        repo_id: str = UNIT_MODEL_REPO_ID,
        revision: str = VINTAGE,
        device: Optional[str] = None,
        prepared_data_dir: Optional[Path] = None,
    ) -> "UnitModel":
        """Load the release and its prepared unit data (prepared from IBL S3 when missing)."""
        from ephysatlas.unit_level_encoder import Config, load_unit_model

        cfg = Config(vintage=revision, device=device or default_device())
        if prepared_data_dir is not None:
            cfg.prepared_data_dir = Path(prepared_data_dir)
        if Path(str(repo_id)).is_dir():
            bundle = load_unit_model(cfg, source="release", release_dir=repo_id)
        else:
            bundle = load_unit_model(cfg, source="hub", repo_id=repo_id, revision=revision)
        depth_path = Path(bundle.cfg.prepared_data_dir) / "unit_local_axial_um_np1_geometry_v2.npy"
        if not depth_path.exists():
            raise FileNotFoundError(
                f"{depth_path} is missing: the unit cost needs each unit's sorter-local axial "
                "position (cluster table axial_um)."
            )
        axial = np.load(depth_path).astype(float)
        if len(axial) != len(bundle.data.pids):
            raise ValueError(
                f"{depth_path.name} describes {len(axial)} units, the prepared data {len(bundle.data.pids)}"
            )
        return cls(
            bundle=bundle,
            z_scaled=np.asarray(bundle.z_scaled, dtype=np.float32),
            pids=np.asarray(bundle.data.pids).astype(str),
            axial_um=axial,
        )

    def units_of(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """``(z_scaled [n, 60], axial_um [n])`` of the units recorded on ``pid``."""
        idx = np.flatnonzero(self.pids == str(pid))
        return self.z_scaled[idx], self.axial_um[idx]

    def log_mixture_weights(self, context_raw: np.ndarray) -> np.ndarray:
        """``[N, K]`` log mixture weights gamma_k(x) at raw contexts."""
        w = self.bundle.context_model.weights_for_context(np.asarray(context_raw, np.float32))
        return np.log(np.maximum(np.asarray(w, dtype=np.float64), 1e-12))
