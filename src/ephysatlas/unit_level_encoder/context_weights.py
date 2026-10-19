"""Context networks: the mixture weights of the unit-level model at a brain position.

A small network maps the standardized molecular context c(x) (MERFISH + AGEA PCs) to mixture
weights. It is trained on the posteriors of the TRAIN units by a soft cross-entropy, with class
weights that keep rare components from being ignored. The unit-level model uses two of them
(:class:`HierarchicalWeightModel`): one over the types and one over the sub-types.
"""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ephysatlas import model_registry

TRANSFORM_FILE = model_registry.UNIT_CONTEXT_TRANSFORM_FILE
TYPE_WEIGHTS_FILE = model_registry.UNIT_TYPE_WEIGHTS_FILE
SUBTYPE_WEIGHTS_FILE = model_registry.UNIT_SUBTYPE_WEIGHTS_FILE


class ContextWeightNet(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, n_components):
        super().__init__()
        seq = []
        d = input_dim
        for _ in range(max(1, int(layers) - 1)):
            seq.extend(
                [
                    nn.Linear(d, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            d = hidden
        seq.append(nn.Linear(d, n_components))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class ContextWeightModel:
    """Mixture weights from context: ``weights(indices)`` over a dataset, or at raw contexts."""

    def __init__(self, net, context_pc, device, transform=None):
        self.net = net
        self.context_pc = np.asarray(context_pc, np.float32)
        self.device = device
        self.transform = transform

    @torch.no_grad()
    def weights_from_pc(self, context_pc):
        x = torch.from_numpy(np.asarray(context_pc, np.float32)).to(self.device)
        return torch.softmax(self.net(x), dim=1).cpu().numpy().astype(np.float32)

    def weights(self, indices):
        return self.weights_from_pc(self.context_pc[np.asarray(indices, int)])

    def weights_for_context(self, context_raw):
        if self.transform is None:
            raise RuntimeError(
                "Context transform unavailable for arbitrary-voxel prediction"
            )
        return self.weights_from_pc(
            self.transform.transform(np.asarray(context_raw, np.float32))
        )


def fit_context_weight_model(
    context_pc, train_mask, val_mask, resp_train, resp_val, component_mass, cfg
):
    """Train a context network on posteriors, keeping the weights with the lowest VALIDATION loss.

    Args:
        context_pc: ``[N, n_context]`` standardized context of every unit.
        train_mask, val_mask: ``[N]`` TRAIN and VALIDATION units.
        resp_train, resp_val: Their posteriors over the components, the training targets.
        component_mass: ``[n_components]`` TRAIN mass of each component, for the class weights.
        cfg: Unit-model configuration.

    Returns:
        tuple: ``(ContextWeightModel, info)``; ``info`` holds the loss history.
    """
    x_train = torch.from_numpy(np.asarray(context_pc[train_mask], np.float32))
    y_train = torch.from_numpy(np.asarray(resp_train, np.float32))
    x_val = torch.from_numpy(np.asarray(context_pc[val_mask], np.float32))
    y_val = torch.from_numpy(np.asarray(resp_val, np.float32))

    mass = np.maximum(np.asarray(component_mass, np.float32), 1e-6)
    class_weight = mass ** (-float(cfg.rare_component_power))
    class_weight /= np.average(class_weight, weights=mass)
    class_weight = np.minimum(
        class_weight, float(cfg.rare_component_weight_cap)
    ).astype(np.float32)
    cw = torch.from_numpy(class_weight).to(cfg.device)

    net = ContextWeightNet(
        x_train.shape[1],
        cfg.context_hidden_dim,
        cfg.context_layers,
        cfg.context_dropout,
        y_train.shape[1],
    ).to(cfg.device)
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.context_weight_lr,
        weight_decay=cfg.context_weight_decay,
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.context_weight_batch_size,
        shuffle=True,
    )

    def soft_ce(logits, target):
        logp = F.log_softmax(logits, dim=1)
        wt = target * cw[None, :]
        wt = wt / wt.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return -(wt * logp).sum(dim=1).mean()

    best = np.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, cfg.context_weight_epochs + 1):
        net.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = soft_ce(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        net.eval()
        with torch.no_grad():
            val_loss = float(
                soft_ce(net(x_val.to(cfg.device)), y_val.to(cfg.device)).cpu()
            )
        row = {
            "epoch": epoch,
            "train_balanced_ce": total / max(n, 1),
            "val_balanced_ce": val_loss,
        }
        history.append(row)
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = copy.deepcopy(net.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg.context_weight_patience:
            break
    if best_state is None:
        raise RuntimeError("Context weight model produced no checkpoint")
    net.load_state_dict(best_state)
    net.eval()
    return ContextWeightModel(net, context_pc, cfg.device), {
        "best_val_balanced_ce": float(best),
        "rare_component_class_weights": class_weight.tolist(),
        "history": history,
    }


class HierarchicalWeightModel:
    """Sub-type weights at a position, ``w_kj(x) = gamma_k(x) gamma_j|k(x)``.

    ``types`` gives the type weights gamma_k(x); the within-type shares of the sub-type weights of
    ``subtypes`` give gamma_j|k(x), so a type's sub-type weights add up to its type weight.

    Args:
        types: :class:`ContextWeightModel` over the types.
        subtypes: :class:`ContextWeightModel` over the sub-types (same contexts and transform).
        parent: ``[n_subtypes]`` type of each sub-type.
    """

    def __init__(self, types: ContextWeightModel, subtypes: ContextWeightModel, parent):
        self.types = types
        self.subtypes = subtypes
        self.parent = np.asarray(parent, np.int64)
        self._onehot = np.eye(int(self.parent.max()) + 1, dtype=np.float64)[self.parent]

    @property
    def context_pc(self) -> np.ndarray:
        return self.types.context_pc

    @property
    def transform(self):
        return self.types.transform

    def combine(self, type_weights, subtype_weights) -> np.ndarray:
        """``[n, n_subtypes]`` gamma_k gamma_j|k from type and sub-type network outputs."""
        wt = np.asarray(type_weights, np.float64)
        ws = np.asarray(subtype_weights, np.float64)
        within = ws / np.maximum((ws @ self._onehot)[:, self.parent], 1e-12)
        return (wt[:, self.parent] * within).astype(np.float32)

    def weights_from_pc(self, context_pc) -> np.ndarray:
        return self.combine(
            self.types.weights_from_pc(context_pc),
            self.subtypes.weights_from_pc(context_pc),
        )

    def weights(self, indices) -> np.ndarray:
        """``[n, n_subtypes]`` sub-type weights of dataset units."""
        return self.combine(self.types.weights(indices), self.subtypes.weights(indices))

    def weights_for_context(self, context_raw) -> np.ndarray:
        """``[n, n_subtypes]`` sub-type weights at raw contexts."""
        return self.weights_from_pc(
            self.transform.transform(np.asarray(context_raw, np.float32))
        )

    def type_weights(self, indices) -> np.ndarray:
        """``[n, n_types]`` type weights of dataset units."""
        return self.types.weights(indices)

    def type_weights_for_context(self, context_raw) -> np.ndarray:
        """``[n, n_types]`` type weights at raw contexts."""
        return self.types.weights_for_context(context_raw)

    def over(self, context_raw) -> "HierarchicalWeightModel":
        """The same model indexed over another dataset's raw contexts (``weights(indices)``)."""
        pc = self.transform.transform(np.asarray(context_raw, np.float32))
        models = [
            ContextWeightModel(m.net, pc, m.device, transform=m.transform)
            for m in (self.types, self.subtypes)
        ]
        return HierarchicalWeightModel(*models, self.parent)

    def save(self, directory, cfg) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transform, directory / TRANSFORM_FILE)
        for model, name in (
            (self.types, TYPE_WEIGHTS_FILE),
            (self.subtypes, SUBTYPE_WEIGHTS_FILE),
        ):
            save_context_net(model, directory / name, cfg)
        return directory

    @classmethod
    def load(
        cls, parent, cfg, context_raw=None, *, transform, types, subtypes
    ) -> "HierarchicalWeightModel":
        """Load the two networks and the context transform.

        Args:
            parent: ``[n_subtypes]`` type of each sub-type.
            cfg: Unit-model ``Config`` (only ``device`` is read).
            context_raw: ``[N, n_context]`` raw contexts of a unit dataset for ``weights(indices)``;
                None when only arbitrary-position prediction is needed.
            transform, types, subtypes: Paths of the context transform and of the two networks.
        """
        transform = joblib.load(transform)
        if context_raw is None:
            context_pc = None
        else:
            context_pc = transform.transform(np.asarray(context_raw, np.float32))
        models = [
            load_context_net(path, context_pc, transform, cfg)
            for path in (types, subtypes)
        ]
        return cls(*models, parent)


def save_context_net(model: ContextWeightModel, path, cfg) -> Path:
    path = Path(path)
    torch.save(
        {
            "model_state_dict": model.net.state_dict(),
            "input_dim": int(model.net.net[0].in_features),
            "hidden": int(cfg.context_hidden_dim),
            "layers": int(cfg.context_layers),
            "dropout": float(cfg.context_dropout),
            "n_components": int(model.net.net[-1].out_features),
        },
        path,
    )
    return path


def load_context_net(path, context_pc, transform, cfg) -> ContextWeightModel:
    payload = torch.load(Path(path), map_location=cfg.device, weights_only=False)
    net = ContextWeightNet(
        payload["input_dim"],
        payload["hidden"],
        payload["layers"],
        payload["dropout"],
        payload["n_components"],
    ).to(cfg.device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    if context_pc is None:
        context_pc = np.zeros((0, int(payload["input_dim"])), np.float32)
    return ContextWeightModel(net, context_pc, cfg.device, transform=transform)
