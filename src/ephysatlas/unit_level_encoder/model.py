from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], latent_dim: int):
        super().__init__()
        n = 1
        for d in input_shape:
            n *= int(d)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, output_shape: tuple[int, ...], latent_dim: int):
        super().__init__()
        n = 1
        for d in output_shape:
            n *= int(d)
        self.output_shape = tuple(output_shape)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, n),
        )

    def forward(self, z):
        return self.net(z).reshape(len(z), *self.output_shape)


class ModalityAutoencoder(nn.Module):
    def __init__(self, shape: tuple[int, ...], latent_dim: int):
        super().__init__()
        self.encoder = MLPEncoder(shape, latent_dim)
        self.decoder = MLPDecoder(shape, latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class WaveformFeatureHead(nn.Module):
    """Predict standardized continuous features and categorical polarity from z_wave."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_continuous: int, n_polarity_classes: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.continuous = nn.Linear(hidden_dim, n_continuous)
        self.polarity = nn.Linear(hidden_dim, n_polarity_classes)

    def forward(self, z):
        h = self.shared(z)
        return self.continuous(h), self.polarity(h)


class UnitAutoencoder(nn.Module):
    """Independent modality AEs, optionally with a waveform-feature auxiliary head."""

    def __init__(self, cfg, n_continuous_features: int = 10, n_polarity_classes: int = 2):
        super().__init__()
        d = int(cfg.modality_latent_dim)
        self.use_acg = bool(cfg.use_acg)
        self.use_stpc = bool(cfg.use_stpc)
        self.feature_fidelity = bool(getattr(cfg, "feature_fidelity", False))

        self.waveform = ModalityAutoencoder(tuple(cfg.waveform_shape), d)
        self.acg = ModalityAutoencoder(tuple(cfg.acg_shape), d) if self.use_acg else None
        self.stpc = ModalityAutoencoder(tuple(cfg.stpc_shape), d) if self.use_stpc else None
        self.feature_head = (
            WaveformFeatureHead(
                d,
                int(cfg.feature_fidelity_hidden_dim),
                int(n_continuous_features),
                int(n_polarity_classes),
            )
            if self.feature_fidelity else None
        )

    def encode(self, waveform, acg=None, stpc=None):
        out = {"waveform": self.waveform.encode(waveform)}
        if self.use_acg:
            out["acg"] = self.acg.encode(acg)
        if self.use_stpc:
            out["stpc"] = self.stpc.encode(stpc)
        return out

    def decode(self, latents: dict[str, torch.Tensor]):
        out = {"waveform": self.waveform.decode(latents["waveform"])}
        if self.use_acg:
            out["acg"] = self.acg.decode(latents["acg"])
        if self.use_stpc:
            out["stpc"] = self.stpc.decode(latents["stpc"])
        return out

    def predict_waveform_features(self, waveform_latent):
        if self.feature_head is None:
            raise RuntimeError("This autoencoder was not configured with feature_fidelity=True")
        return self.feature_head(waveform_latent)

    def split_joint_latent(self, joint: torch.Tensor, latent_dim: int):
        order = ["waveform"] + (["acg"] if self.use_acg else []) + (["stpc"] if self.use_stpc else [])
        chunks = torch.split(joint, int(latent_dim), dim=1)
        return {name: chunk for name, chunk in zip(order, chunks)}


def covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    if len(z) < 2:
        return z.new_zeros(())
    z = z - z.mean(0, keepdim=True)
    cov = z.T @ z / max(len(z) - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return off.square().mean()


def variance_penalty(z: torch.Tensor, target_std: float) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(target_std) - std).mean()
