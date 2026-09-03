from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ephysatlas.unit_level_encoder.config import Config


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (3, 7), padding=(1, 3)),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 5), stride=(1, 2), padding=(1, 2)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, (3, 5), stride=(2, 2), padding=(1, 2)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, (3, 5), stride=(2, 2), padding=(1, 2)),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((5, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 8, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x[:, None]))


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_shape: Tuple[int, int]):
        super().__init__()
        self.output_shape = output_shape
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256 * 5 * 8), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 4), stride=(1, 2), padding=(1, 1)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, (3, 4), stride=(1, 2), padding=(1, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(self.fc(z).reshape(-1, 256, 5, 8))[:, 0]
        channels, time = self.output_shape
        return x[:, :channels, :time]


class ACGDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_shape: Tuple[int, int]):
        super().__init__()
        self.output_shape = output_shape
        rows, _ = output_shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 64 * rows * 26),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, (3, 5), stride=(1, 2), padding=(1, 2), output_padding=(0, 1)
            ),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.ConvTranspose2d(
                32, 16, (3, 5), stride=(1, 2), padding=(1, 2), output_padding=(0, 1)
            ),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, (3, 5), padding=(1, 2)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rows, time = self.output_shape
        x = self.refine(self.fc(z).reshape(len(z), 64, rows, 26))[:, 0]
        if x.shape[-1] != time:
            x = F.interpolate(
                x[:, None], size=(rows, time), mode="bilinear", align_corners=False
            )[:, 0]
        return F.softplus(x)


class SharedProjector(nn.Module):
    def __init__(self, dim: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultimodalAutoencoder(nn.Module):
    """Waveform+ACG multimodal encoder used by the released unit atlas."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.shared_latent_dim
        self.waveform_encoder = ConvEncoder(d)
        self.acg_encoder = ConvEncoder(d)
        self.waveform_decoder = WaveformDecoder(d, cfg.waveform_shape)
        self.acg_decoder = ACGDecoder(d, cfg.acg_shape)
        self.wave_projector = SharedProjector(
            d, cfg.projector_hidden_dim, cfg.projector_dim
        )
        self.acg_projector = SharedProjector(
            d, cfg.projector_hidden_dim, cfg.projector_dim
        )

    def encode(
        self, waveform: torch.Tensor, acg: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_wave = self.waveform_encoder(waveform)
        z_acg = self.acg_encoder(acg)
        return {
            "z_unit_shared": z_wave,
            "z_wave_shared": z_wave,
            "z_acg_shared": z_acg,
            "p_wave": self.wave_projector(z_wave),
            "p_acg": self.acg_projector(z_acg),
        }

    def reconstruct(self, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "waveform_reconstruction": self.waveform_decoder(encoded["z_wave_shared"]),
            "acg_reconstruction": self.acg_decoder(encoded["z_acg_shared"]),
        }

    def decode_waveform_from_shared(self, z_shared: torch.Tensor) -> torch.Tensor:
        return self.waveform_decoder(z_shared)

    def decode_acg_from_shared(self, z_shared: torch.Tensor) -> torch.Tensor:
        return self.acg_decoder(z_shared)


def augment_waveform(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    y = x.clone()
    if cfg.waveform_amplitude_jitter > 0:
        y *= 1 + cfg.waveform_amplitude_jitter * torch.randn(
            len(y), 1, 1, device=y.device
        )
    if cfg.waveform_noise_std > 0:
        y += cfg.waveform_noise_std * torch.randn_like(y)
    if cfg.waveform_time_shift > 0:
        shifts = torch.randint(
            -cfg.waveform_time_shift,
            cfg.waveform_time_shift + 1,
            (len(y),),
            device=y.device,
        )
        y = torch.stack([torch.roll(v, int(s), dims=-1) for v, s in zip(y, shifts)])
    if cfg.waveform_channel_mask_probability > 0:
        mask = (
            torch.rand(y.shape[:2], device=y.device)
            < cfg.waveform_channel_mask_probability
        )
        y = y.masked_fill(mask[..., None], 0)
    if cfg.waveform_time_mask_probability > 0:
        mask = (
            torch.rand((len(y), y.shape[-1]), device=y.device)
            < cfg.waveform_time_mask_probability
        )
        y = y.masked_fill(mask[:, None, :], 0)
    return y.clamp(-1.25, 1.25)


def augment_acg(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    y = x.clone()
    if cfg.acg_noise_std > 0:
        scale = y.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
        y += cfg.acg_noise_std * scale * torch.randn_like(y)
    if cfg.acg_mask_probability > 0:
        y = y.masked_fill(torch.rand_like(y) < cfg.acg_mask_probability, 0)
    return y.clamp_min(0)


def waveform_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return (
        F.mse_loss(pred, target)
        + 0.5
        * F.mse_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
        + 0.25 * F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        + 0.5 * ((1 + 4 * target.abs()) * (pred - target).square()).mean()
    )


def waveform_morphology_loss(
    pred: torch.Tensor, target: torch.Tensor, cfg: Config
) -> torch.Tensor:
    batch, _, time = target.shape
    dominant = target.abs().amax(dim=-1).argmax(dim=1)
    row = torch.arange(batch, device=target.device)
    p = pred[row, dominant]
    t = target[row, dominant]
    temperature = max(float(cfg.feature_softmax_temperature), 1e-3)
    time_axis = torch.linspace(0.0, 1.0, time, device=target.device)[None]

    def soft_min(trace):
        w = F.softmax(-trace / temperature, dim=-1)
        return (w * trace).sum(-1), (w * time_axis).sum(-1)

    def soft_max(trace):
        w = F.softmax(trace / temperature, dim=-1)
        return (w * trace).sum(-1), (w * time_axis).sum(-1)

    pmin, ptmin = soft_min(p)
    tmin, ttmin = soft_min(t)
    pmax, ptmax = soft_max(p)
    tmax, ttmax = soft_max(t)
    p_smooth = F.avg_pool1d(p[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
    t_smooth = F.avg_pool1d(t[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
    return (
        F.smooth_l1_loss(pmin, tmin)
        + F.smooth_l1_loss(pmax, tmax)
        + 0.5 * F.smooth_l1_loss(ptmin, ttmin)
        + 0.5 * F.smooth_l1_loss(ptmax, ttmax)
        + F.mse_loss(
            p_smooth[:, 1:] - p_smooth[:, :-1], t_smooth[:, 1:] - t_smooth[:, :-1]
        )
    )


def acg_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, cfg: Config
) -> torch.Tensor:
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    normalized = target / scale
    weight = 1.0 + cfg.acg_activity_weight * normalized
    value_loss = (weight * (pred - target).square()).mean()
    temporal_loss = F.mse_loss(
        pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1]
    )
    row_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    p_mass = pred.sum(-1)
    t_mass = target.sum(-1)
    mass_loss = F.smooth_l1_loss(torch.log1p(p_mass), torch.log1p(t_mass))
    axis = torch.linspace(0.0, 1.0, target.shape[-1], device=target.device)
    p_centroid = (pred * axis).sum(-1) / p_mass.clamp_min(1e-5)
    t_centroid = (target * axis).sum(-1) / t_mass.clamp_min(1e-5)
    centroid_loss = F.smooth_l1_loss(p_centroid, t_centroid)
    return (
        value_loss
        + 0.5 * temporal_loss
        + 0.15 * row_loss
        + 0.1 * mass_loss
        + 0.1 * centroid_loss
    )


def latent_scale_loss(
    z_wave: torch.Tensor, z_acg: torch.Tensor, cfg: Config
) -> torch.Tensor:
    losses = []
    for z in (z_wave, z_acg):
        mean = z.mean(0)
        std = z.std(0, unbiased=False)
        losses.append(
            mean.square().mean() + (std - cfg.raw_latent_std_target).square().mean()
        )
    return 0.5 * (losses[0] + losses[1])


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Expected square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, cfg: Config):
    inv = F.mse_loss(z1, z2)
    s1 = torch.sqrt(z1.var(0, unbiased=False) + cfg.vicreg_eps)
    s2 = torch.sqrt(z2.var(0, unbiased=False) + cfg.vicreg_eps)
    var = 0.5 * (
        F.relu(cfg.vicreg_std_target - s1).mean()
        + F.relu(cfg.vicreg_std_target - s2).mean()
    )
    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    den = max(len(z1) - 1, 1)
    c1 = z1c.T @ z1c / den
    c2 = z2c.T @ z2c / den
    cov = (off_diagonal(c1).square().sum() + off_diagonal(c2).square().sum()) / (
        2 * z1.shape[1]
    )
    return {"invariance": inv, "variance": var, "covariance": cov}
