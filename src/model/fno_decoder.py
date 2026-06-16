"""
FNO Action Decoder for VLA.

Core idea: spectral convolution in Fourier space + local bypass + FiLM conditioning.
Each layer: x -> rfft -> truncate to k modes -> multiply learned R -> irfft + W*x -> FiLM -> GeLU
Resolution-invariant: change irfft(n=NEW_SIZE) at inference for any control frequency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv1d(nn.Module):
    """Fourier layer: spectral convolution + local linear bypass."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes

        # Learned complex weights for spectral convolution: [modes, width, width]
        scale = 1 / (width * width)
        self.R = nn.Parameter(
            scale * torch.randn(modes, width, width, dtype=torch.cfloat)
        )

        # Local bypass path (equivalent to 1x1 conv / pointwise linear)
        self.W = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, output_size: int | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, H, C] — batch, time steps, channels
            output_size: if provided, irfft outputs this many time steps (resolution-invariance)

        Returns:
            [B, H_out, C] where H_out = output_size or H
        """
        B, H, C = x.shape
        n_out = output_size if output_size is not None else H

        # Spectral path
        x_ft = torch.fft.rfft(x, dim=1)                     # [B, H//2+1, C]
        k = min(self.modes, x_ft.shape[1])                   # handle small H
        x_ft_trunc = x_ft[:, :k, :]                          # [B, k, C]
        # Learned spectral transform: einsum over (modes, in_channels, out_channels)
        out_ft = torch.einsum('bkc,kcd->bkd', x_ft_trunc, self.R[:k])  # [B, k, C]
        # Pad back to full spectrum and inverse FFT
        full_ft = torch.zeros(B, n_out // 2 + 1, C, dtype=torch.cfloat, device=x.device)
        full_ft[:, :k, :] = out_ft
        x_spectral = torch.fft.irfft(full_ft, n=n_out, dim=1)  # [B, n_out, C]

        # Local path — if resolution changed, interpolate x first
        if n_out != H:
            x_interp = F.interpolate(
                x.permute(0, 2, 1), size=n_out, mode='linear', align_corners=False
            ).permute(0, 2, 1)
        else:
            x_interp = x
        x_local = self.W(x_interp)  # [B, n_out, C]

        return x_spectral + x_local


class FNODecoder(nn.Module):
    """
    Fourier Neural Operator action decoder.

    Takes latent z [B, D] + FiLM params -> outputs action chunk [B, H, action_dim].
    Resolution-invariant: pass output_size != chunk_size at inference.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        width: int = 128,
        num_layers: int = 4,
        modes: int = 12,
        action_dim: int = 6,
        chunk_size: int = 10,
    ):
        super().__init__()
        self.width = width
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Lifting: project latent z to FNO width
        self.lift = nn.Linear(latent_dim, width)

        # Sinusoidal positional encoding (not learned — works at any resolution)
        self.register_buffer('_dummy', torch.empty(0))  # for device tracking

        # Fourier layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(width, modes) for _ in range(num_layers)
        ])

        # Projection to action space
        self.proj = nn.Linear(width, action_dim)

        # Per-timestep gripper head (open/close *timing* matters for pick-place,
        # so predict a gripper value at every step from the same per-step
        # features, instead of one binary for the whole chunk).
        self.gripper_proj = nn.Linear(width, 1)

    def _positional_encoding(self, length: int, dim: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal positional encoding, works at any length."""
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # [L, 1]
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / dim)
        )  # [D/2]
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [L, D]

    def forward(
        self,
        z: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]],
        output_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, latent_dim] — fused vision+language+proprio latent
            film_params: list of (gamma, beta) per layer, each [B, width]
            output_size: number of output timesteps (None = use chunk_size)

        Returns:
            actions: [B, H, action_dim] continuous actions
            gripper: [B, H, 1] per-timestep gripper probability
        """
        B = z.shape[0]
        H = output_size if output_size is not None else self.chunk_size
        device = z.device

        # Lift: replicate z across time and project
        x = self.lift(z)                              # [B, width]
        x = x.unsqueeze(1).expand(B, H, -1)          # [B, H, width]

        # Add positional encoding
        pe = self._positional_encoding(H, self.width, device)  # [H, width]
        x = x + pe.unsqueeze(0)                       # [B, H, width]

        # Fourier layers with FiLM conditioning
        for i, conv in enumerate(self.spectral_convs):
            gamma, beta = film_params[i]              # each [B, width]
            x_out = conv(x)                           # [B, H, width]
            # FiLM: gamma * x + beta (broadcast over time dim)
            x = F.gelu(gamma.unsqueeze(1) * x_out + beta.unsqueeze(1))

        # Project to action space + per-timestep gripper
        actions = self.proj(x)                        # [B, H, action_dim]
        gripper = torch.sigmoid(self.gripper_proj(x))  # [B, H, 1]
        return actions, gripper
