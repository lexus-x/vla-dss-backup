"""
Simple diffusion action head baseline for ablation.
DDPM-style denoising with small MLP, similar to Octo's action head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionHead(nn.Module):
    """
    DDPM diffusion action decoder (baseline).
    Iteratively denoises random noise into action chunk.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 6,
        chunk_size: int = 10,
        num_denoise_steps: int = 20,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_denoise_steps = num_denoise_steps

        # Noise prediction network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + chunk_size * action_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

        # DDPM schedule
        betas = torch.linspace(1e-4, 0.02, num_denoise_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward(
        self,
        z: torch.Tensor,
        film_params=None,
        output_size: int | None = None,
    ) -> torch.Tensor:
        """
        At training: add noise to target actions, predict noise (call forward_train instead).
        At inference: iteratively denoise from random noise.
        """
        if self.training:
            raise RuntimeError("Use forward_train() during training")
        return self._inference(z)

    def forward_train(
        self,
        z: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass: noise + denoise.

        Args:
            z: [B, latent_dim]
            target_actions: [B, chunk_size, action_dim]

        Returns:
            noise_pred_loss: scalar
        """
        B = z.shape[0]
        x0 = target_actions.reshape(B, -1)  # [B, chunk*action]

        # Random timestep
        t = torch.randint(0, self.num_denoise_steps, (B,), device=z.device)

        # Add noise
        noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_noisy = sqrt_ac * x0 + sqrt_omac * noise

        # Predict noise
        t_embed = t.float().unsqueeze(-1) / self.num_denoise_steps
        net_input = torch.cat([z, x_noisy, t_embed], dim=-1)
        noise_pred = self.net(net_input)

        return F.mse_loss(noise_pred, noise)

    def _inference(self, z: torch.Tensor) -> torch.Tensor:
        """DDPM reverse process."""
        B = z.shape[0]
        x = torch.randn(B, self.chunk_size * self.action_dim, device=z.device)

        for t in reversed(range(self.num_denoise_steps)):
            t_batch = torch.full((B,), t, device=z.device)
            t_embed = t_batch.float().unsqueeze(-1) / self.num_denoise_steps

            net_input = torch.cat([z, x, t_embed], dim=-1)
            noise_pred = self.net(net_input)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )

            if t > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return x.reshape(B, self.chunk_size, self.action_dim)
