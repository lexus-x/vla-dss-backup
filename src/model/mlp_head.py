"""
MLP action head baseline for ablation.
Same interface as FNODecoder — drop-in replacement to isolate FNO's contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActionGripperHead(nn.Module):
    """
    Param-matched MLP baseline with the SAME output interface as FNODecoder:
    returns (actions [B, chunk, action_dim], gripper [B, chunk, 1]).

    Cannot change resolution (output_size is ignored) — that's the whole point
    of the ablation; resolution-invariance is unique to the FNO head, and the
    MLP baseline must instead rely on spline interpolation at eval time.
    """

    def __init__(self, latent_dim=256, hidden_dim=1024, num_layers=4,
                 action_dim=6, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        layers, in_dim = [], latent_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.action_out = nn.Linear(hidden_dim, chunk_size * action_dim)
        self.gripper_out = nn.Linear(hidden_dim, chunk_size)

    def forward(self, z, film_params=None, output_size=None):
        B = z.shape[0]
        h = self.trunk(z)
        actions = self.action_out(h).reshape(B, self.chunk_size, self.action_dim)
        gripper = torch.sigmoid(self.gripper_out(h)).reshape(B, self.chunk_size, 1)
        return actions, gripper


class MLPHead(nn.Module):
    """
    Standard MLP action decoder (baseline).
    Takes latent z, outputs action chunk. No frequency-domain processing.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        action_dim: int = 6,
        chunk_size: int = 10,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, chunk_size * action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        film_params=None,  # ignored — MLP doesn't use FiLM
        output_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
            film_params: ignored (compatibility with FNO interface)
            output_size: ignored (MLP can't change resolution)

        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = z.shape[0]
        out = self.mlp(z)  # [B, chunk_size * action_dim]
        actions = out.reshape(B, self.chunk_size, self.action_dim)
        return actions


class MLPHeadLarge(nn.Module):
    """
    Parameter-matched MLP baseline.
    Same param count as FNO decoder (~8M) for fair comparison.
    hidden=1400, 6 layers -> ~8.3M params (matches FNO's 8.2M)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 1400,
        num_layers: int = 6,
        action_dim: int = 6,
        chunk_size: int = 10,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, chunk_size * action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        film_params=None,
        output_size: int | None = None,
    ) -> torch.Tensor:
        B = z.shape[0]
        out = self.mlp(z)
        actions = out.reshape(B, self.chunk_size, self.action_dim)
        return actions
