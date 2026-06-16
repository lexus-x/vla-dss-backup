"""
Hybrid Scattering + CNN observation encoder.

Two parallel paths:
  Path A: Scattering Transform → provably stable features (parameter-free)
  Path B: Small ConvNet on raw image → fine-grained details (learned)

Combined → ResBlocks → Self-Attention → rich + stable visual tokens.

This dual-path design ensures:
1. Stability: scattering path provides Lipschitz-continuous features
2. Richness: raw path captures fine textures/edges for grasping
3. Task-specificity: ResBlocks + attention learn manipulation-relevant patterns
"""

import torch
import torch.nn as nn
from kymatio.torch import Scattering2D


class ResBlock(nn.Module):
    """Residual block with BatchNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SpatialAttention(nn.Module):
    """Lightweight self-attention on spatial tokens."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> attend over H*W spatial positions."""
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = tokens + self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens))[0]
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class ScatteringEncoder(nn.Module):
    """
    Dual-path Scattering + Raw Image encoder.

    Path A: Image → Scattering2D (J scales, L orientations) → 1x1 reduce
    Path B: Image → 3-layer ConvNet (stride 8 to match scattering spatial) → features
    Merge:  Concat → ResBlocks → Self-Attention → visual tokens
    """

    def __init__(
        self,
        img_size: int = 128,
        J: int = 3,
        L: int = 12,
        embed_dim: int = 256,
        proj_channels: int = 256,
        num_res_blocks: int = 6,
        separate_views: bool = False,
    ):
        super().__init__()
        self.J = J
        self.L = L
        self.img_size = img_size
        self.embed_dim = embed_dim
        # If True, the two camera views are kept as SEPARATE token streams
        # (concatenated along the token dim) instead of averaged at the
        # feature-map level. Lets downstream fusion attend to the wrist view
        # specifically (grasp-depth cue). Default False = legacy averaging.
        self.separate_views = separate_views

        # ═══ PATH A: Scattering (parameter-free, provably stable) ═══
        self.scattering = Scattering2D(J=J, shape=(img_size, img_size), L=L)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.scattering(dummy)
            self.n_coeffs = out.shape[2]
            self.spatial_h = out.shape[3]
            self.spatial_w = out.shape[4]
            self.scatter_channels = 3 * self.n_coeffs

        # Reduce scattering channels (1407 → proj_channels/2)
        half_ch = proj_channels // 2
        self.scatter_proj = nn.Sequential(
            nn.Conv2d(self.scatter_channels, half_ch, 1, bias=False),
            nn.BatchNorm2d(half_ch),
            nn.GELU(),
        )

        # ═══ PATH B: Raw image features (learned, captures fine details) ═══
        # Downsample 128x128 → 16x16 (stride 8 total) to match scattering spatial
        self.raw_conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2, bias=False),     # 128→64
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),   # 64→32
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, half_ch, 3, stride=2, padding=1, bias=False),  # 32→16
            nn.BatchNorm2d(half_ch),
            nn.GELU(),
        )

        # ═══ MERGE: concat + ResBlocks + attention ═══
        merged_channels = proj_channels  # half_ch + half_ch

        self.res_blocks = nn.Sequential(
            *[ResBlock(merged_channels) for _ in range(num_res_blocks)]
        )

        # Self-attention for spatial relationships
        self.spatial_attn = SpatialAttention(merged_channels, num_heads=4)

        # Final projection to embed_dim
        self.final_proj = nn.Sequential(
            nn.Conv2d(merged_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.num_tokens = self.spatial_h * self.spatial_w  # 16*16 = 256

    def _encode_single(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single camera view. images: [B, 3, H, W]"""
        B = images.shape[0]
        images = images.contiguous()

        # Path A: Scattering (stable)
        Sx = self.scattering(images)
        Sx = Sx.reshape(B, self.scatter_channels, self.spatial_h, self.spatial_w)
        scatter_feat = self.scatter_proj(Sx)

        # Path B: Raw conv (fine details)
        raw_feat = self.raw_conv(images)

        # Merge scatter + raw channels
        return torch.cat([scatter_feat, raw_feat], dim=1)

    def _view_tokens(self, x3: torch.Tensor) -> torch.Tensor:
        """Full per-view pipeline -> [B, num_tokens, embed_dim] (for separate_views)."""
        m = self._encode_single(x3)
        m = self.res_blocks(m)
        m = self.spatial_attn(m)
        return self.final_proj(m).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] single camera
                    OR [B, 6, H, W] two cameras stacked on channel dim
                    OR [B, N_cams, 3, H, W] multiple cameras

        Returns:
            tokens: [B, H*W, embed_dim]

        Views are averaged at the feature-map level (the 53%-model behaviour;
        separating them regressed grasp accuracy in practice).
        """
        # Separate-views: keep each camera as its own token stream (concat).
        if self.separate_views and (images.dim() == 5 or images.shape[1] == 6):
            if images.dim() == 5:
                views = [images[:, i] for i in range(images.shape[1])]
            else:
                views = [images[:, :3], images[:, 3:]]
            return torch.cat([self._view_tokens(v) for v in views], dim=1)
        # Legacy: average views at the feature-map level.
        if images.dim() == 5:
            N = images.shape[1]
            merged = sum(self._encode_single(images[:, i]) for i in range(N)) / N
        elif images.shape[1] == 6:
            merged = (self._encode_single(images[:, :3]) + self._encode_single(images[:, 3:])) / 2
        else:
            merged = self._encode_single(images)
        merged = self.res_blocks(merged)
        merged = self.spatial_attn(merged)
        return self.final_proj(merged).flatten(2).transpose(1, 2)
