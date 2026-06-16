"""
Dual-path vision encoder: frozen pretrained TinyViT (semantic) + Scattering CNN
(stable, learned). Either path can be disabled to form the ablation matrix.

- TinyViT is frozen (0 trainable params) and always kept in eval mode.
- Dropout (p) is applied to the TinyViT *output tokens* during training. This
  prevents the network from collapsing onto TinyViT and ignoring the scattering
  path (which would void the robustness claim).
- Two camera views (stacked as 6 channels) are each passed through the shared
  backbones and averaged, matching the scattering encoder's multi-view handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scattering_encoder import ScatteringEncoder


class DualVisionEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        embed_dim: int = 256,
        use_scattering: bool = True,
        use_tinyvit: bool = True,
        scattering_kwargs: dict | None = None,
        tinyvit_name: str = "tiny_vit_21m_224.dist_in22k_ft_in1k",
        tinyvit_dropout: float = 0.3,
        tinyvit_input: int = 224,
        separate_views: bool = False,
        gated_fusion: bool = False,
    ):
        super().__init__()
        assert use_scattering or use_tinyvit, "at least one vision path must be enabled"
        self.embed_dim = embed_dim
        self.use_scattering = use_scattering
        self.use_tinyvit = use_tinyvit
        self.tv_size = tinyvit_input
        # GMU-style gate (Arevalo et al. 2017): learn a per-channel weight that routes
        # between the scattering (noise-robust) and DINOv3 (blur-robust) streams per input.
        # Zero-init -> g=0.5 -> both streams pass at weight 1.0 -> identical to no-gate at start.
        self.gated_fusion = gated_fusion and use_scattering and use_tinyvit
        # Keep the two views as separate token streams (concat) instead of
        # averaging -> lets fusion attend to the wrist view (grasp-depth cue).
        self.separate_views = separate_views

        if use_scattering:
            self.scattering = ScatteringEncoder(
                img_size=img_size, embed_dim=embed_dim,
                separate_views=separate_views, **(scattering_kwargs or {})
            )

        if use_tinyvit:
            import timm
            # Generic frozen timm backbone: TinyViT conv-features OR a ViT such as
            # DINOv2. For ViTs we pass img_size so the positional embedding is
            # interpolated to our input grid; conv backbones reject that kwarg.
            try:
                self.tinyvit = timm.create_model(
                    tinyvit_name, pretrained=True, num_classes=0, img_size=tinyvit_input)
            except TypeError:
                self.tinyvit = timm.create_model(tinyvit_name, pretrained=True, num_classes=0)
            for p in self.tinyvit.parameters():
                p.requires_grad = False
            self.tinyvit.eval()
            self._tv_dim = self.tinyvit.num_features
            # ViT backbones prepend CLS/register tokens we drop to keep only the
            # spatial patch tokens (num_prefix_tokens = 0 for conv backbones).
            self._tv_prefix = getattr(self.tinyvit, 'num_prefix_tokens', 0)
            self.tv_proj = nn.Linear(self._tv_dim, embed_dim)
            self.tv_drop = nn.Dropout(tinyvit_dropout)

        if self.gated_fusion:
            self.gmu_gate = nn.Linear(2 * embed_dim, embed_dim)
            nn.init.zeros_(self.gmu_gate.weight)   # identity init: g=0.5 -> weights=1.0
            nn.init.zeros_(self.gmu_gate.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_tinyvit:
            self.tinyvit.eval()  # frozen backbone never in train mode (no BN/stat drift)
        return self

    def _tinyvit_tokens(self, x3: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x3, size=(self.tv_size, self.tv_size),
                          mode='bilinear', align_corners=False)
        with torch.no_grad():
            f = self.tinyvit.forward_features(x)
        if f.dim() == 4:                      # conv [B,C,h,w] -> [B,N,C]
            f = f.flatten(2).transpose(1, 2)
        elif f.dim() == 2:                    # [B,C] -> [B,1,C]
            f = f.unsqueeze(1)
        elif f.dim() == 3 and self._tv_prefix:  # ViT: drop CLS/register prefix
            f = f[:, self._tv_prefix:]
        f = self.tv_proj(f)                   # [B,N,embed] (trainable)
        f = self.tv_drop(f)                   # anti-collapse dropout
        return f

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        scat = self.scattering(images) if self.use_scattering else None   # [B, Ns, embed]
        dino = []
        if self.use_tinyvit:
            if images.shape[1] == 6:
                a = self._tinyvit_tokens(images[:, :3]); w = self._tinyvit_tokens(images[:, 3:])
                dino = [a, w] if self.separate_views else [(a + w) / 2]
            else:
                dino = [self._tinyvit_tokens(images)]

        if self.gated_fusion and scat is not None and dino:
            dino_all = torch.cat(dino, dim=1)                              # [B, Nd, embed]
            g = torch.sigmoid(self.gmu_gate(torch.cat([scat.mean(1), dino_all.mean(1)], dim=-1)))  # [B, embed]
            scat = scat * (2.0 * g).unsqueeze(1)                           # scattering weight (1.0 at init)
            dino = [d * (2.0 * (1.0 - g)).unsqueeze(1) for d in dino]      # DINOv3 weight (1.0 at init)

        toks = ([scat] if scat is not None else []) + dino
        return torch.cat(toks, dim=1)
