"""
Fusion module: projects vision/language/proprio to shared dim,
cross-attends, pools to latent z, generates FiLM params for FNO layers.
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Cross-modal fusion + FiLM parameter generation."""

    def __init__(
        self,
        vision_dim: int,
        language_dim: int = 128,
        proprio_dim: int = 128,
        shared_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        latent_dim: int = 256,
        fno_width: int = 128,
        fno_num_layers: int = 4,
        separate_pool: bool = False,
        vision_pool: str = 'mean',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.fno_num_layers = fno_num_layers
        self.fno_width = fno_width
        # separate_pool: pool vision / language / proprio independently so the
        # few language tokens aren't drowned out by ~300 vision tokens in a
        # global mean (the cause of weak instruction grounding on multi-object
        # scenes like LIBERO-Object).
        self.separate_pool = separate_pool
        # vision_pool: 'mean' (bag-of-features, loses 'where') or 'attention'
        # (a learned query attends over the spatial vision tokens -> preserves
        # object localization for reaching).
        self.vision_pool = vision_pool
        if vision_pool == 'attention':
            self.vis_query = nn.Parameter(torch.randn(1, 1, shared_dim) * 0.02)
            self.vis_attn = nn.MultiheadAttention(shared_dim, num_heads, batch_first=True)
            # Proprio-gate the pooling query: the gripper/EEF state shifts WHERE
            # the query looks (e.g. during a grasp approach, bias toward the
            # wrist tokens for depth). Zero-init -> identity at start, so this is
            # backward-compatible with existing attn-pool checkpoints.
            self.q_gate = nn.Linear(shared_dim, shared_dim)
            nn.init.zeros_(self.q_gate.weight)
            nn.init.zeros_(self.q_gate.bias)

        # Project each modality to shared dim
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.language_proj = nn.Linear(language_dim, shared_dim)
        self.proprio_proj = nn.Linear(proprio_dim, shared_dim)

        # Cross-attention transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(shared_dim)

        # Pool to latent z (3*shared_dim when pooling modalities separately)
        pool_in = shared_dim * 3 if separate_pool else shared_dim
        self.pool_proj = nn.Linear(pool_in, latent_dim)

        # FiLM generators: one gamma + one beta per FNO layer
        self.film_gamma = nn.ModuleList([
            nn.Linear(latent_dim, fno_width) for _ in range(fno_num_layers)
        ])
        self.film_beta = nn.ModuleList([
            nn.Linear(latent_dim, fno_width) for _ in range(fno_num_layers)
        ])

        # Initialize FiLM: gamma=1, beta=0 (identity at start)
        for layer in self.film_gamma:
            nn.init.ones_(layer.bias)
            nn.init.zeros_(layer.weight)
        for layer in self.film_beta:
            nn.init.zeros_(layer.bias)
            nn.init.zeros_(layer.weight)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            vision_tokens:  [B, N_v, vision_dim]
            language_tokens: [B, N_l, language_dim]
            proprio_embed:  [B, proprio_dim]

        Returns:
            z: [B, latent_dim]
            film_params: list of (gamma, beta) per FNO layer, each [B, fno_width]
        """
        # Project to shared dim
        v = self.vision_proj(vision_tokens)     # [B, N_v, D]
        l = self.language_proj(language_tokens)  # [B, N_l, D]
        p = self.proprio_proj(proprio_embed)     # [B, D]
        p = p.unsqueeze(1)                       # [B, 1, D]
        n_v, n_l = v.shape[1], l.shape[1]

        # Concatenate all tokens
        tokens = torch.cat([v, l, p], dim=1)     # [B, N_v+N_l+1, D]

        # Cross-attention
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        if self.separate_pool:
            # Per-modality pooling so language keeps equal footing with vision
            vis_tok = tokens[:, :n_v]
            if self.vision_pool == 'attention':
                # learned query attends over spatial vision tokens -> keeps "where".
                # Gated by the (post-transformer) proprio token so the gripper
                # state can steer the query toward the wrist view during grasps.
                q = self.vis_query.expand(vis_tok.shape[0], -1, -1)
                p_ctx = tokens[:, n_v + n_l:]                    # [B, 1, D] proprio token
                q = q + self.q_gate(p_ctx)
                z_v = self.vis_attn(q, vis_tok, vis_tok)[0].squeeze(1)
            else:
                z_v = vis_tok.mean(dim=1)
            z_l = tokens[:, n_v:n_v + n_l].mean(dim=1)
            z_p = tokens[:, n_v + n_l:].mean(dim=1)
            z = self.pool_proj(torch.cat([z_v, z_l, z_p], dim=-1))  # [B, latent_dim]
        else:
            z = self.pool_proj(tokens.mean(dim=1))                  # [B, latent_dim]

        # Generate FiLM params
        film_params = []
        for i in range(self.fno_num_layers):
            gamma = self.film_gamma[i](z)        # [B, fno_width]
            beta = self.film_beta[i](z)          # [B, fno_width]
            film_params.append((gamma, beta))

        return z, film_params
