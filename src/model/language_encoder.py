"""
Tiny language encoder: learned embeddings + small transformer.
Encodes task instructions like "pick up the red cup" into language tokens.
~4.6M params total.
"""

import torch
import torch.nn as nn
import math


class LanguageEncoder(nn.Module):
    """Lightweight language encoder for task instructions."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] integer token IDs
            attention_mask: [B, L] binary mask (1=attend, 0=ignore)

        Returns:
            lang_tokens: [B, L, embed_dim]
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)

        x = self.token_embed(token_ids) + self.pos_embed(positions)

        # Convert attention mask to transformer format (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x


class PretrainedLanguageEncoder(nn.Module):
    """Frozen pretrained sentence encoder (e.g. all-MiniLM-L6-v2) -> one strong
    language token via masked-mean pooling. Gives robust instruction grounding
    that a from-scratch 26-word embedding can't, and the single pooled token
    avoids padding-dilution in the fusion module.
    """

    def __init__(self, model_name: str, out_dim: int = 128, freeze: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.embed_dim = out_dim
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()  # keep frozen backbone in eval (no dropout/stat drift)
        return self

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.model(input_ids=token_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                       # [B, L, hidden]
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean
        else:
            pooled = h.mean(1)
        return self.proj(pooled).unsqueeze(1)            # [B, 1, out_dim]
