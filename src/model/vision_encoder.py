"""
Vision encoder: TinyViT-21M via timm.
Extracts visual tokens from 128x128 RGB images.
"""

import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """TinyViT-21M wrapper that outputs visual tokens."""

    def __init__(self, img_size: int = 128, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        self.img_size = img_size
        self.encoder = timm.create_model(
            'tiny_vit_21m_224.dist_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get output shape by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.encoder.forward_features(dummy)
            # TinyViT outputs [B, C, H, W] spatial feature map
            if out.dim() == 4:
                self.embed_dim = out.shape[1]
                self.num_tokens = out.shape[2] * out.shape[3]
            elif out.dim() == 3:
                self.num_tokens = out.shape[1]
                self.embed_dim = out.shape[2]
            else:
                self.num_tokens = 1
                self.embed_dim = out.shape[1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 128, 128]

        Returns:
            tokens: [B, num_tokens, embed_dim]
        """
        out = self.encoder.forward_features(images)
        if out.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2)  # [B, H*W, C]
        elif out.dim() == 2:
            out = out.unsqueeze(1)  # [B, 1, dim]
        return out
