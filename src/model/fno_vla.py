"""
FNO-VLA: Full model assembly.

Scattering Transform (vision) + tiny transformer (language) + MLP (proprio)
-> Fusion + FiLM -> FNO decoder -> continuous actions + gripper

Two novel components:
1. Scattering Transform observation encoder (provably Lipschitz-stable, 0 learned params)
2. FNO action decoder (resolution-invariant, frequency-domain operator learning)
"""

import torch
import torch.nn as nn
import yaml

from .scattering_encoder import ScatteringEncoder
from .vision_encoder import VisionEncoder
from .dual_vision_encoder import DualVisionEncoder
from .language_encoder import LanguageEncoder, PretrainedLanguageEncoder
from .fusion import FusionModule
from .fno_decoder import FNODecoder
from .mlp_head import MLPActionGripperHead


class FNOVLA(nn.Module):
    """Complete FNO-VLA model with Scattering Transform + FNO."""

    def __init__(self, config: dict):
        super().__init__()
        vc = config['model']['vision']
        lc = config['model']['language']
        pc = config['model']['proprio']
        fc = config['model']['fusion']
        fnoc = config['model']['fno']

        # Vision encoder — config-switchable for the ablation matrix:
        #   scattering-only, tinyvit-only, or dual (tinyvit + scattering).
        sc = config['model'].get('scattering', {})
        tv = config['model'].get('tinyvit', {})
        use_scattering = sc.get('enabled', True)
        use_tinyvit = tv.get('enabled', False)
        scat_kwargs = dict(
            J=sc.get('J', 3), L=sc.get('L', 12),
            proj_channels=sc.get('proj_channels', 256),
            num_res_blocks=sc.get('num_res_blocks', 6),
        )
        # Keep the two camera views as separate token streams (vs averaging) so
        # the fusion transformer can attend to the wrist view for grasp-depth.
        sep = config['model'].get('separate_views', False)

        if use_tinyvit:
            self.vision = DualVisionEncoder(
                img_size=vc['img_size'], embed_dim=fc['shared_dim'],
                use_scattering=use_scattering, use_tinyvit=True,
                scattering_kwargs=scat_kwargs,
                tinyvit_name=tv.get('model_name', 'tiny_vit_21m_224.dist_in22k_ft_in1k'),
                tinyvit_dropout=tv.get('dropout', 0.3),
                separate_views=sep,
                gated_fusion=config['model'].get('gated_fusion', False),
            )
        elif use_scattering:
            self.vision = ScatteringEncoder(
                img_size=vc['img_size'], embed_dim=fc['shared_dim'],
                separate_views=sep, **scat_kwargs,
            )
        else:
            # Legacy TinyViT-from-timm fallback (3-channel only)
            self.vision = VisionEncoder(
                img_size=vc['img_size'],
                pretrained=vc.get('pretrained', True),
                freeze=vc.get('freeze', False),
            )

        # Language encoder — frozen pretrained sentence model (robust grounding)
        # or the from-scratch tiny encoder.
        if lc.get('pretrained', False):
            # freeze defaults True (frozen MiniLM). For a small trainable encoder
            # (e.g. bert-tiny) set language.freeze: false to fine-tune it.
            self.language = PretrainedLanguageEncoder(
                model_name=lc['model_name'], out_dim=lc['embed_dim'],
                freeze=lc.get('freeze', True),
            )
        else:
            self.language = LanguageEncoder(
                vocab_size=lc['vocab_size'],
                embed_dim=lc['embed_dim'],
                num_layers=lc['num_layers'],
                num_heads=lc['num_heads'],
                mlp_dim=lc['mlp_dim'],
                max_seq_len=lc['max_seq_len'],
            )

        # Proprioception encoder: simple 2-layer MLP
        self.proprio = nn.Sequential(
            nn.Linear(pc['input_dim'], pc['hidden_dim']),
            nn.ReLU(),
            nn.Linear(pc['hidden_dim'], pc['output_dim']),
        )

        # Fusion module
        self.fusion = FusionModule(
            vision_dim=self.vision.embed_dim,
            language_dim=lc['embed_dim'],
            proprio_dim=pc['output_dim'],
            shared_dim=fc['shared_dim'],
            num_layers=fc['num_layers'],
            num_heads=fc['num_heads'],
            mlp_dim=fc['mlp_dim'],
            latent_dim=fc['latent_dim'],
            fno_width=fnoc['width'],
            fno_num_layers=fnoc['num_layers'],
            separate_pool=fc.get('separate_pool', False),
            vision_pool=fc.get('vision_pool', 'mean'),
        )

        # Action head — config-switchable (fno | mlp) for the ablation matrix.
        # Both return (actions [B,H,6], gripper [B,H,1]); only FNO is
        # resolution-invariant. Stored as self.fno for checkpoint/key continuity.
        self.head_type = config['model'].get('head', 'fno')
        if self.head_type == 'mlp':
            self.fno = MLPActionGripperHead(
                latent_dim=fc['latent_dim'],
                hidden_dim=fnoc.get('mlp_hidden', 1024),
                num_layers=fnoc.get('mlp_layers', 4),
                action_dim=fnoc['action_dim'],
                chunk_size=fnoc['chunk_size'],
            )
        else:
            self.fno = FNODecoder(
                latent_dim=fc['latent_dim'],
                width=fnoc['width'],
                num_layers=fnoc['num_layers'],
                modes=fnoc['modes'],
                action_dim=fnoc['action_dim'],
                chunk_size=fnoc['chunk_size'],
            )

        # Gripper is predicted per-timestep inside the action head.

        # Auxiliary x-y localization head (TRAIN-ONLY; computed only in .train()
        # mode -> dropped at inference for 0 deploy cost, SwiftVLA-style). Predicts
        # the grasp-target eef x-y from the fused latent z, forcing the
        # representation to encode object location -> sharpens single-shot x-y
        # (the diagnosed high-variance localization failure). aux_xy_dim=2 (x,y).
        self.aux_xy = config['model'].get('aux_xy', False)
        if self.aux_xy:
            self.aux_head = nn.Sequential(
                nn.Linear(fc['latent_dim'], 128), nn.GELU(), nn.Linear(128, 2))

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images:         [B, 3, H, W]
            token_ids:      [B, L]
            proprio:        [B, 8]
            attention_mask:  [B, L] (optional)
            output_size:    int, override action chunk length (resolution-invariance)

        Returns:
            dict with:
                'actions':  [B, H, 6] continuous actions
                'gripper':  [B, 1] gripper probability
        """
        # Encode each modality
        vis_tokens = self.vision(images)                              # [B, N_v, vis_dim]
        lang_tokens = self.language(token_ids, attention_mask)        # [B, L, lang_dim]
        proprio_embed = self.proprio(proprio)                         # [B, proprio_out]

        # Fuse + generate FiLM params
        z, film_params = self.fusion(vis_tokens, lang_tokens, proprio_embed)

        # Decode actions + per-timestep gripper
        actions, gripper = self.fno(z, film_params, output_size=output_size)
        # actions: [B, H, 6], gripper: [B, H, 1]

        out = {'actions': actions, 'gripper': gripper}
        # aux head only in training -> zero inference cost (dropped at deploy)
        if self.aux_xy and self.training:
            out['aux_xy'] = self.aux_head(z)   # [B, 2] predicted grasp-target x-y
        return out

    def count_params(self) -> dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in [
            ('vision', self.vision),
            ('language', self.language),
            ('proprio', self.proprio),
            ('fusion', self.fusion),
            ('fno', self.fno),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    @classmethod
    def from_config(cls, config_path: str) -> 'FNOVLA':
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)
