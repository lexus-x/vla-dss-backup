"""
Training script for FNO-VLA (Scattering + FNO).
"""

import os
import sys
import time
import math
import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import FNOVLA
from src.data.libero_dataset import LIBERODataset


class EMA:
    """Exponential moving average of model parameters.

    Policies trained by imitation are noisy; evaluating/deploying the EMA of
    the weights (as Octo does) is a reliable, low-cost success-rate boost.
    Works with the FNO's complex (cfloat) parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Only track trainable params (skip frozen TinyViT — it never changes).
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def store_and_apply(self, model: nn.Module):
        """Swap EMA weights into the model, backing up the live weights."""
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        """Restore the live weights saved by store_and_apply()."""
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = None


def train(config_path: str, data_dir: str, resume: str = None,
          anneal_epochs: int = 0, checkpoint_dir: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tc = config['training']
    dc = config['data']

    # Resume / explore-then-anneal overrides
    if anneal_epochs > 0:
        tc['max_epochs'] = anneal_epochs   # short run
        tc['warmup_steps'] = 0             # already warmed up
    if checkpoint_dir:
        tc['checkpoint_dir'] = checkpoint_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Fast-training flags for Ampere (RTX A6000): TF32 matmuls + cuDNN autotuner.
    # Big speedup, negligible accuracy impact, fixed input sizes.
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Build model
    print("Building model...")
    model = FNOVLA(config).to(device)
    resume_ckpt = None
    if resume:
        resume_ckpt = torch.load(resume, map_location=device)
        # non-strict: tolerates NEW heads not in the resume ckpt (e.g. aux_xy head),
        # which start randomly-initialized and train from scratch.
        missing, unexpected = model.load_state_dict(resume_ckpt['model_state_dict'], strict=False)
        if missing:    print(f"  [resume] new (random-init) keys: {missing}")
        if unexpected: print(f"  [resume] unused ckpt keys: {unexpected}")
        print(f"Resumed weights from {resume} (epoch {resume_ckpt.get('epoch')}, "
              f"val_loss={resume_ckpt.get('val_loss')})")
    counts = model.count_params()
    print(f"Total params: {counts['total']/1e6:.1f}M")
    for name, count in counts.items():
        if name != 'total':
            print(f"  {name}: {count/1e6:.1f}M")

    # Tokenizer: HF (frozen pretrained LM) or the built-in word tokenizer.
    # On fine-tune (--resume), reuse the pretrain tokenizer so token ids match.
    lc_cfg = config['model']['language']
    tokenizer = None
    if lc_cfg.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tokenizer = HFTokenizerAdapter(lc_cfg['model_name'], max_seq_len=lc_cfg['max_seq_len'])
    elif resume_ckpt is not None and 'tokenizer_word2idx' in resume_ckpt:
        from src.data.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer(max_seq_len=resume_ckpt.get('tokenizer_max_seq_len', 64))
        tokenizer.word2idx = resume_ckpt['tokenizer_word2idx']
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
        print(f"Reused tokenizer from resume ckpt (vocab {tokenizer.vocab_size})")

    # Reuse pretrain action-normalization on fine-tune (same action distribution).
    amean = astd = None
    if resume_ckpt is not None and 'action_mean' in resume_ckpt:
        import numpy as _np
        amean = _np.asarray(resume_ckpt['action_mean'], _np.float32)
        astd = _np.asarray(resume_ckpt['action_std'], _np.float32)
        print("Reused action mean/std from resume ckpt")

    # Load dataset — multi-suite lazy (cross-suite pretrain) or single-suite in-RAM.
    suites = dc.get('suites')
    if suites:
        from src.data.libero_lazy import LIBEROLazyDataset
        print(f"\nLoading cross-suite lazy dataset {suites} (stride {dc.get('stride', 1)})...")
        dataset = LIBEROLazyDataset(
            data_dir=data_dir, suites=suites,
            chunk_size=config['model']['fno']['chunk_size'],
            img_size=dc['img_size'],
            augment=dc['augmentation']['random_crop'],
            tokenizer=tokenizer, action_mean=amean, action_std=astd,
            stride=dc.get('stride', 1),
        )
    else:
        print(f"\nLoading {dc['suite']} dataset...")
        dataset = LIBERODataset(
            data_dir=data_dir, suite=dc['suite'],
            chunk_size=config['model']['fno']['chunk_size'],
            img_size=dc['img_size'],
            augment=dc['augmentation']['random_crop'],
            tokenizer=tokenizer, action_mean=amean, action_std=astd,
            corruption_aug=dc['augmentation'].get('corruption', False),
        )
    print(f"Dataset size: {len(dataset)} samples")

    # Update config vocab size to match tokenizer (only for the built-in encoder)
    if not lc_cfg.get('pretrained', False):
        config['model']['language']['vocab_size'] = max(
            config['model']['language']['vocab_size'],
            dataset.tokenizer.vocab_size
        )

    # Train/val split (90/10)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # On Windows (spawn) the in-RAM dataset is pickled to each worker, so keep
    # workers alive across epochs to pay that cost only once.
    num_workers = tc.get('num_workers', 4)
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=tc['batch_size'],
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set, batch_size=tc['batch_size'],
        shuffle=False, num_workers=max(0, num_workers // 2), pin_memory=True,
        persistent_workers=persistent,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],  # skip frozen TinyViT
        lr=tc['lr'],
        weight_decay=tc['weight_decay'],
    )
    if resume_ckpt is not None and 'optimizer_state_dict' in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            print("Resumed optimizer state (Adam moments preserved)")
        except Exception as e:
            print(f"[warn] could not load optimizer state: {e}")
    total_steps = len(train_loader) * tc['max_epochs']
    warmup_steps = min(tc.get('warmup_steps', 0), max(1, total_steps - 1))
    eta_min_ratio = 1e-6 / max(tc['lr'], 1e-12)

    def lr_lambda(step):
        # Linear warmup, then cosine decay to eta_min. The config declared
        # warmup_steps but the old CosineAnnealingLR ignored it.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return eta_min_ratio + (1 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA of weights (evaluate/deploy this, not the raw weights)
    ema = EMA(model, decay=tc.get('ema_decay', 0.999))

    # Loss functions
    if tc['loss']['action'] == 'smooth_l1':
        action_loss_fn = nn.SmoothL1Loss()
    elif tc['loss']['action'] == 'mse':
        action_loss_fn = nn.MSELoss()
    else:
        action_loss_fn = nn.L1Loss()

    gripper_loss_fn = nn.BCELoss()
    gripper_weight = tc['loss']['gripper_weight']
    aux_weight = tc['loss'].get('aux_xy_weight', 0.0)   # >0 enables aux x-y localization loss

    # Mixed precision
    scaler = GradScaler(enabled=tc['mixed_precision'])

    # Training loop
    save_dir = tc.get('checkpoint_dir') or os.path.join(os.path.dirname(config_path), '..', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints -> {save_dir}")

    best_val_loss = float('inf')

    for epoch in range(tc['max_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_gripper_loss = 0.0
        t0 = time.time()

        n_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{tc['max_epochs']}", ncols=110, leave=True)
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            proprio = batch['proprio'].to(device)
            target_actions = batch['actions'].to(device)
            target_gripper = batch['gripper'].to(device)
            # aux x-y target only exists in single-suite LIBERODataset; the lazy
            # cross-suite pretrain dataset omits it -> guard the access.
            target_grasp = batch['grasp_target'].to(device) if 'grasp_target' in batch else None

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=tc['mixed_precision']):
                out = model(images, token_ids, proprio, attention_mask=attn_mask)

                loss_action = action_loss_fn(out['actions'], target_actions)
                loss_gripper = gripper_loss_fn(out['gripper'].squeeze(-1), target_gripper)
                loss = loss_action + gripper_weight * loss_gripper
                if 'aux_xy' in out and aux_weight > 0 and target_grasp is not None:   # train-only aux x-y localization
                    loss = loss + aux_weight * nn.functional.mse_loss(out['aux_xy'], target_grasp)

            scaler.scale(loss).backward()

            if tc['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            epoch_action_loss += loss_action.item()
            epoch_gripper_loss += loss_gripper.item()

            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        epoch_time = time.time() - t0

        # Validation — on the EMA weights (these are what we deploy/checkpoint)
        ema.store_and_apply(model)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                token_ids = batch['token_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                proprio = batch['proprio'].to(device)
                target_actions = batch['actions'].to(device)
                target_gripper = batch['gripper'].to(device)

                out = model(images, token_ids, proprio, attention_mask=attn_mask)
                loss_action = action_loss_fn(out['actions'], target_actions)
                loss_gripper = gripper_loss_fn(out['gripper'].squeeze(-1), target_gripper)
                val_loss += (loss_action + gripper_weight * loss_gripper).item()

        val_loss /= max(1, len(val_loader))

        print(
            f"Epoch {epoch+1:3d}/{tc['max_epochs']} | "
            f"train_loss: {epoch_loss/n_batches:.4f} "
            f"(action: {epoch_action_loss/n_batches:.4f}, "
            f"gripper: {epoch_gripper_loss/n_batches:.4f}) | "
            f"val_loss(ema): {val_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_time:.1f}s"
        )

        def make_ckpt():
            # EMA weights are currently applied to `model`. Persist the
            # tokenizer info and action-normalization stats so eval can
            # reproduce conditioning and un-normalize the action outputs.
            ck = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # EMA weights
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'action_mean': dataset.action_mean,
                'action_std': dataset.action_std,
                'normalize_actions': dataset.normalize_actions,
                'tokenizer_max_seq_len': dataset.tokenizer.max_seq_len,
            }
            tok = dataset.tokenizer
            if hasattr(tok, 'word2idx'):       # built-in word tokenizer
                ck['tokenizer_word2idx'] = tok.word2idx
            else:                              # HF adapter
                ck['tokenizer_model_name'] = tok.model_name
            return ck

        # Save best model (EMA weights still applied here)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(make_ckpt(), os.path.join(save_dir, 'best.pt'))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint (configurable cadence so we can sim-test mid-training)
        if (epoch + 1) % tc.get('ckpt_every', 50) == 0:
            torch.save(make_ckpt(), os.path.join(save_dir, f'epoch_{epoch+1}.pt'))

        # Restore live (non-EMA) weights for the next training epoch
        ema.restore(model)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--data_dir', default='/home/sarvik/Desktop/VLA-main/LIBERO/datasets')
    parser.add_argument('--resume', default=None, help='checkpoint to warm-start from')
    parser.add_argument('--anneal_epochs', type=int, default=0,
                        help='short run length with LR cosine-annealed to ~0')
    parser.add_argument('--checkpoint_dir', default=None)
    args = parser.parse_args()
    train(args.config, args.data_dir, resume=args.resume,
          anneal_epochs=args.anneal_epochs, checkpoint_dir=args.checkpoint_dir)
