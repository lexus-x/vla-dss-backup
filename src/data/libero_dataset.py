"""
LIBERO dataset loader for FNO-VLA training.
Loads HDF5 demonstration files and produces (image, language, proprio, action) tuples.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import SimpleTokenizer


class _AddGaussianNoise:
    """Picklable Gaussian-noise transform (a module-level class, NOT a lambda,
    so Windows DataLoader workers can pickle the dataset)."""
    def __init__(self, std: float = 0.06):
        self.std = std
    def __call__(self, x):
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)


class LIBERODataset(Dataset):
    """
    Loads LIBERO demonstrations from HDF5 files.
    Each sample: (image, token_ids, attention_mask, proprio, action_chunk, gripper_chunk)
    """

    def __init__(
        self,
        data_dir: str,
        suite: str = "libero_object",
        chunk_size: int = 10,
        img_size: int = 128,
        tokenizer: SimpleTokenizer | None = None,
        augment: bool = True,
        normalize_actions: bool = True,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        corruption_aug: bool = False,
    ):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.corruption_aug = corruption_aug
        self.normalize_actions = normalize_actions

        suite_dir = os.path.join(data_dir, suite)
        hdf5_files = sorted(glob.glob(os.path.join(suite_dir, "*.hdf5")))
        assert len(hdf5_files) > 0, f"No HDF5 files found in {suite_dir}"

        # Extract task names from filenames
        task_names = []
        for f in hdf5_files:
            name = os.path.basename(f).replace("_demo.hdf5", "")
            # DAgger aggregation files (<lang>_dagger_demo.hdf5) share a task's
            # language -> strip the marker so they map to the same task_name.
            name = name.replace("_dagger", "")
            task_names.append(name)

        # Build tokenizer
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_seq_len=64)
            self.tokenizer.fit(task_names)
        else:
            self.tokenizer = tokenizer

        # Load all demonstrations into memory
        self.samples = []  # list of (file_idx, demo_idx, start_step)
        self.data = {}     # file_idx -> {images, proprios, actions, task_name}

        for file_idx, (fpath, task_name) in enumerate(zip(hdf5_files, task_names)):
            with h5py.File(fpath, 'r') as f:
                demos = sorted(f['data'].keys())
                all_images = []
                all_wrist = []
                all_proprios = []
                all_actions = []
                all_grasp_target = []   # self-supervised aux x-y label
                demo_boundaries = []

                for demo_key in demos:
                    demo = f['data'][demo_key]

                    # Images: [T, 128, 128, 3] uint8 — two camera views.
                    # The wrist (eye-in-hand) view is critical for grasp precision
                    # and was previously unused.
                    images = demo['obs']['agentview_rgb'][:]       # [T, H, W, 3]
                    wrist = demo['obs']['eye_in_hand_rgb'][:]      # [T, H, W, 3]

                    # Proprioception: ee_pos(3)+ee_ori(3)+gripper(2)+joint(7) = 15D
                    ee_pos = demo['obs']['ee_pos'][:]      # [T, 3]
                    ee_ori = demo['obs']['ee_ori'][:]       # [T, 3]
                    gripper = demo['obs']['gripper_states'][:]  # [T, 2]
                    joints = demo['obs']['joint_states'][:]     # [T, 7]
                    proprio = np.concatenate([ee_pos, ee_ori, gripper, joints], axis=1)  # [T, 15]

                    # Actions: [T, 7] (6 continuous + 1 gripper)
                    actions = demo['actions'][:]  # [T, 7]

                    T = actions.shape[0]
                    # Aux x-y label: eef x-y at the first gripper-close (the grasp
                    # target). Broadcast to all frames -> the model learns to predict
                    # WHERE it will grasp = the object x-y (sharpens localization).
                    close = np.where(actions[:, 6] > 0)[0]
                    gi = int(close[0]) if len(close) else T - 1
                    grasp_xy = ee_pos[gi, :2].astype(np.float32)            # [2]

                    start = len(all_images)
                    all_images.append(images)
                    all_wrist.append(wrist)
                    all_proprios.append(proprio)
                    all_actions.append(actions)
                    all_grasp_target.append(np.tile(grasp_xy, (T, 1)))     # [T, 2]

                    # Create sample indices: every valid starting point for a chunk
                    for t in range(T - chunk_size + 1):
                        self.samples.append((file_idx, start + t))

                    demo_boundaries.append(start + T)

                self.data[file_idx] = {
                    'images': np.concatenate(all_images, axis=0),
                    'wrist': np.concatenate(all_wrist, axis=0),
                    'proprios': np.concatenate(all_proprios, axis=0).astype(np.float32),
                    'actions': np.concatenate(all_actions, axis=0).astype(np.float32),
                    'grasp_target': np.concatenate(all_grasp_target, axis=0).astype(np.float32),
                    'task_name': task_name,
                    'demo_boundaries': demo_boundaries,
                }

        # Per-dim action normalization stats (continuous dims only, [:6]).
        # The 6 action dims have very different scales (position std ~0.4 vs
        # rotation std ~0.04); without standardization the loss is dominated by
        # position and the model under-learns rotation. Computed from this
        # dataset, or passed in (e.g. reused for a val split / eval).
        if action_mean is not None and action_std is not None:
            self.action_mean = action_mean.astype(np.float32)
            self.action_std = action_std.astype(np.float32)
        else:
            all_cont = np.concatenate(
                [d['actions'][:, :6] for d in self.data.values()], axis=0
            )
            self.action_mean = all_cont.mean(axis=0).astype(np.float32)
            self.action_std = (all_cont.std(axis=0) + 1e-6).astype(np.float32)

        # Image transforms
        if augment and corruption_aug:
            # Corruption augmentation (ImageNet-C-style) for robustness: blur +
            # brightness/contrast + Gaussian noise, randomly applied so the gate
            # learns to route between scattering (noise) and DINOv3 (blur).
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=8),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.4, 2.2))], p=0.35),
                transforms.ToTensor(),
                transforms.RandomApply([_AddGaussianNoise(0.06)], p=0.35),  # picklable (not a lambda)
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=8),
                # Mild photometric jitter (the 53%-model setting; stronger jitter
                # 0.3 coincided with the regressions).
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_idx, t = self.samples[idx]
        d = self.data[file_idx]

        # Two camera views at current timestep, stacked on the channel dim ->
        # [6, H, W]. The scattering encoder splits this back into two views.
        image = self.transform(d['images'][t])   # [3, H, W]
        wrist = self.transform(d['wrist'][t])     # [3, H, W]
        image = torch.cat([image, wrist], dim=0)  # [6, H, W]

        # Proprioception at current timestep (15D)
        proprio = torch.tensor(d['proprios'][t], dtype=torch.float32)  # [15]

        # Action chunk: next chunk_size actions
        action_chunk = d['actions'][t:t + self.chunk_size]  # [chunk, 7]
        continuous = action_chunk[:, :6]
        if self.normalize_actions:
            continuous = (continuous - self.action_mean) / self.action_std
        continuous_actions = torch.tensor(continuous, dtype=torch.float32)  # [chunk, 6]
        # Per-timestep gripper target (-1 open / 1 close -> 0 / 1). Per-step
        # (not one-per-chunk) so the model learns open/close *timing*.
        gripper_seq = (action_chunk[:, 6] > 0).astype(np.float32)  # [chunk]
        gripper_label = torch.tensor(gripper_seq, dtype=torch.float32)  # [chunk]

        # Language tokens
        token_ids, attn_mask = self.tokenizer.encode(d['task_name'])
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)

        return {
            'image': image,               # [6, 128, 128]
            'token_ids': token_ids,        # [max_seq_len]
            'attention_mask': attn_mask,   # [max_seq_len]
            'proprio': proprio,            # [15]
            'actions': continuous_actions,  # [chunk_size, 6]
            'gripper': gripper_label,       # [chunk_size]
            'grasp_target': torch.tensor(d['grasp_target'][t], dtype=torch.float32),  # [2] aux x-y label
        }
