"""
Lazy, multi-suite LIBERO dataset for cross-suite pretraining.

Unlike LIBERODataset (which loads every image into RAM — ~15 GB per suite, so
all ~130 tasks would need ~195 GB), this keeps only the small actions/proprio
arrays in memory and reads the two camera frames per sample from HDF5 on demand.
That fits in RAM and also avoids the Windows spawn pickling-OOM (no large image
array is copied to each worker).

Output dict matches LIBERODataset exactly so the model/training code is unchanged.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import SimpleTokenizer


class LIBEROLazyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        suites: list[str],
        chunk_size: int = 16,
        img_size: int = 128,
        tokenizer: SimpleTokenizer | None = None,
        augment: bool = True,
        normalize_actions: bool = True,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        stride: int = 1,
    ):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.normalize_actions = normalize_actions
        self.stride = max(1, int(stride))
        self._handles: dict[str, h5py.File] = {}   # per-worker lazy HDF5 handles

        # Gather every hdf5 across the requested suites.
        hdf5_files = []
        for suite in suites:
            hdf5_files += sorted(glob.glob(os.path.join(data_dir, suite, "*.hdf5")))
        assert len(hdf5_files) > 0, f"No HDF5 files found for suites {suites} in {data_dir}"

        # Per-file metadata + small arrays (actions/proprio) held in RAM.
        self.files = hdf5_files
        self.task_names = []
        self.ram = {}        # file_idx -> {demo_key -> {actions, proprio}}, task_name
        self.samples = []    # (file_idx, demo_key, t)

        for fi, fpath in enumerate(hdf5_files):
            task_name = os.path.basename(fpath).replace("_demo.hdf5", "").replace("_", " ")
            self.task_names.append(task_name)
            per_demo = {}
            with h5py.File(fpath, 'r') as f:
                for demo_key in sorted(f['data'].keys()):
                    d = f['data'][demo_key]
                    actions = d['actions'][:].astype(np.float32)          # [T,7]
                    ee_pos = d['obs']['ee_pos'][:]
                    ee_ori = d['obs']['ee_ori'][:]
                    gripper = d['obs']['gripper_states'][:]
                    joints = d['obs']['joint_states'][:]
                    proprio = np.concatenate([ee_pos, ee_ori, gripper, joints], axis=1).astype(np.float32)  # [T,15]
                    per_demo[demo_key] = {'actions': actions, 'proprio': proprio}
                    T = actions.shape[0]
                    for t in range(0, T - chunk_size + 1, self.stride):
                        self.samples.append((fi, demo_key, t))
            self.ram[fi] = {'demos': per_demo, 'task_name': task_name}

        # Tokenizer over ALL task names across suites.
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_seq_len=64)
            self.tokenizer.fit(self.task_names)
        else:
            self.tokenizer = tokenizer

        # Per-dim action normalization (continuous dims only) over ALL suites.
        if action_mean is not None and action_std is not None:
            self.action_mean = action_mean.astype(np.float32)
            self.action_std = action_std.astype(np.float32)
        else:
            cont = np.concatenate(
                [dd['actions'][:, :6] for fi in self.ram
                 for dd in self.ram[fi]['demos'].values()], axis=0)
            self.action_mean = cont.mean(axis=0).astype(np.float32)
            self.action_std = (cont.std(axis=0) + 1e-6).astype(np.float32)

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=8),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def _handle(self, fi: int) -> h5py.File:
        # Open lazily and cache per worker (each DataLoader worker gets its own
        # copy of self, so its own handle dict — safe with num_workers>0).
        path = self.files[fi]
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, 'r')
            self._handles[path] = h
        return h

    def __getitem__(self, idx: int) -> dict:
        fi, demo_key, t = self.samples[idx]
        d = self.ram[fi]
        obs = self._handle(fi)['data'][demo_key]['obs']

        # Two camera frames read from disk (only the single timestep we need).
        agent = self.transform(obs['agentview_rgb'][t])      # [3,H,W]
        wrist = self.transform(obs['eye_in_hand_rgb'][t])    # [3,H,W]
        image = torch.cat([agent, wrist], dim=0)             # [6,H,W]

        dd = d['demos'][demo_key]
        proprio = torch.tensor(dd['proprio'][t], dtype=torch.float32)     # [15]

        chunk = dd['actions'][t:t + self.chunk_size]         # [chunk,7]
        cont = chunk[:, :6]
        if self.normalize_actions:
            cont = (cont - self.action_mean) / self.action_std
        actions = torch.tensor(cont, dtype=torch.float32)               # [chunk,6]
        gripper = torch.tensor((chunk[:, 6] > 0).astype(np.float32), dtype=torch.float32)  # [chunk]

        ids, mask = self.tokenizer.encode(d['task_name'])
        return {
            'image': image,
            'token_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'proprio': proprio,
            'actions': actions,
            'gripper': gripper,
        }
