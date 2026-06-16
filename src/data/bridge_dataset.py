"""
Lazy BridgeData V2 loader. Reads the sharded HDF5 produced by
scripts/convert_bridge_to_hdf5.py. Built on the same lazy pattern as
LIBEROLazyDataset (frames read per-sample from disk -> fits in RAM), with two
Bridge-specific differences:
  * language is PER-DEMO (read from the demo's `language` attr), not per-file.
  * grasp_target (aux x-y) is recomputed here from ee_pos + the gripper action.

Output dict matches LIBERODataset, so model/training code is unchanged.
Pass an HFTokenizerAdapter as `tokenizer` and enable PretrainedLanguageEncoder
(language.pretrained: true) in the config for real instruction grounding.
"""
import os, glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .libero_dataset import _AddGaussianNoise


class BridgeLazyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 16,
        img_size: int = 128,
        tokenizer=None,                 # HFTokenizerAdapter (required for Bridge)
        augment: bool = True,
        normalize_actions: bool = True,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        corruption_aug: bool = False,
        stride: int = 1,
    ):
        assert tokenizer is not None, "Bridge needs a tokenizer (HFTokenizerAdapter)"
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.normalize_actions = normalize_actions
        self.tokenizer = tokenizer
        self.stride = max(1, int(stride))
        self._handles: dict[str, h5py.File] = {}

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        assert self.files, f"No bridge shards in {data_dir}"

        self.ram = {}      # fi -> {demo_key -> {actions, proprio, language, grasp_xy}}
        self.samples = []  # (fi, demo_key, t)
        for fi, fpath in enumerate(self.files):
            per_demo = {}
            with h5py.File(fpath, 'r') as f:
                for demo_key in sorted(f['data'].keys()):
                    d = f['data'][demo_key]
                    actions = d['actions'][:].astype(np.float32)              # [T,7]
                    ee_pos = d['obs']['ee_pos'][:]
                    ee_ori = d['obs']['ee_ori'][:]
                    gripper = d['obs']['gripper_states'][:]
                    joints = d['obs']['joint_states'][:]
                    proprio = np.concatenate([ee_pos, ee_ori, gripper, joints], axis=1).astype(np.float32)
                    language = d.attrs.get('language', '')
                    T = actions.shape[0]
                    close = np.where(actions[:, 6] > 0)[0]
                    gi = int(close[0]) if len(close) else T - 1
                    grasp_xy = ee_pos[gi, :2].astype(np.float32)             # [2] aux target
                    per_demo[demo_key] = {'actions': actions, 'proprio': proprio,
                                          'language': str(language), 'grasp_xy': grasp_xy}
                    for t in range(0, T - chunk_size + 1, self.stride):
                        self.samples.append((fi, demo_key, t))
            self.ram[fi] = per_demo

        # per-dim action norm over continuous dims (actions already percentile-clipped
        # at conversion, so plain mean/std is robust here).
        if action_mean is not None and action_std is not None:
            self.action_mean = action_mean.astype(np.float32)
            self.action_std = action_std.astype(np.float32)
        else:
            cont = np.concatenate(
                [dd['actions'][:, :6] for fi in self.ram for dd in self.ram[fi].values()], axis=0)
            self.action_mean = cont.mean(axis=0).astype(np.float32)
            self.action_std = (cont.std(axis=0) + 1e-6).astype(np.float32)

        if augment and corruption_aug:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=8),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.4, 2.2))], p=0.35),
                transforms.ToTensor(),
                transforms.RandomApply([_AddGaussianNoise(0.06)], p=0.35),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=8),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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

    def __len__(self):
        return len(self.samples)

    def _handle(self, fi):
        path = self.files[fi]
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, 'r')
            self._handles[path] = h
        return h

    def __getitem__(self, idx):
        fi, demo_key, t = self.samples[idx]
        dd = self.ram[fi][demo_key]
        obs = self._handle(fi)['data'][demo_key]['obs']

        agent = self.transform(obs['agentview_rgb'][t])
        wrist = self.transform(obs['eye_in_hand_rgb'][t])
        image = torch.cat([agent, wrist], dim=0)                 # [6,H,W]

        proprio = torch.tensor(dd['proprio'][t], dtype=torch.float32)
        chunk = dd['actions'][t:t + self.chunk_size]
        cont = chunk[:, :6]
        if self.normalize_actions:
            cont = (cont - self.action_mean) / self.action_std
        actions = torch.tensor(cont, dtype=torch.float32)
        gripper = torch.tensor((chunk[:, 6] > 0).astype(np.float32), dtype=torch.float32)

        ids, mask = self.tokenizer.encode(dd['language'])
        return {
            'image': image,
            'token_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'proprio': proprio,
            'actions': actions,
            'gripper': gripper,
            'grasp_target': torch.tensor(dd['grasp_xy'], dtype=torch.float32),  # [2] aux x-y
        }
