"""Diagnostic: predicted vs demo action chunks (is the FNO head sharp or blurry/averaged?).
Runs on a trained LIBERO checkpoint (no sim needed). Sharp tracking => no multimodality
problem => generative head NOT justified. Blurry/flat predictions => multimodality => justified.
"""
import os, sys, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.model.fno_vla import FNOVLA
from src.data.libero_dataset import LIBERODataset

CFG = 'configs/finetune_dinov3_aug.yaml'
CKPT = 'E:/fno_data/run_dinov3_aug/epoch_15.pt'
OUT = r'c:\sarvik\fno_backup\action_chunk_diag.png'

cfg = yaml.safe_load(open(CFG))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FNOVLA(cfg).to(dev)
ck = torch.load(CKPT, map_location=dev)
model.load_state_dict(ck['model_state_dict']); model.eval()
amean = np.asarray(ck['action_mean'], np.float32); astd = np.asarray(ck['action_std'], np.float32)

ds = LIBERODataset('E:/fno_data', suite='libero_object', chunk_size=16, img_size=128,
                   augment=False, normalize_actions=True, action_mean=amean, action_std=astd)

# grab 4 samples spread across the dataset (different tasks/states)
idxs = np.linspace(0, len(ds) - 1, 4).astype(int)
preds, gts = [], []
with torch.no_grad():
    for i in idxs:
        s = ds[int(i)]
        img = s['image'].unsqueeze(0).to(dev)
        tid = s['token_ids'].unsqueeze(0).to(dev)
        msk = s['attention_mask'].unsqueeze(0).to(dev)
        pro = s['proprio'].unsqueeze(0).to(dev)
        out = model(img, tid, pro, attention_mask=msk)
        preds.append(out['actions'][0].cpu().numpy())   # [16,6] normalized
        gts.append(s['actions'].numpy())                 # [16,6] normalized

dims = ['dx', 'dy', 'dz']        # show the 3 position deltas (clearest)
fig, axes = plt.subplots(len(idxs), 3, figsize=(12, 11), sharex=True)
mse_all = []
for r in range(len(idxs)):
    p, g = preds[r], gts[r]
    mse_all.append(np.mean((p - g) ** 2))
    for c in range(3):
        ax = axes[r, c]
        ax.plot(g[:, c], 'k--', lw=2, label='demo (GT)')
        ax.plot(p[:, c], color='#2e75b6', lw=2, label='FNO pred')
        if r == 0: ax.set_title(dims[c], fontweight='bold')
        if c == 0: ax.set_ylabel(f'sample {r}\n(norm. action)')
        ax.grid(alpha=0.3)
        if r == 0 and c == 2: ax.legend(frameon=False, fontsize=9)
axes[-1, 1].set_xlabel('chunk timestep (0-15)')
fig.suptitle(f'Predicted vs demo action chunks — aug ep15 (LIBERO-Object)\n'
             f'mean chunk MSE = {np.mean(mse_all):.4f}  (low + tracks shape = SHARP)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches='tight')
print('saved', OUT, '| mean chunk MSE =', round(float(np.mean(mse_all)), 4))
