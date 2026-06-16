"""
Resolution-invariance figure: the FNO action decoder produces a *continuous*
trajectory that can be sampled at ANY control rate from a single forward pass.
We decode the same latent at several output_sizes and overlay them on a
normalized time axis -> they lie on one curve (same underlying function).
An MLP head cannot do this (fixed output length).

Saves: E:/fno_data/fig_resolution_invariance.png
"""
import os, sys, glob
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

CKPT = os.environ.get('CKPT', 'E:/fno_data/run_full/best.pt')
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    ckpt = torch.load(CKPT, map_location='cpu')
    cfg = ckpt['config']
    m = FNOVLA(cfg); m.load_state_dict(ckpt['model_state_dict']); m.eval()
    amean = np.asarray(ckpt['action_mean'], np.float32); astd = np.asarray(ckpt['action_std'], np.float32)
    H = cfg['model']['fno']['chunk_size']

    lc = cfg['model']['language']
    if lc.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tok = HFTokenizerAdapter(lc['model_name'], max_seq_len=lc['max_seq_len'])
    else:
        tok = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
        tok.word2idx = ckpt['tokenizer_word2idx']; tok.idx2word = {v: k for k, v in tok.word2idx.items()}

    # one real observation
    f = sorted(glob.glob('E:/fno_data/libero_object/*.hdf5'))[0]
    name = os.path.basename(f).replace('_demo.hdf5', '').replace('_', ' ')
    with h5py.File(f, 'r') as h:
        d = h['data'][sorted(h['data'].keys())[0]]
        ag, wr = d['obs']['agentview_rgb'][0], d['obs']['eye_in_hand_rgb'][0]
        pr = np.concatenate([d['obs']['ee_pos'][0], d['obs']['ee_ori'][0],
                             d['obs']['gripper_states'][0], d['obs']['joint_states'][0]]).astype(np.float32)
    img = torch.cat([TF(ag), TF(wr)], 0).unsqueeze(0)
    ids, msk = tok.encode(name)
    ids = torch.tensor(ids).unsqueeze(0); msk = torch.tensor(msk).unsqueeze(0)
    prt = torch.tensor(pr).unsqueeze(0)

    sizes = [H, 2 * H, 5 * H]   # e.g. 16, 32, 80  (≈ 1x, 2x, 5x control rate)
    dims = ['dx', 'dy', 'dz']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    with torch.no_grad():
        for sz in sizes:
            a = m(img, ids, prt, attention_mask=msk, output_size=sz)['actions'][0].numpy() * astd + amean
            t = np.linspace(0, 1, sz)
            for j, ax in enumerate(axes):
                ax.plot(t, a[:, j], marker='o', ms=3, label=f'{sz} steps')
    for j, ax in enumerate(axes):
        ax.set_title(dims[j]); ax.set_xlabel('normalized time'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('action'); axes[0].legend(title='output rate', fontsize=8)
    fig.suptitle('FNO resolution-invariance: one forward pass, sampled at any rate (curves coincide). '
                 'An MLP head is locked to a single fixed length.', fontsize=10)
    fig.tight_layout()
    out = 'E:/fno_data/fig_resolution_invariance.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
