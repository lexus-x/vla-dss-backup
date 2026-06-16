"""
Compute trajectory jerk (mean |2nd difference| of the action chunk, real units)
for FNO vs MLP vs the human demos, on a set of real observations.
Lower = smoother.
"""
import os, sys, glob
import numpy as np
import torch
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def jerk(traj):  # traj [T,6]
    return float(np.abs(np.diff(traj, n=2, axis=0)).mean()) if traj.shape[0] >= 3 else float('nan')


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    cfg = ck['config']; m = FNOVLA(cfg); m.load_state_dict(ck['model_state_dict']); m.eval()
    lc = cfg['model']['language']
    if lc.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tok = HFTokenizerAdapter(lc['model_name'], max_seq_len=lc['max_seq_len'])
    else:
        tok = SimpleTokenizer(max_seq_len=ck.get('tokenizer_max_seq_len', 64))
        tok.word2idx = ck['tokenizer_word2idx']; tok.idx2word = {v: k for k, v in tok.word2idx.items()}
    return m, np.asarray(ck['action_mean'], np.float32), np.asarray(ck['action_std'], np.float32), \
           tok, cfg['model']['fno']['chunk_size']


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fno', default='E:/fno_data/run_chunk10_m5/best.pt')
    ap.add_argument('--mlp', default='E:/fno_data/run_mlp/best.pt')
    ap.add_argument('--n', type=int, default=40)
    args = ap.parse_args()

    models = {}
    for tag, p in [('FNO', args.fno), ('MLP', args.mlp)]:
        if os.path.exists(p):
            models[tag] = load_model(p)
            print(f"loaded {tag}: {p}")

    f = sorted(glob.glob('E:/fno_data/libero_object/*.hdf5'))[0]
    name = os.path.basename(f).replace('_demo.hdf5', '').replace('_', ' ')
    fno_j, mlp_j, demo_j = [], [], []
    with h5py.File(f, 'r') as h:
        demos = sorted(h['data'].keys())
        for dk in demos[:args.n]:
            d = h['data'][dk]
            T = d['actions'].shape[0]
            t = T // 2  # mid-trajectory frame
            tag0, (m, amean, astd, tok, H) = next(iter(models.items()))
            img = torch.cat([TF(d['obs']['agentview_rgb'][t]), TF(d['obs']['eye_in_hand_rgb'][t])], 0).unsqueeze(0)
            pr = np.concatenate([d['obs']['ee_pos'][t], d['obs']['ee_ori'][t],
                                 d['obs']['gripper_states'][t], d['obs']['joint_states'][t]]).astype(np.float32)
            ids, msk = tok.encode(name)
            ids = torch.tensor(ids).unsqueeze(0); msk = torch.tensor(msk).unsqueeze(0); prt = torch.tensor(pr).unsqueeze(0)
            # demo (human) jerk over the matching window
            demo_j.append(jerk(d['actions'][t:t + H, :6]))
            with torch.no_grad():
                for tag, (m, amean, astd, tok2, H2) in models.items():
                    a = m(img, ids, prt, attention_mask=msk)['actions'][0].numpy() * astd + amean
                    (fno_j if tag == 'FNO' else mlp_j).append(jerk(a))

    def stat(x): x = [v for v in x if v == v]; return (np.mean(x), np.std(x)) if x else (float('nan'), 0)
    print(f"\n=== Trajectory jerk (mean |2nd diff|, real units; lower=smoother), n={args.n} demos ===")
    for tag, arr in [('Human demos', demo_j), ('FNO head', fno_j), ('MLP head', mlp_j)]:
        mu, sd = stat(arr)
        if mu == mu:
            print(f"  {tag:12s}: {mu:.4f} +/- {sd:.4f}")
    if fno_j and mlp_j:
        mu_f, _ = stat(fno_j); mu_m, _ = stat(mlp_j)
        print(f"\n  FNO is {(1 - mu_f / mu_m) * 100:+.1f}% vs MLP   (negative = FNO smoother)")


if __name__ == '__main__':
    main()
