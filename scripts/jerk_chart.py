"""FNO vs MLP vs human-demo trajectory-jerk bar chart (smoothness novelty)."""
import sys, os, glob, numpy as np, torch, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
W = 8  # matched jerk window (run_mlp is chunk10, run_full chunk16 -> compare first 8 steps)

def load(p):
    ck = torch.load(p, map_location=dev); cfg = ck['config']
    m = FNOVLA(cfg).to(dev); m.load_state_dict(ck['model_state_dict']); m.eval()
    lc = cfg['model']['language']
    if lc.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tok = HFTokenizerAdapter(lc['model_name'], max_seq_len=lc['max_seq_len'])
    else:
        tok = SimpleTokenizer(max_seq_len=ck.get('tokenizer_max_seq_len', 64))
        tok.word2idx = ck['tokenizer_word2idx']; tok.idx2word = {v: k for k, v in tok.word2idx.items()}
    return m, np.asarray(ck['action_mean'], np.float32), np.asarray(ck['action_std'], np.float32), tok

def jerk(traj):
    return float(np.abs(np.diff(traj[:W], n=2, axis=0)).mean())

fno = load('E:/fno_data/run_full/best.pt')
mlp = load('E:/fno_data/run_mlp/best.pt')
files = sorted(glob.glob('E:/fno_data/libero_object/*.hdf5'))[:5]
J = {'Human demos': [], 'FNO (ours)': [], 'MLP baseline': []}
for f in files:
    name = os.path.basename(f).replace('_demo.hdf5', '').replace('_', ' ')
    with h5py.File(f, 'r') as h:
        for dk in sorted(h['data'].keys())[:8]:
            d = h['data'][dk]; T = d['actions'].shape[0]; t = T // 2
            if t + W >= T:
                continue
            img = torch.cat([TF(d['obs']['agentview_rgb'][t]), TF(d['obs']['eye_in_hand_rgb'][t])], 0).unsqueeze(0).to(dev)
            pr = np.concatenate([d['obs']['ee_pos'][t], d['obs']['ee_ori'][t],
                                 d['obs']['gripper_states'][t], d['obs']['joint_states'][t]]).astype(np.float32)
            prt = torch.tensor(pr).unsqueeze(0).to(dev)
            J['Human demos'].append(jerk(d['actions'][t:t + W, :6]))
            for tag, (m, am, as_, tok) in [('FNO (ours)', fno), ('MLP baseline', mlp)]:
                ids, msk = tok.encode(name)
                ids = torch.tensor(ids).unsqueeze(0).to(dev); msk = torch.tensor(msk).unsqueeze(0).to(dev)
                with torch.no_grad():
                    a = m(img, ids, prt, attention_mask=msk)['actions'][0].cpu().numpy() * as_ + am
                J[tag].append(jerk(a))

labels = list(J.keys()); means = [np.mean(J[k]) for k in labels]; stds = [np.std(J[k]) for k in labels]
print("=== mean trajectory jerk (|2nd diff|, 8-step window, lower=smoother) ===")
for k in labels:
    print(f"  {k:14s}: {np.mean(J[k]):.4f} +/- {np.std(J[k]):.4f}  (n={len(J[k])})")
imp = (np.mean(J['MLP baseline']) - np.mean(J['FNO (ours)'])) / np.mean(J['MLP baseline']) * 100
print(f"\nFNO is {imp:.1f}% smoother than MLP")

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(labels, means, yerr=stds, capsize=6,
              color=['#888888', '#2a7fff', '#ff6b6b'], edgecolor='black', linewidth=0.8)
ax.set_ylabel('Trajectory jerk  (mean |2nd difference|)', fontsize=11)
ax.set_title('Action smoothness on LIBERO-Object\n(lower = smoother)', fontsize=12, fontweight='bold')
for b, mv in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(stds) * 0.15, f'{mv:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.text(0.5, 0.92, f'FNO {imp:.0f}% smoother than MLP', transform=ax.transAxes,
        ha='center', fontsize=10, color='#2a7fff', fontweight='bold')
plt.tight_layout(); plt.savefig('E:/fno_data/jerk_comparison.png', dpi=150)
print("saved E:/fno_data/jerk_comparison.png")
