"""Line graph: predicted action trajectories (FNO smooth vs MLP jagged vs human)."""
import sys, os, glob, numpy as np, torch, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
H = 10  # steps to plot (min of the two chunk sizes)

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

def predict(bundle, img, prt, name):
    m, am, as_, tok = bundle
    ids, msk = tok.encode(name)
    ids = torch.tensor(ids).unsqueeze(0).to(dev); msk = torch.tensor(msk).unsqueeze(0).to(dev)
    with torch.no_grad():
        return m(img, ids, prt, attention_mask=msk)['actions'][0].cpu().numpy() * as_ + am

fno = load('E:/fno_data/run_full/best.pt')
mlp = load('E:/fno_data/run_mlp/best.pt')

# pick the demo window with the clearest motion (largest action range) for a vivid plot
f = sorted(glob.glob('E:/fno_data/libero_object/*.hdf5'))[0]
name = os.path.basename(f).replace('_demo.hdf5', '').replace('_', ' ')
best = None
with h5py.File(f, 'r') as h:
    for dk in sorted(h['data'].keys())[:15]:
        d = h['data'][dk]; T = d['actions'].shape[0]; t = T // 3
        if t + H >= T:
            continue
        rng = np.ptp(d['actions'][t:t + H, :3], axis=0).sum()
        if best is None or rng > best[0]:
            img = torch.cat([TF(d['obs']['agentview_rgb'][t]), TF(d['obs']['eye_in_hand_rgb'][t])], 0).unsqueeze(0).to(dev)
            pr = np.concatenate([d['obs']['ee_pos'][t], d['obs']['ee_ori'][t],
                                 d['obs']['gripper_states'][t], d['obs']['joint_states'][t]]).astype(np.float32)
            best = (rng, d['actions'][t:t + H, :6].copy(), img, torch.tensor(pr).unsqueeze(0).to(dev))

_, human, img, prt = best
fno_traj = predict(fno, img, prt, name)[:H]
mlp_traj = predict(mlp, img, prt, name)[:H]
steps = np.arange(H)
dims = [('Δx', 0), ('Δy', 1), ('Δz', 2)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (lbl, d) in zip(axes, dims):
    ax.plot(steps, human[:, d], color='#888888', lw=2, ls='--', marker='o', ms=4, label='Human demo')
    ax.plot(steps, fno_traj[:, d], color='#2a7fff', lw=2.5, marker='o', ms=4, label='FNO (ours)')
    ax.plot(steps, mlp_traj[:, d], color='#ff6b6b', lw=2.5, marker='s', ms=4, label='MLP baseline')
    ax.set_title(f'end-effector {lbl}', fontsize=11, fontweight='bold')
    ax.set_xlabel('timestep in action chunk'); ax.grid(alpha=0.3)
axes[0].set_ylabel('action value (real units)')
axes[0].legend(fontsize=9, loc='best')
fig.suptitle('Predicted action trajectories — FNO follows the demo smoothly, MLP is choppier (LIBERO-Object)',
             fontsize=12.5, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig('E:/fno_data/jerk_graph.png', dpi=150)
print("saved E:/fno_data/jerk_graph.png  (task:", name, ")")
