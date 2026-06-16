"""
Apples-to-apples open-loop prediction jerk:
  Human demos vs old FNO (run_full) vs MLP baseline (run_mlp) vs DINOv3-finetune (run_dinov3_finetune)

Same methodology as jerk_chart.py (W=8 step window, mean |2nd diff|, t = T//2).
"""
import sys, os, glob, numpy as np, torch, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
W = 8

def load(p):
    ck = torch.load(p, map_location=dev, weights_only=False); cfg = ck['config']
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

models = {
    'FNO old (run_full)':       load('E:/fno_data/run_full/best.pt'),
    'MLP baseline (run_mlp)':   load('E:/fno_data/run_mlp/best.pt'),
    'FNO DINOv3 (finetune)':    load('E:/fno_data/run_dinov3_finetune/best.pt'),
}

files = sorted(glob.glob('E:/fno_data/libero_object/*.hdf5'))[:5]
J = {k: [] for k in (['Human demos'] + list(models.keys()))}

for f in files:
    name = os.path.basename(f).replace('_demo.hdf5', '').replace('_', ' ')
    with h5py.File(f, 'r') as h:
        for dk in sorted(h['data'].keys())[:8]:
            d = h['data'][dk]; T = d['actions'].shape[0]; t = T // 2
            if t + W >= T: continue
            img = torch.cat([TF(d['obs']['agentview_rgb'][t]), TF(d['obs']['eye_in_hand_rgb'][t])], 0).unsqueeze(0).to(dev)
            pr = np.concatenate([d['obs']['ee_pos'][t], d['obs']['ee_ori'][t],
                                 d['obs']['gripper_states'][t], d['obs']['joint_states'][t]]).astype(np.float32)
            prt = torch.tensor(pr).unsqueeze(0).to(dev)
            J['Human demos'].append(jerk(d['actions'][t:t + W, :6]))
            for tag, (m, am, as_, tok) in models.items():
                ids, msk = tok.encode(name)
                ids = torch.tensor(ids).unsqueeze(0).to(dev); msk = torch.tensor(msk).unsqueeze(0).to(dev)
                with torch.no_grad():
                    a = m(img, ids, prt, attention_mask=msk)['actions'][0].cpu().numpy() * as_ + am
                J[tag].append(jerk(a))

print("=== mean trajectory jerk (|2nd diff|, 8-step window, lower=smoother, OPEN-LOOP PREDICTION) ===")
for k in J:
    print(f"  {k:30s}: {np.mean(J[k]):.4f} +/- {np.std(J[k]):.4f}  (n={len(J[k])})")
mlp_mean = np.mean(J['MLP baseline (run_mlp)'])
for tag in ['FNO old (run_full)', 'FNO DINOv3 (finetune)']:
    imp = (mlp_mean - np.mean(J[tag])) / mlp_mean * 100
    print(f"  {tag} vs MLP: {imp:+.1f}%  (positive = smoother than MLP)")
