"""
Proxy evaluation (no simulator required).

Reports the metrics we CAN measure without LIBERO rollouts:
  - open-loop action prediction error (real units) on a DEMO-LEVEL holdout
  - per-step gripper accuracy
  - trajectory smoothness (mean |jerk|) vs ground truth
  - inference latency / implied control Hz
  - resolution-invariance consistency (same weights at 10/16/50 steps)
  - parameter count

These are NOT task success rate (that needs the sim). They measure imitation
fit quality, smoothness, speed, and the resolution-invariance property.
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import FNOVLA
from src.data.libero_dataset import LIBERODataset
from src.data.tokenizer import SimpleTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='E:/fno_data/checkpoints/best.pt')
    ap.add_argument('--data_dir', default='E:/fno_data')
    ap.add_argument('--suite', default='libero_object')
    ap.add_argument('--max_test', type=int, default=3000)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt['config']
    H = cfg['model']['fno']['chunk_size']
    amean = np.asarray(ckpt['action_mean'], dtype=np.float32)
    astd = np.asarray(ckpt['action_std'], dtype=np.float32)

    model = FNOVLA(cfg).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])  # EMA weights
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, val_loss(ema)={ckpt['val_loss']:.4f})")

    # tokenizer from checkpoint vocab so token IDs match the trained embeddings
    lc_cfg = cfg['model'].get('language', {})
    if lc_cfg.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tok = HFTokenizerAdapter(lc_cfg['model_name'], max_seq_len=lc_cfg['max_seq_len'])
    else:
        tok = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
        tok.word2idx = ckpt['tokenizer_word2idx']
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}

    # Build dataset (no augmentation; reuse training normalization stats + tokenizer)
    ds = LIBERODataset(args.data_dir, suite=args.suite, chunk_size=H, img_size=128,
                       tokenizer=tok, augment=False,
                       action_mean=amean, action_std=astd)

    # ---- DEMO-LEVEL holdout: last 10% of demos in each file are "test" ----
    test_idx = []
    for s_i, (fidx, gt) in enumerate(ds.samples):
        b = ds.data[fidx]['demo_boundaries']           # cumulative end positions
        d = int(np.searchsorted(b, gt, side='right'))   # which demo this sample is in
        if d >= int(len(b) * 0.9):                       # last 10% of demos
            test_idx.append(s_i)
    rng = np.random.default_rng(0)
    rng.shuffle(test_idx)
    test_idx = test_idx[:args.max_test]
    print(f"Demo-level test samples: {len(test_idx)}")

    # ---- run model over the test set ----
    pred_a, tgt_a, pred_g, tgt_g = [], [], [], []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(test_idx), bs):
            batch = [ds[j] for j in test_idx[i:i+bs]]
            img = torch.stack([b['image'] for b in batch]).to(dev)
            tk = torch.stack([b['token_ids'] for b in batch]).to(dev)
            am = torch.stack([b['attention_mask'] for b in batch]).to(dev)
            pr = torch.stack([b['proprio'] for b in batch]).to(dev)
            out = model(img, tk, pr, attention_mask=am)
            pred_a.append(out['actions'].cpu().numpy())
            tgt_a.append(torch.stack([b['actions'] for b in batch]).numpy())
            pred_g.append(out['gripper'][..., 0].cpu().numpy())
            tgt_g.append(torch.stack([b['gripper'] for b in batch]).numpy())
    pred_a = np.concatenate(pred_a); tgt_a = np.concatenate(tgt_a)   # [N,H,6] normalized
    pred_g = np.concatenate(pred_g); tgt_g = np.concatenate(tgt_g)   # [N,H]

    # un-normalize actions to real units
    pred_real = pred_a * astd + amean
    tgt_real = tgt_a * astd + amean

    mae = np.abs(pred_real - tgt_real).mean(axis=(0, 1))   # per-dim
    rmse = np.sqrt(((pred_real - tgt_real) ** 2).mean())
    g_acc = ((pred_g > 0.5) == (tgt_g > 0.5)).mean()

    # smoothness: mean |2nd difference| along time (jerk proxy), real units
    def jerk(x):  # x:[N,H,6]
        return np.abs(np.diff(x, n=2, axis=1)).mean()
    jerk_pred, jerk_gt = jerk(pred_real), jerk(tgt_real)

    # ---- inference latency (batch=1, deployment) ----
    img1 = ds[test_idx[0]]['image'].unsqueeze(0).to(dev)
    tk1 = ds[test_idx[0]]['token_ids'].unsqueeze(0).to(dev)
    pr1 = ds[test_idx[0]]['proprio'].unsqueeze(0).to(dev)
    with torch.no_grad():
        for _ in range(5):
            model(img1, tk1, pr1)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time(); N = 50
        for _ in range(N):
            model(img1, tk1, pr1)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        lat = (time.time() - t0) / N

    # ---- resolution-invariance consistency ----
    # produce the chunk at 50 steps, subsample to H, compare to native-H output
    with torch.no_grad():
        a_H = model(img1, tk1, pr1, output_size=H)['actions'][0].cpu().numpy()
        a_50 = model(img1, tk1, pr1, output_size=50)['actions'][0].cpu().numpy()
    sub = np.linspace(0, 49, H).round().astype(int)
    res_consist = np.abs(a_50[sub] - a_H).mean()

    print("\n================ PROXY RESULTS (no sim) ================")
    print(f"Params: {model.count_params()['total']/1e6:.2f}M")
    dims = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']
    print("\n-- Open-loop action prediction (demo-level holdout, REAL units) --")
    for d, m in zip(dims, mae):
        print(f"   MAE {d:7s}: {m:.4f}")
    print(f"   MAE overall : {mae.mean():.4f}   RMSE: {rmse:.4f}")
    print(f"\n-- Gripper --\n   per-step accuracy: {g_acc*100:.1f}%")
    print(f"\n-- Smoothness (mean |jerk|, lower=smoother) --")
    print(f"   predicted: {jerk_pred:.4f}   ground-truth: {jerk_gt:.4f}")
    print(f"\n-- Inference speed (batch=1, {dev.type}) --")
    print(f"   latency/forward: {lat*1000:.1f} ms  ->  {1/lat:.0f} forwards/s")
    print(f"   one forward = {H} action steps")
    print(f"\n-- Resolution-invariance --")
    print(f"   mean|a(50hz subsampled) - a(native {H})|: {res_consist:.4f}  (small = same underlying function)")
    print("========================================================")


if __name__ == '__main__':
    main()
