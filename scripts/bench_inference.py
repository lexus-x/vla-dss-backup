"""Onboard-feasibility benchmark: inference latency, throughput, memory, params.
Defends the 'runs in real-time on modest hardware' use case. CPU+GPU, ~30s.
Run: python scripts/bench_inference.py --checkpoint E:/fno_data/run_dinov3_auxxy/epoch_5.pt
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, torch
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='E:/fno_data/run_dinov3_auxxy/epoch_5.pt')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--control_hz', type=float, default=20.0)  # LIBERO control rate
    ap.add_argument('--execute', type=int, default=8)          # actions consumed per inference
    args = ap.parse_args()

    for dev_name in (['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']):
        dev = torch.device(dev_name)
        ckpt = torch.load(args.checkpoint, map_location=dev)
        model = FNOVLA(ckpt['config']).to(dev).eval()
        model.load_state_dict(ckpt['model_state_dict'])
        n_params = sum(p.numel() for p in model.parameters())
        # dummy inputs matching the live eval (2 views stacked = 6ch, 15D proprio)
        img = torch.randn(1, 6, 128, 128, device=dev)
        tok = torch.zeros(1, 64, dtype=torch.long, device=dev)
        attn = torch.ones(1, 64, dtype=torch.long, device=dev)
        prop = torch.zeros(1, 15, device=dev)

        with torch.no_grad():
            for _ in range(args.warmup):
                model(img, tok, prop, attention_mask=attn)
            if dev_name == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                model(img, tok, prop, attention_mask=attn)
            if dev_name == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.iters

        ms = dt * 1000
        # one inference produces `execute` actions -> sustained control rate it can support
        sustained_hz = args.execute / dt
        mem = (torch.cuda.max_memory_allocated() / 1e6) if dev_name == 'cuda' else float('nan')
        print(f"\n=== {dev_name.upper()} ===")
        print(f"  params:            {n_params/1e6:.2f} M")
        print(f"  latency / infer:   {ms:.2f} ms")
        print(f"  throughput:        {1/dt:.1f} inferences/s")
        print(f"  sustains control:  {sustained_hz:.0f} Hz  (execute={args.execute}/infer)")
        print(f"  real-time @ {args.control_hz:.0f}Hz: "
              f"{'YES' if sustained_hz >= args.control_hz else 'NO'} "
              f"({sustained_hz/args.control_hz:.1f}x headroom)")
        if dev_name == 'cuda':
            print(f"  peak GPU memory:   {mem:.0f} MB")
    print("\n(Octo-Small ~27M but transformer-autoregressive; Octo-Base ~93M. "
          "Ours is a single forward pass -> low, constant latency.)")


if __name__ == '__main__':
    main()
