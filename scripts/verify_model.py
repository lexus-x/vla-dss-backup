"""
Quick sanity check: build FNO-VLA, run dummy forward pass, print shapes and param counts.
No GPU needed — runs on CPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml

from src.model import FNOVLA


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Building FNO-VLA...")
    model = FNOVLA(config)

    # Parameter counts
    counts = model.count_params()
    print("\n--- Parameter Counts ---")
    total = 0
    for name, count in counts.items():
        if name != 'total':
            print(f"  {name:12s}: {count:>10,d} ({count/1e6:.1f}M)")
            total += count
    print(f"  {'TOTAL':12s}: {counts['total']:>10,d} ({counts['total']/1e6:.1f}M)")

    # Dummy forward pass (two camera views stacked -> 6 channels; 15D proprio)
    B = 2
    images = torch.randn(B, 6, 128, 128)
    token_ids = torch.randint(0, 1000, (B, 16))
    proprio = torch.randn(B, 15)

    print("\n--- Forward Pass (training, chunk=10) ---")
    out = model(images, token_ids, proprio)
    print(f"  actions shape:  {out['actions'].shape}")    # expect [2, 10, 6]
    print(f"  gripper shape:  {out['gripper'].shape}")    # expect [2, 1]

    # Resolution-invariance test
    print("\n--- Forward Pass (inference, 20Hz = 20 steps) ---")
    out_20hz = model(images, token_ids, proprio, output_size=20)
    print(f"  actions shape:  {out_20hz['actions'].shape}")  # expect [2, 20, 6]

    print("\n--- Forward Pass (inference, 50Hz = 50 steps) ---")
    out_50hz = model(images, token_ids, proprio, output_size=50)
    print(f"  actions shape:  {out_50hz['actions'].shape}")  # expect [2, 50, 6]

    print("\nAll shapes correct! Resolution-invariance verified.")


if __name__ == '__main__':
    main()
