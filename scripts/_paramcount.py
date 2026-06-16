import sys, os
sys.path.insert(0, ".")
from src.model import FNOVLA
print("=== mean-pool (current) ===")
m = FNOVLA.from_config("configs/finetune_dinov3.yaml")
cm = m.count_params(); tm = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"  total {cm['total']/1e6:.3f}M  trainable {tm/1e6:.3f}M  fusion {cm['fusion']/1e6:.3f}M")
print("=== attention-pool (Exp B) ===")
a = FNOVLA.from_config("configs/finetune_dinov3_attnpool.yaml")
ca = a.count_params(); ta = sum(p.numel() for p in a.parameters() if p.requires_grad)
print(f"  total {ca['total']/1e6:.3f}M  trainable {ta/1e6:.3f}M  fusion {ca['fusion']/1e6:.3f}M")
print(f"=== delta: +{(ca['total']-cm['total'])/1e6:.3f}M total, +{(ta-tm)/1e6:.3f}M trainable ===")
print(f"=== under 32M cap: {ca['total'] <= 32e6} ===")
