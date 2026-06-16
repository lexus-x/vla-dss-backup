import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=== model build + param count ===", flush=True)
from src.model import FNOVLA
m = FNOVLA.from_config(os.path.join(os.path.dirname(__file__), "..", "configs", "finetune_dinov3.yaml"))
c = m.count_params()
for k, v in c.items():
    print(f"  {k:10s} {v/1e6:7.3f}M", flush=True)
tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"  trainable  {tr/1e6:7.3f}M", flush=True)
print(f"  under 32M cap: {c['total'] <= 32e6}", flush=True)

print("=== sim deps ===", flush=True)
for mod in ["kymatio", "robosuite", "mujoco", "bddl", "libero"]:
    try:
        __import__(mod)
        print(f"  {mod}: OK", flush=True)
    except Exception as e:
        print(f"  {mod}: FAIL -> {type(e).__name__}: {e}", flush=True)
