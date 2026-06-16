"""X-Y localization root-cause diagnostic (CPU-only, existing eval logs).
Tests: is the x-y error a SYSTEMATIC precision floor (perception limit) or
POSITION-dependent (data coverage)? And how tightly does x-y predict success?"""
import json, numpy as np
rows = [json.loads(l) for l in open(r"E:\fno_data\zsv_ep5diag.jsonl") if l.strip()]
rows = [r for r in rows if r.get("tag")=="sv_ep5d" and "grasp_dxy" in r and r["grasp_dxy"] is not None]
NAMES={0:"soup",1:"cream cheese",2:"salad",3:"bbq",4:"ketchup",5:"tomato",6:"butter",7:"milk",8:"pudding",9:"OJ"}
dxy = np.array([r["grasp_dxy"] for r in rows]); succ = np.array([r["success"] for r in rows])

print("=== 1) X-Y error distribution (mm) ??is there a systematic FLOOR? ===")
print(f"  ALL      : min {dxy.min()*1000:.1f}  median {np.median(dxy)*1000:.1f}  mean {dxy.mean()*1000:.1f}  max {dxy.max()*1000:.1f}")
print(f"  SUCCESS  : min {dxy[succ==1].min()*1000:.1f}  median {np.median(dxy[succ==1])*1000:.1f}  mean {dxy[succ==1].mean()*1000:.1f}")
print(f"  FAILURE  : min {dxy[succ==0].min()*1000:.1f}  median {np.median(dxy[succ==0])*1000:.1f}  mean {dxy[succ==0].mean()*1000:.1f}")
print("  -> if SUCCESS min/median is still ~10-15mm, the model has a systematic precision floor (perception),")
print("     not just bad coverage (which would let some grasps be near-perfect ~2-3mm).")

print("\n=== 2) Success rate vs x-y error bin ??does x-y PREDICT success? ===")
bins = [(0,0.010),(0.010,0.020),(0.020,0.030),(0.030,1.0)]
for lo,hi in bins:
    m=(dxy>=lo)&(dxy<hi); n=m.sum()
    sr=100*succ[m].mean() if n else float('nan')
    print(f"  dxy {int(lo*1000):2d}-{int(hi*1000) if hi<1 else 999:>3}mm : success {sr:5.1f}%  (n={n})")

print("\n=== 3) Per-object: x-y tolerance (do narrow objects need tighter x-y?) ===")
print(f"  {'obj':14s}{'SR':>4}{'dxy_OK':>9}{'dxy_FAIL':>10}{'min_dxy':>9}")
for ti in range(10):
    rs=[r for r in rows if r["task_idx"]==ti]
    if not rs: continue
    d=np.array([r["grasp_dxy"] for r in rs]); s=np.array([r["success"] for r in rs])
    ok = d[s==1]*1000; f = d[s==0]*1000
    print(f"  {NAMES[ti]:14s}{int(100*s.mean()):>4}{(ok.mean() if len(ok) else float('nan')):>9.1f}"
          f"{(f.mean() if len(f) else float('nan')):>10.1f}{d.min()*1000:>9.1f}")

