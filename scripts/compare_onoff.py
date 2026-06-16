"""Scattering ON vs OFF robustness (the FAIR ablation: same data/scale, only scattering differs).
ON  = FNO robustness.jsonl (tag rob)   |   OFF = robustness_noscatter.jsonl (tag robns)."""
import json, csv, os
def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
on  = load(r"E:\fno_data\robustness.jsonl")
off = load(r"D:\eroot\fno_data\robustness_noscatter.jsonl")
if not off: off = load(r"E:\fno_data\robustness_noscatter.jsonl")

def rate(rows, k, s):
    g = [r for r in rows if (r.get("perturb")=="none") ] if k=="_clean" else \
        [r for r in rows if r.get("perturb")==k and int(r.get("severity",0))==s]
    return (round(100*sum(r["success"] for r in g)/len(g)), len(g)) if g else (None,0)

print(f"{'perturb':12s}{'sev':>4}{'ON%':>7}{'OFF%':>7}{'delta':>7}  (pos = scattering HELPS)")
rows=[["perturb","severity","scatterON_pct","scatterON_n","scatterOFF_pct","scatterOFF_n","delta_on_minus_off"]]
o,on_n = rate(on,"_clean",0); f,off_n = rate(off,"_clean",0)
if o is not None and f is not None:
    print(f"{'clean':12s}{0:>4}{o:>7}{f:>7}{o-f:>7}")
    rows.append(["clean",0,o,on_n,f,off_n,o-f])
for k in ["noise","blur","brightness"]:
    for s in [1,2,3]:
        o,onn = rate(on,k,s); f,offn = rate(off,k,s)
        d = (o-f) if (o is not None and f is not None) else ""
        tag = ("" if d=="" else ("scattering helps" if d>0 else ("tie" if d==0 else "OFF more robust")))
        print(f"{k:12s}{s:>4}{str(o if o is not None else '-'):>7}{str(f if f is not None else '-'):>7}{str(d):>7}  {tag}")
        rows.append([k,s,o,onn,f,offn,d])
with open(r"c:\sarvik\fno_backup\robustness_onoff.csv","w",newline="") as fh:
    csv.writer(fh).writerows(rows)
print(f"\nOFF rollouts so far: {len(off)}")
print("wrote robustness_onoff.csv  (delta>0 => scattering improves robustness = your claim)")
