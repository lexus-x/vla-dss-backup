"""Clean FNO-vs-Octo robustness comparison from the two jsonl logs -> table + CSV."""
import json, csv, os
def load(p, tagfilter=None):
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
fno  = load(r"E:\fno_data\robustness.jsonl")           # FNO sweep (tag rob)
octo = load(r"D:\eroot\fno_data\octo_robustness.jsonl") # Octo sweep

def rate(rows, k, s):
    if k == "_clean":
        g = [r for r in rows if r.get("perturb")=="none"]
    else:
        g = [r for r in rows if r.get("perturb")==k and int(r.get("severity",0))==s]
    if not g: return None, 0
    return round(100*sum(r["success"] for r in g)/len(g)), len(g)

print(f"{'perturb':12s}{'sev':>4}{'FNO%':>8}{'Octo%':>9}{'delta':>8}{'(neg=FNO more robust)':>24}")
rows_csv = [["perturb","severity","fno_success_pct","fno_n","octo_success_pct","octo_n","delta_octo_minus_fno"]]
# clean
f0,fn = rate(fno,"_clean",0); o0,on = rate(octo,"_clean",0)
if o0 is not None:
    print(f"{'clean':12s}{0:>4}{f0:>8}{o0:>9}{o0-f0:>8}")
    rows_csv.append(["clean",0,f0,fn,o0,on,o0-f0])
for k in ["noise","blur","brightness"]:
    for s in [1,2,3]:
        f,fn = rate(fno,k,s); o,on = rate(octo,k,s)
        fs = f if f is not None else "-"; os_ = o if o is not None else "-"
        d = (o-f) if (o is not None and f is not None) else ""
        tag = ""
        if d != "": tag = "<-- FNO more robust" if d < 0 else ("tie" if d==0 else "Octo more robust")
        print(f"{k:12s}{s:>4}{str(fs):>8}{str(os_):>9}{str(d):>8}   {tag}")
        rows_csv.append([k,s,f,fn,o,on,d])
with open(r"c:\sarvik\fno_backup\robustness_comparison.csv","w",newline="") as fh:
    csv.writer(fh).writerows(rows_csv)
print(f"\nocto rollouts so far: {len(octo)}   (s4-5 = 0 for both, omitted)")
print("wrote c:\\sarvik\\fno_backup\\robustness_comparison.csv")
