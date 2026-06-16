"""Export resolution-invariance results to graph-ready CSVs at multiple granularities."""
import json, csv, numpy as np
SIZES = [8, 16, 24, 32]
NAMES = {0:"alphabet_soup",1:"cream_cheese",2:"salad_dressing",3:"bbq_sauce",4:"ketchup",
         5:"tomato_sauce",6:"butter",7:"milk",8:"chocolate_pudding",9:"orange_juice"}

def load(s):
    rows = [json.loads(l) for l in open(fr"E:\fno_data\zres_{s}.jsonl", encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("tag") == f"res{s}"]

# 1) PER-ROLLOUT (raw data: every rollout, success + steps + jerk)
with open(r"c:\sarvik\fno_backup\resinvar_per_rollout.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["output_size","region","task_idx","task","rollout","success","steps","jerk"])
    for s in SIZES:
        region = "below_nyquist" if s < 16 else ("native" if s == 16 else "above_nyquist")
        for r in load(s):
            w.writerow([s, region, r["task_idx"], NAMES.get(r["task_idx"],r.get("task","")),
                        r.get("rollout"), r["success"], r.get("steps"), round(float(r.get("jerk",float("nan"))),5)])

# 2) PER-TASK (output_size x object: success count + rate + jerk)
with open(r"c:\sarvik\fno_backup\resinvar_per_task.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["output_size","task","n","successes","success_pct","mean_jerk"])
    for s in SIZES:
        rs = load(s)
        for ti in range(10):
            g = [r for r in rs if r["task_idx"] == ti]
            if not g: continue
            n = len(g); succ = sum(r["success"] for r in g)
            jk = np.nanmean([float(r.get("jerk",float("nan"))) for r in g])
            w.writerow([s, NAMES[ti], n, succ, round(100*succ/n,1), round(jk,5)])

# 3) SUMMARY (one row per resolution: success + CI + jerk)
with open(r"c:\sarvik\fno_backup\resinvar_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["output_size","region","n","successes","success_pct","wilson_ci_halfwidth","mean_jerk"])
    for s in SIZES:
        rs = load(s); n = len(rs); succ = sum(r["success"] for r in rs); p = succ/n
        ci = round(100*1.96*np.sqrt(p*(1-p)/n),1)   # ~Wald/Wilson halfwidth in pp
        jk = round(np.nanmean([float(r.get("jerk",float("nan"))) for r in rs]),5)
        region = "below_nyquist" if s < 16 else ("native" if s == 16 else "above_nyquist")
        w.writerow([s, region, n, succ, round(100*p,1), ci, jk])

print("wrote: resinvar_per_rollout.csv (raw, every rollout)")
print("       resinvar_per_task.csv    (output_size x object)")
print("       resinvar_results.csv     (summary + CI per resolution)")
