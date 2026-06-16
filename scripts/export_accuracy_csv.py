"""Export accuracy results for all models to graph-ready CSVs (raw + per-task + summary)."""
import json, csv, os, numpy as np
NAMES = {0:"alphabet_soup",1:"cream_cheese",2:"salad_dressing",3:"bbq_sauce",4:"ketchup",
         5:"tomato_sauce",6:"butter",7:"milk",8:"chocolate_pudding",9:"orange_juice"}
# (model_label, params, file, tag)
MODELS = [
    ("FNO-VLA_baseline",    "28.9M", r"E:\fno_data\zerror_expA.jsonl", "exec8"),
    ("FNO-VLA_sv_ep5",      "28.9M", r"E:\fno_data\zsv_ep5diag.jsonl",  "sv_ep5d"),
    ("FNO-VLA_sv_ep10",     "28.9M", r"E:\fno_data\zsv_ep10diag.jsonl", "sv_ep10d"),
    ("FNO-VLA_sv_ep44",     "28.9M", r"E:\fno_data\zsv_eval.jsonl",     "sv_final"),
    ("Octo-Small",          "137M",  r"E:\fno_data\octo_results.jsonl", None),
]
def load(path, tag):
    if not os.path.exists(path): return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if (tag is None or r.get("tag") == tag)]

data = {lbl: (params, load(path, tag)) for lbl, params, path, tag in MODELS}

# 1) PER-ROLLOUT (raw)
with open(r"c:\sarvik\fno_backup\accuracy_per_rollout.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model","params","task_idx","task","rollout","success","steps","jerk"])
    for lbl,(params,rows) in data.items():
        for r in rows:
            jk = r.get("jerk"); jk = round(float(jk),5) if jk not in (None,) and not (isinstance(jk,float) and np.isnan(jk)) else ""
            w.writerow([lbl,params,r.get("task_idx"),NAMES.get(r.get("task_idx"),""),
                        r.get("rollout"),r.get("success"),r.get("steps"),jk])

# 2) PER-TASK (model x object)
with open(r"c:\sarvik\fno_backup\accuracy_per_task.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model","task","n","successes","success_pct"])
    for lbl,(params,rows) in data.items():
        for ti in range(10):
            g=[r for r in rows if r.get("task_idx")==ti]
            if not g: continue
            n=len(g); s=sum(r["success"] for r in g)
            w.writerow([lbl,NAMES[ti],n,s,round(100*s/n,1)])

# 3) SUMMARY (model overall + CI)
with open(r"c:\sarvik\fno_backup\accuracy_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model","params","n","successes","success_pct","wilson_ci_halfwidth"])
    for lbl,(params,rows) in data.items():
        if not rows: continue
        n=len(rows); s=sum(r["success"] for r in rows); p=s/n
        ci=round(100*1.96*np.sqrt(p*(1-p)/n),1)
        w.writerow([lbl,params,n,s,round(100*p,1),ci])

print("wrote: accuracy_per_rollout.csv / accuracy_per_task.csv / accuracy_summary.csv")
