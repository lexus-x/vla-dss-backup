"""Export robustness sweep to graph-ready CSVs (degradation curves)."""
import json, csv, os, numpy as np
JF = r"E:\fno_data\robustness.jsonl"
NAMES = {0:"alphabet_soup",1:"cream_cheese",2:"salad_dressing",3:"bbq_sauce",4:"ketchup",
         5:"tomato_sauce",6:"butter",7:"milk",8:"chocolate_pudding",9:"orange_juice"}
rows = [json.loads(l) for l in open(JF, encoding="utf-8")] if os.path.exists(JF) else []

# normalize: clean (none,0) is the severity-0 point for EVERY perturbation curve
def cfgkey(r): return (r.get("perturb","none"), int(r.get("severity",0)))
clean = [r for r in rows if cfgkey(r) == ("none", 0)]

# 1) PER-ROLLOUT raw
with open(r"c:\sarvik\fno_backup\robustness_per_rollout.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["perturb","severity","task_idx","task","rollout","success","steps","jerk"])
    for r in rows:
        w.writerow([r.get("perturb"),r.get("severity"),r.get("task_idx"),NAMES.get(r.get("task_idx"),""),
                    r.get("rollout"),r.get("success"),r.get("steps"),round(float(r.get("jerk",float("nan"))),5)])

# 2) SUMMARY: degradation curve (perturb x severity -> success% + CI + jerk)
#    clean reproduced as severity 0 for each perturbation for easy plotting.
def summ(rs):
    n=len(rs); s=sum(x["success"] for x in rs); p=s/n if n else 0
    ci=round(100*1.96*np.sqrt(p*(1-p)/n),1) if n else 0
    jk=round(np.nanmean([float(x.get("jerk",float("nan"))) for x in rs]),5) if n else 0
    return n,s,round(100*p,1),ci,jk

with open(r"c:\sarvik\fno_backup\robustness_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["perturb","severity","n","successes","success_pct","wilson_ci_halfwidth","mean_jerk"])
    for k in ["noise","blur","brightness"]:
        # severity 0 = clean reference
        if clean:
            n,s,sp,ci,jk = summ(clean); w.writerow([k,0,n,s,sp,ci,jk])
        for sev in range(1,6):
            rs=[r for r in rows if cfgkey(r)==(k,sev)]
            if rs: n,s,sp,ci,jk=summ(rs); w.writerow([k,sev,n,s,sp,ci,jk])

# 3) PER-TASK (perturb x severity x object)
with open(r"c:\sarvik\fno_backup\robustness_per_task.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["perturb","severity","task","n","successes","success_pct"])
    for k in ["noise","blur","brightness"]:
        for sev in range(1,6):
            rs=[r for r in rows if cfgkey(r)==(k,sev)]
            for ti in range(10):
                g=[r for r in rs if r.get("task_idx")==ti]
                if g: w.writerow([k,sev,NAMES[ti],len(g),sum(x["success"] for x in g),round(100*sum(x["success"] for x in g)/len(g),1)])

print(f"rows read: {len(rows)}")
print("wrote: robustness_summary.csv (curves), robustness_per_rollout.csv (raw), robustness_per_task.csv")
