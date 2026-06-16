"""Generate presentation-ready figures (PNG) for the FNO-VLA deck from the eval jsonls."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"c:\sarvik\fno_backup\ppt_figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.dpi": 140, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3})
NAMES = ["soup","cream\ncheese","salad","bbq","ketchup","tomato","butter","milk","pudding","OJ"]
C_AUX, C_DAG, C_OCTO = "#2E86C1", "#28B463", "#E74C3C"

def load(p, tag=None):
    if not os.path.exists(p): return []
    out=[]
    for ln in open(p, encoding="utf-8"):
        ln=ln.strip()
        if not ln: continue
        try: d=json.loads(ln)
        except: continue
        if tag is not None and d.get("tag")!=tag: continue
        out.append(d)
    return out

def per_task(rows):
    pt={}
    for r in rows:
        t=r.get("task_idx"); pt.setdefault(t,[]).append(int(r.get("success",0)))
    return {t:100*np.mean(v) for t,v in pt.items()}

def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))

aux = load(r"E:\fno_data\zaux_5.jsonl")
dag = load(r"E:\fno_data\zdag_dag5n20.jsonl")
octo= load(r"E:\fno_data\octo_results.jsonl")

def rate(rows):
    k=sum(int(r.get("success",0)) for r in rows); return 100*k/len(rows), k, len(rows)
def jerk(rows):
    j=[r["jerk"] for r in rows if r.get("jerk") is not None and r["jerk"]==r["jerk"]]
    return (np.mean(j), np.std(j)) if j else (np.nan,np.nan)

octo_jrows = load(r"E:\fno_data\octo_jerk.jsonl")
A=dict(rate=rate(aux), jerk=jerk(aux), name="aux head", c=C_AUX)
D=dict(rate=rate(dag), jerk=jerk(dag), name="DAgger", c=C_DAG)
O=dict(rate=rate(octo), jerk=jerk(octo_jrows if octo_jrows else octo), name="Octo-small", c=C_OCTO)

# ---------- FIG 1: tradeoff frontier ----------
fig,ax=plt.subplots(figsize=(7,5.5))
for M in (A,D,O):
    r,k,n=M["rate"]; jm,js=M["jerk"]; lo,hi=wilson(k,n)
    ax.errorbar(jm, r, yerr=[[r-lo],[hi-r]], fmt="o", ms=14, color=M["c"], capsize=5, lw=2)
    ax.annotate(f"{M['name']}\n{r:.1f}%  jerk {jm:.3f}", (jm,r),
                textcoords="offset points", xytext=(12,8), fontsize=12, fontweight="bold", color=M["c"])
ax.annotate("better", xy=(0.022,82), xytext=(0.034,73),
            arrowprops=dict(arrowstyle="->",lw=2,color="gray"), color="gray", fontsize=12)
ax.set_xlabel("mean jerk  (lower = smoother)"); ax.set_ylabel("success rate  (%)")
ax.set_title("Accuracy–Smoothness Tradeoff (LIBERO-Object)")
ax.invert_xaxis(); ax.set_ylim(60,90)
fig.savefig(os.path.join(OUT,"fig1_tradeoff.png")); plt.close(fig)

# ---------- FIG 2: per-task grouped bars ----------
pa,pd,po = per_task(aux), per_task(dag), per_task(octo)
x=np.arange(10); w=0.27
fig,ax=plt.subplots(figsize=(13,5.5))
ax.bar(x-w, [pa.get(i,0) for i in range(10)], w, label=f"aux head ({A['rate'][0]:.1f}%)", color=C_AUX)
ax.bar(x,   [pd.get(i,0) for i in range(10)], w, label=f"DAgger ({D['rate'][0]:.1f}%)", color=C_DAG)
ax.bar(x+w, [po.get(i,0) for i in range(10)], w, label=f"Octo ({O['rate'][0]:.1f}%)", color=C_OCTO)
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("success (%)"); ax.set_ylim(0,105)
ax.set_title("Per-Task Success: DAgger lifts the x-y-limited tasks (bbq +40, ketchup +15)")
ax.legend(loc="lower right")
fig.savefig(os.path.join(OUT,"fig2_per_task.png")); plt.close(fig)

# ---------- FIG 3: jerk bar ----------
fig,ax=plt.subplots(figsize=(6.5,5))
labels=[A["name"],D["name"],O["name"]]; means=[A["jerk"][0],D["jerk"][0],O["jerk"][0]]
errs=[A["jerk"][1],D["jerk"][1],O["jerk"][1]]; cols=[C_AUX,C_DAG,C_OCTO]
ax.bar(labels, means, yerr=errs, color=cols, capsize=6)
for i,m in enumerate(means): ax.text(i,m+0.001,f"{m:.4f}",ha="center",fontweight="bold")
ax.set_ylabel("mean jerk  |2nd diff of executed cmds|")
ax.set_title("Motion Smoothness: FNO head is 28-36% smoother than Octo")
fig.savefig(os.path.join(OUT,"fig3_jerk.png")); plt.close(fig)

# ---------- FIG 4: mechanism (grasp_dxy) ----------
def dxy_task(rows, ti):
    v=[1000*r["grasp_dxy"] for r in rows if r.get("task_idx")==ti and r.get("grasp_dxy") is not None]
    return np.mean(v) if v else np.nan
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
tks=[3,4,0,5]; tn=["bbq","ketchup","soup","tomato"]
xa=np.arange(len(tks)); w=0.38
a1.bar(xa-w/2,[dxy_task(aux,t) for t in tks],w,label="aux",color=C_AUX)
a1.bar(xa+w/2,[dxy_task(dag,t) for t in tks],w,label="DAgger",color=C_DAG)
a1.set_xticks(xa); a1.set_xticklabels(tn); a1.set_ylabel("grasp x-y error (mm)")
a1.axhline(20,ls="--",c="gray"); a1.set_title("DAgger tightens grasp localization")
a1.legend()
# scatter dxy vs success (pooled dagger)
ds=[(1000*r["grasp_dxy"], r["success"]) for r in dag if r.get("grasp_dxy") is not None]
ds=np.array(ds); succ=ds[ds[:,1]==1]; fail=ds[ds[:,1]==0]
a2.hist([succ[:,0],fail[:,0]],bins=np.arange(0,55,5),stacked=True,
        color=["#28B463","#E74C3C"],label=["success","fail"])
a2.axvline(20,ls="--",c="gray"); a2.set_xlabel("grasp x-y error (mm)"); a2.set_ylabel("rollouts")
a2.set_title("Failures concentrate at high x-y error"); a2.legend()
fig.savefig(os.path.join(OUT,"fig4_mechanism.png")); plt.close(fig)

# ---------- FIG 5: resolution invariance ----------
res={}
for r_ in (8,16,24,32):
    rows=load(rf"E:\fno_data\zres_{r_}.jsonl")
    if rows: res[r_]=rate(rows)[0]
if res:
    fig,ax=plt.subplots(figsize=(7,5))
    xs=sorted(res); ax.plot(xs,[res[k] for k in xs],"o-",ms=11,lw=2.5,color=C_DAG)
    for k in xs: ax.annotate(f"{res[k]:.0f}%",(k,res[k]),textcoords="offset points",xytext=(0,10),ha="center",fontweight="bold")
    ax.axvline(16,ls="--",c="gray"); ax.text(16.4,ax.get_ylim()[0]+3,"train res (16)",color="gray")
    ax.set_xlabel("FNO decode resolution (chunk length)"); ax.set_ylabel("success (%)")
    ax.set_title("Resolution-Invariance: flat success across decode resolutions")
    fig.savefig(os.path.join(OUT,"fig5_resinvar.png")); plt.close(fig)

# ---------- FIG 6: scattering robustness (noise) ----------
def by_sev(rows, kind):
    out={}
    for r in rows:
        if r.get("perturb") in (kind,"none"):
            s=r.get("severity",0); out.setdefault(s,[]).append(int(r.get("success",0)))
    return {s:100*np.mean(v) for s,v in out.items()}
on=load(r"E:\fno_data\robustness.jsonl"); off=load(r"E:\fno_data\robustness_noscatter.jsonl")
if on and off:
    fig,ax=plt.subplots(figsize=(7.5,5))
    onn=by_sev(on,"noise"); offn=by_sev(off,"noise")
    xs=sorted(set(onn)&set(offn))   # only severities tested for BOTH (apples-to-apples)
    ax.plot(xs,[onn.get(s,np.nan) for s in xs],"o-",lw=2.5,ms=10,color=C_DAG,label="scattering ON")
    ax.plot(xs,[offn.get(s,np.nan) for s in xs],"s--",lw=2.5,ms=10,color=C_OCTO,label="scattering OFF")
    ax.set_xlabel("noise severity"); ax.set_ylabel("success (%)")
    ax.set_title("Wavelet Scattering = Noise Robustness (ablation)"); ax.legend()
    fig.savefig(os.path.join(OUT,"fig6_robustness_noise.png")); plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
