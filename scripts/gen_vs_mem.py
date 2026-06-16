"""Generalization vs memorization test:
   does success depend on distance from each eval scene to the NEAREST training scene?
   Memorizing -> success drops with distance. Generalizing -> success flat.
   Uses the Object aux run (zaux_5.jsonl) + LIBERO eval/train init states. CPU only.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.environ.get('LIBERO_SRC', r'C:\code\LIBERO'))
import torch, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from libero.libero import benchmark, get_libero_path

OBJ_FILE = {  # task_idx -> training demo hdf5 (object suite)
}
bench = benchmark.get_benchmark_dict()['libero_object']()
init_root = get_libero_path('init_states')
data_dir = r'E:\fno_data\libero_object'

# build per-task (eval states, train states) and object (dynamic) dims
rows = [json.loads(l) for l in open(r'E:\fno_data\zaux_5.jsonl', encoding='utf-8') if l.strip()]
recs = []  # (dist_to_nearest_train, success)
for ti in range(10):
    task = bench.get_task(ti)
    ev = np.asarray(torch.load(os.path.join(init_root, task.problem_folder, task.init_states_file)))  # [50,D]
    # find the matching training hdf5 by language
    fn = task.language.replace(' ', '_') + '_demo.hdf5'
    hp = os.path.join(data_dir, fn)
    if not os.path.exists(hp):
        cands = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        hp = os.path.join(data_dir, cands[ti])
    with h5py.File(hp, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        tr = np.stack([f['data'][d]['states'][0] for d in demos])  # [50,D]
    dyn = np.where(ev.std(0) > 1e-3)[0]          # dims that vary = object/scene config
    # nearest-train distance for each eval init (in object dims)
    nn = np.array([np.abs(tr[:, dyn] - ev[i, dyn]).sum(1).min() for i in range(ev.shape[0])])
    for r in rows:
        if r.get('task_idx') == ti and r.get('init_idx') is not None:
            recs.append((nn[r['init_idx']], int(r.get('success', 0))))

recs = np.array(recs)
d, s = recs[:, 0], recs[:, 1]
# correlation: if memorizing, corr(distance, success) strongly NEGATIVE
corr = np.corrcoef(d, s)[0, 1]
# quartile bins
q = np.quantile(d, [0, .25, .5, .75, 1.0])
labels, rates, ns = [], [], []
for i in range(4):
    m = (d >= q[i]) & (d <= q[i + 1]) if i == 3 else (d >= q[i]) & (d < q[i + 1])
    labels.append(f"Q{i+1}\n{q[i]:.2f}-{q[i+1]:.2f}")
    rates.append(100 * s[m].mean() if m.sum() else np.nan); ns.append(int(m.sum()))

print(f"n={len(recs)} rollouts | corr(distance, success) = {corr:+.3f}")
print("success by distance-to-nearest-training-scene (quartiles, near->far):")
for L, r, n in zip(labels, rates, ns):
    print(f"  {L.splitlines()[0]} (dist {L.splitlines()[1]}): {r:.0f}%  (n={n})")

INK="#1b2a41"; TEAL="#13a07a"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.titlesize":15,
                     "axes.titleweight":"bold","figure.dpi":220,"savefig.bbox":"tight"})
fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e7ebef")
ax.bar(range(4), rates, color=TEAL, width=0.6, edgecolor="white", lw=1.5)
for i, (r, n) in enumerate(zip(rates, ns)):
    ax.text(i, r+1.5, f"{r:.0f}%", ha="center", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(["nearest\n25%","","", "farthest\n25%"])
ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100)
ax.set_title(f"Success vs distance to nearest TRAINING scene\n(flat = generalizing; corr={corr:+.2f})")
ax.text(0.5, -0.22, "x-axis: eval scenes binned by how FAR they are from any training scene",
        transform=ax.transAxes, ha="center", fontsize=10, color="#7f8c8d")
fig.savefig(r"c:\sarvik\fno_backup\ppt_figures\gen_vs_mem.png")
print("-> ppt_figures/gen_vs_mem.png")
