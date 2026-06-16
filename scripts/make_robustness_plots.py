"""Robustness ablation figures: DAgger (no aug) vs aug ep15.
Reads E:/fno_data/robustness_ablation.csv -> clean 2-panel PNG."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'E:\fno_data\robustness_ablation.csv'
OUT = r'c:\sarvik\fno_backup\robustness_plot.png'

with open(CSV, newline='', encoding='utf-8-sig') as f:
    rows = {r['condition']: r for r in csv.DictReader(f)}
def get(cond, col):
    r = rows.get(cond)
    if not r: return np.nan
    try: return float(r[col])
    except (ValueError, TypeError): return np.nan

C_DAG, C_AUG = '#c0504d', '#2e75b6'   # muted red / muted blue
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: blur degradation curve (the headline) ----
sev = [0, 1, 2, 3]
dag_blur = [get('clean','dagger_pct'), get('blur1','dagger_pct'), get('blur2','dagger_pct'), get('blur3','dagger_pct')]
aug_blur = [get('clean','aug_pct'),    get('blur1','aug_pct'),    get('blur2','aug_pct'),    get('blur3','aug_pct')]
ax1.plot(sev, dag_blur, 'o-', color=C_DAG, lw=2.5, ms=9, label='DAgger (no aug)')
ax1.plot(sev, aug_blur, 's-', color=C_AUG, lw=2.5, ms=9, label='+ corruption aug')
ax1.fill_between(sev, dag_blur, aug_blur, where=~np.isnan(dag_blur), color=C_AUG, alpha=0.08)
ax1.set_xlabel('Blur severity'); ax1.set_ylabel('Success rate (%)')
ax1.set_title('Blur robustness: aug is invariant, DAgger collapses', fontsize=13, fontweight='bold')
ax1.set_xticks(sev); ax1.set_ylim(0, 100); ax1.grid(alpha=0.3); ax1.legend(frameon=False)
ax1.annotate('DAgger collapses\n42 → 12%', xy=(2, 12), xytext=(2.05, 32),
             color=C_DAG, fontsize=10, arrowprops=dict(arrowstyle='->', color=C_DAG))
ax1.annotate('aug stays ~75%', xy=(3, 74), xytext=(1.9, 88), color=C_AUG, fontsize=10)

# ---- Panel B: matched-condition bars (both models present) ----
conds = ['clean', 'noise1', 'blur1', 'blur2']
labels = ['clean', 'noise-1', 'blur-1', 'blur-2']
dag = [get(c,'dagger_pct') for c in conds]
aug = [get(c,'aug_pct') for c in conds]
x = np.arange(len(conds)); w = 0.38
ax2.bar(x - w/2, dag, w, color=C_DAG, label='DAgger (no aug)')
ax2.bar(x + w/2, aug, w, color=C_AUG, label='+ corruption aug')
for i,(d,a) in enumerate(zip(dag,aug)):
    ax2.text(i - w/2, d+1.5, f'{d:.0f}', ha='center', fontsize=9, color=C_DAG)
    ax2.text(i + w/2, a+1.5, f'{a:.0f}', ha='center', fontsize=9, color=C_AUG)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel('Success rate (%)'); ax2.set_ylim(0, 100)
ax2.set_title('Per-condition (matched, N=5/task)', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='y'); ax2.legend(frameon=False)

fig.suptitle('Corruption-augmentation robustness ablation (LIBERO-Object)', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print('saved', OUT)
