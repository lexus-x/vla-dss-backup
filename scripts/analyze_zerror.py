"""
Analyze the per-rollout z-error JSONL from eval_sim.py per PREREGISTRATION.md.

Pairs episodes across two tags by (task_idx, init_idx), imputes failed episodes'
z-error to a frozen ceiling (95th pct of pooled completed z_err), then runs the
pre-registered tests:
  - paired Wilcoxon signed-rank on combined z_err  (primary)
  - McNemar on paired binary success
and reports jerk + success deltas with Wilson CIs.

Usage:
  python scripts/analyze_zerror.py --jsonl E:/fno_data/zerror_expA.jsonl \
      --arm_a exec4 --arm_b exec8
"""
import argparse, json, math
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(b, c):
    # b = A-success & B-fail ; c = A-fail & B-success. Exact binomial (two-sided).
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative='two-sided').pvalue


def load(jsonl):
    rows = defaultdict(list)
    with open(jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r['tag']].append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--arm_a', required=True)
    ap.add_argument('--arm_b', required=True)
    args = ap.parse_args()

    rows = load(args.jsonl)
    for arm in (args.arm_a, args.arm_b):
        assert arm in rows, f"tag '{arm}' not in {args.jsonl}. Tags present: {list(rows)}"

    def key(r):
        return (r['task_idx'], r['init_idx'])
    A = {key(r): r for r in rows[args.arm_a]}
    B = {key(r): r for r in rows[args.arm_b]}
    keys = sorted(set(A) & set(B))
    print(f"paired episodes: {len(keys)}  (arm_a={args.arm_a} n={len(A)}, arm_b={args.arm_b} n={len(B)})")

    # Freeze ceiling = 95th pct of pooled completed (grasp_z_gap_min not None) z_err.
    completed = [r['grasp_z_gap_min'] for r in rows[args.arm_a] + rows[args.arm_b]
                 if r.get('grasp_z_gap_min') is not None]
    Z_CEIL = float(np.percentile(completed, 95)) if completed else float('nan')
    print(f"Z_CEIL (frozen, 95th pct of completed grasp_z_gap): {Z_CEIL:.4f} m\n")

    def zerr(r):
        g = r.get('grasp_z_gap_min')
        return Z_CEIL if (g is None or r['success'] == 0) else g

    za = np.array([zerr(A[k]) for k in keys])
    zb = np.array([zerr(B[k]) for k in keys])
    sa = np.array([A[k]['success'] for k in keys])
    sb = np.array([B[k]['success'] for k in keys])
    ja = np.array([A[k]['jerk'] for k in keys], float)
    jb = np.array([B[k]['jerk'] for k in keys], float)

    # Primary: paired Wilcoxon on combined z_err (arm_a - arm_b)
    diff = za - zb
    nz = diff[diff != 0]
    try:
        w_p = wilcoxon(za, zb).pvalue if len(nz) else 1.0
    except ValueError:
        w_p = 1.0
    print("=== PRIMARY: combined (imputed) grasp-height z-error ===")
    print(f"  {args.arm_a}: {za.mean():.4f} m   {args.arm_b}: {zb.mean():.4f} m   "
          f"(delta {za.mean()-zb.mean():+.4f})")
    print(f"  paired Wilcoxon p = {w_p:.4f}   (n_nonzero={len(nz)})\n")

    # Binary success: McNemar
    b = int(((sa == 1) & (sb == 0)).sum())
    c = int(((sa == 0) & (sb == 1)).sum())
    m_p = mcnemar(b, c)
    la, ha = wilson(sa.sum(), len(sa))
    lb, hb = wilson(sb.sum(), len(sb))
    print("=== SECONDARY: success rate ===")
    print(f"  {args.arm_a}: {100*sa.mean():.1f}% [{100*la:.1f}, {100*ha:.1f}]   "
          f"{args.arm_b}: {100*sb.mean():.1f}% [{100*lb:.1f}, {100*hb:.1f}]")
    print(f"  McNemar discordant b={b} c={c}  p = {m_p:.4f}\n")

    print("=== SECONDARY: jerk (smoothness must not regress) ===")
    print(f"  {args.arm_a}: {np.nanmean(ja):.4f}   {args.arm_b}: {np.nanmean(jb):.4f}   "
          f"(delta {np.nanmean(ja)-np.nanmean(jb):+.4f}, "
          f"{100*(np.nanmean(ja)-np.nanmean(jb))/np.nanmean(jb):+.1f}%)\n")

    # Decomposition (explanation only)
    print("=== decomposition (NOT headline): conditional z_err among completed ===")
    for tag, R in ((args.arm_a, rows[args.arm_a]), (args.arm_b, rows[args.arm_b])):
        comp = [r['grasp_z_gap_min'] for r in R if r.get('grasp_z_gap_min') is not None]
        cr = np.mean([r['success'] for r in R])
        print(f"  {tag}: conditional z_err {np.mean(comp):.4f} m (n={len(comp)}), "
              f"completion/success {100*cr:.1f}%")


if __name__ == '__main__':
    main()
