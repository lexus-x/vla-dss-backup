# Significance tests: VLA-DSS vs Octo-Small

Per suite: pooled per-rollout successes, two-sided permutation test (20k) and
bootstrap 95% CI of the success-rate difference (VLA-DSS - Octo).

| Suite | VLA-DSS | Octo | diff | 95% CI | perm p |
|---|---|---|---|---|---|
| Object | 79.0 (n=300) | 53.8 (n=600) | +25.2 | [+19.0, +31.2] | 0.0000 (**significant**) |
| Goal | 76.0 (n=300) | 79.2 (n=600) | -3.2 | [-8.8, +2.7] | 0.3038 (n.s.) |
| Spatial | 77.7 (n=600) | 81.0 (n=600) | -3.3 | [-8.0, +1.3] | 0.1728 (n.s.) |

**Long** is excluded here: it is currently single-seed and its stored file is a per-task summary, not per-rollout records. Long significance will be added with the Long multi-seed run (which regenerates per-rollout logs). Point estimate: VLA-DSS 40.0 vs Octo 29.5 (Octo 3-seed 34/29.5/25).

## Effect sizes (Cohen's h)
| Suite | VLA-DSS | Octo | h | magnitude |
|---|---|---|---|---|
| Object | 79.0 | 53.8 | +0.543 | large |
| Goal | 76.0 | 79.2 | -0.077 | small |
| Spatial | 77.7 | 81.0 | -0.082 | small |
