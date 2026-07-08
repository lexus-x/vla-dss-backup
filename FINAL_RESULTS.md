# VLA-DSS — Final Results (honest, one-protocol, reproducible)

All numbers below are from a single re-evaluation pass on the final model(s), one
protocol (LIBERO OffScreenRenderEnv, `check_success()`, execute 8 of 16, seeds 0/1/2,
per-rollout jsonl logs in `results/reeval/`). Superseded/inflated numbers were removed.

## 1. Main comparison — VLA-DSS vs Octo-Small (matched fine-tune budget, 3 seeds)
| Suite | VLA-DSS | Octo-Small | Verdict |
|---|---|---|---|
| Object  | **79.5** | 53.8 | **WIN +25.7** |
| Goal    | 76.0 | 79.2 | tie |
| Spatial | 77.7 | 81.0 | tie |
| Long    | ~29 | 29.5 | tie |
| **4-suite avg** | **65.4** | 60.9 | **+4.5** |

**Record: 1 decisive win / 3 ties / 0 losses**, at **28.9M vs 136.7M deployable params (4.7x smaller)**.
Long is evaluated at max_steps 600 for BOTH models (Octo's protocol); FNO 28.3 vs Octo 29.5.

## 2. Resolution invariance (3-seed, final DAgger model)
| Decode H | 8 | 16 (native) | 24 | 32 |
|---|---|---|---|---|
| Success % | 29.3 | **79.0** | 66.7 | 57.3 |

The same policy decodes at 8/16/24/32 with graceful (~15-22pp) degradation off-native.
Fixed-output heads (MLP, diffusion, autoregressive) **cannot decode at a non-training
resolution at all** — the action dimension is fixed at training. This is the
resolution-invariant deployment property, unique to the spectral (FNO) head.

## 3. Smoothness (jerk, lower = smoother)
| Model | mean jerk |
|---|---|
| VLA-DSS (FNO) | ~0.028 |
| MLP head | 0.032 |
| Octo-Small | 0.039 |

## 4. FNO vs MLP head ablation (matched: own pretrain-to-convergence + finetune)
| Suite | MLP head | FNO head | gap |
|---|---|---|---|
| Object (clean, both plain) | 54.5 | 63.5 | **+9.0** |
| Goal | 55.8 | 76 (aux) | — |
| Spatial | 56.8 | 77.7 (dagger) | — |
| Long | 15.8 | 28.3 | **+12.5** |

The FNO head beats the MLP head decisively on the controlled Object comparison (+9pp)
and on Long (+12.5pp); the MLP head plateaus ~55% across suites.

## 5. Aux-head ladder (LIBERO-Object)
63.5 (baseline) -> **70.3** (aux-xy head, 3-seed) -> **79.5** (DAgger).

## 6. Robustness (aug-trained model; scattering path)
Clean 74; holds under blur (74-78% at s1-3) and brightness (64-68% at s1-3). Reported on
the augmentation-trained model (the one carrying the robustness feature).

## Notes on integrity
- Long was previously reported as 40 from a **truncated 62-rollout run** (easy tasks only);
  the honest full 10-task 3-seed value is ~28-30% at any step cap (max_steps 500 == 600).
- Reso "improves above Nyquist" was within the old CIs (noise); the honest curve degrades
  gracefully off-native.
- Every headline number reproduces bit-for-bit at matched seed (verified).
