# Reproducibility

## Evaluation protocol
- Env: LIBERO `OffScreenRenderEnv`, 128x128 (VLA-DSS) / 256x256 primary (Octo).
- Success: LIBERO `env.check_success()` (identical for both models).
- Rollouts: 10 tasks x 20 rollouts = 200 per seed; 3 seeds (0,1,2). Per-seed the
  init-state order is shuffled by the seed; perturbation RNG is seeded.
- Execution: action chunk executed open-loop with execute-horizon 8 (VLA-DSS) /
  4 (Octo, its native horizon); max steps 500 (VLA-DSS) / 400 (Octo).
- CIs: Wilson score interval at the reported N.

## Training protocol
- Pretrain: all 5 LIBERO suites (~130 tasks), 80 epochs, batch 128, DINOv3 frozen,
  stride 4, lr 2e-4, warmup 1000, cosine schedule.
- Fine-tune: per suite, resume from the pretrain checkpoint, lr 1e-4, batch 128,
  max_epochs 100, validation-selected best checkpoint (early-stop). Loss: smooth-L1
  (actions) + BCE (gripper).
- Octo baseline: official JAX fine-tune (`octo-small-1.5`) on `modified_libero_rlds`,
  matched to <= our step budget, 3 seeds. Config + eval script in `octo_comparison/`.

## Matched-budget fairness
Octo is fine-tuned for at least as many steps as VLA-DSS on every suite (verified;
more on Object/Goal). The head ablation trains the MLP head to the FNO baseline's exact
ep63 val-best checkpoint — same budget, not less, not more.

## Key files
| Result | File |
|---|---|
| VLA-DSS 3-seed (Object) | `results/zseed_libero_object_s{0,1,2}.jsonl`, `results/multiseed_results.csv` |
| Octo 3-seed (all suites) | `octo_comparison/.../{suite}_10000_s{1,2}.jsonl` |
| Size verification | `octo_comparison/evidence/verify_octo_params.py` (+ output) |
| Frozen-DINOv3 proof | `octo_comparison/evidence/freeze_proof.txt` |
| Scattering / robustness | `ablations/results/robustness_onoff.csv`, `scattering_stability.csv` |
| Resolution invariance | `results/resinvar_results.csv` |
| Jerk | `results/jerk_comparison.csv` |
| Head ablation | `results/zhead2_{fno,mlp}_s{0,1,2}.jsonl` (pending) |
| DAgger curve | `ablations/results/zseed_spdag_ep{5,10,15,20}*.jsonl` |

## Configs
`configs/pretrain_dinov3.yaml`, `configs/finetune_dinov3_*.yaml`,
`configs/mlp_dinov3_c16.yaml` (head ablation), `octo_comparison/libero_finetune_config.py`.
Random seeds fixed at 0/1/2 throughout.
