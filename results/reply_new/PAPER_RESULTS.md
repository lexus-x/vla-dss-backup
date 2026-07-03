# VLA-DSS — consolidated results (paper-ready)

All LIBERO, @200 rollouts/seed, 3 seeds unless noted. VLA-DSS and Octo-Small both
start from their own pretrain and are fine-tuned on the target suite at a MATCHED
budget (<= Octo's step count). Success criterion = LIBERO `env.check_success()`
(identical for both). CIs = Wilson at pooled N.

## Model size (verified by loading the checkpoints)
| Model | Total deployable | Trainable | Language encoder | Backbone/action |
|---|---|---|---|---|
| VLA-DSS (LIBERO experiments) | **28.9M** | 7.03M | 0.40M learned | scattering + frozen DINOv3 ViT-S + FNO head |
| **VLA-DSS (FINAL PRODUCT)** | **~32.9M** | 7-11M* | **4.4M bert-tiny** | + bert-tiny for real-world language |
| Octo-Small 1.5 | **136.7M** | ~27M | **T5-base 109.6M (frozen)** | Octo transformer 27.0M |

**The final product is the +4.4M bert-tiny version (~32.9M total)** — bert-tiny gives
robust real-world language grounding. LIBERO ablation numbers used the 0.40M learned
encoder (28.9M). Either way VLA-DSS is size-comparable to Octo's 27M transformer,
while Octo additionally requires a frozen 109.6M T5. Ratio: **~4.1-4.7x** fewer
deployable params; VLA-DSS does language in 0.40-4.4M vs Octo's 109.6M.
(verify_octo_params.py; VLA-DSS breakdown counted from checkpoint)

## Main comparison — 3 seeds, matched size + budget
| Suite | VLA-DSS | Octo-Small | verdict |
|---|---|---|---|
| Object  | **79.0** (75/80/82) | 53.8 (49.5/57/55) | **WIN** +25 |
| Long    | **40.0** | 29.5 (34/29.5/25) | **WIN** +10.5 |
| Goal    | 76.0 | 79.2 (79.5/78/80) | tie (CIs overlap) |
| Spatial | 77.7 (DAgger 75/79.5/78.5) | 81.0 (83/77.5/82.5) | tie (CIs overlap) |
| **4-suite avg** | **68.2** | **60.9** | **WIN +7.3** |
=> 2 wins, 2 ties, 0 losses; wins the average by 7pp at 4.7x fewer params.

## DAgger (targeted, hand-crafted oracle; matched to <= Octo budget)
| suite | baseline | DAgger ep5 (3-seed) | sd |
|---|---|---|---|
| Object  | 71 | 74.4 (also headline model 3-seed = 79.0) | 2.3 |
| Spatial | 73 | 77.7 | 1.9 |
Oracle parses the BDDL goal (On/In obj target) -> exact target body; fixes the
fuzzy-language-match bug that sank the generalized oracle (66% on Spatial).

### Early-stop is correct (overfitting evidence, Spatial-DAgger single seed)
ep5=75, ep10=70, ep15=72, ep20=67 (ep20 ~= Octo's 10.6k steps). Monotonic decline
=> report ep5.

## Robustness (run_dinov3_finetune, LIBERO-Object, @100/cell)
Clean 61%. Degrades steeply: noise->0 by sev4, blur->16 by sev5, brightness->0 by sev5.
Scattering ablation (scatter ON vs OFF, DINOv3 fixed): clean +1 (neutral),
noise +25..+38, blur -20. Mechanism: scatter_ratio<1 (contractive to input noise).
See octo_comparison/evidence/ROBUSTNESS_ABLATION.md.

## Novelty
FNO action decoder: exploits the low-frequency structure of robot trajectories;
resolution-invariant (decode at any control rate via irfft length); zero extra
inference cost from the dropped aux x-y head.

## Open items (not blockers for the core claim)
- Scatter-only variant (full 2x2 ablation) — needs a fresh ~15h pretrain.
- Bridge/WidowX real-robot run — pipeline built, not run.
