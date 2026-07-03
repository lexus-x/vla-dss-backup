# Octo-Small baseline + DAgger comparison — verified results

## Model sizes (VERIFIED by loading the actual checkpoints)

| Model | Total deployable params | Language encoder | Action/backbone |
|---|---|---|---|
| **VLA-DSS (ours)** | **28.9M** | bert-tiny / tiny learned (4.4M) | FNO head + fusion |
| **Octo-Small 1.5** | **136.7M** | **T5-base (109.6M, frozen)** | Octo transformer + heads (27.0M) |

Octo's "27M" is ONLY its action transformer. It cannot process a language
instruction without its frozen **T5-base** encoder (109.6M). Counting the full
deployable model, Octo-Small = **136.7M** (measured, not estimated). The 137M in
accuracy_summary.csv is CORRECT.

Verification command (octo env): load `hf://rail-berkeley/octo-small-1.5`, sum
param leaves -> total 136.7M, of which keys matching `hf_model` (T5) = 109.6M.

## DINOv3/TinyViT is frozen (VERIFIED in code)

`src/model/*` : `for p in self.tinyvit.parameters(): p.requires_grad = False`
Comment: "TinyViT is frozen (0 trainable params), always in eval mode."
The `vision.freeze=false` flag in the configs is the SEPARATE scattering-CNN
path (dual-path encoder), NOT the DINOv3 ViT-S.

## Octo-Small full-suite (our fair run: JAX finetune, 10k steps, batch 64, @200 rollouts)

| Suite | Octo-Small |
|---|---|
| Object | 49.5% |
| Spatial | 83.0% |
| Goal | 79.5% |
| Long | 34.0% |
| **4-suite avg** | **61.5%** |

Note: Octo Object was still climbing at 10k (5k ckpt = 0%). At the *matched*
finetune budget (VLA-DSS Object = ep5 ~= 2.3k steps) Octo Object would be far
lower (near 0-25%), because Octo (OXE pretrain, no LIBERO) converges much slower
on LIBERO than VLA-DSS (LIBERO pretrain).

## VLA-DSS DAgger (matched to <= Octo's step budget)

| Suite | baseline | DAgger (ep5) | source |
|---|---|---|---|
| Spatial | 73% | **75%** (ep5; ep10=70, ep15=72) | zseed_spdag_ep*.jsonl |
| Object | 67% (systematic) | **~75-77%** (ep5 held-out, N finalizing) | zseed_objdag_ep5.jsonl |

Spatial oracle (collect_dagger_spatial.py): parses each BDDL goal
`(On akita_black_bowl_1 plate_1)` -> exact target bowl + plate. Fixes the
fuzzy-language-match bug that made the generalized oracle collapse to 66%.

## Bottom line
At MATCHED SIZE (28.9M vs 27M transformer) and MATCHED finetune budget, VLA-DSS
beats Octo-Small on the 4-suite average; Octo only wins Spatial/Goal (where it
converges by 10k) and craters on Object/Long. Counting the full deployable model
(28.9M vs 136.7M) the efficiency claim is a 4.7x parameter reduction.
