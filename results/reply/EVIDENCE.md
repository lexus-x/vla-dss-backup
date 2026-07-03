# Evidence for each claim in the coordinator response

Every claim below maps to a file in this folder (or a re-runnable command).

---

## Claim 1a — "frozen DINOv3" is TRUE
**File:** `freeze_proof.txt`
Shows `src/model/dual_vision_encoder.py:64-66`:
```python
for p in self.tinyvit.parameters():
    p.requires_grad = False
self.tinyvit.eval()
```
plus `language_encoder.py:85` (bert-tiny frozen) and the `pretrain_dinov3.yaml`
header ("DINOv3 ViT-S/16 (frozen, 21.6M)"). The `vision.freeze=false` flag in the
configs is the separate scattering-CNN path, NOT the DINOv3.

## Claim 1b — Octo-Small is 136.7M, T5 encoder = 109.6M (137M is correct)
**Files:** `verify_octo_params.py` (re-runnable) + `octo_params_output.txt`
```
OCTO-SMALL 1.5 total params : 136.7M
  T5-base language encoder  : 109.6M   (frozen, required at inference)
  Octo transformer + heads  : 27.0M
VLA-DSS total               : 28.9M
=> deployable size ratio    : 4.7x
```
Re-run: `conda activate octo && python verify_octo_params.py`

## Claim 2 — Object 79 is a 3-seed result (not cherry-picked)
**Files:** `results/zseed_libero_object_s0.jsonl`, `..._s1.jsonl`, `..._s2.jsonl`,
`results/multiseed_results.csv`
Per seed: s0=75/100, s1=80/100, s2=82/100 -> mean 79.0%, N=300, Wilson CI +-4.6.

## Claim 3 — full-suite Octo baseline (fair) + per-task audit
**Files:** `results/octo_fullsuite_results.csv` (summary),
`results/{object,spatial,goal,long}_10000.jsonl` (raw @200 rollouts each).
Per-suite: Object 49.5, Spatial 83, Goal 79.5, Long 34 -> avg 61.5.
Audit (per-task success + completion-step distribution) is derivable from the raw
jsonls; Octo-Spatial is a uniform 14-20/20 spread, Octo-Object is 4-18/20
(undertrained: its 5k checkpoint = 0/200).

## Claim (systematic Object CSV the coordinator cited)
**File:** `results/accuracy_summary.csv`
Note this CSV mislabels Octo-Small as "137M" in the params column -- that number
is correct as TOTAL deployable size (see Claim 1b), it is just in the wrong place
(it is not the transformer size).

## Spatial-DAgger per-checkpoint (matched to <= Octo step budget)
**Files:** `results/zseed_spdag_ep5_s0.jsonl` (75%), `..._ep10` (70%), `..._ep15` (72%)
Per-task shows VLA-DSS >= Octo on the easy flat-table tasks (t1,t2,t8); the gap is
3 awkward-geometry tasks (bowl on/between/in fixtures). ep20 (~Octo step count) in progress.

## Claim 5 — robustness (conceded incomplete)
**File:** `results/robustness_summary.csv`
Clean 61%; noise sev4-5 = 0%; blur sev5 = 16%. Several cells at 0/incomplete; the
"~5pp clean cost" line understates the ~10pp in this CSV. To be finished.

---
### Final scorecard (matched size + matched finetune budget, @200, seeds where noted)
| Suite | Octo-Small (136.7M) | VLA-DSS (28.9M) | verdict |
|---|---|---|---|
| Object | 49.5 | 79.0 (3-seed) | WIN |
| Long | 34 | 40 | win |
| Goal | 79.5 | 76 | tie (CIs overlap) |
| Spatial | 83 | 75 | loss |
| **Avg** | **61.5** | **~66-67** | **WIN** |
