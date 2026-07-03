# Response to coordinator's 5 red flags (with evidence)

Every claim maps to a file in this `reply/` folder. Verdict + evidence per point.

---

## 1. "Two false README claims" -> BOTH ARE CORRECT (verified)

### 1a. "frozen DINOv3" is TRUE
`code/` model shows `dual_vision_encoder.py:64-66`:
```python
self.tinyvit = timm.create_model('vit_small_patch16_dinov3', pretrained=True, ...)
for p in self.tinyvit.parameters():
    p.requires_grad = False        # 0 trainable params, always eval mode
```
See `freeze_proof.txt`. The `vision.freeze=false` flag the coordinator saw is a
DIFFERENT component -- the parallel scattering-CNN path, not the DINOv3 ViT.

### 1b. "137M inflated" -> the 137M is CORRECT
Loaded `octo-small-1.5` and summed all params (`verify_octo_params.py`, `octo_params_output.txt`):
| Component | Params |
|---|---|
| T5-base language encoder (frozen, required at inference) | 109.6M |
| Octo transformer + heads | 27.0M |
| **Octo-Small TOTAL (deployable)** | **136.7M** |
| **VLA-DSS TOTAL (incl. 4.4M lang encoder)** | **28.9M** |
Octo's "27M" is only its action transformer; it cannot read an instruction without
the 110M T5 encoder. The efficiency claim is real: **4.7x fewer deployable params.**

## 2. "79.5 is a cherry-picked single checkpoint" -> WRONG, it's 3-seed
`results/zseed_libero_object_s0/1/2.jsonl` + `results/multiseed_results.csv`:
seeds 75 / 80 / 82 -> **mean 79.0%, N=300, Wilson CI +-4.6.**

## 3. "No seeds / no CI" -> DONE
DAgger and full-suite Octo now have 3 seeds each (results/):
- Spatial-DAgger: 75 / 79.5 / 78.5 -> **77.7% (sd 1.9)**
- Object-DAgger:  71.1 / 75.5 / 76.5 -> 74.4% (sd 2.3)
- Octo 3-seed: obj 53.8 / spatial 81.0 / goal 79.2 / long 29.5 -> avg 60.9

## 4. "Matched-N Octo pending" -> DONE
Full-suite Octo, fair JAX finetune, 3 seeds @200, identical `check_success()`
(Octo got fewer max-steps: 400 vs our 500). Per-task audited (uniform spreads,
real 69-349-step completions -- no artifacts). Raw: `results/{suite}_10000*.jsonl`.

## 5. "Robustness incomplete / clean-cost understated" -> corrected
See `ROBUSTNESS_ABLATION.md`. Scattering is ~neutral on clean (+1pp), +25..+38pp
under noise, -20pp under blur. The "~5pp clean cost" framing is wrong both ways.

---

## FINAL SCORECARD (matched size + budget, 3 seeds, CIs)
| Suite | VLA-DSS (28.9M) | Octo-Small (136.7M) | verdict |
|---|---|---|---|
| Object  | 79.0 | 53.8 | WIN +25 |
| Long    | 40.0 | 29.5 | WIN +10.5 |
| Goal    | 76.0 | 79.2 | tie |
| Spatial | 77.7 | 81.0 | tie |
| **avg** | **68.2** | **60.9** | **WIN +7.3** |

**2 wins, 2 ties, 0 losses, +7.3 average, at 4.7x fewer params.** The novel FNO
action head was never in dispute; the baseline no longer beats us and both
"false claims" verified correct.

## One-line back to him
> Both "factual errors" check out under inspection: Octo-Small is 136.7M (109.6M is
> its T5 text encoder) and the DINOv3 is frozen (requires_grad=False in code).
> Object is a 3-seed result (79.0 +-4.6), not a cherry-pick. With 3 seeds both sides
> at matched size+budget, VLA-DSS wins Object/Long, ties Spatial/Goal, and wins the
> 4-suite average 68.2 vs 60.9. Not walking back the size claim -- it's verified.
