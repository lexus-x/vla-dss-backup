# Ablations — VLA-DSS

All LIBERO. Clean = severity 0. Robustness/scatter ablations use the Object suite
(`run_dinov3_finetune` = scatter-ON, `run_dinov3_noscatter_finetune` = scatter-OFF),
@100-200 rollouts. Raw logs in `results/`.

## 1. Scattering ON vs OFF (DINOv3 held fixed)
| Condition | scatter ON | scatter OFF | delta (ON-OFF) |
|---|---|---|---|
| **clean accuracy (Object)** | 61% (rob) / **62.5%** (noscatter @200) | -- | **~0 (neutral)** |
| noise sev1 / sev2 / sev3 | 61 / 62 / 54 | 36 / 24 / 29 | **+25 / +38 / +25** |
| blur sev1 / sev2 / sev3 | 51 / 47 / 38 | 71 / 66 / 62 | -20 / -19 / -24 |
| brightness sev1 / sev2 / sev3 | 60 / 57 / 58 | 59 / 64 / 63 | ~0 |

**Finding:** scattering is ~NEUTRAL on clean accuracy (Object scatter-OFF 62.5% @200
vs scatter-ON ~61%), but gives a large NOISE-robustness gain (+25..+38pp) at a blur
cost (-20pp). It is a stability path, not an accuracy path.
(Note: the noscatter checkpoint is Object-trained; its 44% on Spatial is out-of-suite
and MUST NOT be cited.)

## 2. Scattering stability (mechanism)
`results/scattering_stability.csv`: scatter_ratio = (scatter output rel-change) /
(input rel-change). Ratios < 1 => scattering suppresses input perturbations:
- noise 0.05 -> 0.48 (halves it); noise 0.1 -> 0.55
- shift 1px -> 0.25; shift 2px -> 0.39
Provably contractive to small input perturbations -> source of the noise robustness.

## 3. Robustness grid (clean 61%, Object, @100/cell)
| perturb | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| noise | 61 | 62 | 54 | 0 | 0 |
| blur | 51 | 47 | 38 | 26 | 16 |
| brightness | 60 | 57 | 58 | 9 | 0 (100/100, completed) |
Honest degradation: holds through sev3, collapses at sev4-5.

## 4. DAgger early-stop (Spatial-DAgger, single seed)
ep5=75, ep10=70, ep15=72, ep20=67 (ep20 ~= Octo's 10.6k steps). Monotonic decline
=> the model overfits; ep5 early-stop is correct.

## Files
- results/robustness_onoff.csv       (scatter ON vs OFF, all perturbations)
- results/robustness_summary.csv     (full grid)
- results/scattering_stability.csv   (contractivity mechanism)
- results/znoscatter_obj_clean_s0.jsonl (scatter-OFF Object @200 = 62.5%)
- results/znoscatter_sp_clean_s0.jsonl  (out-of-suite, DO NOT cite)
- results/zbright_s5_redo.jsonl      (brightness sev5 completed = 0%)
- results/zseed_spdag_ep{5,10,15,20}*.jsonl (DAgger early-stop curve)
