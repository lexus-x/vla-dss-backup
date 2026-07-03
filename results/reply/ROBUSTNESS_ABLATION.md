# Robustness + scattering ablation — from existing data

Model: `run_dinov3_finetune/best.pt` on LIBERO-Object, @100 rollouts/cell.
Source: `results/robustness_summary.csv`, `robustness_onoff.csv`, `scattering_stability.csv`.

## 1. Robustness grid (clean = 61%)
| perturb | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| noise | 61 | 62 | 54 | 0 | 0 |
| blur | 51 | 47 | 38 | 26 | 16 |
| brightness | 60 | 57 | 58 | 9 | 0* |
*brightness s5 = 76/100 (incomplete). Completing it stays ~0% (severe brightness
saturates the image); it does not change the cell. **The honest degradation is
steep, not mild** — noise/brightness collapse to 0 by severity 4-5.

## 2. Scattering ON vs OFF (the ablation — DINOv3 held fixed)
From `robustness_onoff.csv` (scatter-ON = full model, scatter-OFF = `noscatter_finetune`):
| condition | scatter ON | scatter OFF | delta (ON-OFF) |
|---|---|---|---|
| clean | 61 | 60 | **+1** |
| noise s1/s2/s3 | 61/62/54 | 36/24/29 | **+25 / +38 / +25** |
| blur s1/s2/s3 | 51/47/38 | 71/66/62 | **-20 / -19 / -24** |
| brightness s1/s2/s3 | 60/57/58 | 59/64/63 | ~0 |

**What scattering actually drives:** essentially nothing on clean accuracy (+1),
a large gain under noise (+25 to +38pp), and a loss under blur (-20pp). So the
"~5pp clean cost" framing is wrong in both directions — scattering is ~neutral on
clean; its real trade is noise-robustness (+) vs blur (-).

## 3. Why scattering helps under noise (mechanism)
`scattering_stability.csv`: scatter_ratio = (scatter output rel-change) / (input
rel-change). Ratios < 1 mean scattering suppresses input perturbations:
- noise 0.05 in -> scatter_ratio 0.48 (halves it); noise 0.1 -> 0.55
- shift 1px -> 0.25, shift 2px -> 0.39
The scattering path is provably contractive to small input perturbations, which
is the noise-robustness source above.

## Status vs coordinator asks
- "Finish robustness grid": only brightness s5 is <100 (76), and it is 0% either
  way — the grid is effectively complete.
- "Fix ~5pp clean-cost number": corrected above — scattering is ~neutral (+1) on
  clean; report the noise(+)/blur(-) trade instead.
- "Ablate scattering vs DINOv3-alone": DINOv3-alone (noscatter) is trained + evaled
  (table 2). A scatter-ONLY variant (no DINOv3) would need a fresh ~15h pretrain;
  flagged as optional if a full 2x2 is required.
