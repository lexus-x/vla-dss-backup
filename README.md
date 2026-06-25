# VLA-DSS: Vision-Language-Action via Dynamic Spectral Synthesis

A **compact, ~28.9M-parameter, RGB-only** Vision-Language-Action model for robot
manipulation, built on **two spectral operators**: **wavelet scattering** for vision and a
**Fourier Neural Operator (FNO)** for actions. The FNO *synthesizes* action trajectories in
the frequency domain — the way a synthesizer builds sound from frequencies — giving
**smooth, low-jerk, resolution-invariant** motion at a fraction of the parameters of
generalist VLAs. Designed for the efficiency frontier: strong manipulation at low cost.

> **DSS = Dynamic Spectral Synthesis** — the model produces *dynamic* (time-varying) action
> trajectories by *synthesizing* them from a band-limited *spectral* representation.

## Highlights
- **Compact & efficient** — 28.9M params, RGB-only, runs real-time on weak hardware.
- **Dual spectral streams** — a wavelet-scattering observation encoder (Lipschitz-stable) +
  an FNO action head that synthesizes band-limited, resolution-invariant action chunks.
- **Smooth motion** — band-limited mode truncation acts as a low-pass filter → low jerk
  (≈ human level; ~28–36% smoother than Octo). *Smoothness is architectural, not learned.*
- **Resolution-invariance** — one model decodes at **any** control rate / chunk length
  without retraining (flat 61–68% at the training rate or higher on LIBERO).
- **LIBERO results** — Object **79.5%** (DAgger, N=200); aux-x-y variant Object **71%** /
  Spatial **73%** / Goal **72%** (N=100). *(Across runs, ~74–80% typical; report mean ± std.)*
- **Robustness** — scattering → noise robustness (ON vs OFF ablation, proven); corruption
  augmentation → blur-invariance (blur-2: 12% → 78%) at ~5pp clean cost. See
  `results/robustness_ablation.csv`.
- **Efficiency comparison** — competitive with Octo-Small (27M) at matched size; far smaller
  end-to-end than SmolVLA (450M) / Octo+t5.

## Architecture
```
RGB (2 views) ─► Wavelet Scattering + frozen DINOv3 ViT-S ──┐   ┌─ "spectral stream 1"
language      ─► text encoder (MiniLM / tiny) ──────────────┤──►│ FiLM fusion ─► FNO head ─► action chunk + gripper
proprio       ─► MLP ───────────────────────────────────────┘   └─ "spectral stream 2" (Fourier operator)
                  (train-only aux x-y grasp head, dropped at inference)
```
Both perception (wavelet scattering) and action generation (Fourier operator) live in the
frequency domain — that's the *dual spectral synthesis* at the core of VLA-DSS.

## Repo structure
```
src/             model, training, data loaders
configs/         training configs (LIBERO + Bridge)
scripts/         eval, conversion, plotting, diagnostics
bridge_handoff/  BridgeData V2 → WidowX recipe (START_HERE.md) + robot deploy/HG-DAgger skeletons
env/             setup_a100.sh, requirements, A100 migration + Bridge recipe
docs/            results summaries, recipes, manifests
results/         metrics CSVs, plots, figures
```

## Setup
```bash
bash env/setup_a100.sh        # conda env 'fnovla', torch 2.1 cu121, deps, LIBERO
conda activate fnovla
```
Requires **h5py==3.9.0** (not 3.16 — ABI). See `env/requirements_a100.txt`.

## Data & weights (NOT in this repo)
- **LIBERO** suites — download from the LIBERO benchmark.
- **BridgeData V2** — use **`bridge_orig`** (RLDS, from the official RAIL site, per the
  OpenVLA README). The OXE `bridge` GCS copy is out of date; the raw 400GB numpy is unnecessary.
- **Trained checkpoints** — hosted separately (too large for git); see releases / external storage.

## Train / eval
```bash
# train (example)
python -u src/train.py --config configs/finetune_dinov3_aug.yaml --data_dir <data> --checkpoint_dir <out>
# eval in sim
python -u scripts/eval_sim.py --checkpoint <ckpt> --suite libero_object --n_rollouts 20
```
Bridge V2 → WidowX pipeline: see `bridge_handoff/START_HERE.md`.

## License / citation
Research code. Cite the component works: Fourier Neural Operator (Li et al. 2021),
Wavelet Scattering (Mallat 2012), DINOv3, FiLM (Perez et al. 2018), DAgger (Ross et al. 2011).

If you use VLA-DSS, please also cite this project as
*"VLA-DSS: Vision-Language-Action via Dynamic Spectral Synthesis."*
