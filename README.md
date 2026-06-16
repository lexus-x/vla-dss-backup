# FNO-VLA: A Compact RGB-Only Vision-Language-Action Model

A **28.9M-parameter, RGB-only** Vision-Language-Action model for robot manipulation,
combining **wavelet scattering + frozen DINOv3** vision, **FiLM** fusion, and a
**Fourier-Neural-Operator (FNO)** action head. Designed for the efficiency frontier:
strong manipulation performance at a fraction of the parameters of generalist VLAs.

## Highlights
- **Compact & efficient** — 28.9M params, RGB-only, runs real-time on weak hardware.
- **Novel components** — wavelet-scattering observation encoder (Lipschitz-stable) +
  FNO action head (band-limited, resolution-invariant action chunks).
- **LIBERO results** — Object **79.5%** (DAgger, N=200); aux-x-y variant Object **71%** /
  Spatial **73%** / Goal **72%** (N=100).
- **Robustness ablation** — corruption augmentation makes the policy **blur-invariant**
  (blur-2: 12% → 78%) at ~5pp clean cost; noise-neutral. See `results/robustness_ablation.csv`.
- **Efficiency comparison** — competitive with Octo-Small (27M) at matched size; far
  smaller end-to-end than SmolVLA (450M) / Octo+t5.

## Architecture
```
RGB (2 views) ──► Wavelet Scattering + frozen DINOv3 ViT-S ──┐
language     ──► text encoder (MiniLM / tiny) ───────────────┤──► FiLM fusion ──► FNO head ──► action chunk + gripper
proprio      ──► MLP ────────────────────────────────────────┘
                 (train-only aux x-y grasp head, dropped at inference)
```

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
