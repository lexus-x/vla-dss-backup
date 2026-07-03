# content/ — paper-ready section drafts (VLA-DSS)

Thorough, grounded write-ups for each part of the paper. Every number traces to a
file in `results/`, `octo_comparison/`, or `ablations/`. Written to be dropped
into the IEEE draft with light editing.

| File | Paper section |
|---|---|
| `01_method.md` | Method / Architecture (+ FNO head math) |
| `02_results.md` | Experiments & Results (main tables) |
| `03_ablations.md` | Ablation studies (scattering, resolution, jerk, head, DAgger) |
| `04_related_work.md` | Related Work |
| `05_discussion.md` | Analysis, Discussion, Limitations, Future Work |
| `06_repro.md` | Reproducibility (protocol, configs, seeds) |
| `07_novelty_vs_priorart.md` | Differentiation vs arXiv:2604.03449 (DeepONet) — the Novelty lever |

## One-paragraph abstract seed
We present **VLA-DSS**, a 28.9M-parameter RGB-only vision-language-action model whose
action decoder is a **Fourier Neural Operator (FNO)**. Robot trajectories are smooth,
low-frequency signals; an FNO decoder represents them in the frequency domain, making
smoothness an architectural prior rather than something learned from scratch. This
yields three properties no baseline action head has together: (i) **resolution-invariant**
decoding (execute the same trajectory at any control rate), (ii) **low-jerk** motion by
construction, and (iii) a compact parameter footprint. On the LIBERO benchmark, at
matched model size and matched fine-tuning budget with three seeds, VLA-DSS **beats
Octo-Small on the four-suite average (68.2 vs 60.9)** while using **~4x fewer deployable
parameters** (Octo requires a frozen 109.6M T5 text encoder). We ablate the FNO head
against an MLP head under an identical backbone, and analyze the wavelet-scattering
vision path as a noise-robustness (not accuracy) mechanism.
