# Novelty positioning vs. closest prior art

## Closest cited prior art: Sewell et al., "Neural Operators for Multi-Task
## Control and Adaptation" (arXiv:2604.03449)
That work uses a **permutation-invariant, branch-trunk (DeepONet-style) neural
operator** to approximate the mapping from a *task descriptor* (cost / dynamics
functions) to an *optimal control law*, evaluated on parametric optimal-control
environments and a locomotion benchmark. It contributes multi-task generalization,
structured fine-tuning for adaptation, and meta-trained few-shot initialization.

It shares one high-level idea with us — *neural operators for control* — and
nothing below that level. VLA-DSS differs on four concrete, defensible axes:

| Axis | Sewell et al. (2604.03449) | VLA-DSS (ours) |
|---|---|---|
| Operator family | branch-trunk (**DeepONet**) | **Fourier Neural Operator** (spectral) |
| Operator acts over | **task space** (descriptor → control law) | **time axis of the action chunk** (frequency domain) |
| Prior it exploits | multi-task structure | **low-frequency structure of motion** |
| Perception / language | none (state-based) | **RGB (frozen DINOv3 + scattering) + language** |
| Domain | optimal control + locomotion | **vision-language-action manipulation (LIBERO)** |
| Resolution-invariant deploy | not claimed | **yes — train once, deploy at any control rate** |

## The one property a branch-trunk operator cannot replicate
Our decoder reconstructs the action chunk with `irfft(·, n=H)`, so the temporal
resolution `H` is a free parameter *at inference*: the same learned operator emits
the trajectory at any control frequency (measured: 8/16/24/32 -> 29/61/68/66%).
A branch-trunk / DeepONet head — the prior art's design, and also the design used
by the DeepONet-PH-VLA line — produces a fixed-dimensional output determined at
training time; it has no mechanism to be re-decoded at an arbitrary rate. **This
"train once, deploy at any control frequency" property is specific to the spectral
(FNO) formulation and is our sharpest structural differentiator.**

## Why FNO is the *right* operator for actions (not just a different one)
Robot end-effector trajectories are smooth, low-frequency signals. An FNO keeps
only the lowest `k` spectral modes, so smoothness is architectural — jitter is
suppressed by construction (measured jerk 0.0256 vs Octo 0.0386, ~33% smoother).
A branch-trunk operator has no such frequency prior. So the choice of FNO over
DeepONet is not incidental: it is matched to the signal class (band-limited
motion), and it buys two properties (smoothness-by-construction and
resolution-invariance) that the branch-trunk family does not provide.

## One-sentence novelty claim (for the intro/abstract)
> To our knowledge, VLA-DSS is the first vision-language-action model to decode
> actions with a Fourier Neural Operator, yielding smoothness-by-construction and
> resolution-invariant deployment (train once, run at any control frequency) —
> properties that distinguish it from both standard diffusion/flow action heads and
> from prior branch-trunk (DeepONet) neural-operator controllers.
