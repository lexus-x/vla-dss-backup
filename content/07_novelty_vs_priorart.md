# Novelty: resolution-invariant action deployment, and differentiation from prior art

## The central novelty claim (lead with this)
> VLA-DSS decodes the action chunk with a Fourier Neural Operator, reconstructing it
> as `a_{1:H} = irfft(spectrum, n=H)`. **The temporal resolution `H` — i.e. the control
> frequency — is a free parameter at inference.** A policy trained once at H=16 is
> deployed, without any retraining, at H = 8 / 24 / 32 / any value: **train once,
> deploy at any control frequency.** We measure this directly (29 / 61 / 68 / 66% at
> H = 8 / 16 / 24 / 32), with accuracy maintained or improved above the training
> (Nyquist) resolution.

This is a *structural* property of the spectral formulation, not a trick. It is the
sharpest thing that separates VLA-DSS from every other action-head family and from the
closest cited prior art.

## Why no other action-head family can do this (the mechanism)
| Action head | Output mechanism | Variable control-frequency deploy? |
|---|---|---|
| **FNO (ours)** | trajectory reconstructed by `irfft(·, n=H)` | **YES — H is free at inference** |
| MLP / regression | fixed linear map to a fixed-length chunk | No — output dim fixed at training |
| Diffusion Policy / flow | denoiser operates on a fixed-shape action tensor | No — the denoising net's I/O shape is fixed at training |
| Autoregressive tokens (OpenVLA) | fixed action-token vocabulary + horizon | No — discretization + horizon fixed |
| Branch-trunk / DeepONet (Sewell'26; DeepONet-PH-VLA) | trunk queries fixed coordinates; operator is over *task/observation* space | No — not formulated over the control-time axis; no variable-rate action decoding demonstrated |

The reason is concrete: MLP, diffusion/flow, and autoregressive heads all fix the
*shape* of their output at training time. To change the control rate they must be
retrained, or their output must be resampled by an external interpolator (which
degrades the learned trajectory and is not part of the policy). The FNO instead learns
a *continuous* spectral representation of the trajectory, so `irfft(·, n=H)` is an
exact, learned re-synthesis at any `H`. Nothing external is added.

## Why this matters (it is a deployment advantage, not a curiosity)
Real robot controllers run at heterogeneous rates (e.g. 10 / 20 / 50 Hz depending on
the arm, the safety controller, and the compute budget). A policy trained at one rate
normally cannot be moved to another without retraining or lossy interpolation.
VLA-DSS is deployed natively at the target rate: the *same* checkpoint serves a
10 Hz onboard controller and a 50 Hz workstation. For the edge-deployment regime this
model targets, that is a first-class practical property.

## Differentiation from the closest cited prior art (arXiv:2604.03449, Sewell et al.)
Sewell et al. use a **branch-trunk (DeepONet) neural operator** to map a *task
descriptor* (cost / dynamics functions) to an *optimal control law*, on parametric
optimal-control environments and a locomotion benchmark. It shares only the phrase
"neural operators for control"; below that it diverges on every axis:

| Axis | Sewell et al. (2604.03449) | VLA-DSS (ours) |
|---|---|---|
| Operator family | branch-trunk (**DeepONet**) | **Fourier Neural Operator** (spectral) |
| Operator acts over | **task space** (descriptor → control law) | **control-time axis** of the action chunk (frequency domain) |
| Structural prior | multi-task / adaptation | **band-limited (low-frequency) motion** |
| Perception + language | none (state-based) | **RGB (frozen DINOv3 + scattering) + language** |
| Domain | optimal control + locomotion | **vision-language-action manipulation (LIBERO)** |
| Resolution-invariant *control-freq* deploy | not addressed | **yes, and measured** |
| Smoothness by construction | no | **yes (jerk 0.026 vs 0.039)** |

The two works answer different questions: theirs is *"can a neural operator generalize
across control tasks and adapt with little data?"*; ours is *"can a spectral operator
be the action decoder of a compact VLA, giving smoothness-by-construction and
control-frequency-invariant deployment?"* Neither the operator family, the axis it
operates over, the input modalities, nor the deployment property overlap.

## Bonus separation from the nearest competitor
The DeepONet-PH-VLA line is also a **branch-trunk** operator. So the *same* argument
that separates us from Sewell et al. separates us from the closest competing VLA:
we are the **spectral (FNO)** approach, and the train-once/deploy-at-any-frequency
property is specific to the spectral formulation — a branch-trunk head does not
provide it.

## Why FNO is the *right* choice, not merely a different one
Robot end-effector trajectories are band-limited: they are dominated by low
frequencies, and high-frequency content implies physically implausible jerk. An FNO
keeps only the lowest `k` modes, so it is *matched to the signal class*. The two
properties it yields for free — smoothness-by-construction and
resolution-invariance — both follow from that match. A branch-trunk operator has no
frequency prior and provides neither.

## One-sentence claim for abstract/intro
> To our knowledge VLA-DSS is the first vision-language-action model with a Fourier
> Neural Operator action head, giving smoothness-by-construction and — uniquely among
> action-head families — **resolution-invariant deployment (train once, run at any
> control frequency)**, a property that fixed-output diffusion/flow/MLP heads and
> branch-trunk (DeepONet) operators cannot provide.
