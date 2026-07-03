# Analysis, Discussion, Limitations, Future Work

## Why a Fourier action head helps
Robot end-effector trajectories are **smooth and low-frequency**: position and
orientation vary slowly relative to the control rate, and abrupt high-frequency
components are physically implausible (they imply large accelerations/jerk). A generic
MLP or transformer head must *learn* this structure from data, spending capacity to
discover that trajectories are smooth. The FNO head instead **bakes it into the
architecture**: by keeping only the lowest `k` spectral modes, high-frequency jitter is
suppressed by construction, and the remaining capacity is spent on task reasoning. Two
consequences we measure directly:
1. **Smoothness for free** — mean jerk 0.0256 vs Octo's 0.0386 (~33% smoother), at
   comparable or better success.
2. **Resolution invariance** — the trajectory is reconstructed via `irfft(·, n=H)`, so
   the same operator decodes at any H (8/16/24/32 -> 29/61/68/66%). This is a property
   of the operator, not of the data, and no fixed-output head has it.

## Efficiency as a design outcome, not an afterthought
Because smoothness is architectural, VLA-DSS spends few parameters on the decoder
(2.43M) and freezes its semantic vision (DINOv3) and (in the LIBERO setting) uses a
0.40M language encoder. The result is a **28.9M deployable model** that matches or beats
a baseline needing a **136.7M** deployment (Octo's frozen 109.6M T5 dominates its
footprint). This matters for the target regime — edge/onboard robot hardware.

## Failure analysis
Failures are dominated by **grasp acquisition** (the policy reaches correct grasp
*height* but lands slightly off in x-y, toppling narrow objects), which motivated both
the aux x-y head and the DAgger corrections. Suite-wise, VLA-DSS is strongest where the
grasp is stereotyped (Object) and weakest on long-horizon compositional tasks (Long),
where error compounds across sub-goals.

## When the FNO helps least
On suites where success is not smoothness-limited (e.g., where Octo already converges,
Spatial/Goal), the FNO's smoothness prior yields ties rather than wins. The FNO's
advantage is clearest where open-loop trajectory quality matters (Object) and in the
*qualitative* properties (jerk, resolution invariance) that are architecture-level, not
suite-specific.

## Limitations
- **Simulation only (this version).** Real-robot validation on a low-cost SO-100 arm is
  planned; the fine-tuning pipeline (sim-free aux head, light language encoder) is built.
- **Two of four suites are ties, not wins.** The average win is carried by Object/Long.
- **Robustness degrades steeply at high perturbation severity** (noise/brightness -> 0
  by severity 4-5); the scattering path helps under noise but costs under blur.
- **DAgger uses a privileged-pose oracle** (sim only); the corrected policy uses no
  privileged state at test time, but the correction data generation does.

## Future work
Real-robot deployment (SO-100), a diffusion-head arm of the head ablation, learned
mode-selection in the FNO, and extending the frequency-domain decoder to force/torque
(contact-rich) tasks where smoothness priors are even more valuable.
