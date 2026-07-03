# Related Work

## Vision-Language-Action models
**Octo** (Octo-Small 27M transformer + frozen T5-base 109.6M; Octo-Base 93M) is an
open, transformer-based VLA with a diffusion action head, trained on Open X-Embodiment.
**OpenVLA** (7B, LLaMA-2 + SigLIP 400M) autoregresses discretized actions and is far
larger and heavier to deploy. **RT-1/RT-2** established the VLA paradigm at large scale.
**SmolVLA** and **Pi0** target smaller/efficient regimes. VLA-DSS differs on the
**action decoder**: where these use diffusion, autoregressive token, or MLP heads,
we use a Fourier Neural Operator that encodes the frequency structure of motion.

## Action decoders in imitation learning
**ACT** (action chunking transformer) and **Diffusion Policy** are the dominant chunked
action heads. Both predict a fixed-length chunk and have **no explicit smoothness prior**
and **no resolution invariance** — the output length is fixed at training. Diffusion
heads additionally pay iterative denoising cost at inference. Our FNO head predicts a
chunk in one shot, is smooth by construction, and can be decoded at any temporal
resolution.

## Neural operators
**Fourier Neural Operators (FNO)** were introduced for learning solution operators of
PDEs, exploiting that many physical fields are dominated by low frequencies and that
operators can be parameterized in the spectral domain (resolution-invariant by
construction). We repurpose this for **action generation**: robot trajectories are
smooth low-frequency signals, so an FNO is a natural, parameter-efficient decoder. To
our knowledge this is the first use of a neural-operator action head in a VLA.

## Wavelet scattering
**Scattering transforms** provide translation-invariant, deformation-stable,
noise-robust representations with no learned parameters, and are provably non-expansive
to small input perturbations. We use scattering as a stability path alongside a frozen
semantic ViT, and show (Ablations) that it contributes noise robustness rather than
clean accuracy.

## Corrective data (DAgger)
DAgger addresses covariate shift by aggregating expert corrections on policy-visited
states. We use a hand-crafted, privileged-pose oracle (parsed from each task's BDDL
goal) to generate grasp-phase corrections, which lifts Object and Spatial success while
keeping the model unchanged and adding zero inference cost.

## Positioning
VLA-DSS is, to our knowledge, the smallest deployable VLA to (a) use a neural-operator
action head, (b) demonstrate resolution-invariant action decoding, and (c) beat
Octo-Small on the LIBERO four-suite average at matched size and budget with ~4x fewer
deployable parameters.
