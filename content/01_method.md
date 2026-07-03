# Method

## 1. Overview
VLA-DSS maps a single RGB observation, a language instruction, and proprioception to
a chunk of continuous end-effector actions. It has four parts: a **dual-path vision
encoder**, a **compact language encoder**, a **FiLM cross-attention fusion module**,
and a **Fourier Neural Operator (FNO) action decoder**. Total: **28.9M parameters
(7.03M trainable)**; the deployed product swaps in a 4.4M bert-tiny language encoder
(~32.9M). All vision/language backbones are frozen; only fusion + FNO head + the small
learned components train.

Parameter breakdown (counted from checkpoint):

| Component | Params | Trainable |
|---|---|---|
| Vision (DINOv3 ViT-S 21.6M frozen + scattering CNN) | 22.91M | 1.33M |
| Language (learned 2-layer, 128-d) | 0.40M | 0.40M |
| Fusion (FiLM cross-attention) | 2.83M | 2.83M |
| FNO action head | 2.43M | 2.43M |
| Proprio + aux x-y head | 0.04M | 0.04M |

## 2. Dual-path vision encoder
Two complementary streams over a 128x128 RGB frame:
- **Semantic path:** a **frozen DINOv3 ViT-S/16** (`requires_grad=False`, eval mode),
  giving strong pretrained semantic features at zero training cost.
- **Stability path:** a **wavelet scattering CNN** (J=3, L=12). Scattering is provably
  contractive to small input perturbations (Section: Ablations), providing noise
  robustness that a learned CNN does not.
The two token sets are gated-fused. Dropout (p=0.3) is applied to the DINOv3 tokens so
the network cannot collapse onto the semantic path and ignore scattering.

## 3. Language & proprioception
- **Language:** for LIBERO's templated instructions, a 0.40M learned 2-layer
  transformer (embed 128, vocab 1000) suffices. The deployed product uses a 4.4M
  bert-tiny encoder for robust real-world / novel-instruction grounding. Either is
  tiny next to Octo's **frozen 109.6M T5**.
- **Proprio:** 15-D state -> 2-layer MLP -> 128-D.

## 4. FiLM cross-attention fusion
Project vision/language/proprio tokens to a shared 256-D space, concatenate, pass
through a 3-layer, 8-head cross-attention transformer, and mean-pool to a latent
`z in R^256`. From `z` we generate FiLM parameters `(gamma_l, beta_l)` for each Fourier
layer of the action head. Thus the instruction/scene conditions the decoder through
feature-wise modulation rather than token concatenation.

## 5. FNO action decoder (the contribution)
The decoder predicts an action chunk `a_{1:H} in R^{H x 6}` (H = 16 at train time)
plus a gripper sequence. Motivation: robot trajectories are **smooth, low-frequency**
signals, so we decode them in the **frequency domain**.

**Lifting.** Replicate `z` across H steps, add sinusoidal positional encoding, and
project to a per-step feature `x in R^{H x d}`.

**Fourier layers (x4).** Each layer computes
```
  X = rfft(x, axis=time)                 # to frequency domain
  X'[:k] = R_l · X[:k]                    # keep lowest k modes, learned complex R_l
  y = irfft(X', n=H) + W_l · x            # back to time + local linear (skip)
  x = GeLU( gamma_l ⊙ y + beta_l )        # FiLM conditioning from z
```
Keeping only the lowest `k` modes makes the output **smooth by construction** — high
frequencies (jitter) are architecturally suppressed, not penalized.

**Projection.** Linear to 6-D continuous actions per step. A parallel gripper head
maps `z` to a per-step open/close logit.

**Resolution invariance.** Because the trajectory is reconstructed by `irfft(·, n=H)`,
H is a free parameter at inference: the *same* learned operator decodes the trajectory
at any temporal resolution (8/16/24/32), i.e. any control rate. No MLP or diffusion
head has this property (their output dimensionality is fixed at training).

## 6. Auxiliary x-y head (zero inference cost)
During training a small head regresses the target object's x-y position from `z`
(supervised by privileged sim pose). It sharpens grasp localization but is **dropped
at inference**, so it costs nothing at deployment.

## 7. Training
Cross-suite pretrain on all five LIBERO suites (~130 tasks, 80 epochs, batch 128,
DINOv3 frozen), then per-suite fine-tune (standard LIBERO protocol; OpenVLA-OFT and
SmolVLA fine-tune per suite likewise). Loss: smooth-L1 on continuous actions + BCE on
the gripper sequence. Headline checkpoints are early-stopped (validation), which we
show is correct in the Ablations (later checkpoints overfit).
