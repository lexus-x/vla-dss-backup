"""Render the FNO-VLA architecture as a clean block diagram (matplotlib)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK="#1b2a41"; TEAL="#13a07a"; BLUE="#2e6fae"; PURP="#7d5ba6"; GREY="#8896a6"
GOLD="#e0a13a"; ORANGE="#e1783a"; GREEN="#2a9d6f"
TINT={"teal":"#dcf2ec","blue":"#dde9f4","purp":"#ece3f2","grey":"#e9edf1",
      "gold":"#fcefd6","orange":"#fbe5d8","green":"#dcf0e6"}

plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":230,"savefig.bbox":"tight"})
fig,ax=plt.subplots(figsize=(13.5,11)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

def box(cx,cy,w,h,title,sub,fc,ec,dashed=False,bold=False,sub_fs=8.5,t_fs=11):
    p=FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.6,rounding_size=2.2",
                     fc=fc,ec=ec,lw=2.4 if bold else 1.7,ls="--" if dashed else "-",zorder=3)
    ax.add_patch(p)
    yt=cy+h/2-h*0.30 if sub else cy
    ax.text(cx,yt,title,ha="center",va="center",fontsize=t_fs,fontweight="bold",color=INK,zorder=4)
    if sub: ax.text(cx,cy-h*0.16,sub,ha="center",va="center",fontsize=sub_fs,color="#3a4a5a",zorder=4)

def arr(x1,y1,x2,y2,dashed=False,color=GREY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,
                 lw=1.8,color=color,ls="--" if dashed else "-",
                 shrinkA=2,shrinkB=2,zorder=2,connectionstyle="arc3,rad=0"))

def tag(cx,cy,s,color):
    ax.text(cx,cy,s,ha="center",va="center",fontsize=8,style="italic",color=color,zorder=5,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=color,lw=1.1))

ax.text(50,97.5,"FNO-VLA  —  Scattering + DINOv3 vision · FiLM fusion · Fourier-Neural-Operator action head",
        ha="center",fontsize=14.5,fontweight="bold",color=INK)
ax.text(50,94.2,"28.9M parameters · RGB-only · LIBERO manipulation",ha="center",fontsize=10,color=GREY)

# ---- Row 1: inputs ----
iy=86
box(11,iy,17,8,"Agentview Camera","RGB 128×128×3",TINT["grey"],GREY)
box(29,iy,17,8,"Wrist Camera","eye-in-hand 128×128×3",TINT["grey"],GREY)
box(62,iy,18,8,"Language","'pick up the soup…'",TINT["grey"],GREY)
box(85,iy,18,8,"Proprioception","15-D (ee_pos,ori,grip,joints)",TINT["grey"],GREY)
ax.text(20,80.3,"both views stacked (6-ch) → both vision encoders",ha="center",fontsize=8,style="italic",color=GREY)

# ---- Row 2: encoders ----
ey=66
box(11,ey,18,11,"Wavelet Scattering","kymatio J=3, L=12\n+ CNN proj (128, 3 res-blocks)",TINT["teal"],TEAL,sub_fs=7.5)
box(29,ey,18,11,"DINOv3 ViT-S/16  ❄","frozen · 21.6M\nsemantic features",TINT["blue"],BLUE,sub_fs=7.5)
box(62,ey,18,11,"Language Encoder","learned 2-layer Transformer (dim 128)",TINT["purp"],PURP,sub_fs=8)
box(85,ey,18,11,"Proprio MLP","15 → 64 → 128",TINT["grey"],GREY,sub_fs=8)
ax.text(11,58.6,"Lipschitz-stable (noise-robust)",ha="center",fontsize=8,style="italic",color=TEAL)
# input->encoder arrows
arr(11,82,11,71.5); arr(29,82,29,71.5); arr(62,82,62,71.5); arr(85,82,85,71.5)

# ---- Row 3: fusion ----
fy=44
box(48,fy,72,10,"Multimodal Fusion Transformer",
    "3 layers · 8 heads · dim 256   |   proprio-gated attention pool over views   |   FiLM language conditioning",
    TINT["gold"],GOLD,sub_fs=8.5,t_fs=12)
for ex in (11,29,62,85): arr(ex,60.5,min(max(ex,20),80),49.2)
arr(48,39,48,34,color=INK)
ax.text(50.6,36.4,"latent z (256-D)",ha="left",fontsize=9,fontweight="bold",color=INK)

# ---- Row 4: heads ----
hy=25
box(33,hy,32,11,"FNO Action Decoder",
    "Fourier Neural Operator · width 256 · 4 layers · 8 retained modes · spectral conv + iFFT",
    TINT["orange"],ORANGE,bold=True,sub_fs=8,t_fs=12)
box(76,hy,24,11,"Aux x-y Head","Linear 256→128 → GELU → 128→2",TINT["grey"],GREY,dashed=True,sub_fs=8)
tag(33,17.8,"band-limited → smooth + resolution-invariant (decode at any rate)",ORANGE)
tag(76,18.0,"TRAIN-ONLY — dropped at inference (0 deploy cost)",GREY)
arr(45,32,37,31)          # z -> FNO
arr(51,32,72,31,dashed=True)   # z -> aux (dashed)

# ---- Row 5: outputs ----
oy=6.5
box(33,oy,34,8,"Action Trajectory + Gripper","chunk 16 × 6-DoF (Δx Δy Δz Δr Δp Δyaw) + 16 gripper · execute 8/16 @ 20 Hz",
    TINT["green"],GREEN,sub_fs=7.8,t_fs=10.5)
box(76,oy,24,8,"Predicted grasp (x,y)","(used only in training)",TINT["green"],GREEN,dashed=True,sub_fs=8,t_fs=10)
arr(33,19.3,33,10.7); arr(76,19.4,76,10.7,dashed=True)

# ---- legend ----
ax.text(2,2.0,"❄ frozen (not trained)      —— train + inference      - - - training only (removed at deployment)",
        ha="left",fontsize=8.5,color=INK)
fig.savefig(r"c:\sarvik\fno_backup\ppt_figures\architecture.png")
print("-> ppt_figures/architecture.png")
