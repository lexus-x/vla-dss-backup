"""Render the robustness perturbations at every severity on a real frame -> grid PNG."""
import numpy as np, cv2, glob, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# perturb_img: IDENTICAL to eval_sim.py
def perturb_img(img, kind, sev):
    if kind == 'none' or sev <= 0: return img
    im = img.astype(np.float32)
    if kind == 'noise':
        std = [0,5,10,16,24,34][min(sev,5)]; im = im + np.random.normal(0,std,im.shape)
    elif kind == 'blur':
        sigma = [0,0.6,1.1,1.8,2.7,4.0][min(sev,5)]; kk = max(1,int(2*round(sigma)+1))|1
        im = cv2.GaussianBlur(im,(kk,kk),sigma)
    elif kind == 'brightness':
        im = im + [0,18,36,58,84,115][min(sev,5)]
    return np.clip(im,0,255).astype(np.uint8)

# real agentview frame from a demo (a mid-trajectory frame = has the objects in view)
fs = glob.glob(r"E:\fno_data\libero_object\*demo.hdf5")
with h5py.File(fs[0],"r") as f:
    d = f["data"][sorted(f["data"].keys())[0]]
    frames = d["obs"]["agentview_rgb"][:]
img = frames[len(frames)//3]  # a frame partway in
img = cv2.resize(np.ascontiguousarray(img), (256,256))

np.random.seed(0)
kinds = ["noise","blur","brightness"]
fig, ax = plt.subplots(3, 6, figsize=(15, 8))
for r,k in enumerate(kinds):
    for s in range(6):
        ax[r,s].imshow(perturb_img(img, k, s))
        ax[r,s].axis("off")
        if r==0: ax[r,s].set_title(f"severity {s}", fontsize=11)
    ax[r,0].set_ylabel(k, fontsize=13, rotation=90)
    # row label (since axis off, use text)
    ax[r,0].text(-0.15, 0.5, k, transform=ax[r,0].transAxes, fontsize=13,
                 va="center", ha="right", rotation=90, weight="bold")
fig.suptitle("Robustness perturbations across severities (LIBERO-Object agentview)", fontsize=14)
plt.tight_layout(rect=[0.02,0,1,0.97])
out = r"c:\sarvik\fno_backup\perturbation_examples.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
