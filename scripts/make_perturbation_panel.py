"""Showcase image: what the model sees — clean vs severity-3 noise / blur / brightness."""
import os, glob, numpy as np, h5py, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def perturb(img, kind, sev):
    im = img.astype(np.float32)
    if kind == "noise":      im = im + np.random.normal(0, [0,5,10,16,24,34][sev], im.shape)
    elif kind == "blur":     s=[0,0.6,1.1,1.8,2.7,4.0][sev]; k=max(1,int(2*round(s)+1))|1; im=cv2.GaussianBlur(im,(k,k),s)
    elif kind == "brightness": im = im + [0,18,36,58,84,115][sev]
    return np.clip(im, 0, 255).astype(np.uint8)

np.random.seed(0)
hp = glob.glob(r"E:\fno_data\libero_object\*.hdf5")[0]
with h5py.File(hp, "r") as f:
    k = sorted(f["data"].keys())[0]
    frame = f["data"][k]["obs"]["agentview_rgb"][30]   # a mid-approach frame
if frame.mean() < 1: frame = (frame*255)
frame = np.ascontiguousarray(frame[::-1])  # upright

panels = [("Clean (severity 0)", frame),
          ("Gaussian Noise (severity 3)", perturb(frame, "noise", 3)),
          ("Blur (severity 3)", perturb(frame, "blur", 3)),
          ("Brightness (severity 3)", perturb(frame, "brightness", 3))]

plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":220,"savefig.bbox":"tight"})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
for ax,(title,im) in zip(axes, panels):
    ax.imshow(im); ax.set_title(title, fontsize=13, fontweight="bold", color="#26313f"); ax.axis("off")
fig.suptitle("What the policy sees under input corruption (agentview)  —  scattering keeps grasping under noise",
             fontsize=14, fontweight="bold", color="#26313f", y=1.04)
out = r"c:\sarvik\fno_backup\ppt_figures\perturbation_examples.png"
fig.savefig(out); print("wrote", out)
