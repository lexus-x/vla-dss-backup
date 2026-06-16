"""Verify the scattering transform is WORKING = Lipschitz-stable.
Tests the defining property: scattering output changes LESS than raw pixels under
(a) gaussian noise and (b) small spatial deformation/shift. CPU-only."""
import numpy as np, torch, glob, h5py
from kymatio.torch import Scattering2D

J, L, S = 3, 12, 128
scat = Scattering2D(J=J, shape=(S, S), L=L)  # same params as the model

# real image from a demo (fall back to structured synthetic)
fs = glob.glob(r"E:\fno_data\libero_object\*cream_cheese*demo.hdf5")
img = None
if fs:
    with h5py.File(fs[0], "r") as f:
        d = f["data"][sorted(f["data"].keys())[0]]
        im = d["obs"]["agentview_rgb"][0]  # HWC uint8
    import cv2
    img = cv2.resize(im, (S, S)).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,S,S]
else:
    img = torch.rand(1, 3, S, S)

def Sx(x):
    with torch.no_grad():
        return scat(x.contiguous()).reshape(x.shape[0], -1)

img = img.contiguous()
base = Sx(img)
with torch.no_grad():
    _shape = tuple(scat(img).shape)
print(f"scattering output: shape {_shape}  (J={J}, L={L})  -> {base.shape[1]} coeffs/channel")
print(f"deterministic: {torch.allclose(Sx(img), base)}\n")

# Plain conv feature (random-init, matches the model's raw_conv path) for contrast
import torch.nn as nn
torch.manual_seed(0)
conv = nn.Sequential(
    nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.ReLU(),
    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(128, 128, 3, stride=2, padding=1)).eval()
def Cx(x):
    with torch.no_grad():
        return conv(x.contiguous()).reshape(x.shape[0], -1)
cbase = Cx(img)
def relchg(a, b): return (torch.norm(a-b)/torch.norm(a)).item()

print("=== STABILITY: output-change / input-change (ratio<1 = stable/contractive) ===\n")
print(f"  {'perturb':16s}{'input':>9s}{'scatter':>9s}{'s_ratio':>9s}{'conv':>9s}{'c_ratio':>9s}")
import csv
rows = []
def test(label, xp):
    ic = relchg(img.flatten(1), xp.flatten(1))
    sc = relchg(base, Sx(xp)); cc = relchg(cbase, Cx(xp))
    print(f"  {label:16s}{ic:>9.4f}{sc:>9.4f}{sc/ic:>9.2f}{cc:>9.4f}{cc/ic:>9.2f}")
    rows.append([label, round(ic,4), round(sc,4), round(sc/ic,3), round(cc,4), round(cc/ic,3)])
for std in [0.02, 0.05, 0.10, 0.20]:
    test(f"noise_{std}", torch.clamp(img + torch.randn_like(img)*std, 0, 1))
for sh in [1, 2, 4, 8]:
    test(f"shift_{sh}px", torch.roll(img, shifts=(sh, sh), dims=(2, 3)))

with open(r"c:\sarvik\fno_backup\scattering_stability.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["perturb","input_relchg","scatter_relchg","scatter_ratio","conv_relchg","conv_ratio"])
    w.writerows(rows)
print("\n  scatter ratio < conv ratio  => scattering is MORE stable than a conv feature.")
print("  wrote c:\\sarvik\\fno_backup\\scattering_stability.csv")
