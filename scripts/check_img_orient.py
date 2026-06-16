"""Dump HDF5 agentview_rgb[0] RAW (as the trainer feeds it) so we can see if the
stored demo image is upside-down (robosuite-native, matches live env) or upright
(mismatch -> model sees flipped images at eval -> systematic spatial error)."""
import os, glob, h5py, cv2, numpy as np

root = "E:/fno_data/libero_object"
if not os.path.isdir(root):
    root = "D:/eroot/fno_data/libero_object"
f = sorted(glob.glob(os.path.join(root, "*.hdf5")))[0]
print("file:", os.path.basename(f))
with h5py.File(f, "r") as h:
    demo = list(h["data"].keys())[0]
    print("demo:", demo, "| obs keys:", list(h[f"data/{demo}/obs"].keys()))
    ds = h[f"data/{demo}/obs/agentview_rgb"]
    print("agentview_rgb shape:", ds.shape, "dtype:", ds.dtype)
    av = ds[:][0]                                    # read full then frame 0 (matches loader)
    wr = h[f"data/{demo}/obs/eye_in_hand_rgb"][:][0]
    print("frame0 shape:", av.shape, "dtype:", av.dtype)
# stack RAW (top) vs vertical-flip (bottom) so orientation is obvious
def lab(img, t):
    img = np.ascontiguousarray(img[..., ::-1])  # RGB->BGR for cv2
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    cv2.putText(img, t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return img
out = np.hstack([
    np.vstack([lab(av, "AGENT raw(HDF5)"), lab(av[::-1], "AGENT vflip")]),
    np.vstack([lab(wr, "WRIST raw(HDF5)"), lab(wr[::-1], "WRIST vflip")]),
])
cv2.imwrite("C:/sarvik/fno_backup/logs/hdf5_orient.png", out)
print("wrote C:/sarvik/fno_backup/logs/hdf5_orient.png")
