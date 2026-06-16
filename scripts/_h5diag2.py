import numpy as np, h5py, glob
print("numpy", np.__version__, "| uint8 itemsize", np.dtype("uint8").itemsize, "| float32 itemsize", np.dtype("float32").itemsize)
f = sorted(glob.glob("D:/eroot/fno_data/libero_object/*.hdf5"))[0]
h = h5py.File(f, "r")
for path in ["data/demo_0/actions","data/demo_0/obs/ee_pos","data/demo_0/obs/joint_states","data/demo_0/obs/agentview_rgb"]:
    ds = h[path]
    try:
        r = ds[:]
        print(f"  {path}: OK shape={r.shape} dtype={r.dtype}")
    except Exception as e:
        print(f"  {path}: FAIL {type(e).__name__}: {e}")
h.close()
