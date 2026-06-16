import h5py, numpy as np, glob, os
print("h5py", h5py.__version__, "| numpy", np.__version__, "| hdf5", h5py.version.hdf5_version)
f = sorted(glob.glob("D:/eroot/fno_data/libero_object/*.hdf5"))[0]
h = h5py.File(f, "r")
ds = h["data/demo_0/obs/agentview_rgb"]
print("shape", ds.shape, "dtype", ds.dtype, "compression", ds.compression, "chunks", ds.chunks)
for how, fn in [("astype-uint8", lambda: ds.astype("uint8")[0]),
                ("read_direct", None),
                ("slice 0:1", lambda: ds[0:1]),
                ("np.asarray", lambda: np.asarray(ds[...]))]:
    try:
        if how == "read_direct":
            buf = np.empty(ds.shape, ds.dtype); ds.read_direct(buf); r = buf[0]
        else:
            r = fn()
        print(f"  {how}: OK {getattr(r,'shape',None)}")
    except Exception as e:
        print(f"  {how}: FAIL {type(e).__name__}: {e}")
h.close()
