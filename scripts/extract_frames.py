"""Dump an evenly-spaced montage of frames from a rollout mp4, so we can SEE
what actually happened. Output: one tall PNG (rows = time, top->bottom)."""
import sys, cv2, numpy as np

src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
start_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cap = cv2.VideoCapture(src)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(int(start_frac * (total - 1)), max(total - 1, 0), n).astype(int)
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    if ok:
        f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        cv2.putText(f, f"t={i}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        frames.append(f)
cap.release()
montage = np.vstack(frames)
cv2.imwrite(out, montage)
print(f"{src}: {total} frames -> {out}  ({montage.shape[1]}x{montage.shape[0]})")
