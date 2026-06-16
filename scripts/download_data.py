"""
Download a single LIBERO suite from HuggingFace (yifengzhu-hf/LIBERO-datasets)
without importing the heavy `libero` package (no mujoco/robosuite needed).

Usage:
    python scripts/download_data.py --suite libero_object --out data
"""
import argparse
import os
from huggingface_hub import snapshot_download

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {args.suite} from {HF_REPO_ID} -> {args.out}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=f"{args.suite}/*",
    )

    suite_dir = os.path.join(args.out, args.suite)
    files = [f for f in os.listdir(suite_dir) if f.endswith(".hdf5")] if os.path.isdir(suite_dir) else []
    print(f"Done. {len(files)} hdf5 files in {suite_dir}")
    for f in sorted(files):
        print("  ", f)


if __name__ == "__main__":
    main()
