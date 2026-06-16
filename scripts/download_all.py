"""
Download all LIBERO suites from HuggingFace into a single root.
libero_object first (smallest / needed for eval+finetune), then the rest.

Usage:
    python scripts/download_all.py --out E:/fno_data
"""
import argparse, os, time
from huggingface_hub import snapshot_download

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
# object first so training/eval can start before the big pretrain suites land
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10", "libero_90"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/fno_data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for suite in SUITES:
        t0 = time.time()
        print(f"=== downloading {suite} -> {args.out} ===", flush=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=args.out,
            allow_patterns=f"{suite}/*",
            max_workers=8,
        )
        d = os.path.join(args.out, suite)
        n = len([f for f in os.listdir(d) if f.endswith(".hdf5")]) if os.path.isdir(d) else 0
        print(f"=== {suite} done: {n} hdf5 files in {(time.time()-t0)/60:.1f} min ===", flush=True)

    print("=== ALL SUITES DOWNLOADED ===", flush=True)


if __name__ == "__main__":
    main()
