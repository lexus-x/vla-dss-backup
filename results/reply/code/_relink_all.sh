#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate octo
python - <<'PY'
from huggingface_hub import snapshot_download
import os, glob
suites = ["libero_object_no_noops","libero_spatial_no_noops","libero_goal_no_noops","libero_10_no_noops"]
for s in suites:
    p = snapshot_download(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
                          allow_patterns=[f"{s}/*"])
    d = os.path.join(p, s, "1.0.0")
    tfr = glob.glob(os.path.join(d, "*.tfrecord*"))
    ok = os.path.isfile(os.path.join(d,"dataset_info.json"))
    print(f"{s}: tfrecords={len(tfr)} info={ok}")
PY
