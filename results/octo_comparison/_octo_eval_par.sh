#!/bin/bash
# Parallel Octo eval: all 4 suites' 10k ckpts concurrently (headline), then all 4
# 5k ckpts concurrently. 200 rollouts each. Single 48GB GPU / 48 cores.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate octo
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
EV=/mnt/c/sarvik/fno_backup/octo_libero_eval.py
OUT=/home/islab/octo_ft_eval; mkdir -p $OUT
PROG=$OUT/EVAL_PROGRESS.log

one() {  # tag suite maxsteps step
  local tag=$1 suite=$2 maxs=$3 step=$4
  local ckdir=$(ls -d /home/islab/octo_ft/$tag/octo_libero/experiment_* 2>/dev/null | head -1)
  local log=$OUT/${tag}_${step}.jsonl
  rm -f "$log"
  python $EV --finetuned_path "$ckdir" --step $step --suite $suite \
     --n_tasks 10 --n_rollouts 20 --max_steps $maxs --exec_horizon 4 --log "$log" \
     > $OUT/${tag}_${step}.out 2>&1
}

summ() {  # tag step
  python - "$OUT/$1_$2.jsonl" "$1" "$2" <<'PY' | tee -a $PROG
import json,sys
p,tag,step=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    rows=[json.loads(l) for l in open(p) if l.strip()]
    n=len(rows); s=sum(r["success"] for r in rows)
    print(f"  RESULT {tag} step{step}: {100*s/max(n,1):.1f}%  ({s}/{n})")
except Exception as e:
    print(f"  RESULT {tag} step{step}: ERROR {e}")
PY
}

wave() {  # step
  local step=$1
  echo "[$(date '+%H:%M:%S')] WAVE step $step START (4 parallel)" | tee -a $PROG
  one object  libero_object  400 $step &
  one spatial libero_spatial 400 $step &
  one goal    libero_goal    400 $step &
  one long    libero_10      600 $step &
  wait
  for t in object spatial goal long; do summ $t $step; done
  echo "[$(date '+%H:%M:%S')] WAVE step $step DONE" | tee -a $PROG
}

echo "===== OCTO PARALLEL EVAL START $(date) =====" | tee -a $PROG
wave 10000
wave 5000
echo "===== OCTO EVAL COMPLETE $(date) =====" | tee -a $PROG
