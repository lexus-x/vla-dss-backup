#!/bin/bash
# Sequential Octo-small full finetune on all 4 LIBERO suites (50k steps each,
# checkpoint every 10k). Single A6000. Logs per suite. Robust: continues to next
# suite even if one errors (marks .FAIL).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate octo
cd /home/islab/octo
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /home/islab/octo_ft_logs

train_one() {
  local suite=$1          # rlds name, e.g. libero_spatial_no_noops
  local tag=$2            # short tag for save dir
  local save=/home/islab/octo_ft/$tag
  local log=/home/islab/octo_ft_logs/${tag}.log
  echo "===== TRAIN $tag ($suite) 50k steps  $(date) =====" | tee -a /home/islab/octo_ft_logs/PROGRESS.log
  rm -rf "$save"; mkdir -p "$save"
  stdbuf -oL python scripts/finetune.py \
    --config=scripts/configs/libero_finetune_config.py:"$suite" \
    --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 \
    --config.save_dir="$save" \
    --config.num_steps=10000 \
    --config.batch_size=64 \
    --config.save_interval=5000 \
    --config.optimizer.learning_rate.warmup_steps=1000 \
    --debug > "$log" 2>&1
  if [ $? -eq 0 ]; then
    echo "===== DONE $tag  $(date) =====" | tee -a /home/islab/octo_ft_logs/PROGRESS.log
  else
    echo "===== FAIL $tag  $(date) =====" | tee -a /home/islab/octo_ft_logs/PROGRESS.log
    touch "$save/.FAIL"
  fi
}

train_one libero_spatial_no_noops spatial
train_one libero_goal_no_noops    goal
train_one libero_10_no_noops      long
train_one libero_object_no_noops  object
echo "===== ALL SUITES COMPLETE  $(date) =====" | tee -a /home/islab/octo_ft_logs/PROGRESS.log
