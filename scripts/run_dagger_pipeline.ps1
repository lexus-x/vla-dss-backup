# X-Y FIX #3 DAgger pipeline. RUN ONLY WHEN THE GPU IS FREE (no eval/train running).
#   STAGE 1  collect : roll out aux-xy ep5, relabel policy-visited states with the
#                      privileged-pose oracle -> <lang>_dagger_demo.hdf5 (all 10 tasks)
#   STAGE 2  train   : finetune the aux+dagger recipe from the DINOv3 pretrain on
#                      (50 orig demos + dagger demos), ckpt every 5
# Eval (N=100 vs 71%) is a SEPARATE step after we inspect the val curve.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'DAgger pipeline (collect -> train)'
$logd = 'c:\sarvik\fno_backup\logs'

Write-Host "=== STAGE 1/2: collect DAgger data (aux-xy ep5, all 10 tasks) ===" -ForegroundColor Cyan
& $py -u scripts/collect_dagger.py `
    --checkpoint E:/fno_data/run_dinov3_auxxy/epoch_5.pt `
    --task_indices 0,1,2,3,4,5,6,7,8,9 --rollouts_per_task 50 `
    --out_dir E:/fno_data/libero_object_dagger `
    > "$logd\_dagger_collect.txt" 2>&1
Write-Host "  collect done" -ForegroundColor Green

# rebuild the dir listing (originals already hardlinked by setup_dagger_dir.ps1)
powershell -ExecutionPolicy Bypass -File scripts/setup_dagger_dir.ps1

Write-Host "`n=== STAGE 2/2: finetune aux+DAgger from pretrain (ckpt every 5) ===" -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_dagger.yaml `
    --data_dir E:/fno_data/libero_object_dagger `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "$logd\_train_dagger.txt" 2>&1
Write-Host "=== DAgger pipeline complete -> eval run_dinov3_dagger checkpoints next ===" -ForegroundColor Green
