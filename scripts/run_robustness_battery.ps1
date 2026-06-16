# Shared robustness battery: clean + noise/blur/brightness at sev 1,2,3, N=5/task.
# Run the SAME battery on each ablation model -> attribution table.
#   powershell -File run_robustness_battery.ps1 -ckpt E:/fno_data/run_dinov3_aug/epoch_5.pt -tag aug
param([string]$ckpt, [string]$tag)
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = "battery $tag"
# clean (sev0)
& $py -u scripts/eval_sim.py --checkpoint $ckpt --suite libero_object --n_rollouts 5 --execute 8 `
    --perturb none --severity 0 --log_jsonl "E:/fno_data/zbat_${tag}_clean.jsonl" --tag "${tag}_clean" `
    > "c:\sarvik\fno_backup\logs\_bat_${tag}_clean.txt" 2>&1
foreach($kind in 'noise','blur','brightness'){
  foreach($sev in 1,2,3){
    & $py -u scripts/eval_sim.py --checkpoint $ckpt --suite libero_object --n_rollouts 5 --execute 8 `
        --perturb $kind --severity $sev --log_jsonl "E:/fno_data/zbat_${tag}_${kind}${sev}.jsonl" --tag "${tag}_${kind}${sev}" `
        > "c:\sarvik\fno_backup\logs\_bat_${tag}_${kind}${sev}.txt" 2>&1
    Write-Host "  done $tag $kind sev$sev"
  }
}
Write-Host "=== battery $tag complete ===" -ForegroundColor Green
