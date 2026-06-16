# AUG (Step 1, ep15) robustness battery — BLUR FIRST (the decisive comparison vs DAgger's
# blur collapse 42->17%). Then noise, then brightness. N=5/task. Writes zbat_aug_*.jsonl.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'battery AUG ep15 (blur-first)'
$ck = 'E:/fno_data/run_dinov3_aug/epoch_15.pt'
# clean
Remove-Item 'E:\fno_data\zbat_aug_clean.jsonl' -Force -EA SilentlyContinue
& $py -u scripts/eval_sim.py --checkpoint $ck --suite libero_object --n_rollouts 5 --execute 8 `
    --perturb none --severity 0 --log_jsonl 'E:/fno_data/zbat_aug_clean.jsonl' --tag 'aug_clean' `
    > 'c:\sarvik\fno_backup\logs\_bat_aug_clean.txt' 2>&1
Write-Host '  aug clean done' -ForegroundColor Green
# blur first, then noise, then brightness
foreach($kind in 'blur','noise','brightness'){
  foreach($sev in 1,2,3){
    Remove-Item "E:\fno_data\zbat_aug_$kind$sev.jsonl" -Force -EA SilentlyContinue
    & $py -u scripts/eval_sim.py --checkpoint $ck --suite libero_object --n_rollouts 5 --execute 8 `
        --perturb $kind --severity $sev --log_jsonl "E:/fno_data/zbat_aug_$kind$sev.jsonl" --tag "aug_$kind$sev" `
        > "c:\sarvik\fno_backup\logs\_bat_aug_$kind$sev.txt" 2>&1
    Write-Host "  aug $kind sev$sev done" -ForegroundColor Green
  }
}
Write-Host '=== aug battery complete ===' -ForegroundColor Green
