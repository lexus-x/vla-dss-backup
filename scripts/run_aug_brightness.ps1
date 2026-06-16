# AUG ep15 BRIGHTNESS only (sev 1,2,3). Noise skipped (it's a tie). N=5/task.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'aug brightness'
$ck = 'E:/fno_data/run_dinov3_aug/epoch_15.pt'
foreach($sev in 1,2,3){
  Remove-Item "E:\fno_data\zbat_aug_brightness$sev.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $ck --suite libero_object --n_rollouts 5 --execute 8 `
      --perturb brightness --severity $sev --log_jsonl "E:/fno_data/zbat_aug_brightness$sev.jsonl" --tag "aug_brightness$sev" `
      > "c:\sarvik\fno_backup\logs\_bat_aug_brightness$sev.txt" 2>&1
  Write-Host "  aug brightness sev$sev done" -ForegroundColor Green
}
Write-Host '=== aug brightness complete ===' -ForegroundColor Green
