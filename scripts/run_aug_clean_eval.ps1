# Floor check: aug-only (Step 1) clean Object accuracy on ep5/10/15 (N=10/task).
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'aug clean floor-check'
foreach ($ep in 5,10,15) {
  Write-Host "=== aug ep$ep clean ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zaugclean_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_aug/epoch_$ep.pt" `
      --suite libero_object --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zaugclean_$ep.jsonl" --tag "augclean$ep" `
      > "c:\sarvik\fno_backup\logs\_augclean_$ep.txt" 2>&1
  Write-Host "  done ep$ep" -ForegroundColor Green
}
Write-Host '=== aug clean floor-check complete ===' -ForegroundColor Green
