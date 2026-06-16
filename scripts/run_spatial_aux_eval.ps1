# Find the success peak for Spatial+aux: eval the EARLY ladder (ep5/10/15/20).
# Late ckpts (ep100, val 0.0008) are overtrained -> skip. N=10/task, exec8, 1 video/task.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'EVAL spatial+aux'
foreach ($ep in 5,10,15,20) {
  Write-Host "=== eval spa$ep (libero_spatial, aux, N=10) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zspa_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_spatial_aux/epoch_$ep.pt" `
      --suite libero_spatial --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zspa_$ep.jsonl" --tag "spa$ep" `
      --save_videos --video_dir "E:/fno_data/eval_videos_spatial_aux/spa$ep" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_spa$ep.txt" 2>&1
  Write-Host "  done spa$ep" -ForegroundColor Green
}
Write-Host '=== spatial+aux ladder eval complete ===' -ForegroundColor Green
