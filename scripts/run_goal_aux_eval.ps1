# Find the success peak for Goal+aux: eval ep5/10/15/20 at N=10/task, exec8, 1 video/task.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'EVAL goal+aux'
foreach ($ep in 5,10,15,20) {
  Write-Host "=== eval goa$ep (libero_goal, aux, N=10) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zgoa_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_goal_aux/epoch_$ep.pt" `
      --suite libero_goal --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zgoa_$ep.jsonl" --tag "goa$ep" `
      --save_videos --video_dir "E:/fno_data/eval_videos_goal_aux/goa$ep" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_goa$ep.txt" 2>&1
  Write-Host "  done goa$ep" -ForegroundColor Green
}
Write-Host '=== goal+aux ladder eval complete ===' -ForegroundColor Green
