# Find the success peak for Long+aux (libero_10): eval ep5/10/15/20 at N=10/task, exec8.
# Long-horizon -> slower rollouts (up to 500 steps) and lower success expected.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'EVAL long+aux'
foreach ($ep in 5,10,15,20) {
  Write-Host "=== eval lng$ep (libero_10, aux, N=10) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zlng_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_long_aux/epoch_$ep.pt" `
      --suite libero_10 --n_rollouts 10 --execute 8 --max_steps 520 `
      --log_jsonl "E:/fno_data/zlng_$ep.jsonl" --tag "lng$ep" `
      --save_videos --video_dir "E:/fno_data/eval_videos_long_aux/lng$ep" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_lng$ep.txt" 2>&1
  Write-Host "  done lng$ep" -ForegroundColor Green
}
Write-Host '=== long+aux ladder eval complete ===' -ForegroundColor Green
