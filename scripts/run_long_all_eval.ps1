# Screen ALL Long checkpoints to find the success peak (late-first; Long peaks late).
# N=5/task = 50/ckpt, exec8, no videos (speed). -> zlngall_<ep>.jsonl
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'EVAL long ALL ckpts'
foreach ($ep in 30,25,20,15,10,5) {
  Write-Host "=== eval long ep$ep (libero_10, N=5) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zlngall_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_long_aux/epoch_$ep.pt" `
      --suite libero_10 --n_rollouts 5 --execute 8 --max_steps 520 `
      --log_jsonl "E:/fno_data/zlngall_$ep.jsonl" --tag "lngall$ep" `
      > "c:\sarvik\fno_backup\logs\_eval_lngall_$ep.txt" 2>&1
  Write-Host "  done ep$ep" -ForegroundColor Green
}
Write-Host '=== long all-checkpoint screen complete ===' -ForegroundColor Green
