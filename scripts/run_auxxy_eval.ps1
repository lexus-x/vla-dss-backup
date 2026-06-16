# Eval aux-xy checkpoints (ep5/15/25) to find the success-optimal point.
# N=10, exec8 (matched harness), all 10 tasks. -> zaux_<ep>.jsonl
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'aux-xy checkpoint eval'
foreach ($ep in 5,15,25) {
  Write-Host "=== eval epoch_$ep ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zaux_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_auxxy/epoch_$ep.pt" `
      --suite libero_object --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zaux_$ep.jsonl" --tag "aux$ep" `
      --save_videos --video_dir "E:/fno_data/eval_videos" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_aux_$ep.txt" 2>&1
  Write-Host "  done epoch_$ep" -ForegroundColor Green
}
Write-Host '=== aux-xy checkpoint eval complete ===' -ForegroundColor Green
