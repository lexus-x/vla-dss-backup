# Eval LIBERO-Spatial checkpoints (ep5, ep10, best=ep14) to find the success peak.
# N=10 x 10 tasks = 100 each, exec8 matched harness, 1 video/task. -> zsp_<tag>.jsonl
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'EVAL spatial'
$ckpts = @(@{p='E:/fno_data/run_dinov3_spatial/epoch_5.pt';  t='sp5'},
           @{p='E:/fno_data/run_dinov3_spatial/epoch_10.pt'; t='sp10'},
           @{p='E:/fno_data/run_dinov3_spatial/best.pt';     t='spbest'})
foreach ($c in $ckpts) {
  Write-Host "=== eval $($c.t) (libero_spatial, N=10/task) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zsp_$($c.t).jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $c.p `
      --suite libero_spatial --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zsp_$($c.t).jsonl" --tag $c.t `
      --save_videos --video_dir "E:/fno_data/eval_videos_spatial/$($c.t)" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_$($c.t).txt" 2>&1
  Write-Host "  done $($c.t)" -ForegroundColor Green
}
Write-Host '=== spatial eval complete ===' -ForegroundColor Green
