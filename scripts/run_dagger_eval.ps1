# Eval DAgger checkpoints (ep5 + best/ep8) vs the 71% aux-xy headline.
# N=10 x 10 tasks = 100, exec8 (matched harness), 1 video/task. -> zdag_<tag>.jsonl
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'DAgger eval vs 71%'
$ckpts = @(@{p='E:/fno_data/run_dinov3_dagger/epoch_5.pt'; t='dag5'},
           @{p='E:/fno_data/run_dinov3_dagger/best.pt';   t='dagbest'})
foreach ($c in $ckpts) {
  Write-Host "=== eval $($c.t) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zdag_$($c.t).jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $c.p `
      --suite libero_object --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zdag_$($c.t).jsonl" --tag $c.t `
      --save_videos --video_dir "E:/fno_data/eval_videos_dagger" --videos_per_task 1 `
      > "c:\sarvik\fno_backup\logs\_eval_$($c.t).txt" 2>&1
  Write-Host "  done $($c.t)" -ForegroundColor Green
}
Write-Host '=== DAgger eval complete ===' -ForegroundColor Green
