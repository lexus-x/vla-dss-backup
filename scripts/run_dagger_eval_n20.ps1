# REPORTABLE BENCHMARK: DAgger ep5 + best(ep8) at N=20/task (200 rollouts each),
# exec8 matched harness, EVERY rollout saved to video. -> zdag_<tag>n20.jsonl
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'DAgger N=20 reportable'
$ckpts = @(@{p='E:/fno_data/run_dinov3_dagger/epoch_5.pt'; t='dag5n20'},
           @{p='E:/fno_data/run_dinov3_dagger/best.pt';   t='dagbestn20'})
foreach ($c in $ckpts) {
  Write-Host "=== eval $($c.t)  (N=20/task = 200, all videos) ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zdag_$($c.t).jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $c.p `
      --suite libero_object --n_rollouts 20 --execute 8 `
      --log_jsonl "E:/fno_data/zdag_$($c.t).jsonl" --tag $c.t `
      --save_videos --video_dir "E:/fno_data/eval_videos_dagger_n20/$($c.t)" --videos_per_task 20 `
      > "c:\sarvik\fno_backup\logs\_eval_$($c.t).txt" 2>&1
  Write-Host "  done $($c.t)" -ForegroundColor Green
}
Write-Host '=== DAgger N=20 reportable benchmark complete ===' -ForegroundColor Green
