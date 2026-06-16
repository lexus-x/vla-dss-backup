# Resolution-invariance sweep: decode the FNO at output_size in {8,16,24,32},
# resample to exec_res=16, run baseline (63.5%) N=20 each. success ~flat above
# Nyquist (16) = resolution-invariance; 8 (below Nyquist) shows the aliasing floor.
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$ckpt = 'E:/fno_data/run_dinov3_finetune/best.pt'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'FNO resolution-invariance sweep'
foreach ($s in 8,16,24,32) {
  Write-Host "=== output_size = $s ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zres_$s.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $ckpt --suite libero_object `
      --n_rollouts 10 --execute 8 --output_size $s `
      --log_jsonl "E:/fno_data/zres_$s.jsonl" --tag "res$s" `
      > "c:\sarvik\fno_backup\logs\_res_$s.txt" 2>&1
  Write-Host "  done output_size=$s" -ForegroundColor Green
}
Write-Host "=== resolution sweep complete ===" -ForegroundColor Green