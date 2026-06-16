# Robustness sweep: baseline (63.5%) under image perturbations.
# clean (none,0) + {noise,blur,brightness} x severity {1..5}, N=10 each.
# All -> E:/fno_data/robustness.jsonl (records carry perturb+severity fields).
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$ckpt = 'E:/fno_data/run_dinov3_finetune/best.pt'
$jf   = 'E:/fno_data/robustness.jsonl'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
Remove-Item $jf -Force -EA SilentlyContinue
$host.UI.RawUI.WindowTitle = 'FNO robustness sweep'

function RunOne($kind, $sev) {
  Write-Host "=== perturb=$kind severity=$sev ===" -ForegroundColor Cyan
  & $py -u scripts/eval_sim.py --checkpoint $ckpt --suite libero_object `
      --n_rollouts 10 --execute 8 --perturb $kind --severity $sev `
      --log_jsonl $jf --tag rob > "c:\sarvik\fno_backup\logs\_rob_${kind}_$sev.txt" 2>&1
  Write-Host "  done $kind/$sev" -ForegroundColor Green
}

RunOne 'none' 0                                   # clean reference (= ~63.5%)
foreach ($k in 'noise','blur','brightness') {
  foreach ($s in 1,2,3,4,5) { RunOne $k $s }
}
Write-Host '=== robustness sweep complete ===' -ForegroundColor Green
