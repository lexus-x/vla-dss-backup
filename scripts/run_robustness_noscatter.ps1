# Robustness sweep on the NO-SCATTER model -- IDENTICAL perturbations to the FNO sweep
# (clean + noise/blur/brightness x sev1-5, N=10) for an airtight ON/OFF comparison.
# Pass the finetuned no-scatter checkpoint as $args[0], else uses best.pt.
param([string]$ckpt = 'E:/fno_data/run_dinov3_noscatter_finetune/best.pt')
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$jf = 'E:/fno_data/robustness_noscatter.jsonl'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
Set-Location 'c:\sarvik\fno_backup\code'
Remove-Item $jf -Force -EA SilentlyContinue
$host.UI.RawUI.WindowTitle = 'NO-SCATTER robustness sweep'
function RunOne($k,$s){
  Write-Host "=== noscatter perturb=$k severity=$s ===" -ForegroundColor Cyan
  & $py -u scripts/eval_sim.py --checkpoint $ckpt --suite libero_object `
      --n_rollouts 10 --execute 8 --perturb $k --severity $s `
      --log_jsonl $jf --tag robns > "c:\sarvik\fno_backup\logs\_robns_${k}_$s.txt" 2>&1
  Write-Host "  done $k/$s" -ForegroundColor Green
}
RunOne 'none' 0
foreach($k in 'noise','blur','brightness'){ foreach($s in 1,2,3){ RunOne $k $s } }  # informative range (quick ON/OFF read); s4-5 -> 0 for both
Write-Host '=== no-scatter robustness sweep complete (s1-3) ===' -ForegroundColor Green
