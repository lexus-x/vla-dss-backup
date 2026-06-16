# Full robustness battery on BOTH ablation models -> attribution table.
#   Step 0 = DAgger (no aug)  tag=dag   |  Step 1 = aug ep15  tag=aug
# Each: clean + noise/blur/brightness x sev 1,2,3  (N=5/task).
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'robustness batteries (dag + aug)'
$models = @(@{ck='E:/fno_data/run_dinov3_dagger/epoch_5.pt'; t='dag'},
            @{ck='E:/fno_data/run_dinov3_aug/epoch_15.pt';   t='aug'})
foreach($m in $models){
  Write-Host "=== battery $($m.t) : clean ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zbat_$($m.t)_clean.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint $m.ck --suite libero_object --n_rollouts 5 --execute 8 `
      --perturb none --severity 0 --log_jsonl "E:/fno_data/zbat_$($m.t)_clean.jsonl" --tag "$($m.t)_clean" `
      > "c:\sarvik\fno_backup\logs\_bat_$($m.t)_clean.txt" 2>&1
  foreach($kind in 'noise','blur','brightness'){
    foreach($sev in 1,2,3){
      Remove-Item "E:\fno_data\zbat_$($m.t)_$kind$sev.jsonl" -Force -EA SilentlyContinue
      & $py -u scripts/eval_sim.py --checkpoint $m.ck --suite libero_object --n_rollouts 5 --execute 8 `
          --perturb $kind --severity $sev --log_jsonl "E:/fno_data/zbat_$($m.t)_$kind$sev.jsonl" --tag "$($m.t)_$kind$sev" `
          > "c:\sarvik\fno_backup\logs\_bat_$($m.t)_$kind$sev.txt" 2>&1
      Write-Host "  $($m.t) $kind sev$sev done" -ForegroundColor Green
    }
  }
}
Write-Host '=== all batteries complete ===' -ForegroundColor Green
