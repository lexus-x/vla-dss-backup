# STAGE 2 (autonomous): waits for the dag+aug battery to finish, then:
#   (a) aug clean ep20/25/30  (clean was still rising at ep15 -> find true peak)
#   (b) Step 2 combo training (gate ON) from pretrain on DAgger data
# Stage 3 (combo floor-check + battery) is launched separately after this, since the
# combo battery checkpoint must be chosen from (a)'s-style floor-check results.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'STAGE2 (aug clean 20/25/30 + combo train)'

Write-Host '=== STAGE 2 waiting for battery (zbat_aug_brightness3 >= 45) ===' -ForegroundColor Yellow
while($true){
  $f = 'E:\fno_data\zbat_aug_brightness3.jsonl'
  $n = if(Test-Path $f){ (Get-Content $f -EA SilentlyContinue | Where-Object{$_}).Count } else { 0 }
  if($n -ge 45){ break }
  Start-Sleep 120
}
Write-Host '=== battery done -> STAGE 2 starting ===' -ForegroundColor Green

# (a) aug clean ep20/25/30
foreach($ep in 20,25,30){
  Write-Host "=== aug ep$ep clean ===" -ForegroundColor Cyan
  Remove-Item "E:\fno_data\zaugclean_$ep.jsonl" -Force -EA SilentlyContinue
  & $py -u scripts/eval_sim.py --checkpoint "E:/fno_data/run_dinov3_aug/epoch_$ep.pt" `
      --suite libero_object --n_rollouts 10 --execute 8 `
      --log_jsonl "E:/fno_data/zaugclean_$ep.jsonl" --tag "augclean$ep" `
      > "c:\sarvik\fno_backup\logs\_augclean_$ep.txt" 2>&1
  Write-Host "  done aug clean ep$ep" -ForegroundColor Green
}

# (b) Step 2 combo training (gate ON)
Write-Host '=== STEP 2 combo training (gate ON) ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_combo.yaml `
    --data_dir E:/fno_data/libero_object_dagger `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "c:\sarvik\fno_backup\logs\_train_combo.txt" 2>&1
Write-Host '=== STAGE 2 complete (combo trained) ===' -ForegroundColor Green
