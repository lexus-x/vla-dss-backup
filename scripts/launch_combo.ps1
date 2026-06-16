# COMBO best-model training: gated fusion + aux head + corruption aug + DAgger data.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'TRAIN combo (gate+aug+dagger)'
Write-Host '=== COMBO: gated fusion + aux + corruption-aug, on DAgger data, from cross-suite pretrain ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_combo.yaml `
    --data_dir E:/fno_data/libero_object_dagger `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "c:\sarvik\fno_backup\logs\_train_combo.txt" 2>&1
Write-Host '=== combo training done -> eval run_dinov3_combo (clean + robustness) ===' -ForegroundColor Green
