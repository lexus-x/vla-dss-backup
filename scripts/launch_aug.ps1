# STEP 1: aug-only (aux + DAgger data + corruption aug, NO gate) from cross-suite pretrain.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'TRAIN aug-only (step1)'
Write-Host '=== STEP 1: aug-only (no gate), DAgger data, from pretrain ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_aug.yaml `
    --data_dir E:/fno_data/libero_object_dagger `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "c:\sarvik\fno_backup\logs\_train_aug.txt" 2>&1
Write-Host '=== aug-only training done ===' -ForegroundColor Green
