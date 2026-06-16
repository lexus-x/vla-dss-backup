# Multi-suite + aux: finetune FNO-VLA (aux x-y) on LIBERO-Long (libero_10) from cross-suite pretrain.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'TRAIN long+aux'
Write-Host '=== finetune aux-head model on LIBERO-Long (libero_10), resume cross-suite pretrain ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_long_aux.yaml `
    --data_dir E:/fno_data `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "c:\sarvik\fno_backup\logs\_train_long_aux.txt" 2>&1
Write-Host '=== long+aux training done -> eval run_dinov3_long_aux ladder ===' -ForegroundColor Green
