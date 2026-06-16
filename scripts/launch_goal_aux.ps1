# Multi-suite + aux: finetune FNO-VLA (aux x-y) on LIBERO-Goal from cross-suite pretrain.
# max_epochs=30 (success peaks early; no point overtraining to 100 like spatial did).
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'TRAIN goal+aux'
Write-Host '=== finetune aux-head model on LIBERO-Goal (resume cross-suite pretrain) ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_goal_aux.yaml `
    --data_dir E:/fno_data `
    --resume E:/fno_data/run_dinov3_pretrain/best.pt `
    > "c:\sarvik\fno_backup\logs\_train_goal_aux.txt" 2>&1
Write-Host '=== goal+aux training done -> eval run_dinov3_goal_aux ladder ===' -ForegroundColor Green
