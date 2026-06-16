# Resume the _sv finetune after the GPU-contention crash, from the ep5 checkpoint.
# NEVER run eval on this GPU while this is training -- it deadlocks/crashes the run.
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$code = 'c:\sarvik\fno_backup\code'
$logd = 'c:\sarvik\fno_backup\logs'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:NUMBA_DISABLE_JIT    = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location $code
$resume = 'E:/fno_data/run_dinov3_attnpool_sv/best.pt'

$host.UI.RawUI.WindowTitle = 'FNO _sv FINETUNE (resumed from ep5)'
Write-Host "=== RESUME _sv finetune from $resume (GPU alone - no eval!) ===" -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_attnpool_sv.yaml `
      --data_dir E:/fno_data --resume $resume > "$logd\_train_sv_finetune2.txt" 2>&1
Write-Host "=== finetune finished. ===" -ForegroundColor Green
Read-Host 'Press Enter to close'
