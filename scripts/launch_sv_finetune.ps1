# Stage 2 only: finetune _sv on libero_object from the ep26 pretrain best.pt.
# (Pretrain was stopped early -- it had overfit since ep26, best.pt frozen there.)
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$code = 'c:\sarvik\fno_backup\code'
$logd = 'c:\sarvik\fno_backup\logs'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:NUMBA_DISABLE_JIT    = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location $code
$preCkpt = 'E:/fno_data/run_dinov3_attnpool_sv_pretrain/best.pt'

$host.UI.RawUI.WindowTitle = 'FNO _sv FINETUNE (object) from ep26 pretrain'
Write-Host "=== FINETUNE _sv on libero_object (100 ep, ckpt every 5), resume $preCkpt ===" -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_attnpool_sv.yaml `
      --data_dir E:/fno_data --resume $preCkpt > "$logd\_train_sv_finetune.txt" 2>&1
Write-Host "=== finetune finished. Eval run_dinov3_attnpool_sv vs 63.5% baseline. ===" -ForegroundColor Green
Read-Host 'Press Enter to close'
