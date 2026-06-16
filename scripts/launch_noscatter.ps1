# Scattering-OFF ablation: pretrain (5 suites) -> finetune (object), DINOv3-only.
# The fair robustness control (same data/scale, only scattering removed).
# NOTE: no ErrorActionPreference='Stop' (PyTorch UserWarning would abort it); use > log 2>&1.
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$logd = 'c:\sarvik\fno_backup\logs'
$env:CUDA_VISIBLE_DEVICES = '0'; $env:NUMBA_DISABLE_JIT = '1'; $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location 'c:\sarvik\fno_backup\code'
$preCkpt = 'E:/fno_data/run_dinov3_noscatter_pretrain/best.pt'
$host.UI.RawUI.WindowTitle = 'NO-SCATTER ablation: pretrain -> finetune'

Write-Host '=== STAGE 1/2: PRETRAIN no-scatter (5 suites) ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/pretrain_dinov3_noscatter.yaml `
      --data_dir E:/fno_data > "$logd\_train_noscatter_pretrain.txt" 2>&1

if (-not (Test-Path $preCkpt)) {
  Write-Host "PRETRAIN produced no best.pt at $preCkpt -- stopping." -ForegroundColor Red
  Read-Host 'Enter to close'; exit 1
}
Write-Host "`n=== STAGE 2/2: FINETUNE no-scatter (object), resume $preCkpt ===" -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_noscatter.yaml `
      --data_dir E:/fno_data --resume $preCkpt > "$logd\_train_noscatter_finetune.txt" 2>&1
Write-Host "`n=== no-scatter training done. Next: robustness sweep on this ckpt. ===" -ForegroundColor Green
Read-Host 'Enter to close'
