# Full _sv pipeline on all 130 tasks: pretrain (5 suites) -> finetune (object).
# separate_views + proprio-gated attention-pool. Single GPU (RTX A6000, index 0).
# NOTE: do NOT set ErrorActionPreference='Stop' -- PyTorch prints a UserWarning to
# stderr during build, which PS wraps as NativeCommandError and would abort the run.
# Use cmd-style '> log 2>&1' redirection (process-level) instead of Tee + 2>&1.
$py   = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$code = 'c:\sarvik\fno_backup\code'
$logd = 'c:\sarvik\fno_backup\logs'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:NUMBA_DISABLE_JIT    = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location $code

$preCkpt = 'E:/fno_data/run_dinov3_attnpool_sv_pretrain/best.pt'

$host.UI.RawUI.WindowTitle = 'FNO _sv PRETRAIN (5 suites) -> FINETUNE (object)'
Write-Host '=== STAGE 1/2: PRETRAIN _sv on all 5 suites (80 ep, ~44h) ===' -ForegroundColor Cyan
& $py -u src/train.py --config configs/pretrain_dinov3_attnpool_sv.yaml `
      --data_dir E:/fno_data > "$logd\_train_sv_pretrain.txt" 2>&1

if (-not (Test-Path $preCkpt)) {
  Write-Host "PRETRAIN did not produce $preCkpt -- stopping (not starting finetune)." -ForegroundColor Red
  Read-Host 'Press Enter to close'; exit 1
}

Write-Host "`n=== STAGE 2/2: FINETUNE _sv on libero_object (100 ep, ~17h), resume $preCkpt ===" -ForegroundColor Cyan
& $py -u src/train.py --config configs/finetune_dinov3_attnpool_sv.yaml `
      --data_dir E:/fno_data --resume $preCkpt > "$logd\_train_sv_finetune.txt" 2>&1

Write-Host "`n=== _sv pipeline finished. Eval: run_dinov3_attnpool_sv (paired vs 63.5% mean-pool). ===" -ForegroundColor Green
Read-Host 'Press Enter to close'
