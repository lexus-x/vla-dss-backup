$host.UI.RawUI.WindowTitle = 'FNO FINETUNE 100ep (GPU1) - tqdm live'
$env:CUDA_VISIBLE_DEVICES = '1'
Set-Location 'c:\Users\DELL\Desktop\fno'
Write-Host '=== FNO-VLA fine-tune on libero_object: 100 epochs, lr 1e-4, resume from best.pt ep24 ===' -ForegroundColor Cyan
Write-Host '=== GPU 1 | tqdm live | Ctrl+C to stop ===' -ForegroundColor Cyan
Write-Host ''
& 'C:\Users\DELL\anaconda3\envs\mmdetection\python.exe' -u src/train.py --config configs/finetune_dinov3.yaml --data_dir E:/fno_data --resume E:/fno_data/run_dinov3_pretrain/best.pt
Write-Host ''
Write-Host '=== Training finished. Window stays open. ===' -ForegroundColor Green
