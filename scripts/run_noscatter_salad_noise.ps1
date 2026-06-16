# NO-scattering model on SALAD under severity-3 noise, many rollouts to catch a FAIL.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'noscatter salad noise'
& $py -u scripts/eval_sim.py --checkpoint 'E:/fno_data/run_dinov3_noscatter_finetune/best.pt' `
    --suite libero_object --n_rollouts 8 --execute 8 --task_indices 2 `
    --perturb noise --severity 3 `
    --save_videos --video_dir "E:/fno_data/eval_videos_noscatter_noise/salad_n3" --videos_per_task 8 `
    --log_jsonl "E:/fno_data/znoscatter_salad_n3.jsonl" --tag noscat_salad `
    > "c:\sarvik\fno_backup\logs\_noscatter_salad.txt" 2>&1
Write-Host '=== noscatter salad noise-3 done ===' -ForegroundColor Green
