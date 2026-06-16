# Ablation contrast: NO-SCATTERING model FAILS under severity-3 noise (cream cheese).
# Pairs with the scattering model surviving noise. Video shows the noisy input.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'noscatter noise video'
$ck = 'E:/fno_data/run_dinov3_noscatter_finetune/best.pt'
# cream cheese (task 1) + salad (task 2, a task scattering passes) under noise-3, no-scattering model
& $py -u scripts/eval_sim.py --checkpoint $ck `
    --suite libero_object --n_rollouts 3 --execute 8 --task_indices 1,2 `
    --perturb noise --severity 3 `
    --save_videos --video_dir "E:/fno_data/eval_videos_noscatter_noise/noise3" --videos_per_task 3 `
    --log_jsonl "E:/fno_data/znoscatter_noise3.jsonl" --tag noscat_n3 `
    > "c:\sarvik\fno_backup\logs\_noscatter_noise3.txt" 2>&1
Write-Host '=== noscatter noise-3 video done ===' -ForegroundColor Green
