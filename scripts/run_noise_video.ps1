# Noise-robustness DEMO video: policy operating under severity-3 Gaussian noise.
# Video now shows the PERTURBED input (what the model sees). Easy tasks -> successes under noise.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'noise demo video'
# clean reference (severity 0)
& $py -u scripts/eval_sim.py --checkpoint E:/fno_data/run_dinov3_dagger/epoch_5.pt `
    --suite libero_object --n_rollouts 2 --execute 8 --task_indices 0,2,6 `
    --perturb noise --severity 0 `
    --save_videos --video_dir "E:/fno_data/eval_videos_noise/clean" --videos_per_task 2 `
    --log_jsonl "E:/fno_data/znoise_clean.jsonl" --tag noiseclean `
    > "c:\sarvik\fno_backup\logs\_noise_clean.txt" 2>&1
# severity-3 noise
& $py -u scripts/eval_sim.py --checkpoint E:/fno_data/run_dinov3_dagger/epoch_5.pt `
    --suite libero_object --n_rollouts 2 --execute 8 --task_indices 0,2,6 `
    --perturb noise --severity 3 `
    --save_videos --video_dir "E:/fno_data/eval_videos_noise/noise3" --videos_per_task 2 `
    --log_jsonl "E:/fno_data/znoise_3.jsonl" --tag noise3 `
    > "c:\sarvik\fno_backup\logs\_noise_3.txt" 2>&1
Write-Host '=== noise demo videos done ===' -ForegroundColor Green
