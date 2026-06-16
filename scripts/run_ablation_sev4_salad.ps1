# Fair ablation at the SAME severity-4 noise, salad: scattering ON vs OFF (models differ ONLY in scattering).
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$env:MUJOCO_GL='glfw'; $env:NUMBA_DISABLE_JIT='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:CUDA_VISIBLE_DEVICES='0'
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
Set-Location 'c:\sarvik\fno_backup\code'
$host.UI.RawUI.WindowTitle = 'ablation sev4 salad'
# scattering ON (base scattering model)
& $py -u scripts/eval_sim.py --checkpoint 'E:/fno_data/run_dinov3_finetune/best.pt' `
    --suite libero_object --n_rollouts 5 --execute 8 --task_indices 2 --perturb noise --severity 4 `
    --save_videos --video_dir "E:/fno_data/eval_videos_ablation_sev4/scatON" --videos_per_task 5 `
    --log_jsonl "E:/fno_data/zabl_sev4_on.jsonl" --tag ablON `
    > "c:\sarvik\fno_backup\logs\_abl_sev4_on.txt" 2>&1
# scattering OFF (noscatter model)
& $py -u scripts/eval_sim.py --checkpoint 'E:/fno_data/run_dinov3_noscatter_finetune/best.pt' `
    --suite libero_object --n_rollouts 5 --execute 8 --task_indices 2 --perturb noise --severity 4 `
    --save_videos --video_dir "E:/fno_data/eval_videos_ablation_sev4/scatOFF" --videos_per_task 5 `
    --log_jsonl "E:/fno_data/zabl_sev4_off.jsonl" --tag ablOFF `
    > "c:\sarvik\fno_backup\logs\_abl_sev4_off.txt" 2>&1
Write-Host '=== sev4 salad ablation done ===' -ForegroundColor Green
