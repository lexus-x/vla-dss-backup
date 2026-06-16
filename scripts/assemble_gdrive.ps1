# Assemble gdrive_sarvik = everything precious EXCEPT the dataset + regenerable env/caches.
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
$dst = 'D:\gdrive_sarvik'
$rc = '/E','/NFL','/NDL','/NJH','/NJS','/R:1','/W:1'
New-Item -ItemType Directory -Force "$dst\code","$dst\checkpoints","$dst\eval_videos","$dst\results","$dst\env","$dst\LIBERO_source","$dst\memory","$dst\logs" | Out-Null

# code (exclude pycache)
robocopy 'c:\sarvik\fno_backup\code' "$dst\code" @rc /XD __pycache__ | Out-Null
# trained checkpoints: all run_* on E: + the two C: checkpoint dirs
Get-ChildItem 'E:\fno_data' -Directory | Where-Object{$_.Name -like 'run_*'} | ForEach-Object { robocopy $_.FullName "$dst\checkpoints\$($_.Name)" @rc | Out-Null }
if(Test-Path 'c:\sarvik\fno_backup\checkpoints'){ robocopy 'c:\sarvik\fno_backup\checkpoints' "$dst\checkpoints\_c_checkpoints" @rc | Out-Null }
if(Test-Path 'c:\sarvik\fno_backup\checkpoints_extra'){ robocopy 'c:\sarvik\fno_backup\checkpoints_extra' "$dst\checkpoints\_c_checkpoints_extra" @rc | Out-Null }
# demo videos (E: + C:)
Get-ChildItem 'E:\fno_data' -Directory | Where-Object{$_.Name -like 'eval_videos*'} | ForEach-Object { robocopy $_.FullName "$dst\eval_videos\$($_.Name)" @rc | Out-Null }
foreach($d in 'eval_videos','eval_videos_extra'){ if(Test-Path "c:\sarvik\fno_backup\$d"){ robocopy "c:\sarvik\fno_backup\$d" "$dst\eval_videos\_c_$d" @rc | Out-Null } }
# results / figures / docs
foreach($d in 'results_json','figures','ppt_figures','showcase'){ if(Test-Path "c:\sarvik\fno_backup\$d"){ robocopy "c:\sarvik\fno_backup\$d" "$dst\results\$d" @rc | Out-Null } }
Copy-Item 'c:\sarvik\fno_backup\*.csv','c:\sarvik\fno_backup\*.png','c:\sarvik\fno_backup\*.md','c:\sarvik\fno_backup\*.txt' "$dst\results\" -Force -EA SilentlyContinue
Copy-Item 'E:\fno_data\robustness_ablation.csv' "$dst\results\" -Force -EA SilentlyContinue
# env setup, LIBERO benchmark code, logs
robocopy 'c:\sarvik\fno_backup\env' "$dst\env" @rc | Out-Null
robocopy 'c:\sarvik\fno_backup\LIBERO_source' "$dst\LIBERO_source" @rc /XD __pycache__ | Out-Null
robocopy 'c:\sarvik\fno_backup\logs' "$dst\logs" @rc | Out-Null
# memory (project claude_memory + the auto-memory store)
if(Test-Path 'c:\sarvik\fno_backup\claude_memory'){ robocopy 'c:\sarvik\fno_backup\claude_memory' "$dst\memory" @rc | Out-Null }
robocopy 'C:\Users\islab\.claude\projects\c--sarvik-fno-backup\memory' "$dst\memory\claude_auto_memory" @rc | Out-Null

$gb=[math]::Round((Get-ChildItem $dst -Recurse -File -EA SilentlyContinue|Measure-Object Length -Sum).Sum/1GB,2)
$n=(Get-ChildItem $dst -Recurse -File -EA SilentlyContinue).Count
Write-Host ("=== gdrive_sarvik assembled: {0} GB, {1} files ===" -f $gb,$n) -ForegroundColor Green
$global:LASTEXITCODE = 0
