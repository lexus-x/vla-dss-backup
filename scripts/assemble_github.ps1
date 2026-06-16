# Assemble github_sarvik = code + env + docs + small results, GitHub-safe.
# Excludes dataset, checkpoints (>100MB, git-incompatible), videos, caches.
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
$dst = 'c:\sarvik\fno_backup\github_sarvik'
$rc = '/E','/NFL','/NDL','/NJH','/NJS','/R:1','/W:1'
New-Item -ItemType Directory -Force "$dst\src","$dst\configs","$dst\scripts","$dst\bridge_handoff","$dst\env","$dst\docs","$dst\results" | Out-Null

robocopy 'c:\sarvik\fno_backup\code\src' "$dst\src" @rc /XD __pycache__ | Out-Null
robocopy 'c:\sarvik\fno_backup\code\configs' "$dst\configs" @rc | Out-Null
robocopy 'c:\sarvik\fno_backup\code\scripts' "$dst\scripts" @rc | Out-Null
robocopy 'c:\sarvik\fno_backup\code\bridge_handoff' "$dst\bridge_handoff" @rc | Out-Null
robocopy 'c:\sarvik\fno_backup\env' "$dst\env" @rc | Out-Null
Copy-Item 'c:\sarvik\fno_backup\code\requirements.txt' "$dst\" -Force -EA SilentlyContinue
# docs (markdown)
Copy-Item 'c:\sarvik\fno_backup\*.md' "$dst\docs\" -Force -EA SilentlyContinue
Copy-Item 'c:\sarvik\fno_backup\code\*.md' "$dst\docs\" -Force -EA SilentlyContinue
# small results: csv + png + figure folders
Copy-Item 'c:\sarvik\fno_backup\*.csv','c:\sarvik\fno_backup\*.png' "$dst\results\" -Force -EA SilentlyContinue
Copy-Item 'E:\fno_data\robustness_ablation.csv' "$dst\results\" -Force -EA SilentlyContinue
foreach($d in 'figures','ppt_figures','showcase'){ if(Test-Path "c:\sarvik\fno_backup\$d"){ robocopy "c:\sarvik\fno_backup\$d" "$dst\results\$d" @rc | Out-Null } }

# SAFETY: strip anything GitHub can't take (big binaries / weights / data / video)
Get-ChildItem $dst -Recurse -Include *.pt,*.pth,*.ckpt,*.hdf5,*.h5,*.mp4,*.npz,*.npy -EA SilentlyContinue | Remove-Item -Force -EA SilentlyContinue
$big = Get-ChildItem $dst -Recurse -File -EA SilentlyContinue | Where-Object{$_.Length -gt 100MB}
$mb = [math]::Round((Get-ChildItem $dst -Recurse -File -EA SilentlyContinue|Measure-Object Length -Sum).Sum/1MB,1)
Write-Host ("=== github_sarvik = {0} MB, {1} files ===" -f $mb, (Get-ChildItem $dst -Recurse -File).Count) -ForegroundColor Green
Write-Host ("files >100MB (GitHub blocks these): {0}" -f $big.Count)
if($big){ $big | ForEach-Object{ Write-Host ("  BIG: {0} ({1} MB)" -f $_.Name,[math]::Round($_.Length/1MB)) } }
$global:LASTEXITCODE = 0
