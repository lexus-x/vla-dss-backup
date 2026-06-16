# Waits for the Octo full eval to finish (200 rollouts) AND the GPU to free,
# then auto-launches the _sv pretrain->finetune pipeline. Poll every 60s.
$octo   = 'D:\eroot\fno_data\octo_results.jsonl'
$target = 200            # 10 tasks x 20 rollouts
$launch = 'c:\sarvik\fno_backup\code\scripts\launch_sv.ps1'
$host.UI.RawUI.WindowTitle = 'QUEUE: waiting for Octo eval -> _sv pipeline'

function GpuMemUsedMiB {
  try { [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n")[0].Trim() }
  catch { -1 }
}

$lowStreak = 0
while ($true) {
  Clear-Host
  $n = 0
  if (Test-Path $octo) { $n = @(Get-Content $octo -EA SilentlyContinue | Where-Object {$_}).Count }
  $mem = GpuMemUsedMiB
  Write-Host ("=== waiting for Octo eval  $(Get-Date -Format 'HH:mm:ss') ===") -ForegroundColor Cyan
  Write-Host ("  octo rollouts : {0}/{1}" -f $n,$target)
  Write-Host ("  GPU mem used  : {0} MiB" -f $mem)

  # Done condition: all rollouts logged, OR GPU has been idle (<6GB) for 3 polls
  if ($mem -ge 0 -and $mem -lt 6000) { $lowStreak++ } else { $lowStreak = 0 }
  $rolloutsDone = ($n -ge $target)
  $gpuFreed     = ($lowStreak -ge 3)

  if ($rolloutsDone -or $gpuFreed) {
    $why = if ($rolloutsDone) { "all $target rollouts logged" } else { "GPU idle 3x (<6GB)" }
    Write-Host ("`n  -> Octo done ({0}). Launching _sv pipeline..." -f $why) -ForegroundColor Green
    Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File',$launch
    Write-Host '  launched. This window can be closed.' -ForegroundColor Green
    break
  }
  Write-Host "`n  (still running -- checking again in 60s)" -ForegroundColor DarkGray
  Start-Sleep 60
}
