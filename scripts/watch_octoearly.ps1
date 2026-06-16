$f = "D:\eroot\fno_data\octo_early.jsonl"
$log = "C:\sarvik\fno_backup\logs\_octo_early.txt"
$host.UI.RawUI.WindowTitle = "OCTO 20k checkpoint - early read"
while ($true) {
  Clear-Host
  Write-Host "=== OCTO-Small @20k step - early read (3 tasks x 3)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $f)) {
    Write-Host "`n  loading model + env (slow under fine-tune contention)..." -ForegroundColor DarkGray
    $tl = Get-Content $log -EA SilentlyContinue | Where-Object {$_ -match 'loading|task|fail|OK|libero|Construct'} | Select-Object -Last 2
    $tl | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    Start-Sleep 4; continue
  }
  $rows = Get-Content $f -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "`n  waiting for first rollout..."; Start-Sleep 4; continue }
  Write-Host ("`n  rollouts: {0}/9`n" -f @($rows).Count)
  $rows | ForEach-Object {
    $t = ($_.task -replace 'pick up the ','' -replace ' and place it in the basket','')
    if ($_.success -eq 1){ Write-Host ("  {0,-16} r{1}  SUCCESS  ({2} steps)" -f $t,$_.rollout,$_.steps) -ForegroundColor Green }
    else { Write-Host ("  {0,-16} r{1}  fail     ({2} steps)" -f $t,$_.rollout,$_.steps) -ForegroundColor Red }
  }
  Write-Host "`n  --- per-task ---" -ForegroundColor Yellow
  $rows | Group-Object task_idx | Sort-Object {[int]$_.Name} | ForEach-Object {
    $t = ($_.Group[0].task -replace 'pick up the ','' -replace ' and place it in the basket','')
    "  {0,-16} {1:N0}%" -f $t, (100*(($_.Group|Measure-Object success -Average).Average))
  }
  "  {0,-16} {1:N1}%" -f "AVG(3 tasks)", (100*(($rows|Measure-Object success -Average).Average))
  Start-Sleep 4
}
