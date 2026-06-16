# Live monitor for the Octo jerk eval (WSL) vs FNO-VLA jerk reference.
$jf = "D:\eroot\fno_data\octo_jerk.jsonl"
$log = "c:\sarvik\fno_backup\logs\_octo_jerk.txt"
$host.UI.RawUI.WindowTitle = "OCTO JERK eval - vs FNO 0.026"
$FNO_SV = 0.0256; $FNO_BASE = 0.0276
while ($true) {
  Clear-Host
  Write-Host "=== OCTO-Small JERK vs FNO-VLA   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  identical formula: mean |2nd diff| of executed 6D commands`n" -ForegroundColor DarkGray
  if (-not (Test-Path $jf)) { Write-Host "  loading Octo model (T5 + diffusion, slow)..." -ForegroundColor DarkGray; Get-Content $log -Tail 2 -EA SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }; Start-Sleep 6; continue }
  $r = Get-Content $jf -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $r) { Write-Host "  waiting for first rollout..."; Start-Sleep 6; continue }
  $n = @($r).Count
  $jvals = $r | Where-Object { $_.jerk -ne $null -and -not [double]::IsNaN([double]$_.jerk) } | ForEach-Object { [double]$_.jerk }
  $jmean = if ($jvals) { [math]::Round(($jvals | Measure-Object -Average).Average, 4) } else { 0 }
  $sr = [math]::Round(100*(($r | Measure-Object success -Average).Average), 0)
  Write-Host ("  rollouts: {0}/50   success {1}%`n" -f $n, $sr)
  Write-Host "  -- JERK comparison --" -ForegroundColor Yellow
  Write-Host ("    Octo-Small (running)  {0}" -f $jmean) -ForegroundColor $(if($jmean -gt $FNO_SV){"Red"}else{"Green"})
  Write-Host ("    FNO-VLA _sv           {0}" -f $FNO_SV) -ForegroundColor Green
  Write-Host ("    FNO-VLA baseline      {0}" -f $FNO_BASE) -ForegroundColor Green
  if ($jmean -gt 0) {
    $ratio = [math]::Round($jmean / $FNO_SV, 2)
    $pct = [math]::Round(100*($jmean - $FNO_SV)/$jmean, 0)
    Write-Host ("`n    -> Octo is {0}x jerkier ({1}% rougher). FNO smoother = claim holds." -f $ratio,$pct) -ForegroundColor Cyan
  }
  Write-Host "`n  -- recent rollouts --" -ForegroundColor Yellow
  $r | Select-Object -Last 6 | ForEach-Object {
    $t = ($_.task -replace 'pick up the ','' -replace ' and place it in the basket','')
    "    {0,-16} {1}  jerk {2}" -f $t, $(if($_.success){"OK  "}else{"fail"}), [math]::Round([double]$_.jerk,4)
  }
  Write-Host "`n  (refresh 6s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 6
}
