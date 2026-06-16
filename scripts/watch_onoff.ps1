# Live monitor: scattering ON (FNO, done) vs OFF (no-scatter, filling in) robustness.
$jf = "E:\fno_data\robustness_noscatter.jsonl"
$log = "c:\sarvik\fno_backup\logs\_robns_none_0.txt"
$host.UI.RawUI.WindowTitle = "scattering ON vs OFF robustness"
# scatter-ON reference (FNO sweep, completed). severity 0 = clean = 61.
$ON = @{
  clean=61
  noise      = @{1=61;2=62;3=54}
  blur       = @{1=51;2=47;3=38}
  brightness = @{1=60;2=57;3=58}
}
while ($true) {
  Clear-Host
  Write-Host "=== Scattering ON (FNO) vs OFF (no-scatter)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  delta = ON - OFF;  POSITIVE = scattering HELPS robustness (your claim)`n" -ForegroundColor DarkGray
  if (-not (Test-Path $jf)) { Write-Host "  loading no-scatter model..." -ForegroundColor DarkGray; Get-Content $log -Tail 2 -EA SilentlyContinue | ForEach-Object {Write-Host "  $_" -ForegroundColor DarkGray}; Start-Sleep 6; continue }
  $rows = Get-Content $jf -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "  waiting for first rollout..."; Start-Sleep 6; continue }
  Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f "perturb","sev","OFF%","ON%","delta") -ForegroundColor Yellow
  $cl = $rows | Where-Object { $_.perturb -eq 'none' }
  if ($cl) { $n=@($cl).Count; $sr=[math]::Round(100*(($cl|Measure-Object success -Average).Average)); $d=$ON.clean-$sr; Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f "clean",0,"$sr% ($n)","$($ON.clean)","$(if($d -ge 0){'+'})$d") }
  foreach ($k in 'noise','blur','brightness') {
    foreach ($s in 1,2,3) {
      $g = $rows | Where-Object { $_.perturb -eq $k -and $_.severity -eq $s }
      if ($g) {
        $n=@($g).Count; $off=[math]::Round(100*(($g|Measure-Object success -Average).Average)); $onv=$ON[$k][$s]; $d=$onv-$off
        $col = if ($d -gt 0) {"Green"} elseif ($d -lt 0) {"Red"} else {"Gray"}  # ON more robust = green = scattering helps
        Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f $k,$s,"$off% ($n)","$onv%","$(if($d -ge 0){'+'})$d") -ForegroundColor $col
      } else {
        Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9}" -f $k,$s,"--","$($ON[$k][$s])%") -ForegroundColor DarkGray
      }
    }
  }
  Write-Host ("`n  no-scatter rollouts: {0} / ~1000 (10 configs x 100)" -f @($rows).Count)
  Write-Host "  GREEN = scattering ON more robust = scattering HELPS = claim survives." -ForegroundColor DarkGray
  Write-Host "`n  (refresh 8s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 8
}
