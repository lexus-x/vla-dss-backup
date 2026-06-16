# Live monitor: Octo robustness sweep vs FNO-VLA (the "more robust than competitor" check).
$jf = "D:\eroot\fno_data\octo_robustness.jsonl"
$log = "c:\sarvik\fno_backup\logs\_octorob_none_0.txt"
$host.UI.RawUI.WindowTitle = "OCTO robustness vs FNO"
# FNO reference (from completed FNO sweep), severity 0 = clean = 61
$FNO = @{
  noise      = @{0=61;1=61;2=62;3=54}
  blur       = @{0=61;1=51;2=47;3=38}
  brightness = @{0=61;1=60;2=57;3=58}
}
while ($true) {
  Clear-Host
  Write-Host "=== OCTO robustness  vs  FNO-VLA   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  (delta = Octo - FNO; NEGATIVE = Octo worse = FNO MORE robust = your claim)`n" -ForegroundColor DarkGray
  if (-not (Test-Path $jf)) { Write-Host "  loading Octo (T5+diffusion)..." -ForegroundColor DarkGray; Get-Content $log -Tail 2 -EA SilentlyContinue | ForEach-Object {Write-Host "  $_" -ForegroundColor DarkGray}; Start-Sleep 6; continue }
  $rows = Get-Content $jf -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "  waiting for first rollout..."; Start-Sleep 6; continue }
  Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f "perturb","sev","octo%","fno%","delta") -ForegroundColor Yellow
  # clean ref (none,0)
  $cl = $rows | Where-Object { $_.perturb -eq 'none' }
  if ($cl) { $n=@($cl).Count; $sr=[math]::Round(100*(($cl|Measure-Object success -Average).Average)); Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f "clean",0,"$sr% ($n)","61","$($sr-61)") }
  foreach ($k in 'noise','blur','brightness') {
    foreach ($s in 1,2,3) {
      $g = $rows | Where-Object { $_.perturb -eq $k -and $_.severity -eq $s }
      if ($g) {
        $n=@($g).Count; $sr=[math]::Round(100*(($g|Measure-Object success -Average).Average))
        $fno=$FNO[$k][$s]; $d=$sr-$fno
        $col = if ($d -lt 0) {"Green"} elseif ($d -gt 0) {"Red"} else {"Gray"}  # octo worse = green (good for us)
        Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9} {4}" -f $k,$s,"$sr% ($n)","$fno%",("{0}{1}" -f $(if($d -ge 0){'+'}),$d)) -ForegroundColor $col
      } else {
        Write-Host ("  {0,-12} {1,-4} {2,-12} {3,-9}" -f $k,$s,"--","$($FNO[$k][$s])%") -ForegroundColor DarkGray
      }
    }
  }
  $tot=@($rows).Count
  Write-Host ("`n  octo rollouts: {0} / ~500 (10 configs x 50)" -f $tot)
  Write-Host "  GREEN rows = Octo degraded MORE than FNO = FNO more robust there." -ForegroundColor DarkGray
  Write-Host "`n  (refresh 8s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 8
}
