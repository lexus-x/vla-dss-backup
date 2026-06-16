# Live monitor for the robustness sweep: refreshes the CSV + shows degradation curves.
$py = 'C:\Users\islab\anaconda3\envs\mmdetection\python.exe'
$csv = 'c:\sarvik\fno_backup\robustness_summary.csv'
$host.UI.RawUI.WindowTitle = 'FNO robustness sweep - live curves'
while ($true) {
  Clear-Host
  Write-Host "=== FNO-VLA ROBUSTNESS (baseline, N=10/config)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  scattering-stability claim: graceful degradation vs perturbation severity`n" -ForegroundColor DarkGray
  # refresh CSV from jsonl
  & $py 'c:\sarvik\fno_backup\code\scripts\export_robustness_csv.py' *> $null
  if (-not (Test-Path $csv)) { Write-Host "  waiting for first results..."; Start-Sleep 8; continue }
  $rows = Import-Csv $csv
  foreach ($k in 'noise','blur','brightness') {
    $cur = $rows | Where-Object { $_.perturb -eq $k } | Sort-Object {[int]$_.severity}
    if ($cur) {
      $line = ($cur | ForEach-Object { "s$($_.severity):$($_.success_pct)%" }) -join "  "
      Write-Host ("  {0,-11} {1}" -f $k, $line) -ForegroundColor Green
    } else {
      Write-Host ("  {0,-11} (pending)" -f $k) -ForegroundColor DarkGray
    }
  }
  # progress: which config running
  $jf = "E:\fno_data\robustness.jsonl"
  $tot = if (Test-Path $jf) { @(Get-Content $jf -EA SilentlyContinue | Where-Object {$_}).Count } else { 0 }
  Write-Host ("`n  total rollouts logged: {0} / ~1600 (16 configs x 100)" -f $tot) -ForegroundColor Yellow
  Write-Host ("  CSV: {0}  (refreshed each tick)" -f $csv) -ForegroundColor Cyan
  Write-Host "`n  (refresh 12s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 12
}
