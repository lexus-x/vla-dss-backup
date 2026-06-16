$d = "D:\eroot\fno_data\zdiag.jsonl"
$host.UI.RawUI.WindowTitle = "DIAG: grasp vs basket-edge"
while ($true) {
  Clear-Host
  Write-Host "=== DIAG: is failure GRASP or BASKET-EDGE?   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $d)) { Write-Host "waiting for first record (model loading ~30s)..."; Start-Sleep 3; continue }
  $rows = Get-Content $d -ErrorAction SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "waiting..."; Start-Sleep 3; continue }
  Write-Host ("records: {0}/50`n" -f @($rows).Count)

  Write-Host "--- per-task success ---" -ForegroundColor Yellow
  $rows | Group-Object task_idx | Sort-Object {[int]$_.Name} | ForEach-Object {
    $t = ($_.Group[0].task -replace 'pick up the ','' -replace ' and place it in the basket','')
    "  [{0}] {1,-14} {2}/{3}" -f $_.Name, $t, (($_.Group|Measure-Object success -Sum).Sum), $_.Group.Count
  }

  $fail = $rows | Where-Object {$_.success -eq 0}
  $succ = $rows | Where-Object {$_.success -eq 1}
  Write-Host "`n--- DECISIVE: did failures pick up + reach basket? ---" -ForegroundColor Yellow
  foreach ($g in @(@('FAIL',$fail),@('SUCCESS',$succ))) {
    $name=$g[0]; $set=$g[1]
    if ($set) {
      $lift = ($set | Where-Object {$null -ne $_.max_lift} | Measure-Object max_lift -Average).Average
      $bdxy = ($set | Where-Object {$null -ne $_.min_basket_dxy} | Measure-Object min_basket_dxy -Average).Average
      $gdxy = ($set | Where-Object {$null -ne $_.grasp_dxy} | Measure-Object grasp_dxy -Average).Average
      "  {0,-8} n={1,-3} max_lift={2:N3}m  min_basket_dxy={3:N3}m  grasp_dxy={4:N3}m" -f $name,$set.Count,$lift,$bdxy,$gdxy
    }
  }

  if ($fail) {
    $fl = ($fail | Where-Object {$null -ne $_.max_lift} | Measure-Object max_lift -Average).Average
    $fb = ($fail | Where-Object {$null -ne $_.min_basket_dxy} | Measure-Object min_basket_dxy -Average).Average
    Write-Host ""
    if ($fl -gt 0.05 -and $fb -lt 0.15) {
      Write-Host "  VERDICT leaning: PLACEMENT/EDGE - picks up, reaches basket, dies at edge (you were right)" -ForegroundColor Green
    } elseif ($fl -lt 0.03) {
      Write-Host "  VERDICT leaning: GRASP - object barely lifts; never acquired" -ForegroundColor Red
    } else {
      Write-Host "  VERDICT leaning: MIXED - see per-task" -ForegroundColor Magenta
    }
  }
  Write-Host "`n(refreshes 4s - lift>0.05 & basket_dxy<0.15 on FAILs => your edge theory)" -ForegroundColor DarkGray
  Start-Sleep 4
}
