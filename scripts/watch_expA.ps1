$j = "E:\fno_data\zerror_expA.jsonl"
$host.UI.RawUI.WindowTitle = "Exp A live - per-rollout SUCCESS/FAIL"
while ($true) {
  Clear-Host
  Write-Host "=== Exp A live (execute=4 vs 8)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $j)) { Write-Host "waiting for first record..."; Start-Sleep 3; continue }
  $rows = Get-Content $j -ErrorAction SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "waiting..."; Start-Sleep 3; continue }

  Write-Host ("total: {0}/400 rollouts (200 per arm)`n" -f @($rows).Count)

  # per-arm rollup
  $rows | Group-Object tag | Sort-Object Name | ForEach-Object {
    "{0,-6} n={1,-4} success={2,5:P0}  meanJerk={3:N4}  meanGraspZ={4:N4}m" -f `
      $_.Name, $_.Count, (($_.Group | Measure-Object success -Average).Average),
      (($_.Group | Measure-Object jerk -Average).Average),
      (($_.Group | Where-Object {$null -ne $_.grasp_z_gap_min} | Measure-Object grasp_z_gap_min -Average).Average)
  }

  # per-task tally
  Write-Host "`n--- per-task (sum/n by arm) ---" -ForegroundColor Yellow
  $rows | Group-Object task_idx | Sort-Object {[int]$_.Name} | ForEach-Object {
    $t = ($_.Group[0].task -replace 'pick up the ','' -replace ' and place it in the basket','')
    $byTag = $_.Group | Group-Object tag | Sort-Object Name | ForEach-Object {
      "{0}:{1}/{2}" -f $_.Name, (($_.Group | Measure-Object success -Sum).Sum), $_.Count }
    "  [{0,2}] {1,-16} {2}" -f $_.Name, $t, ($byTag -join '  ')
  }

  # live per-rollout feed (newest 22)
  Write-Host "`n--- recent rollouts (newest at bottom) ---" -ForegroundColor Yellow
  $rows | Select-Object -Last 22 | ForEach-Object {
    $t = ($_.task -replace 'pick up the ','' -replace ' and place it in the basket','')
    $z = if ($null -ne $_.grasp_z_gap_min) { "{0:N4}" -f $_.grasp_z_gap_min } else { "  -  " }
    if ($_.success -eq 1) { $res='SUCCESS'; $col='Green' } else { $res='FAIL'; $col='Red' }
    Write-Host ("  {0,-6} [{1,2}] {2,-16} r{3,2}  {4,-7}  z={5}  jerk={6:N4}" -f `
      $_.tag, $_.task_idx, $t, $_.rollout, $res, $z, $_.jerk) -ForegroundColor $col
  }
  Write-Host "`n(refreshes every 3s - close window to stop)" -ForegroundColor DarkGray
  Start-Sleep 3
}
