$f = "D:\eroot\fno_data\octo_results.jsonl"
$host.UI.RawUI.WindowTitle = "OCTO-Small per-task success"
while ($true) {
  Clear-Host
  if (-not (Test-Path $f)) { Write-Host "no results yet"; Start-Sleep 5; continue }
  $rows = Get-Content $f -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
  if (-not $rows) { Write-Host "no results yet"; Start-Sleep 5; continue }
  $rows | Group-Object task_idx | Sort-Object {[int]$_.Name} | ForEach-Object {
    $t = ($_.Group[0].task -replace 'pick up the ','' -replace ' and place it in the basket','')
    "{0,-16} {1,4:N0}%" -f $t, (100*(($_.Group | Measure-Object success -Average).Average))
  }
  "{0,-16} {1,4:N1}%" -f "AVERAGE", (100*(($rows | Measure-Object success -Average).Average))
  Start-Sleep 5
}
