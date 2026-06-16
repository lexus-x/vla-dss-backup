$log = "C:\sarvik\fno_backup\logs\_octo_finetune.txt"
$host.UI.RawUI.WindowTitle = "OCTO-Small finetune (object, 30k)"
function Bar($f,$w){ $fl=[int]([math]::Max(0,[math]::Min(1,$f))*$w); ("#"*$fl)+("-"*($w-$fl)) }
while ($true) {
  Clear-Host
  Write-Host "=== OCTO-Small fine-tune: object, 30k steps   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $log)) { Write-Host "waiting..."; Start-Sleep 4; continue }
  $raw = Get-Content $log -Raw -EA SilentlyContinue
  $m = [regex]::Matches($raw, '(\d+)/30000 \[([\d:]+)<([\d:]+),\s*([\d.]+)it/s')
  if ($m.Count -eq 0) {
    if ($raw -match 'CUDNN|XlaRuntime') { Write-Host "  TRAINING FAILED (cuDNN). See log." -ForegroundColor Red }
    else { Write-Host "  starting (compiling)..." -ForegroundColor DarkGray }
    Start-Sleep 4; continue
  }
  $g = $m[$m.Count-1].Groups
  $step=[int]$g[1].Value; $el=$g[2].Value; $eta=$g[3].Value; $rate=$g[4].Value
  Write-Host ""
  Write-Host ("  step {0,6}/30000  [{1}] {2:P1}" -f $step,(Bar ($step/30000) 44),($step/30000)) -ForegroundColor Green
  Write-Host ("  rate {0} it/s   elapsed {1}   ETA {2}" -f $rate,$el,$eta) -ForegroundColor Yellow
  $loss = [regex]::Matches($raw, 'loss[=:]\s*([\d.]+)')
  if ($loss.Count) { Write-Host ("  last loss: {0}" -f $loss[$loss.Count-1].Groups[1].Value) }
  Write-Host "`n  on finish -> auto sim-eval -> Octo panel fills with per-task success" -ForegroundColor DarkGray
  Start-Sleep 5
}
