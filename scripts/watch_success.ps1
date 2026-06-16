$succ = "D:\eroot\fno_data\attnpool_success.jsonl"
$BASE = 63.5   # mean-pool baseline (N=20)
$host.UI.RawUI.WindowTitle = "SUCCESS%: attn-pool checkpoints"
function Bar($pct, $width) {
  $f = [int]([math]::Max(0,[math]::Min(100,$pct))/100*$width)
  return ("#"*$f) + ("-"*($width-$f))
}
while ($true) {
  Clear-Host
  Write-Host "=== SUCCESS% by checkpoint  (attn-pool finetune)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host ("  baseline (mean-pool, N=20): {0}%   |  these evals = 5 rollouts/task (quick, noisy +/-7pp)" -f $BASE) -ForegroundColor DarkGray
  if (-not (Test-Path $succ)) {
    Write-Host "`n  no checkpoint sim-evaled yet. First (epoch_5) ~40min in; watcher evals each as it lands." -ForegroundColor DarkGray
    Start-Sleep 6; continue
  }
  $rows = Get-Content $succ -ErrorAction SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json } | Sort-Object epoch
  if (-not $rows) { Write-Host "`n  waiting for first eval..."; Start-Sleep 6; continue }
  Write-Host ""
  $best=0; $bestep=0
  foreach ($r in $rows) {
    $pct = [double]$r.success * 100
    if ($pct -gt $best) { $best=$pct; $bestep=$r.epoch }
    $beat = if ($pct -ge $BASE) { " >= baseline" } else { "" }
    $col = if ($pct -ge $BASE) { 'Green' } elseif ($pct -ge $BASE-7) { 'Yellow' } else { 'Gray' }
    Write-Host ("  ep {0,3}  {1,5:N1}%  |{2}|  jerk {3:N4}{4}" -f $r.epoch,$pct,(Bar $pct 30),[double]$r.jerk,$beat) -ForegroundColor $col
  }
  Write-Host ("`n  best so far: {0:N1}%  @ ep{1}   (baseline {2}%)" -f $best,$bestep,$BASE) -ForegroundColor Yellow
  Write-Host "  (final pick gets full N=20 paired eval vs baseline)" -ForegroundColor DarkGray
  Start-Sleep 6
}
