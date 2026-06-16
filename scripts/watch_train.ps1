$log = "C:\sarvik\fno_backup\logs\_train_attnpool.txt"
$host.UI.RawUI.WindowTitle = "TRAIN: attention-pool pretrain"
function Bar($frac, $width) {
  $frac = [math]::Max(0, [math]::Min(1, $frac))
  $fill = [int]($frac * $width)
  return ("#" * $fill) + ("-" * ($width - $fill))
}
while ($true) {
  Clear-Host
  Write-Host "=== TRAIN attention-pool (all LIBERO tasks)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $log)) { Write-Host "waiting for training to start..."; Start-Sleep 4; continue }
  $lines = Get-Content $log -ErrorAction SilentlyContinue
  $epochs = $lines | Select-String -Pattern "Epoch\s+(\d+)/\s*(\d+).*train_loss:\s*([\d.]+).*val_loss\(ema\):\s*([\d.]+).*\|\s*([\d.]+)s"
  if (-not $epochs) {
    $tq = ($lines | Select-String -Pattern "Epoch\s+(\d+)/(\d+):\s+(\d+)%.*?(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it.*loss=([\d.]+)")
    if ($tq) {
      $g = $tq[-1].Matches[0].Groups; $bi=[int]$g[4].Value; $bt=[int]$g[5].Value
      Write-Host ("`n  epoch {0}/{1}  (no epoch finished yet)" -f [int]$g[1].Value, [int]$g[2].Value)
      Write-Host ("  [{0}] {1,3}%  batch {2}/{3}  elapsed {4} / eta {5}  {6}s/it  loss={7}" -f `
        (Bar ($bi/$bt) 40), [int]$g[3].Value, $bi, $bt, $g[6].Value, $g[7].Value, $g[8].Value, $g[9].Value) -ForegroundColor Cyan
    } else {
      Write-Host "training started, waiting for first batch..."
      ($lines | Select-Object -Last 3) | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
    Start-Sleep 4; continue
  }
  $last = $epochs[-1].Matches[0].Groups
  $cur = [int]$last[1].Value; $tot = [int]$last[2].Value
  $trl = [double]$last[3].Value; $val = [double]$last[4].Value; $sec = [double]$last[5].Value
  $best = ($epochs | ForEach-Object { [double]$_.Matches[0].Groups[4].Value } | Measure-Object -Minimum).Minimum
  $eta = [TimeSpan]::FromSeconds($sec * ($tot - $cur))

  Write-Host ""
  Write-Host ("  epochs done {0,3}/{1}  [{2}] {3:P0}" -f $cur, $tot, (Bar ($cur/$tot) 40), ($cur/$tot)) -ForegroundColor Green

  # live within-epoch batch progress (latest tqdm line)
  $tq = ($lines | Select-String -Pattern "Epoch\s+(\d+)/(\d+):\s+(\d+)%.*?(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it.*loss=([\d.]+)")
  if ($tq) {
    $g = $tq[-1].Matches[0].Groups
    $bi = [int]$g[4].Value; $bt = [int]$g[5].Value
    Write-Host ("  cur epoch   {0,3}     [{1}] {2,3}%  batch {3}/{4}  {5}s/it  loss={6}" -f `
      [int]$g[1].Value, (Bar ($bi/$bt) 40), [int]$g[3].Value, $bi, $bt, $g[8].Value, $g[9].Value) -ForegroundColor Cyan
  }
  Write-Host ("  train_loss : {0:N4}" -f $trl)
  Write-Host ("  val_loss   : {0:N4}   best: {1:N4}" -f $val, $best) -ForegroundColor Yellow
  Write-Host ("  epoch time : {0:N1}s    ETA: {1:hh\:mm\:ss}" -f $sec, $eta)

  Write-Host "`n  --- val_loss trend (recent epochs, shorter=better) ---" -ForegroundColor DarkGray
  $recent = $epochs | Select-Object -Last 12
  $vals = $recent | ForEach-Object { [double]$_.Matches[0].Groups[4].Value }
  $mx = ($vals | Measure-Object -Maximum).Maximum
  foreach ($e in $recent) {
    $g = $e.Matches[0].Groups; $ep = [int]$g[1].Value; $v = [double]$g[4].Value
    $isbest = ($v -eq $best)
    $col = if ($isbest) { 'Green' } else { 'Gray' }
    Write-Host ("  e{0,3} {1:N4} |{2}{3}" -f $ep, $v, (Bar ($v/$mx) 30), $(if($isbest){"  <- best"}else{""})) -ForegroundColor $col
  }
  Write-Host "`n(refreshes 5s)" -ForegroundColor DarkGray
  Start-Sleep 5
}
