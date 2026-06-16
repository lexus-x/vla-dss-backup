# Epoch-wise watch panel for the _sv finetune, VAL-LOSS focused: per-epoch
# train + val(ema) + delta-val + trend, best.pt flag, reduction summary, live bar.
$f = "c:\sarvik\fno_backup\logs\_train_sv_finetune2.txt"
$host.UI.RawUI.WindowTitle = "FNO _sv FINETUNE - val-loss tracker"
while ($true) {
  Clear-Host
  Write-Host "=== FNO _sv FINETUNE - val(ema) tracker   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $f)) { Write-Host "`n  waiting for log..." -ForegroundColor DarkGray; Start-Sleep 5; continue }
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'}

  $rows = @()
  foreach ($l in ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'})) {
    if ($l -match 'Epoch\s+(\d+)/(\d+).*train_loss:\s*([0-9.]+).*val_loss\(ema\):\s*([0-9.]+).*lr:\s*([0-9.eE+-]+)') {
      $rows += [pscustomobject]@{ ep=[int]$matches[1]; T=[int]$matches[2]; tr=[double]$matches[3]; vl=[double]$matches[4]; lr=$matches[5] }
    }
  }

  if ($rows.Count) {
    $best = ($rows | Measure-Object vl -Minimum).Minimum
    $start = $rows[0].vl
    Write-Host ("`n  {0,-6} {1,-10} {2,-11} {3,-11} {4}" -f "epoch","train","val(ema)","dval","lr") -ForegroundColor Yellow
    $prev = $null
    foreach ($r in ($rows | Select-Object -Last 16)) {
      $isBest = ($r.vl -le $best)
      if ($null -ne $prev) {
        $d = $r.vl - $prev
        $arrow = if ($d -lt 0) {"v"} elseif ($d -gt 0) {"^"} else {"="}
        $dstr = "{0} {1:N4}" -f $arrow, [math]::Abs($d)
        $dcol = if ($d -lt 0) {"Green"} elseif ($d -gt 0) {"Red"} else {"Gray"}
      } else { $dstr = "  --"; $dcol = "Gray" }
      $tag = if ($isBest) {"  <- BEST"} else {""}
      $line = "  {0,-6} {1,-10:N4} {2,-11:N4} {3,-11} {4}{5}" -f $r.ep,$r.tr,$r.vl,$dstr,$r.lr,$tag
      $col = if ($isBest) {"Green"} else {"Gray"}
      Write-Host $line -ForegroundColor $col
      $prev = $r.vl
    }
    $pctDrop = if ($start -gt 0) { 100*($start-$best)/$start } else { 0 }
    Write-Host ("`n  val(ema):  start {0:N4}  ->  BEST {1:N4}   (reduced {2:N1}%)" -f $start,$best,$pctDrop) -ForegroundColor Cyan
    # plateau / overfit watch
    $bestIdx = 0; for ($i=0; $i -lt $rows.Count; $i++){ if ($rows[$i].vl -le $best){ $bestIdx=$i } }
    $sinceBest = ($rows.Count-1) - $bestIdx
    if ($sinceBest -ge 1) {
      $msg = "  val not improved for {0} epoch(s)" -f $sinceBest
      if ($rows[-1].vl -gt $best) { $msg += "  (rising -> possible overfit, near stop point)" }
      Write-Host $msg -ForegroundColor DarkYellow
    } else {
      Write-Host "  val still dropping -> keep training" -ForegroundColor Green
    }
  } else {
    Write-Host "`n  epoch 1 still running (val prints only when an epoch finishes ~11 min)..." -ForegroundColor DarkGray
  }

  # in-progress epoch bar
  $bar = ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
  if ($bar) {
    $ep  = if ($bar -match 'Epoch\s+(\d+)/(\d+):') {"$($matches[1])/$($matches[2])"} else {"?"}
    $pct = if ($bar -match '(\d+)%') {[int]$matches[1]} else {0}
    $rem = if ($bar -match '<([0-9:]+),') {$matches[1]} else {"?"}
    $ls  = if ($bar -match 'loss=([0-9.]+)') {$matches[1]} else {"?"}
    $w=34; $fill=[int]($pct/100*$w); $vbar=("#"*$fill)+("-"*($w-$fill))
    Write-Host ("`n  Epoch {0}  [{1}] {2,3}%   train-loss {3}  eta {4}" -f $ep,$vbar,$pct,$ls,$rem) -ForegroundColor Green
  }
  Write-Host "`n  --- GPU ---" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  Write-Host "`n  (val updates each completed epoch | refresh 6s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 6
}
