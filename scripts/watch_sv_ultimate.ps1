# ULTIMATE _sv finetune dashboard: all loss components (train/action/gripper/val),
# val-loss tracking + reduction + stop-signal, epoch timing + ETA, live bar, GPU.
# -Once: print a single snapshot and exit (for inline/terminal view).
param([switch]$Once)
$f = "c:\sarvik\fno_backup\logs\_train_sv_finetune2.txt"
$host.UI.RawUI.WindowTitle = "FNO _sv  -  ULTIMATE dashboard"
function Bar($pct,$w){ $fill=[int]($pct/100*$w); ("#"*$fill)+("-"*($w-$fill)) }

while ($true) {
  Clear-Host
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = if ($raw) { $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'} } else { @() }

  Write-Host ("================ FNO-VLA  _sv FINETUNE  -  ULTIMATE  ===============  {0}" -f (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan

  # ---- parse completed-epoch summary lines (train, action, gripper, val, lr, time) ----
  $rows = @()
  foreach ($l in ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'})) {
    if ($l -match 'Epoch\s+(\d+)/(\d+)\s*\|\s*train_loss:\s*([0-9.]+)\s*\(action:\s*([0-9.]+),\s*gripper:\s*([0-9.]+)\)\s*\|\s*val_loss\(ema\):\s*([0-9.]+)\s*\|\s*lr:\s*([0-9.eE+-]+)\s*\|\s*([0-9.]+)s') {
      $rows += [pscustomobject]@{ ep=[int]$matches[1]; T=[int]$matches[2]; tr=[double]$matches[3];
        act=[double]$matches[4]; grp=[double]$matches[5]; vl=[double]$matches[6]; lr=$matches[7]; sec=[double]$matches[8] }
    }
  }

  if ($rows.Count) {
    $best = ($rows | Measure-Object vl -Minimum).Minimum
    $start = $rows[0].vl
    $avgSec = ($rows | Measure-Object sec -Average).Average
    Write-Host "`n  -- per-epoch losses --" -ForegroundColor Yellow
    Write-Host ("  {0,-5} {1,-9} {2,-9} {3,-9} {4,-10} {5,-9} {6,-9} {7}" -f "ep","train","action","gripper","val(ema)","dval","lr","min") -ForegroundColor DarkGray
    $prev=$null
    foreach ($r in ($rows | Select-Object -Last 14)) {
      $isBest = ($r.vl -le $best)
      if ($null -ne $prev) { $d=$r.vl-$prev; $ar= if($d -lt 0){"v"}elseif($d -gt 0){"^"}else{"="}; $dstr="{0}{1:N4}" -f $ar,[math]::Abs($d) } else { $dstr="  --" }
      $tag = if ($isBest) {"*"} else {" "}
      $col = if ($isBest) {"Green"} elseif ($null -ne $prev -and $r.vl -gt $prev) {"Gray"} else {"White"}
      Write-Host ("{0} {1,-5} {2,-9:N4} {3,-9:N4} {4,-9:N4} {5,-10:N4} {6,-9} {7,-9} {8,4:N0}" -f $tag,$r.ep,$r.tr,$r.act,$r.grp,$r.vl,$dstr,$r.lr,($r.sec/60)) -ForegroundColor $col

      $prev=$r.vl
    }

    # ---- val summary + stop signal ----
    $pctDrop = if ($start -gt 0) { 100*($start-$best)/$start } else { 0 }
    $bestIdx=0; for($i=0;$i -lt $rows.Count;$i++){ if($rows[$i].vl -le $best){$bestIdx=$i} }
    $sinceBest = ($rows.Count-1)-$bestIdx
    Write-Host "`n  -- val(ema) tracking --" -ForegroundColor Yellow
    Write-Host ("  start {0:N4}   ->   BEST {1:N4} (ep {2})    reduced {3:N1}%" -f $start,$best,$rows[$bestIdx].ep,$pctDrop) -ForegroundColor Cyan
    # stop rule: 3 consecutive epochs with no new best val (plateau or rising)
    if ($sinceBest -eq 0) { Write-Host "  STATUS: val still dropping  ->  KEEP TRAINING" -ForegroundColor Green }
    elseif ($sinceBest -ge 3) { Write-Host ("  STATUS: {0} ep no new best  ->  CONVERGED, STOP & EVAL" -f $sinceBest) -ForegroundColor Red }
    else { Write-Host ("  STATUS: val not improved {0}/3 ep  ->  keep watching (stop at 3)" -f $sinceBest) -ForegroundColor DarkYellow }

    # ---- timing ----
    Write-Host "`n  -- timing --" -ForegroundColor Yellow
    Write-Host ("  {0} epochs done   avg {1:N1} min/ep   (stops at val plateau, not ep{2})" -f $rows.Count,($avgSec/60),$rows[0].T) -ForegroundColor Gray
  } else {
    Write-Host "`n  epoch 1 still running -- val prints when it finishes (~11 min)..." -ForegroundColor DarkGray
  }

  # ---- live in-progress epoch ----
  $bar = ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
  if ($bar) {
    $ep = if($bar -match 'Epoch\s+(\d+)/(\d+):'){"$($matches[1])/$($matches[2])"}else{"?"}
    $pct= if($bar -match '(\d+)%'){[int]$matches[1]}else{0}
    $it = if($bar -match '\|\s*(\d+)/(\d+)\s*\['){"$($matches[1])/$($matches[2])"}else{"?"}
    $rem= if($bar -match '<([0-9:]+),'){$matches[1]}else{"?"}
    $spit=if($bar -match '([0-9.]+)s/it'){$matches[1]}else{"?"}
    $ls = if($bar -match 'loss=([0-9.]+)'){$matches[1]}else{"?"}
    Write-Host "`n  -- live epoch --" -ForegroundColor Yellow
    Write-Host ("  Epoch {0}  [{1}] {2,3}%" -f $ep,(Bar $pct 38),$pct) -ForegroundColor Green
    Write-Host ("  iter {0}   train-loss {1}   {2}s/it   eta {3}" -f $it,$ls,$spit,$rem) -ForegroundColor DarkGray
  }

  # ---- GPU ----
  Write-Host "`n  -- GPU --" -ForegroundColor Yellow
  $g = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>$null)
  $g | ForEach-Object { Write-Host "  $_" }
  $alive = (Get-Process python -EA SilentlyContinue | Measure-Object).Count
  $hcol = if ($alive -gt 0) {"Green"} else {"Red"}
  Write-Host ("  python procs: {0}  {1}" -f $alive, $(if($alive -gt 0){"(training alive)"}else{"(!! TRAINING DEAD !!)"})) -ForegroundColor $hcol

  if ($Once) { Write-Host "`n  (snapshot | run watch_sv_ultimate.ps1 with no args for live refresh)" -ForegroundColor DarkGray; break }
  Write-Host "`n  (refresh 6s | * = best.pt | Ctrl+C to close)" -ForegroundColor DarkGray
  Start-Sleep 6
}
