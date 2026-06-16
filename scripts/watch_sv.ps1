# Live watch panel for the _sv pretrain -> finetune pipeline.
$pre = "c:\sarvik\fno_backup\logs\_train_sv_pretrain.txt"
$ft  = "c:\sarvik\fno_backup\logs\_train_sv_finetune.txt"
$host.UI.RawUI.WindowTitle = "FNO _sv pipeline - live"
while ($true) {
  Clear-Host
  # finetune log appears only after pretrain finishes -> shows which stage we're in
  $stage = "PRETRAIN (5 suites)"; $f = $pre
  if (Test-Path $ft) { $stage = "FINETUNE (object)"; $f = $ft }
  Write-Host "=== FNO _sv : $stage    $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan

  if (-not (Test-Path $f)) { Write-Host "`n  waiting for log..." -ForegroundColor DarkGray; Start-Sleep 5; continue }
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'}
  $bar = ($lines | Where-Object {$_ -match 'Epoch'} | Select-Object -Last 1)

  if ($bar) {
    # parse "Epoch  E/T:  P%|...| C/Tot [elapsed<remain,  s/it, loss=L, lr=R]"
    $ep   = if ($bar -match 'Epoch\s+(\d+)/(\d+)')      { "$($matches[1])/$($matches[2])" } else { "?" }
    $pct  = if ($bar -match '(\d+)%')                   { [int]$matches[1] } else { 0 }
    $it   = if ($bar -match '(\d+)/(\d+)\s*\[')         { "$($matches[1])/$($matches[2])" } else { "?" }
    $rem  = if ($bar -match '<([0-9:]+),')              { $matches[1] } else { "?" }
    $spit = if ($bar -match '([0-9.]+)s/it')            { [double]$matches[1] } else { 0 }
    $loss = if ($bar -match 'loss=([0-9.eE+-]+)')       { $matches[1] } else { "?" }
    $lr   = if ($bar -match 'lr=([0-9.eE+-]+)')         { $matches[1] } else { "?" }

    $filled = [int]($pct/5); $bartxt = ("#"*$filled).PadRight(20)
    Write-Host ""
    Write-Host ("  Epoch {0,-7}  [{1}] {2,3}%   iter {3}" -f $ep,$bartxt,$pct,$it) -ForegroundColor Green
    Write-Host ("  loss {0,-10} lr {1,-10} {2}s/it   epoch ETA {3}" -f $loss,$lr,$spit,$rem)

    # whole-run ETA: epochs left * (iters/epoch * s/it)
    if ($ep -match '(\d+)/(\d+)' ) {
      $e=[int]$matches[1]; $T=[int]$matches[2]
      if ($it -match '(\d+)/(\d+)') {
        $cur=[int]$matches[1]; $tot=[int]$matches[2]
        $epSec = $tot*$spit
        $left = ($T-$e)*$epSec + ($tot-$cur)*$spit
        $h=[int]($left/3600); $m=[int](($left%3600)/60)
        Write-Host ("  stage ETA ~ {0}h {1}m  ({2} epochs left)" -f $h,$m,($T-$e)) -ForegroundColor Yellow
      }
    }
    # loss trend: last 5 distinct loss values
    $losses = ($lines | Where-Object {$_ -match 'loss='} | ForEach-Object { if($_ -match 'loss=([0-9.]+)'){$matches[1]} } | Select-Object -Last 60)
    $trend = @($losses | Select-Object -Last 5) -join "  ->  "
    Write-Host ("  loss trend: {0}" -f $trend) -ForegroundColor DarkGray
  } else {
    Write-Host "`n  building model / loading data..." -ForegroundColor DarkGray
    $lines | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
  }

  Write-Host "`n  --- GPU ---" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  Write-Host "`n  (refresh 5s | Ctrl+C to close)" -ForegroundColor DarkGray
  Start-Sleep 5
}
