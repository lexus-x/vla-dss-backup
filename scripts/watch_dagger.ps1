# Live monitor: DAgger pipeline (collect -> train). Auto-switches phase.
$host.UI.RawUI.WindowTitle = "DAgger pipeline"
$clog = "c:\sarvik\fno_backup\logs\_dagger_collect.txt"
$tlog = "c:\sarvik\fno_backup\logs\_train_dagger.txt"
$names = @{0="soup";1="cream cheese";2="salad";3="bbq";4="ketchup";5="tomato";6="butter";7="milk";8="pudding";9="OJ"}
$RPT = 50
function Bar($pct,$w){ $fl=[int]($pct/100*$w); ("#"*$fl)+("-"*($w-$fl)) }
while ($true) {
  Clear-Host
  $train = (Test-Path $tlog) -and ((Get-Item $tlog -EA SilentlyContinue).Length -gt 0)

  # ---------- STAGE 1: COLLECT ----------
  Write-Host "=== DAgger COLLECT (aux-xy ep5, 50/task = 500)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  $lines = Get-Content $clog -EA SilentlyContinue
  $roll = @{}; foreach($ti in 0..9){ $roll[$ti]=[pscustomobject]@{done=0;ok=0;kept=0} }
  $total=0; $tok=0; $tkept=0
  foreach($l in $lines){
    if($l -match 'task\s+(\d+)\s+r\s*(\d+):\s+(OK|fail)\s+steps=(\d+)\s+maxdiv=([0-9.]+)\s+kept=(True|False)'){
      $ti=[int]$matches[1]; $ok=($matches[3] -eq 'OK'); $kept=($matches[6] -eq 'True')
      $roll[$ti].done++; if($ok){$roll[$ti].ok++}; if($kept){$roll[$ti].kept++}
      $total++; if($ok){$tok++}; if($kept){$tkept++}
    }
  }
  $pct=[math]::Round(100*$total/(10*$RPT),1)
  Write-Host ("  overall {0}/{1} rollouts  [{2}] {3}%   OK {4}  kept {5}" -f $total,(10*$RPT),(Bar $pct 28),$pct,$tok,$tkept) -ForegroundColor Green
  Write-Host ("  {0,-13} {1,-9} {2,-7} {3}" -f "task","done/50","OK","kept") -ForegroundColor Yellow
  foreach($ti in 0..9){
    $r=$roll[$ti]; $c=if($r.done -ge $RPT){"DarkGray"}elseif($r.done -gt 0){"White"}else{"DarkGray"}
    Write-Host ("  {0,-13} {1,-9} {2,-7} {3}" -f $names[$ti],"$($r.done)/$RPT",$r.ok,$r.kept) -ForegroundColor $c
  }
  # per-task file-written confirmations
  $done = $lines | Select-String '=== task \d+ .*-> .*dagger_demo.hdf5' | Select-Object -Last 3
  if($done){ Write-Host "`n  last files written:" -ForegroundColor DarkGray; $done | ForEach-Object { Write-Host ("   " + ($_ -replace '\s+',' ').Trim()) -ForegroundColor DarkGray } }
  $nf=(Get-ChildItem 'E:\fno_data\libero_object_dagger\libero_object\*_dagger_demo.hdf5' -EA SilentlyContinue|Measure-Object).Count
  Write-Host ("  dagger files on disk: {0}/10" -f $nf) -ForegroundColor DarkGray

  # ---------- STAGE 2: TRAIN (once it begins) ----------
  if($train){
    Write-Host "`n=== STAGE 2: TRAIN aux+DAgger ===" -ForegroundColor Cyan
    $tl = Get-Content $tlog -Raw -EA SilentlyContinue
    $tr = ($tl -split "[`r`n]+") | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'}
    $rows=@()
    foreach($l in $tr){ if($l -match 'Epoch\s+(\d+)/(\d+).*train_loss:\s*([0-9.]+).*val_loss\(ema\):\s*([0-9.]+)'){ $rows+=[pscustomobject]@{ep=[int]$matches[1];tr=[double]$matches[3];vl=[double]$matches[4]} } }
    if($rows.Count){
      $best=($rows|Measure-Object vl -Minimum).Minimum
      Write-Host ("  {0,-7} {1,-10} {2,-10}" -f "epoch","train","val(ema)") -ForegroundColor Yellow
      foreach($r in ($rows|Select-Object -Last 8)){ $isb=$r.vl -le $best
        Write-Host ("  {0,-7} {1,-10:N4} {2,-10:N4} {3}" -f $r.ep,$r.tr,$r.vl,$(if($isb){"<-best"}else{""})) -ForegroundColor $(if($isb){"Green"}else{"Gray"}) }
      Write-Host ("  BEST val {0:N4}   [early-stop ~ep5]" -f $best) -ForegroundColor Cyan
    } else { Write-Host "  epoch 1 running (val prints at epoch end)..." -ForegroundColor DarkGray }
    $bar=($tl -split "[`r`n]+" | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
    if($bar -and $bar -match '(\d+)%'){ Write-Host ("  cur epoch {0}%" -f $matches[1]) -ForegroundColor DarkGray }
  }

  Write-Host "`n  -- GPU --" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  $pc=(Get-Process python -EA SilentlyContinue|Measure-Object).Count
  Write-Host ("  python: {0} {1}" -f $pc,$(if($pc -gt 0){"(alive)"}else{"(idle/done)"})) -ForegroundColor $(if($pc -gt 0){"Green"}else{"Red"})
  Write-Host "`n  (refresh 8s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 8
}
