# Clean training monitor (aux-xy x-y fix #1). Training info only.
$f = "c:\sarvik\fno_backup\logs\_train_auxxy.txt"
$host.UI.RawUI.WindowTitle = "FNO-VLA training (aux-xy)"
function Bar($pct,$w){ $fl=[int]($pct/100*$w); ("#"*$fl)+("-"*($w-$fl)) }
while ($true) {
  Clear-Host
  Write-Host "=== aux-xy finetune (x-y fix #1)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = if($raw){ $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'} } else { @() }
  $rows=@()
  foreach($l in ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'})){
    if($l -match 'Epoch\s+(\d+)/(\d+).*train_loss:\s*([0-9.]+)\s*\(action:\s*([0-9.]+),\s*gripper:\s*([0-9.]+)\).*val_loss\(ema\):\s*([0-9.]+)'){
      $rows+=[pscustomobject]@{ep=[int]$matches[1];T=[int]$matches[2];tr=[double]$matches[3];act=[double]$matches[4];grp=[double]$matches[5];vl=[double]$matches[6]} }
  }
  if($rows.Count){
    $best=($rows|Measure-Object vl -Minimum).Minimum; $prev=$null
    Write-Host ("`n  {0,-6} {1,-9} {2,-9} {3,-9} {4,-10} {5}" -f "epoch","train","action","gripper","val(ema)","dval") -ForegroundColor Yellow
    foreach($r in ($rows | Select-Object -Last 14)){
      $isb=$r.vl -le $best
      if($null -ne $prev){ $d=$r.vl-$prev; $ar=if($d -lt 0){"v"}elseif($d -gt 0){"^"}else{"="}; $ds="{0}{1:N4}" -f $ar,[math]::Abs($d) } else { $ds="  --" }
      $tag=if($isb){"  <-best"}else{""}
      Write-Host ("  {0,-6} {1,-9:N4} {2,-9:N4} {3,-9:N4} {4,-10:N4} {5}{6}" -f "$($r.ep)/$($r.T)",$r.tr,$r.act,$r.grp,$r.vl,$ds,$tag) -ForegroundColor $(if($isb){"Green"}else{"Gray"})
      $prev=$r.vl
    }
    $bi=0; for($i=0;$i -lt $rows.Count;$i++){ if($rows[$i].vl -le $best){$bi=$i} }
    Write-Host ("`n  BEST val(ema) {0:N4} @ ep {1}   ({2} epochs)   [early-stop ~ep5]" -f $best,$rows[$bi].ep,$rows.Count) -ForegroundColor Cyan
  } else { Write-Host "`n  epoch 1 running (val prints at epoch end)..." -ForegroundColor DarkGray }

  $bar = ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
  if($bar){
    $ep=if($bar -match 'Epoch\s+(\d+)/(\d+):'){"$($matches[1])/$($matches[2])"}else{"?"}
    $pct=if($bar -match '(\d+)%'){[int]$matches[1]}else{0}; $it=if($bar -match '\|\s*(\d+)/(\d+)\s*\['){"$($matches[1])/$($matches[2])"}else{"?"}
    $ls=if($bar -match 'loss=([0-9.]+)'){$matches[1]}else{"?"}; $rem=if($bar -match '<([0-9:]+),'){$matches[1]}else{"?"}; $sp=if($bar -match '([0-9.]+)s/it'){$matches[1]}else{"?"}
    Write-Host ("`n  Epoch {0}  [{1}] {2,3}%" -f $ep,(Bar $pct 38),$pct) -ForegroundColor Green
    Write-Host ("  iter {0}  loss {1}  {2}s/it  eta {3}" -f $it,$ls,$sp,$rem) -ForegroundColor DarkGray
  }

  Write-Host "`n  -- GPU --" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  $pc=(Get-Process python -EA SilentlyContinue | Measure-Object).Count
  Write-Host ("  python: {0} {1}" -f $pc,$(if($pc -gt 0){"(alive)"}else{"(DEAD)"})) -ForegroundColor $(if($pc -gt 0){"Green"}else{"Red"})
  Write-Host "`n  (refresh 6s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 6
}
