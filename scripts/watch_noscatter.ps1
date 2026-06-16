# Epoch-wise dashboard for the no-scatter ablation (pretrain -> finetune).
$pre = "c:\sarvik\fno_backup\logs\_train_noscatter_pretrain.txt"
$ft  = "c:\sarvik\fno_backup\logs\_train_noscatter_finetune.txt"
$host.UI.RawUI.WindowTitle = "NO-SCATTER ablation - epoch-wise"
function Bar($pct,$w){ $f=[int]($pct/100*$w); ("#"*$f)+("-"*($w-$f)) }
while ($true) {
  Clear-Host
  $stage = "PRETRAIN (5 suites)"; $f = $pre
  if (Test-Path $ft) { $stage = "FINETUNE (object)"; $f = $ft }
  Write-Host "=== NO-SCATTER ablation : $stage   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $f)) { Write-Host "`n  building / loading..." -ForegroundColor DarkGray; Start-Sleep 5; continue }
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'}

  $rows=@()
  foreach($l in ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'})){
    if($l -match 'Epoch\s+(\d+)/(\d+).*train_loss:\s*([0-9.]+).*val_loss\(ema\):\s*([0-9.]+)'){
      $rows += [pscustomobject]@{ep=[int]$matches[1];T=[int]$matches[2];tr=[double]$matches[3];vl=[double]$matches[4]} }
  }
  if($rows.Count){
    $best=($rows|Measure-Object vl -Minimum).Minimum
    Write-Host ("`n  {0,-7} {1,-10} {2,-11} {3}" -f "epoch","train","val(ema)","dval") -ForegroundColor Yellow
    $prev=$null
    foreach($r in ($rows | Select-Object -Last 14)){
      $isb = $r.vl -le $best
      if($null -ne $prev){ $d=$r.vl-$prev; $ar=if($d -lt 0){"v"}elseif($d -gt 0){"^"}else{"="}; $ds="{0}{1:N4}" -f $ar,[math]::Abs($d) } else { $ds="  --" }
      $tag = if($isb){"  <- best"} else {""}
      $col = if($isb){"Green"}else{"Gray"}
      Write-Host ("  {0,-7} {1,-10:N4} {2,-11:N4} {3}{4}" -f "$($r.ep)/$($r.T)",$r.tr,$r.vl,$ds,$tag) -ForegroundColor $col
      $prev=$r.vl
    }
    $bi=0; for($i=0;$i -lt $rows.Count;$i++){ if($rows[$i].vl -le $best){$bi=$i} }
    Write-Host ("`n  BEST val(ema) {0:N4} @ ep {1}   ({2} epochs done)" -f $best,$rows[$bi].ep,$rows.Count) -ForegroundColor Cyan
  } else { Write-Host "`n  epoch 1 running (val prints at epoch end)..." -ForegroundColor DarkGray }

  $bar = ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
  if($bar){
    $ep=if($bar -match 'Epoch\s+(\d+)/(\d+):'){"$($matches[1])/$($matches[2])"}else{"?"}
    $pct=if($bar -match '(\d+)%'){[int]$matches[1]}else{0}
    $rem=if($bar -match '<([0-9:]+),'){$matches[1]}else{"?"}
    $sp =if($bar -match '([0-9.]+)(it/s|s/it)'){$matches[0]}else{"?"}
    $ls =if($bar -match 'loss=([0-9.]+)'){$matches[1]}else{"?"}
    Write-Host ("`n  Epoch {0} [{1}] {2,3}%  loss {3}  {4}  eta {5}" -f $ep,(Bar $pct 30),$pct,$ls,$sp,$rem) -ForegroundColor Green
  }
  Write-Host "`n  --- GPU ---" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  Write-Host ("  python procs: {0}" -f (Get-Process python -EA SilentlyContinue | Measure-Object).Count) -ForegroundColor DarkGray
  Write-Host "`n  (refresh 6s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 6
}
