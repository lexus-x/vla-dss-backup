# Training monitor for LIBERO-Long (libero_10) + aux finetune (max 30 epochs).
$f = "c:\sarvik\fno_backup\logs\_train_long_aux.txt"
$host.UI.RawUI.WindowTitle = "TRAIN long+aux"
function Bar($pct,$w){ $fl=[int]($pct/100*$w); ("#"*$fl)+("-"*($w-$fl)) }
while ($true) {
  Clear-Host
  Write-Host "=== LIBERO-Long+aux finetune (HARD suite)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  ref: Object 71 / Spatial 73 / Goal 72 | Long is long-horizon -> expect lower (~45-55%)`n" -ForegroundColor DarkGray
  $raw = Get-Content $f -Raw -EA SilentlyContinue
  $lines = if($raw){ $raw -split "[`r`n]+" | Where-Object {$_ -match '\S'} } else { @() }
  $rows=@()
  foreach($l in ($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+\s*\|\s*train_loss'})){
    if($l -match 'Epoch\s+(\d+)/(\d+).*train_loss:\s*([0-9.]+).*val_loss\(ema\):\s*([0-9.]+)'){
      $rows+=[pscustomobject]@{ep=[int]$matches[1];T=[int]$matches[2];tr=[double]$matches[3];vl=[double]$matches[4]} }
  }
  if($rows.Count){
    $best=($rows|Measure-Object vl -Minimum).Minimum
    Write-Host ("  {0,-7} {1,-10} {2,-10} {3}" -f "epoch","train","val(ema)","") -ForegroundColor Yellow
    foreach($r in ($rows|Select-Object -Last 12)){ $isb=$r.vl -le $best
      Write-Host ("  {0,-7} {1,-10:N4} {2,-10:N4} {3}" -f "$($r.ep)/$($r.T)",$r.tr,$r.vl,$(if($isb){"<-best"}else{""})) -ForegroundColor $(if($isb){"Green"}else{"Gray"}) }
    Write-Host ("`n  BEST val(ema) {0:N4}   ({1} ep)   [eval ep5/10/15/20 for peak]" -f $best,$rows.Count) -ForegroundColor Cyan
  } else { Write-Host "  epoch 1 running..." -ForegroundColor DarkGray }
  $bar=($lines | Where-Object {$_ -match 'Epoch\s+\d+/\d+:\s+\d+%'} | Select-Object -Last 1)
  if($bar){
    $ep=if($bar -match 'Epoch\s+(\d+)/(\d+):'){"$($matches[1])/$($matches[2])"}else{"?"}
    $pct=if($bar -match '(\d+)%'){[int]$matches[1]}else{0}; $rem=if($bar -match '<([0-9:]+),'){$matches[1]}else{"?"}
    Write-Host ("`n  Epoch {0}  [{1}] {2,3}%   eta {3}" -f $ep,(Bar $pct 34),$pct,$rem) -ForegroundColor Green
  }
  Write-Host "`n  -- GPU --" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  $pc=(Get-Process python -EA SilentlyContinue|Measure-Object).Count
  Write-Host ("  python: {0} {1}" -f $pc,$(if($pc -gt 0){"(alive)"}else{"(DONE)"})) -ForegroundColor $(if($pc -gt 0){"Green"}else{"Red"})
  Write-Host "`n  (refresh 6s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 6
}
