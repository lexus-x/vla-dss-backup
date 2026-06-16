$log = "C:\sarvik\fno_backup\logs\_train_finetune.txt"
$host.UI.RawUI.WindowTitle = "FINETUNE: attn-pool on libero_object"
function Bar($frac, $width) {
  $frac = [math]::Max(0, [math]::Min(1, $frac)); $fill = [int]($frac * $width)
  return ("#" * $fill) + ("-" * ($width - $fill))
}
while ($true) {
  Clear-Host
  Write-Host "=== FINETUNE attn-pool / libero_object   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $log)) { Write-Host "waiting..."; Start-Sleep 4; continue }
  $lines = Get-Content $log -ErrorAction SilentlyContinue
  $epochs = $lines | Select-String -Pattern "Epoch\s+(\d+)/\s*(\d+).*train_loss:\s*([\d.]+).*val_loss\(ema\):\s*([\d.]+).*\|\s*([\d.]+)s"
  $tqPat = "Epoch\s+(\d+)/(\d+):\s+(\d+)%.*?(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it.*loss=([\d.]+)"
  if (-not $epochs) {
    $tq = ($lines | Select-String -Pattern $tqPat)
    if ($tq) { $g=$tq[-1].Matches[0].Groups; $bi=[int]$g[4].Value; $bt=[int]$g[5].Value
      Write-Host ("`n  epoch {0}/{1} (none finished yet)" -f [int]$g[1].Value,[int]$g[2].Value)
      Write-Host ("  [{0}] {1,3}%  batch {2}/{3}  elapsed {4}/eta {5}  {6}s/it  loss={7}" -f (Bar ($bi/$bt) 40),[int]$g[3].Value,$bi,$bt,$g[6].Value,$g[7].Value,$g[8].Value,$g[9].Value) -ForegroundColor Cyan }
    else { Write-Host "waiting for first batch..."; ($lines|Select-Object -Last 3)|ForEach-Object{Write-Host "  $_" -ForegroundColor DarkGray} }
    Start-Sleep 4; continue
  }
  $last=$epochs[-1].Matches[0].Groups; $cur=[int]$last[1].Value; $tot=[int]$last[2].Value
  $val=[double]$last[4].Value; $sec=[double]$last[5].Value
  $best=($epochs|ForEach-Object{[double]$_.Matches[0].Groups[4].Value}|Measure-Object -Minimum).Minimum
  Write-Host ""
  Write-Host ("  epochs done {0,3}/{1}  [{2}] {3:P0}" -f $cur,$tot,(Bar ($cur/$tot) 40),($cur/$tot)) -ForegroundColor Green
  $tq=($lines|Select-String -Pattern $tqPat)
  if ($tq){ $g=$tq[-1].Matches[0].Groups; $bi=[int]$g[4].Value; $bt=[int]$g[5].Value
    Write-Host ("  cur epoch   {0,3}     [{1}] {2,3}%  {3}/{4}  loss={5}" -f [int]$g[1].Value,(Bar ($bi/$bt) 40),[int]$g[3].Value,$bi,$bt,$g[9].Value) -ForegroundColor Cyan }
  Write-Host ("  val(ema)    : {0:N4}   best {1:N4}   ({2:N1}s/epoch)" -f $val,$best,$sec) -ForegroundColor Yellow
  Write-Host "`n  --- val trend (recent) ---" -ForegroundColor DarkGray
  $epochs | Select-Object -Last 10 | ForEach-Object { $g=$_.Matches[0].Groups; $v=[double]$g[4].Value
    Write-Host ("  e{0,3} {1:N4} {2}" -f [int]$g[1].Value,$v,$(if($v -eq $best){"<- best"}else{""})) -ForegroundColor $(if($v -eq $best){'Green'}else{'Gray'}) }
  Write-Host "`n  NOTE: val != success. Real metric = sim-eval of checkpoints (separate)." -ForegroundColor DarkGray
  Start-Sleep 5
}
