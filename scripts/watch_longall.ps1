# Live monitor: Long all-checkpoint screen (find the success peak). Late-first.
$host.UI.RawUI.WindowTitle = "Long ALL ckpts"
while ($true) {
  Clear-Host
  Write-Host "=== LIBERO-Long — all checkpoints (find peak)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  ref: Object 71 / Spatial 73 / Goal 72 | Long = long-horizon (hard) | ep5 was 0%`n" -ForegroundColor DarkGray
  Write-Host ("  {0,-7} {1,-15} {2}" -f "ckpt","success","progress") -ForegroundColor Yellow
  foreach($ep in 30,25,20,15,10,5){
    $f="E:\fno_data\zlngall_$ep.jsonl"
    if(Test-Path $f){
      $r=Get-Content $f -EA SilentlyContinue|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
      if($r){ $n=@($r).Count;$s=(@($r)|Measure-Object success -Sum).Sum;$sr=[math]::Round(100*$s/$n,1)
        $col=if($sr -ge 30){"Green"}elseif($sr -ge 10){"Yellow"}elseif($sr -gt 0){"White"}else{"DarkGray"}
        Write-Host ("  ep{0,-5} {1,-15} {2}/50" -f $ep,"$sr% ($s/$n)",$n) -ForegroundColor $col }
      else { Write-Host ("  ep{0,-5} (starting)" -f $ep) -ForegroundColor DarkGray }
    } else { Write-Host ("  ep{0,-5} (pending)" -f $ep) -ForegroundColor DarkGray }
  }
  $latest=$null
  foreach($ep in 30,25,20,15,10,5){ if(Test-Path "E:\fno_data\zlngall_$ep.jsonl"){ $rr=Get-Content "E:\fno_data\zlngall_$ep.jsonl" -EA SilentlyContinue|Where-Object{$_}; if(@($rr).Count -gt 0){ $latest=$ep; break } } }
  if($latest){
    $r=Get-Content "E:\fno_data\zlngall_$latest.jsonl"|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
    Write-Host ("`n  -- per-task (ep$latest, the one running) --") -ForegroundColor Yellow
    foreach($ti in 0..9){
      $g=$r|Where-Object{$_.task_idx -eq $ti}
      if($g){ $n=@($g).Count;$s=(@($g)|Measure-Object success -Sum).Sum
        $nm=$g[0].task; if($nm.Length -gt 44){$nm=$nm.Substring(0,44)}
        $c=if($s -gt 0){"Green"}else{"DarkGray"}
        Write-Host ("    t{0,-2} {1}/{2}  {3}" -f $ti,$s,$n,$nm) -ForegroundColor $c }
    }
  }
  Write-Host "`n  python: $((Get-Process python -EA SilentlyContinue|Measure-Object).Count)  (refresh 10s | Ctrl+C | stop anytime a ckpt shows life)" -ForegroundColor DarkGray
  Start-Sleep 10
}
