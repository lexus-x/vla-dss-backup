# Live monitor: LIBERO-Goal+aux ladder eval (ep5/10/15/20) -> find the success peak.
$host.UI.RawUI.WindowTitle = "Goal+aux eval"
while ($true) {
  Clear-Host
  Write-Host "=== LIBERO-Goal+aux ladder (3rd suite)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  ref: Object aux 71% | Spatial aux 73% | guess Goal ~70%`n" -ForegroundColor DarkGray
  Write-Host ("  {0,-7} {1,-16} {2,-9} {3}" -f "ckpt","success","jerk","prog") -ForegroundColor Yellow
  foreach($ep in 5,10,15,20){
    $f="E:\fno_data\zgoa_$ep.jsonl"
    if(Test-Path $f){
      $r=Get-Content $f -EA SilentlyContinue|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
      if($r){ $n=@($r).Count;$s=(@($r)|Measure-Object success -Sum).Sum;$sr=[math]::Round(100*$s/$n,1)
        $jm=[math]::Round((@($r)|Where-Object{$_.jerk -ne $null}|ForEach-Object{[double]$_.jerk}|Measure-Object -Average).Average,4)
        $col=if($sr -ge 73){"Green"}elseif($sr -ge 63){"Yellow"}else{"White"}
        Write-Host ("  ep{0,-5} {1,-16} {2,-9} {3}/100" -f $ep,"$sr% ($s/$n)",$jm,$n) -ForegroundColor $col }
      else { Write-Host ("  ep{0,-5} (starting)" -f $ep) -ForegroundColor DarkGray }
    } else { Write-Host ("  ep{0,-5} (pending)" -f $ep) -ForegroundColor DarkGray }
  }
  $latest=$null
  foreach($ep in 20,15,10,5){ if(Test-Path "E:\fno_data\zgoa_$ep.jsonl"){ $rr=Get-Content "E:\fno_data\zgoa_$ep.jsonl" -EA SilentlyContinue|Where-Object{$_}; if(@($rr).Count -gt 0){ $latest=$ep; break } } }
  if($latest){
    $r=Get-Content "E:\fno_data\zgoa_$latest.jsonl"|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
    Write-Host ("`n  -- per-task (ep$latest) --") -ForegroundColor Yellow
    foreach($ti in 0..9){
      $g=$r|Where-Object{$_.task_idx -eq $ti}
      if($g){ $n=@($g).Count;$s=(@($g)|Measure-Object success -Sum).Sum
        $nm=$g[0].task; if($nm.Length -gt 40){$nm=$nm.Substring(0,40)}
        $c=if(100*$s/$n -ge 70){"Green"}elseif(100*$s/$n -ge 40){"Gray"}else{"Red"}
        Write-Host ("    t{0,-2} {1,3}% ({2}/{3})  {4}" -f $ti,[math]::Round(100*$s/$n),$s,$n,$nm) -ForegroundColor $c }
      else { Write-Host ("    t{0,-2} pending" -f $ti) -ForegroundColor DarkGray }
    }
  }
  Write-Host "`n  python: $((Get-Process python -EA SilentlyContinue|Measure-Object).Count)  (refresh 8s | Ctrl+C)" -ForegroundColor DarkGray
  Start-Sleep 8
}
