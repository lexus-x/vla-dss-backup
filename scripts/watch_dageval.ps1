# Live monitor: DAgger eval (ep5 + best) vs the 71% aux-xy headline.
$host.UI.RawUI.WindowTitle = "DAgger eval vs 71%"
# aux-xy 71% per-task profile (the thing DAgger must beat)
$aux = @{0=70;1=40;2=100;3=50;4=20;5=70;6=100;7=100;8=80;9=80}
$names = @{0="soup";1="cream cheese";2="salad";3="bbq";4="ketchup";5="tomato";6="butter";7="milk";8="pudding";9="OJ"}
while ($true) {
  Clear-Host
  Write-Host "=== DAgger N=20 REPORTABLE (x-y fix #3)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  $v5=(Get-ChildItem 'E:\fno_data\eval_videos_dagger_n20\dag5n20\*.mp4' -EA SilentlyContinue|Measure-Object).Count
  $vb=(Get-ChildItem 'E:\fno_data\eval_videos_dagger_n20\dagbestn20\*.mp4' -EA SilentlyContinue|Measure-Object).Count
  Write-Host "  beat 71% (aux) | 20/task=200 | all videos: dag5 $v5  best $vb | watch KETCHUP`n" -ForegroundColor DarkGray
  Write-Host ("  {0,-10} {1,-15} {2}" -f "checkpoint","success","vs 71%") -ForegroundColor Yellow
  foreach($t in 'dag5n20','dagbestn20'){
    $f="E:\fno_data\zdag_$t.jsonl"
    if(Test-Path $f){
      $r=Get-Content $f -EA SilentlyContinue|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
      if($r){ $n=@($r).Count;$s=(@($r)|Measure-Object success -Sum).Sum;$sr=[math]::Round(100*$s/$n,1)
        $d=$sr-71;$c=if($d -ge 0){"Green"}elseif($d -ge -3){"Yellow"}else{"Red"}
        Write-Host ("  {0,-10} {1,-15} {2}{3}" -f $t,"$sr% ($s/$n)",$(if($d -ge 0){'+'}),[math]::Round($d,1)) -ForegroundColor $c }
      else { Write-Host ("  {0,-10} (starting)" -f $t) -ForegroundColor DarkGray }
    } else { Write-Host ("  {0,-10} (pending)" -f $t) -ForegroundColor DarkGray }
  }
  # per-task for the most-complete checkpoint
  $latest=$null
  foreach($t in 'dagbestn20','dag5n20'){ if(Test-Path "E:\fno_data\zdag_$t.jsonl"){ $rr=Get-Content "E:\fno_data\zdag_$t.jsonl" -EA SilentlyContinue|Where-Object{$_}; if(@($rr).Count -gt 0){ $latest=$t; break } } }
  if($latest){
    $r=Get-Content "E:\fno_data\zdag_$latest.jsonl"|Where-Object{$_}|ForEach-Object{$_|ConvertFrom-Json}
    Write-Host ("`n  -- per-task ($latest) vs aux-71 | grasp_dxy(mm) --") -ForegroundColor Yellow
    foreach($ti in 0..9){
      $g=$r|Where-Object{$_.task_idx -eq $ti}
      if($g){ $n=@($g).Count;$s=(@($g)|Measure-Object success -Sum).Sum;$sr=[math]::Round(100*$s/$n)
        $dxy=[math]::Round(1000*(($g|Measure-Object grasp_dxy -Average).Average),1)
        $a=$aux[$ti];$dl=$sr-$a;$c=if($dl -gt 0){"Green"}elseif($dl -lt 0){"Red"}else{"Gray"}
        $star=if($ti -eq 4){" <-KETCHUP"}else{""}
        Write-Host ("    {0,-13} {1,4}% ({2}/{3})  aux {4,3}%  dxy {5}mm{6}" -f $names[$ti],$sr,$s,$n,$a,$dxy,$star) -ForegroundColor $c }
      else { Write-Host ("    {0,-13} pending" -f $names[$ti]) -ForegroundColor DarkGray }
    }
  }
  Write-Host "`n  GPU:" -ForegroundColor Yellow
  (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>$null) | ForEach-Object { Write-Host "  $_" }
  Write-Host ("  python: {0}" -f (Get-Process python -EA SilentlyContinue|Measure-Object).Count) -ForegroundColor DarkGray
  Write-Host "  (refresh 8s | Ctrl+C | green=beats aux-71)" -ForegroundColor DarkGray
  Start-Sleep 8
}
