# Live monitor: aux-xy checkpoint eval (ep5/15/25) vs 67% headline + 63.5 baseline.
$host.UI.RawUI.WindowTitle = "aux-xy eval vs 67%"
$base = @{0=95;1=40;2=100;3=30;4=30;5=75;6=95;7=70;8=70;9=30}  # 63.5% baseline per-task
$names = @{0="soup";1="cream cheese";2="salad";3="bbq";4="ketchup";5="tomato";6="butter";7="milk";8="pudding";9="OJ"}
while ($true) {
  Clear-Host
  Write-Host "=== aux-xy (x-y fix #1) checkpoint eval   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  target: beat 67% (your headline) | baseline 63.5%`n" -ForegroundColor DarkGray
  Write-Host ("  {0,-10} {1,-14} {2}" -f "checkpoint","success","vs 67%") -ForegroundColor Yellow
  foreach ($ep in 5,15,25) {
    $f = "E:\fno_data\zaux_$ep.jsonl"
    if (Test-Path $f) {
      $r = Get-Content $f -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object {$_.tag -eq "aux$ep"}
      if ($r) {
        $n=@($r).Count; $s=(@($r)|Measure-Object success -Sum).Sum; $sr=[math]::Round(100*$s/$n,1)
        $d=$sr-67; $col=if($d -ge 0){"Green"}elseif($d -ge -3){"Yellow"}else{"Red"}
        Write-Host ("  ep{0,-8} {1,-14} {2}" -f $ep,"$sr% ($s/$n)",("{0}{1}" -f $(if($d -ge 0){'+'}),[math]::Round($d,1))) -ForegroundColor $col
      } else { Write-Host ("  ep{0,-8} (starting)" -f $ep) -ForegroundColor DarkGray }
    } else { Write-Host ("  ep{0,-8} (pending)" -f $ep) -ForegroundColor DarkGray }
  }
  # per-task + grasp_dxy for the most-complete checkpoint
  $latest=$null
  foreach($ep in 25,15,5){ if(Test-Path "E:\fno_data\zaux_$ep.jsonl"){ $rr=Get-Content "E:\fno_data\zaux_$ep.jsonl" -EA SilentlyContinue|Where-Object{$_}; if(@($rr).Count -gt 0){ $latest=$ep; break } } }
  if($latest){
    $r = Get-Content "E:\fno_data\zaux_$latest.jsonl" | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
    Write-Host ("`n  -- per-task (ep$latest) vs baseline | grasp_dxy(mm) --") -ForegroundColor Yellow
    foreach($ti in 0..9){
      $g=$r|Where-Object{$_.task_idx -eq $ti}
      if($g){ $n=@($g).Count;$s=(@($g)|Measure-Object success -Sum).Sum;$sr=[math]::Round(100*$s/$n)
        $dxy=[math]::Round(1000*(($g|Measure-Object grasp_dxy -Average).Average),1)
        $b=$base[$ti];$dl=$sr-$b;$c=if($dl -gt 0){"Green"}elseif($dl -lt 0){"Red"}else{"Gray"}
        Write-Host ("    {0,-13} {1,4}%  (base {2}%)  dxy {3}mm" -f $names[$ti],$sr,$b,$dxy) -ForegroundColor $c }
    }
  }
  Write-Host "`n  (refresh 8s | Ctrl+C | green=beats 67% | dxy<=20mm=good)" -ForegroundColor DarkGray
  Start-Sleep 8
}
