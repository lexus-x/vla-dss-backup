# Live watch for the _sv checkpoint eval: per-task success as rollouts land,
# side-by-side vs the 63.5% baseline, weak-4 cluster highlighted.
$jf  = "E:\fno_data\zsv_cl1.jsonl"
$log = "c:\sarvik\fno_backup\logs\_eval_sv_cl1.txt"
$host.UI.RawUI.WindowTitle = "FNO _sv CLOSED-LOOP EVAL (exec=1) - vs ep5 67%"
# baseline (63.5%, N=20) per task_idx
$base = @{0=90;1=35;2=90;3=5;4=60;5=35;6=85;7=90;8=95;9=85}  # ep5 open-loop (67%) for delta
$names = @{0="alphabet soup";1="cream cheese";2="salad dressing";3="bbq sauce";4="ketchup";
           5="tomato sauce";6="butter";7="milk";8="chocolate pudding";9="orange juice"}
$weak = @(1,3,4,9)
while ($true) {
  Clear-Host
  Write-Host "=== FNO _sv FINAL EVAL (ep44, exec8, N=20, all tasks)  vs 63.5% baseline   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  if (-not (Test-Path $jf)) {
    Write-Host "`n  loading model + env (no rollouts yet)..." -ForegroundColor DarkGray
    Get-Content $log -Tail 2 -EA SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    Start-Sleep 5; continue
  }
  $rows = Get-Content $jf -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object {$_.tag -eq 'sv_cl1'}
  if (-not $rows) { Write-Host "`n  waiting for first rollout..."; Start-Sleep 5; continue }

  Write-Host ("`n  {0,-18} {1,-12} {2,-10} {3}" -f "object","_sv (now)","baseline","delta") -ForegroundColor Yellow
  $done = 0; $succ = 0
  foreach ($ti in 0..9) {
    $g = $rows | Where-Object {$_.task_idx -eq $ti}
    if ($g) {
      $n=@($g).Count; $s=(@($g)|Measure-Object success -Sum).Sum; $sr=[math]::Round(100*$s/$n)
      $done += $n; $succ += $s
      $b=$base[$ti]; $d=$sr-$b; $dstr= if($d -ge 0){"+$d"}else{"$d"}
      $col= if($d -gt 0){"Green"}elseif($d -lt 0){"Red"}else{"Gray"}
      $mark= if($weak -contains $ti){"*"}else{" "}
      Write-Host ("{0}{1,-18} {2,-12} {3,-10} {4}" -f $mark,$names[$ti],("$sr% ($s/$n)"),"$b%",$dstr) -ForegroundColor $col
    } else {
      $mark= if($weak -contains $ti){"*"}else{" "}
      Write-Host ("{0}{1,-18} {2,-12} {3,-10}" -f $mark,$names[$ti],"--","$($base[$ti])%") -ForegroundColor DarkGray
    }
  }
  $tasksDone = (@($rows | Group-Object task_idx)).Count
  Write-Host ("  {0}" -f ('-'*52))
  if ($done -gt 0) {
    $runAvg=[math]::Round(100*$succ/$done,1)
    Write-Host ("  running avg: {0}%  ({1} rollouts, {2}/10 tasks started)" -f $runAvg,$done,$tasksDone) -ForegroundColor Cyan
  }
  Write-Host "  (* = weak-4 cluster -- the objects _sv is meant to fix)" -ForegroundColor DarkGray
  Write-Host "`n  (refresh 5s | Ctrl+C to close)" -ForegroundColor DarkGray
  Start-Sleep 5
}
