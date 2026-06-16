# Live monitor for the resolution-invariance sweep + writes a graph-ready CSV.
# CSV: c:\sarvik\fno_backup\resinvar_results.csv  (output_size, n, success_pct, mean_jerk)
$sizes = 8,16,24,32
$csv   = "c:\sarvik\fno_backup\resinvar_results.csv"
$host.UI.RawUI.WindowTitle = "FNO resolution-invariance - live + CSV"
function stat($s){
  $f = "E:\fno_data\zres_$s.jsonl"
  if(-not(Test-Path $f)){ return $null }
  $r = Get-Content $f -EA SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object {$_.tag -eq "res$s"}
  if(-not $r){ return $null }
  [pscustomobject]@{
    size=$s; n=@($r).Count
    succ=[math]::Round(100*(($r|Measure-Object success -Average).Average),1)
    jerk=[math]::Round(($r|Measure-Object jerk -Average).Average,4)
  }
}
while ($true) {
  Clear-Host
  Write-Host "=== FNO RESOLUTION-INVARIANCE (baseline, N=10/size)   $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
  Write-Host "  trained@chunk16/modes8 (Nyquist=16). flat success above 16 = resolution-invariant`n" -ForegroundColor DarkGray
  Write-Host ("  {0,-12} {1,-8} {2,-12} {3,-10} {4}" -f "output_size","n/100","success%","jerk","bar") -ForegroundColor Yellow
  $csvrows = @("output_size,n,success_pct,mean_jerk")
  foreach($s in $sizes){
    $st = stat $s
    if($st){
      $bar = ("#" * [int]($st.succ/3)).PadRight(34)
      $note = if($s -lt 16){" (below Nyquist)"} elseif($s -eq 16){" (native/control)"} else {""}
      Write-Host ("  {0,-12} {1,-8} {2,-12} {3,-10} {4}{5}" -f $s,("$($st.n)/100"),"$($st.succ)%",$st.jerk,$bar,$note) -ForegroundColor Green
      $csvrows += "{0},{1},{2},{3}" -f $s,$st.n,$st.succ,$st.jerk
    } else {
      Write-Host ("  {0,-12} {1,-8} {2}" -f $s,"-","(pending)") -ForegroundColor DarkGray
      $csvrows += "{0},0,," -f $s
    }
  }
  # write graph-ready CSV
  $csvrows | Out-File -FilePath $csv -Encoding utf8
  Write-Host ("`n  CSV (refresh each tick): {0}" -f $csv) -ForegroundColor Cyan
  # which size is running
  $cur = $null; foreach($s in $sizes){ $st=stat $s; if($st -and $st.n -lt 100){ $cur=$s; break } }
  if($cur){ Write-Host ("  currently running: output_size=$cur") -ForegroundColor DarkGray }
  else { $allDone = ($sizes | ForEach-Object { $st=stat $_; $st -and $st.n -ge 100 }) -notcontains $false
         if($allDone){ Write-Host "  ALL DONE - CSV ready for plotting" -ForegroundColor Green } }
  Write-Host "`n  (refresh 8s | Ctrl+C to close)" -ForegroundColor DarkGray
  Start-Sleep 8
}
