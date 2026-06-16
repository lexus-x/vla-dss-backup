# Live monitor for the robustness batteries (dag vs aug) -> attribution table.
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
$host.UI.RawUI.WindowTitle = 'WATCH batteries (dag vs aug)'
$conds = 'clean','noise1','noise2','noise3','blur1','blur2','blur3','brightness1','brightness2','brightness3'

function Rate($tag,$c){
  $f = 'E:\fno_data\zbat_' + $tag + '_' + $c + '.jsonl'
  if(-not (Test-Path $f)){ return '-' }
  $lines = @(Get-Content $f | Where-Object { $_ })
  $n = $lines.Count
  if($n -eq 0){ return '..' }
  $s = 0
  foreach($ln in $lines){ if( ($ln | ConvertFrom-Json).success ){ $s = $s + 1 } }
  $pct = [math]::Round(100 * $s / $n)
  return ('' + $pct + '% (' + $s + '/' + $n + ')')
}

while($true){
  Clear-Host
  Write-Host '=== ROBUSTNESS BATTERY  N=5/task (10 tasks = 50/cond) ==='
  Write-Host ('{0,-13}{1,-16}{2}' -f 'condition','DAgger','aug ep15')
  foreach($c in $conds){
    $d = Rate 'dag' $c
    $a = Rate 'aug' $c
    Write-Host ('{0,-13}{1,-16}{2}' -f $c, $d, $a)
  }
  Write-Host '(blur rows = the key gap aug should close.  Ctrl+C to stop)'
  Start-Sleep 25
}
