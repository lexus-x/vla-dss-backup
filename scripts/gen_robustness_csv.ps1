# Build the robustness-ablation attribution CSV from the battery jsonl files.
# DAgger (no aug) vs aug ep15, per condition. Re-run anytime to refresh.
if(-not (Test-Path 'E:\')){ subst E: D:\eroot }
$out = 'E:\fno_data\robustness_ablation.csv'
function Stat($tag,$c){
  $f = 'E:\fno_data\zbat_'+$tag+'_'+$c+'.jsonl'
  if(-not (Test-Path $f)){ return @{n=0;s=0} }
  $l = @(Get-Content $f | Where-Object {$_})
  $s = 0; foreach($x in $l){ if(($x|ConvertFrom-Json).success){ $s++ } }
  return @{ n=$l.Count; s=$s }
}
$rows = @()
foreach($c in 'clean','noise1','noise2','noise3','blur1','blur2','blur3','brightness1','brightness2','brightness3'){
  $d = Stat 'dag' $c; $a = Stat 'aug' $c
  $dp = if($d.n){[math]::Round(100*$d.s/$d.n,1)} else {''}
  $ap = if($a.n){[math]::Round(100*$a.s/$a.n,1)} else {''}
  $delta = if($d.n -and $a.n){[math]::Round($ap-$dp,1)} else {''}
  $rows += [pscustomobject]@{
    condition=$c; dagger_pct=$dp; dagger_n="$($d.s)/$($d.n)";
    aug_pct=$ap; aug_n="$($a.s)/$($a.n)"; delta_pp=$delta
  }
}
$rows | Export-Csv -Path $out -NoTypeInformation -Encoding utf8
Write-Host "wrote $out" -ForegroundColor Green
$rows | Format-Table -Auto
