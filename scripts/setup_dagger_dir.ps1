# Build the DAgger training suite dir: hardlink the 10 original task files in
# alongside the <lang>_dagger_demo.hdf5 files that collect_dagger.py wrote.
# Hardlinks are instant + zero extra disk (same E: volume) and leave originals
# untouched. Result: each task = 50 original demos + its DAgger demos.
$src = 'E:\fno_data\libero_object'
$dst = 'E:\fno_data\libero_object_dagger\libero_object'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$n = 0
foreach ($f in Get-ChildItem "$src\*_demo.hdf5") {
  $link = Join-Path $dst $f.Name
  if (-not (Test-Path $link)) {
    New-Item -ItemType HardLink -Path $link -Target $f.FullName | Out-Null
    $n++
  }
}
Write-Host "hardlinked $n original files into $dst"
Write-Host "--- contents (dagger files should appear after collect_dagger runs) ---"
Get-ChildItem "$dst\*.hdf5" | Select-Object Name,@{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -Auto
