# Background watcher: sim-eval each new finetune checkpoint, append success% to a log.
# Light passes (5 rollouts/task) so it doesn't starve the running finetune.
$dir  = "D:\eroot\fno_data\run_dinov3_attnpool_finetune"
$succ = "D:\eroot\fno_data\attnpool_success.jsonl"
$doneFile = "D:\eroot\fno_data\_evaled.txt"
if (-not (Test-Path $doneFile)) { New-Item -ItemType File $doneFile | Out-Null }
$env:LIBERO_SRC="C:/code/LIBERO"; $env:MUJOCO_GL="glfw"; $env:CUDA_VISIBLE_DEVICES="0"; $env:NUMBA_DISABLE_JIT="1"
Set-Location C:\sarvik\fno_backup\code
while ($true) {
  $ckpts = Get-ChildItem $dir -Filter "epoch_*.pt" -ErrorAction SilentlyContinue | Sort-Object { [int]([regex]::Match($_.Name,'epoch_(\d+)').Groups[1].Value) }
  $done = Get-Content $doneFile -ErrorAction SilentlyContinue
  foreach ($c in $ckpts) {
    if ($done -contains $c.Name) { continue }
    $ep = [int]([regex]::Match($c.Name,'epoch_(\d+)').Groups[1].Value)
    $tmp = "D:\eroot\fno_data\_evaltmp.jsonl"
    Remove-Item $tmp -ErrorAction SilentlyContinue
    echo "N" | conda run -n mmdetection --no-capture-output python -u scripts/eval_sim.py --checkpoint $c.FullName --suite libero_object --n_rollouts 5 --execute 8 --tag "ep$ep" --log_jsonl $tmp 2>&1 | Out-Null
    $rows = Get-Content $tmp -ErrorAction SilentlyContinue | Where-Object {$_} | ForEach-Object { $_ | ConvertFrom-Json }
    if ($rows) {
      $s = ($rows | Measure-Object success -Average).Average
      $j = ($rows | Measure-Object jerk -Average).Average
      ('{{"epoch":{0},"success":{1:N4},"jerk":{2:N4},"n":{3}}}' -f $ep,$s,$j,$rows.Count) | Add-Content $succ
    }
    Add-Content $doneFile $c.Name
  }
  Start-Sleep 90
}
