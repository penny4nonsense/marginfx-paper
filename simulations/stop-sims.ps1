# stop-sims.ps1 -- stop the simulation queue before gaming.
#
# Order matters. The queue script watches its current stage and moves to the
# next one as soon as it sees that stage exit, so killing python first makes
# it helpfully start the NEXT simulation instead of stopping. Kill the queue
# script first, then the workers.
#
# Stopping is cheap: every stage checkpoints its partial results after each
# batch, so at most a few minutes of work is lost. Restart with start-sims.ps1.

$queue = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*run-queue.ps1*' -and $_.ProcessId -ne $PID })

if ($queue.Count -gt 0) {
    foreach ($q in $queue) {
        Write-Host "stopping queue script (PID $($q.ProcessId))"
        Stop-Process -Id $q.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 3
} else {
    Write-Host "queue script was not running"
}

$py = @(Get-Process python -ErrorAction SilentlyContinue)
if ($py.Count -gt 0) {
    Write-Host "stopping $($py.Count) python worker(s)"
    $py | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
} else {
    Write-Host "no python workers running"
}

$left = @(Get-Process python -ErrorAction SilentlyContinue)
$free = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)
if ($left.Count -eq 0) {
    Write-Host "all stopped. $free GB free. Have fun."
} else {
    Write-Host "WARNING: $($left.Count) python process(es) still alive; re-run this script."
}
