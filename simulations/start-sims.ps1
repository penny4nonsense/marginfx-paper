# start-sims.ps1 -- restart the simulation queue after gaming.
#
# Safe to run any time. Every stage skips combinations whose result file
# already exists and resumes partial ones from their last checkpoint, so this
# picks up where the last stop left off rather than redoing work.
#
# Refuses to start a second copy if one is already running.

$sim   = 'C:\Users\jason\Dropbox\projects\marginfx-paper\simulations'
$queue = Join-Path $sim 'run-queue.ps1'

$running = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*run-queue.ps1*' -and $_.ProcessId -ne $PID })

if ($running.Count -gt 0) {
    Write-Host "queue is already running (PID $($running[0].ProcessId)) -- nothing to do"
    return
}

$py = @(Get-Process python -ErrorAction SilentlyContinue)
if ($py.Count -gt 0) {
    Write-Host "WARNING: $($py.Count) python process(es) still running from a previous run."
    Write-Host "Run stop-sims.ps1 first, then try again."
    return
}

Start-Process powershell `
    -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $queue `
    -WindowStyle Hidden

Start-Sleep -Seconds 20

$py = @(Get-Process python -ErrorAction SilentlyContinue)
Write-Host "queue restarted; $($py.Count) worker(s) up"

# Show where it picked up.
$closed = @(Get-ChildItem "$sim\sim3_debiased\results"       -Filter '*.parquet' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*partial*' })
$sieve  = @(Get-ChildItem "$sim\sim3_debiased\results_sieve" -Filter '*.parquet' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*partial*' })
$sim2   = @(Get-ChildItem "$sim\sim2_se_calibration\results" -Filter '*.parquet' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*partial*' })
Write-Host ("  sim3 closed {0}/120   sim3 sieve {1}/120   sim2 {2}" -f $closed.Count, $sieve.Count, $sim2.Count)
