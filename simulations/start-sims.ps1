# start-sims.ps1 -- start (or resume) a simulation run.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File start-sims.ps1
#     powershell -NoProfile -ExecutionPolicy Bypass -File start-sims.ps1 run-queue
#
# With no argument it starts CURRENT_RUN below, which is whichever run is
# actively being worked on. Pass a launcher name to start a different one; the
# available launchers are listed if the name does not match.
#
# Every run resumes from its checkpoints, so starting one that was interrupted
# picks up where it stopped rather than beginning again. Starting a run that is
# already complete is harmless: it skips every finished cell and exits.
#
# Stop with stop-sims.ps1.

param([string]$Run = '')

$ErrorActionPreference = 'SilentlyContinue'

# The run currently being worked on. Update this when the active work moves.
$CURRENT_RUN = 'run-sim3-uniform.ps1'

$sim = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- resolve which launcher to start ---------------------------------------
if ([string]::IsNullOrWhiteSpace($Run)) {
    $name = $CURRENT_RUN
} else {
    $name = $Run
    if (-not $name.EndsWith('.ps1')) { $name = "$name.ps1" }
}

$launcher = Join-Path $sim $name
if (-not (Test-Path $launcher)) {
    Write-Host "no launcher named '$name'. Available:"
    Get-ChildItem -Path $sim -Filter 'run-*.ps1' |
        ForEach-Object {
            $marker = if ($_.Name -eq $CURRENT_RUN) { '  <- current' } else { '' }
            Write-Host ("    {0}{1}" -f $_.Name, $marker)
        }
    exit 1
}

# --- refuse to start on top of a run already going --------------------------
# Two launchers at once would fight for the machine and, worse, could write to
# the same results directory from two processes.
$running = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -like '*marginfx-paper*' -and
        $_.CommandLine -like '*.ps1*' -and
        $_.CommandLine -notlike '*start-sims*'
    })

if ($running.Count -gt 0) {
    Write-Host "a run is already going:"
    foreach ($r in $running) {
        Write-Host ("    {0} (PID {1})" -f ($r.CommandLine -split '\\')[-1], $r.ProcessId)
    }
    Write-Host "stop it first with stop-sims.ps1, or let it finish."
    exit 1
}

$py = @(Get-Process python -ErrorAction SilentlyContinue)
if ($py.Count -gt 0) {
    Write-Host "WARNING: $($py.Count) python process(es) already running."
    Write-Host "If those are leftover workers, run stop-sims.ps1 first."
}

# --- launch detached, so it outlives this shell -----------------------------
Start-Process powershell `
    -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcher `
    -WindowStyle Hidden

Start-Sleep -Seconds 3
Write-Host "started $name"
Write-Host "stop it any time with stop-sims.ps1 -- progress is checkpointed."
