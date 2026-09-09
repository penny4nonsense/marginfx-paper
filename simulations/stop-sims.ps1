# stop-sims.ps1 -- stop any marginfx simulation or empirical run.
#
# Run this before gaming, or any time the machine needs to be quiet. It is
# safe to run when nothing is going: it will simply say so.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File stop-sims.ps1
#
# Stopping is cheap. Every run checkpoints its partial results after each
# batch, so at most a few minutes of work is lost and restarting resumes from
# the checkpoint rather than from the beginning.
#
# ---------------------------------------------------------------------------
# Why this is more than "kill python"
# ---------------------------------------------------------------------------
#
# Two things make the naive version fail, and both were observed in practice.
#
# 1. Order. A launcher script watches its stage and starts the NEXT one as
#    soon as it sees the current stage exit. Killing python first makes the
#    launcher helpfully begin another simulation. Launchers die first.
#
# 2. Respawn. joblib's loky backend keeps a worker pool alive: kill a worker
#    while its parent is still running and the parent immediately starts a
#    replacement. Killing processes in arbitrary order therefore looks like
#    the run is unkillable -- processes keep reappearing no matter how many
#    times you stop them. Parents must die before their workers, so this
#    script kills the top-level python processes first, then sweeps whatever
#    is left, and repeats until nothing remains.
#
# The previous version of this script had both faults, and matched only one
# launcher by name, so it reported "queue script was not running" while a run
# was in fact going.

$ErrorActionPreference = 'SilentlyContinue'

# Any launcher living in the paper tree, rather than one hard-coded name.
$LAUNCHER_PATTERN = '*marginfx-paper*'

function Get-Launchers {
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -like $LAUNCHER_PATTERN -and
            $_.CommandLine -like '*.ps1*' -and
            $_.CommandLine -notlike '*stop-sims*'
        })
}

function Get-Pythons {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")
}

function Get-LauncherParents {
    # CommandLine is not always readable, so a launcher can be invisible to a
    # name match. It is, however, the parent of the top-level python, and that
    # relationship is always visible. Returns powershell PIDs owning a python.
    $py = @(Get-Pythons)
    if ($py.Count -eq 0) { return @() }
    $pyPids = $py | ForEach-Object { $_.ProcessId }
    $parentPids = $py |
        Where-Object { $pyPids -notcontains $_.ParentProcessId } |
        ForEach-Object { $_.ParentProcessId } |
        Sort-Object -Unique
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $parentPids -contains $_.ProcessId -and $_.ProcessId -ne $PID })
}

# --- 1. launchers first, so nothing starts a replacement stage --------------
$launchers = @(@(Get-Launchers) + @(Get-LauncherParents) |
    Sort-Object ProcessId -Unique)
if ($launchers.Count -gt 0) {
    foreach ($l in $launchers) {
        $script = ($l.CommandLine -split '\\')[-1]
        Write-Host "stopping launcher: $script (PID $($l.ProcessId))"
        Stop-Process -Id $l.ProcessId -Force
    }
    Start-Sleep -Seconds 2
} else {
    Write-Host "no launcher script running"
}

# --- 2. python, parents before children, repeating until none remain --------
$round = 0
while ($true) {
    $py = @(Get-Pythons)
    if ($py.Count -eq 0) { break }

    $round++
    if ($round -gt 6) {
        Write-Host "still $($py.Count) python process(es) after $round rounds; giving up"
        break
    }

    $pids = $py | ForEach-Object { $_.ProcessId }
    # A top-level python is one whose parent is not itself a python process.
    # Those own the worker pools, so they must go first or the pool refills.
    $parents = @($py | Where-Object { $pids -notcontains $_.ParentProcessId })
    $workers = @($py | Where-Object { $pids -contains $_.ParentProcessId })

    Write-Host ("round {0}: {1} python ({2} top-level, {3} worker)" -f `
        $round, $py.Count, $parents.Count, $workers.Count)

    foreach ($p in $parents) { Stop-Process -Id $p.ProcessId -Force }
    Start-Sleep -Seconds 2
    foreach ($w in $workers) { Stop-Process -Id $w.ProcessId -Force }
    Start-Sleep -Seconds 3
}

# --- 3. report --------------------------------------------------------------
$left = @(Get-Pythons)
$stillLaunching = @(@(Get-Launchers) + @(Get-LauncherParents) |
    Sort-Object ProcessId -Unique)
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)

if ($left.Count -eq 0 -and $stillLaunching.Count -eq 0) {
    Write-Host ""
    Write-Host "all stopped. $free GB free. Have fun."
} else {
    Write-Host ""
    Write-Host "WARNING: $($left.Count) python and $($stillLaunching.Count) launcher(s) still alive."
    Write-Host "Re-run this script. If they persist, from an ADMIN PowerShell:"
    Write-Host "    taskkill /F /T /IM python.exe"
    Write-Host "A restart is never necessary -- results are checkpointed."
}
