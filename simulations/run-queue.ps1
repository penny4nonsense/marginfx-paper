# run-queue.ps1 -- the full simulation queue, start to finish.
#
# Runs, in order:
#   1. Simulation 3, closed-form Riesz representer
#   2. Simulation 3, representer estimated by sieve
#   3. Simulation 2, classification calibration
#   4. Simulation 2, regression calibration
#
# Every stage is resumable: completed combinations are skipped and partial
# results are picked up, so re-running this script after an interruption
# continues rather than starting over.
#
# Launch DETACHED so it outlives the shell that starts it:
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass', `
#       '-File','<this file>' -WindowStyle Hidden
#
# Logs: queue.log holds the stage timeline; each stage's own output goes to
# <stage>.log. Stage output is redirected by the child process rather than
# piped through PowerShell, because piping into Add-Content holds an exclusive
# lock on the file for the whole run and makes it unreadable while the job is
# going -- useless for something that runs for days.
#
# To stop: kill THIS script first, otherwise it advances to the next stage
# when it sees the current one exit.
#
#   Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#       Where-Object { $_.CommandLine -like '*run-queue.ps1*' } |
#       ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
#   Get-Process python | Stop-Process -Force

$ErrorActionPreference = 'Continue'

# Not in %TEMP%. The previous environment lived there and Windows cleaned it
# out mid-run on 2026-09-01, deleting parts of numpy, joblib and scikit-learn;
# the regression stage died with ModuleNotFoundError partway through a cell.
# Not in Dropbox either, which locks site-packages during sync.
$py  = 'C:\Users\jason\venvs\marginfx\Scripts\python.exe'
$sim = 'C:\Users\jason\Dropbox\projects\marginfx-paper\simulations'
$log = Join-Path $sim 'queue.log'

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONUNBUFFERED = '1'

function Say($msg) {
    # Add-Content opens and closes per call, so queue.log stays readable.
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Invoke-Stage($name, $workdir, $script, $stageArgs) {
    $out = Join-Path $sim "$name.log"
    $err = Join-Path $sim "$name.err.log"
    Say "$name : start"
    $argList = @('-u', $script) + $stageArgs
    $proc = Start-Process -FilePath $py -ArgumentList $argList `
        -WorkingDirectory $workdir `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

    # Run below normal priority. Twenty-four workers at normal priority take
    # every physical core and leave nothing for the desktop -- enough to
    # freeze the machine hard enough that you cannot open a window to stop
    # them. Below normal costs almost nothing when the machine is idle, since
    # there is nothing else to schedule, but yields immediately when anything
    # else wants the CPU. joblib's workers inherit the class from this
    # process, and any that start later are caught by the sweep below.
    try { $proc.PriorityClass = 'BelowNormal' } catch { }

    while (-not $proc.HasExited) {
        foreach ($w in @(Get-Process python -ErrorAction SilentlyContinue)) {
            if ($w.PriorityClass -ne 'BelowNormal') {
                try { $w.PriorityClass = 'BelowNormal' } catch { }
            }
        }
        Start-Sleep -Seconds 20
    }
    # Refresh before reading ExitCode: a Process object from Start-Process
    # -PassThru does not reliably have it populated, which is why an earlier
    # run logged a bare "exit=" and then announced the queue was finished
    # after the regression stage had in fact crashed with five cells to go.
    try { $proc.Refresh() } catch { }
    $code = $proc.ExitCode
    if ($null -eq $code) { $code = -1 }

    if ($code -eq 0) {
        Say "$name : done (exit 0)"
    } else {
        Say "$name : FAILED (exit $code) -- see $name.err.log"
    }
    return $code
}

Say '================ queue start ================'

$sim3dir = Join-Path $sim 'sim3_debiased'
$sim2dir = Join-Path $sim 'sim2_se_calibration'

# --- 1 and 2. Simulation 3, both representer arms ---------------------------
$failed = @()
if ((Invoke-Stage 'sim3-closed' $sim3dir 'run_debiased.py' @('--riesz','closed')) -ne 0) { $failed += 'sim3-closed' }
if ((Invoke-Stage 'sim3-sieve'  $sim3dir 'run_debiased.py' @('--riesz','sieve'))  -ne 0) { $failed += 'sim3-sieve' }

# --- 3. Simulation 2 --------------------------------------------------------
# The stored results came from a bootstrap that never refit the forest, and
# from the superseded replication counts. Move them aside rather than delete:
# the runner skips any combination whose result file still exists, so they
# cannot stay in place, but the old numbers stay available for comparison.
$results    = Join-Path $sim2dir 'results'
$superseded = Join-Path $sim2dir 'results_superseded'

# Guard on the destination directory. This script is re-run every time the
# queue is restarted, and without the guard each restart archives whatever
# sim2 had computed since the last one -- silently destroying its own
# progress, and overwriting the genuinely superseded files it archived the
# first time. Archive once; after that the directory's existence is the
# record that it was done.
if ((Test-Path $results) -and -not (Test-Path $superseded)) {
    $stale = @(Get-ChildItem -Path $results -Filter '*.parquet' -ErrorAction SilentlyContinue)
    if ($stale.Count -gt 0) {
        New-Item -ItemType Directory -Path $superseded | Out-Null
        Say "sim2 : moving $($stale.Count) superseded result file(s) aside"
        foreach ($f in $stale) {
            Move-Item -Path $f.FullName -Destination $superseded -Force
        }
    }
} elseif (Test-Path $superseded) {
    Say 'sim2 : superseded results already archived; leaving results/ alone'
}

if ((Invoke-Stage 'sim2-classification' $sim2dir 'run_calibration.py' @())            -ne 0) { $failed += 'sim2-classification' }
if ((Invoke-Stage 'sim2-regression'     $sim2dir 'run_calibration_regression.py' @()) -ne 0) { $failed += 'sim2-regression' }

if ($failed.Count -gt 0) {
    Say ("=========== queue ended with FAILURES: {0} ===========" -f ($failed -join ', '))
    Say 'Re-run start-sims.ps1 after fixing; completed cells are skipped.'
} else {
    Say '================ queue done ================='
}
