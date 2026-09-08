# run-icdm-sim2.ps1 -- Simulation 2 for the ICDM camera-ready.
#
# Cells run learner-major: logistic, random forest and XGBoost across all three
# sample sizes first (minutes each), then the three Keras cells (roughly a day
# each at 500 iterations x 200 resamples).
#
# Resumable: finished cells are skipped, partial cells resume from their last
# checkpoint. Writes to sim2_se_calibration/results_icdm/, so the journal
# paper's results are untouched.
#
# To stop:
#   Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#       Where-Object { $_.CommandLine -like '*run-icdm-sim2*' } |
#       ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
#   Get-Process python | Stop-Process -Force

$ErrorActionPreference = 'Continue'

$py  = 'C:\Users\jason\venvs\marginfx\Scripts\python.exe'
$sim = 'C:\Users\jason\Dropbox\projects\marginfx-paper\simulations'
$wd  = Join-Path $sim 'sim2_se_calibration'
$log = Join-Path $sim 'icdm-sim2.log'

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONUNBUFFERED = '1'

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

Say '============ ICDM simulation 2 start ============'

$proc = Start-Process -FilePath $py `
    -ArgumentList @('-u', 'run_calibration_icdm.py') `
    -WorkingDirectory $wd `
    -RedirectStandardOutput (Join-Path $sim 'icdm-sim2.out.log') `
    -RedirectStandardError  (Join-Path $sim 'icdm-sim2.err.log') `
    -WindowStyle Hidden -PassThru

# Retain the handle, else ExitCode reads back null once the process ends.
$null = $proc.Handle
try { $proc.PriorityClass = 'BelowNormal' } catch { }

Write-Host ("icdm sim2 launched, PID {0}" -f $proc.Id)

while (-not $proc.HasExited) {
    foreach ($w in @(Get-Process python -ErrorAction SilentlyContinue)) {
        if ($w.PriorityClass -ne 'BelowNormal') {
            try { $w.PriorityClass = 'BelowNormal' } catch { }
        }
    }
    Start-Sleep -Seconds 20
}
try { $proc.Refresh() } catch { }
$code = $proc.ExitCode
if ($null -eq $code) { $code = -1 }
if ($code -eq 0) { Say 'icdm sim2 : done (exit 0)' }
else { Say "icdm sim2 : FAILED (exit $code) -- see icdm-sim2.err.log" }
