# run-regression-only.ps1 -- launch just the Simulation 2 regression stage.
#
# Used when the rest of the queue is already complete and only this stage
# remains, so the queue wrapper is not in the critical path.
#
# Resumable in the usual way: finished cells are skipped and partial ones
# resume from their last checkpoint.

$sim = 'C:\Users\jason\Dropbox\projects\marginfx-paper\simulations'
$py  = 'C:\Users\jason\venvs\marginfx\Scripts\python.exe'
$wd  = Join-Path $sim 'sim2_se_calibration'

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONUNBUFFERED = '1'

$proc = Start-Process -FilePath $py `
    -ArgumentList '-u', 'run_calibration_regression.py' `
    -WorkingDirectory $wd `
    -RedirectStandardOutput (Join-Path $sim 'sim2-regression.log') `
    -RedirectStandardError  (Join-Path $sim 'sim2-regression.err.log') `
    -WindowStyle Hidden -PassThru

# Touching Handle makes .NET retain the process handle, without which
# ExitCode comes back null after the process ends.
$null = $proc.Handle
try { $proc.PriorityClass = 'BelowNormal' } catch { }

Write-Host ("regression stage launched, PID {0}" -f $proc.Id)
