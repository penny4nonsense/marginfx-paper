# run-sim3-uniform.ps1 -- Simulation 3, bounded-support arm.
#
# The Gaussian arms satisfy the moment-condition assumption but not the
# density-bounded one: a Gaussian density is not bounded below and the
# representer e^{-h^2/2} sinh(h u_j)/h is unbounded. This arm draws the
# covariates uniformly on a box, where the density is constant, the representer
# is the bounded step function +/- 1/(2h) on the two boundary shells, and the
# trimming weight is active rather than vacuous. It therefore exhibits the
# theorems under their literal hypotheses.
#
# One design (linear_uniform) x two outcome types x five sample sizes x four
# learners = 40 cells. The closed arm's 120 cells took about 19 hours, so
# expect roughly 6 to 7 hours here.
#
# Resumable: finished cells are skipped. Writes to sim3_debiased/results_uniform.
#
# To stop:
#   Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#       Where-Object { $_.CommandLine -like '*run-sim3-uniform*' } |
#       ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
#   Get-Process python | Stop-Process -Force

$ErrorActionPreference = 'Continue'

$py  = 'C:\Users\jason\venvs\marginfx\Scripts\python.exe'
$sim = 'C:\Users\jason\Dropbox\projects\marginfx-paper\simulations'
$wd  = Join-Path $sim 'sim3_debiased'
$log = Join-Path $sim 'sim3-uniform.log'

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONUNBUFFERED = '1'

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

Say '============ sim3 uniform arm start ============'

$proc = Start-Process -FilePath $py `
    -ArgumentList @('-u', 'run_debiased.py', '--riesz', 'uniform',
                    '--outcome', 'both') `
    -WorkingDirectory $wd `
    -RedirectStandardOutput (Join-Path $sim 'sim3-uniform.out.log') `
    -RedirectStandardError  (Join-Path $sim 'sim3-uniform.err.log') `
    -WindowStyle Hidden -PassThru

# Retain the handle, else ExitCode reads back null once the process ends.
$null = $proc.Handle
try { $proc.PriorityClass = 'BelowNormal' } catch { }

Write-Host ("sim3 uniform arm launched, PID {0}" -f $proc.Id)

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
if ($code -eq 0) { Say 'sim3 uniform : done (exit 0)' }
else { Say "sim3 uniform : FAILED (exit $code) -- see sim3-uniform.err.log" }
