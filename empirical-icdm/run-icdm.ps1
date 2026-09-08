# run-icdm.ps1 -- regenerate the ICDM camera-ready empirical results.
#
# Runs Ames Housing first (small, and the only regression dataset, so it
# exercises the rescaling-wrapper path early), then Adult.
#
# Both stages are resumable: compute_ames_all_specs writes a partial parquet
# after each (spec, model) cell and skips cells already present, so re-running
# this script after an interruption continues rather than starting over.
#
# SHAP and PDP are not recomputed. Neither uses the bootstrap, so neither is
# affected by the refit defects being corrected here; the ICDM-era parquets are
# already in place and the drivers skip those stages when they exist.
#
# To stop:  Get-Process python | Stop-Process -Force

$ErrorActionPreference = 'Continue'

$py  = 'C:\Users\jason\venvs\marginfx\Scripts\python.exe'
$emp = 'C:\Users\jason\Dropbox\projects\marginfx-paper\empirical-icdm'
$log = Join-Path $emp 'icdm.log'

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONUNBUFFERED = '1'

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Invoke-Stage($name, $workdir, $script) {
    $out = Join-Path $emp "$name.log"
    $err = Join-Path $emp "$name.err.log"
    Say "$name : start"
    $proc = Start-Process -FilePath $py -ArgumentList @('-u', $script) `
        -WorkingDirectory $workdir `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru
    $null = $proc.Handle   # retain the handle, else ExitCode comes back null
    try { $proc.PriorityClass = 'BelowNormal' } catch { }

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
    if ($code -eq 0) { Say "$name : done (exit 0)" }
    else { Say "$name : FAILED (exit $code) -- see $name.err.log" }
    return $code
}

Say '============ ICDM camera-ready run start ============'

$failed = @()
if ((Invoke-Stage 'ames' (Join-Path $emp 'ames_housing') 'analyze_ames_housing.py') -ne 0) {
    $failed += 'ames'
}
if ((Invoke-Stage 'adult' (Join-Path $emp 'adult') 'analyze_adult.py') -ne 0) {
    $failed += 'adult'
}

if ($failed.Count -gt 0) {
    Say ("=========== ended with FAILURES: {0} ===========" -f ($failed -join ', '))
} else {
    Say '============ ICDM camera-ready run done ============'
}
