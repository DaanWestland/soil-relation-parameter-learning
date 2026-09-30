# Run one config with up to N attempts. Finished configurations are skipped and failed ones are retried
# on the next attempt, so this is the unattended version of `python -m pedopilot run <config>`.
# Start it as a separate process that writes to a log file (it keeps running when the shell that
# started it closes):
#
#   New-Item -ItemType Directory -Force logs | Out-Null
#   Start-Process powershell -WorkingDirectory (Get-Location) -WindowStyle Minimized `
#     -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\windows\run_with_retry.ps1','-Run','gpu_quick' `
#     -RedirectStandardOutput logs\gpu_quick.log -RedirectStandardError logs\gpu_quick.err.log
#
# Progress: count results\<config>\meta_*.json (gpu_smoke 3, gpu_quick 50). The last line of the log
# is "run_with_retry: <config> finished with exit code <n>"; 0 means every configuration is done.
# -Folds 0 (or 0,1) restricts the run to those folds, e.g. to finish a fold with a failed configuration.
param([string]$Run = "gpu_quick", [int]$Attempts = 3, [string]$Folds = "")
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)      # scripts\windows\ -> repository root
Set-Location $root
$venv = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venv) { $venv } else { "python" }
$env:PYTHONUNBUFFERED = "1"      # the log file follows the run instead of lagging behind
$env:PYTHONUTF8 = "1"
$extra = if ($Folds) { @("--folds", $Folds) } else { @() }
$code = 1
for ($i = 1; $i -le $Attempts; $i++) {
    Write-Output "run_with_retry: attempt $i of $Attempts, $py -m pedopilot run $Run $extra ($(Get-Date -Format s))"
    & $py -m pedopilot run $Run @extra
    $code = $LASTEXITCODE
    Write-Output "run_with_retry: attempt $i ended with exit code $code ($(Get-Date -Format s))"
    if ($code -eq 0) { break }
}
Write-Output "run_with_retry: $Run finished with exit code $code ($(Get-Date -Format s))"
exit $code
