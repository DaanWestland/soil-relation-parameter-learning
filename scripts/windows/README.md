# Windows PowerShell helper

`run_with_retry.ps1` runs `python -m pedopilot run <config>` up to N times (finished configurations are
skipped, failed ones retried), with the project's `.venv` interpreter. It is meant to run as a separate
process that writes a log file.

Usage (from the repository root):

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
Start-Process powershell -WorkingDirectory (Get-Location) -WindowStyle Minimized `
  -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\windows\run_with_retry.ps1','-Run','gpu_full' `
  -RedirectStandardOutput logs\gpu_full.log -RedirectStandardError logs\gpu_full.err.log
```

The last log line is `run_with_retry: gpu_full finished with exit code <n>`; 0 means every configuration
is done. On Linux or macOS the equivalent is a shell loop (see README.md, Compute).
