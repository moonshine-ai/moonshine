[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Asset,
    # Per-attempt wall-clock timeout. `gh release upload` has no client-side
    # timeout, so a stalled TLS connection to GitHub can leave it blocked
    # forever (see the ~22 min hang observed on the Windows release box). A
    # 31 MB asset uploads in seconds on a healthy link, so a few minutes is
    # generous headroom.
    [int]$TimeoutSec = 180,
    [int]$Retries = 5
)

$ErrorActionPreference = 'Stop'
$tag = "v$Version"

# Recursively terminate a process and any children it spawned so a stalled
# upload can't leave orphaned handles on the asset before the next attempt.
function Stop-ProcessTree {
    param([int]$ProcessId)
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree -ProcessId $_.ProcessId }
    try { Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue } catch {}
}

for ($attempt = 1; $attempt -le $Retries; $attempt++) {
    Write-Host "[gh-upload-retry] Attempt $attempt/$Retries: uploading '$Asset' to release $tag (timeout ${TimeoutSec}s)..."

    $proc = Start-Process -FilePath 'gh' `
        -ArgumentList @('release', 'upload', $tag, $Asset, '--clobber') `
        -NoNewWindow -PassThru
    # Touch .Handle so the object caches it; otherwise .ExitCode can come back
    # $null after the process exits when Start-Process is used without -Wait.
    $null = $proc.Handle

    if ($proc.WaitForExit($TimeoutSec * 1000)) {
        if ($proc.ExitCode -eq 0) {
            Write-Host "[gh-upload-retry] Upload succeeded on attempt $attempt."
            exit 0
        }
        Write-Host "[gh-upload-retry] Attempt $attempt failed with exit code $($proc.ExitCode)."
    }
    else {
        Write-Host "[gh-upload-retry] Attempt $attempt stalled past ${TimeoutSec}s; killing gh (PID $($proc.Id)) and retrying..."
        Stop-ProcessTree -ProcessId $proc.Id
    }

    if ($attempt -lt $Retries) {
        Start-Sleep -Seconds 5
    }
}

Write-Host "[gh-upload-retry] ERROR: upload of '$Asset' to $tag failed after $Retries attempts."
exit 1
