[CmdletBinding()]
param(
    # Pass "upload" to the publish/build steps that support it.
    [switch]$Upload,
    # How often the background resource sampler records RAM / top processes.
    [int]$ResourceSampleSeconds = 15
)

# Orchestrates the Windows release/CI steps with heavy logging so failures are
# diagnosable even when the SSH connection to the box drops (which is what
# happens when the parallel MSVC build thrashes/OOMs: the compiler dies with a
# code like MSB6006 "CL.exe exited with code 4" and the VM stops responding,
# so everything printed after that point is lost on the client side).
#
# Everything is written to a per-line-timestamped log file *on the VM* under
# build-logs/ (git-ignored via the build-*/ rule), plus a separate resource log
# sampled from a background job. After a disconnect, scp those files off the box
# to see exactly where and when it died.

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot 'build-logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "windows-ci-$stamp.log"
$resourceLog = Join-Path $logDir "windows-ci-$stamp.resources.log"

# AutoFlush so the last lines survive an abrupt disconnect / freeze.
$script:writer = [System.IO.StreamWriter]::new($logFile, $true)
$script:writer.AutoFlush = $true

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $Level, $Message
    [Console]::Out.WriteLine($line)
    $script:writer.WriteLine($line)
}

function Get-MemorySnapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeMB = [math]::Round($os.FreePhysicalMemory / 1024)
    $totalMB = [math]::Round($os.TotalVisibleMemorySize / 1024)
    $usedMB = $totalMB - $freeMB
    $freeVirtMB = [math]::Round($os.FreeVirtualMemory / 1024)
    return "RAM used ${usedMB}/${totalMB} MB (free ${freeMB} MB), free virtual ${freeVirtMB} MB"
}

Write-Log "================ Windows CI run START ================"
Write-Log "Log file:     $logFile"
Write-Log "Resource log: $resourceLog"
Write-Log ("Host: {0}  User: {1}  CWD: {2}" -f $env:COMPUTERNAME, $env:USERNAME, (Get-Location).Path)
try {
    $cs = Get-CimInstance Win32_ComputerSystem
    Write-Log ("Logical CPUs: {0}  Physical RAM: {1} MB" -f $env:NUMBER_OF_PROCESSORS, [math]::Round($cs.TotalPhysicalMemory / 1MB))
} catch { Write-Log "Could not read computer system info: $_" 'WARN' }
Write-Log ("Initial memory: {0}" -f (Get-MemorySnapshot))

# Background sampler: records memory + the three heaviest processes on an
# interval to an always-flushed log. If the box thrashes/freezes, the final
# samples on disk reveal the trajectory into the failure.
$sampler = Start-Job -ScriptBlock {
    param($path, $intervalSec)
    $sw = [System.IO.StreamWriter]::new($path, $true)
    $sw.AutoFlush = $true
    try {
        while ($true) {
            $os = Get-CimInstance Win32_OperatingSystem
            $freeMB = [math]::Round($os.FreePhysicalMemory / 1024)
            $totalMB = [math]::Round($os.TotalVisibleMemorySize / 1024)
            $usedMB = $totalMB - $freeMB
            $freeVirtMB = [math]::Round($os.FreeVirtualMemory / 1024)
            $top = Get-Process |
                Sort-Object WorkingSet64 -Descending |
                Select-Object -First 3 |
                ForEach-Object { '{0}({1})={2}MB' -f $_.ProcessName, $_.Id, [math]::Round($_.WorkingSet64 / 1MB) }
            $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss.fff')
            $sw.WriteLine("[$ts] RAM used ${usedMB}/${totalMB} MB, free virt ${freeVirtMB} MB | top: $($top -join ', ')")
            Start-Sleep -Seconds $intervalSec
        }
    } finally {
        $sw.Dispose()
    }
} -ArgumentList $resourceLog, $ResourceSampleSeconds

$uploadArg = ''
if ($Upload) { $uploadArg = 'upload' }

$steps = @(
    [pscustomobject]@{ Name = 'test-core'; Command = 'scripts\test-core.bat' }
    [pscustomobject]@{ Name = 'publish-binary'; Command = ('scripts\publish-binary.bat ' + $uploadArg).Trim() }
    [pscustomobject]@{ Name = 'publish-examples'; Command = ('scripts\publish-examples.bat ' + $uploadArg).Trim() }
    [pscustomobject]@{ Name = 'build-pip'; Command = ('scripts\build-pip.bat ' + $uploadArg).Trim() }
)

$overall = 0
foreach ($step in $steps) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Log "---------------- STEP START: $($step.Name)  [$($step.Command)] ----------------"
    Write-Log ("memory before {0}: {1}" -f $step.Name, (Get-MemorySnapshot))

    # Run the batch step, merging stderr, and tee every line to the log with a
    # per-line timestamp while still streaming to the console (SSH).
    & cmd.exe /c $step.Command 2>&1 | ForEach-Object {
        $line = [string]$_
        $ts = (Get-Date -Format 'HH:mm:ss.fff')
        [Console]::Out.WriteLine($line)
        $script:writer.WriteLine("[$ts] $line")
    }
    $code = $LASTEXITCODE
    $sw.Stop()

    Write-Log ("memory after {0}: {1}" -f $step.Name, (Get-MemorySnapshot))
    if ($code -ne 0) {
        Write-Log "STEP FAILED: $($step.Name) exited with code $code after $($sw.Elapsed.ToString())" 'ERROR'
        $overall = $code
        break
    }
    Write-Log "STEP OK: $($step.Name) in $($sw.Elapsed.ToString())"
}

Stop-Job $sampler -ErrorAction SilentlyContinue | Out-Null
Remove-Job $sampler -Force -ErrorAction SilentlyContinue | Out-Null

if ($overall -eq 0) {
    Write-Log "================ Windows CI run SUCCESS ================"
} else {
    Write-Log "================ Windows CI run FAILED (exit $overall) ================" 'ERROR'
    Write-Log "Full log:         $logFile"
    Write-Log "Resource samples: $resourceLog"
}

$script:writer.Dispose()
exit $overall
