# Installs everything the windows stage of build-all-platforms.sh needs, into a
# guest that create-vm.sh has just brought up. Run it over SSH from the build
# host; it is idempotent, so re-run it after any partial failure.
#
#   ssh pete@moonshine-win "powershell -NoProfile -ExecutionPolicy Bypass -File C:\provision-toolchain.ps1"
#
# Deliberately not part of the unattended install: this is the long, network-
# dependent half, and running it over SSH means its failures are visible instead
# of buried in a log inside a guest nobody can reach yet.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-Step { param([string]$m) Write-Host "`n== $m ==" -ForegroundColor Cyan }

Write-Step 'Chocolatey'
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString(
        'https://community.chocolatey.org/install.ps1'))
    $env:PATH += ';C:\ProgramData\chocolatey\bin'
} else {
    Write-Host 'already installed'
}

Write-Step 'Build tools'
# The VC workload is what supplies the v143 x64 toolset that test-core.bat and
# publish-binary.bat pin explicitly; the rest is what the .bat steps shell out
# to. cmake comes from Chocolatey rather than the one bundled inside Visual
# Studio because the scripts expect it on PATH.
choco install -y --no-progress git cmake gh uv python3 powershell-core
if ($LASTEXITCODE -ne 0) { throw "choco install of base tools failed ($LASTEXITCODE)" }

choco install -y --no-progress visualstudio2022buildtools
if ($LASTEXITCODE -ne 0) { throw "choco install of VS build tools failed ($LASTEXITCODE)" }
choco install -y --no-progress visualstudio2022-workload-vctools `
    --package-parameters '--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621'
if ($LASTEXITCODE -ne 0) { throw "choco install of the VC workload failed ($LASTEXITCODE)" }

Write-Step 'PowerShell 7 as the SSH login shell'
# build-all-platforms.sh invokes pwsh for the CI orchestrator and sends its
# remote command in PowerShell syntax, so the login shell has to be PowerShell.
$pwshPath = 'C:\Program Files\PowerShell\7\pwsh.exe'
if (Test-Path $pwshPath) {
    New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
        -Value $pwshPath -PropertyType String -Force | Out-Null
    Write-Host "default shell -> $pwshPath"
} else {
    Write-Host "pwsh not found at $pwshPath; leaving Windows PowerShell as the shell"
}

Write-Step 'Repository'
# Public over HTTPS, so no credentials are needed to clone or fetch. Only the
# release upload steps need secrets, and those are set up separately.
$repo = Join-Path $env:USERPROFILE 'moonshine'
if (-not (Test-Path $repo)) {
    & 'C:\Program Files\Git\cmd\git.exe' clone `
        https://github.com/moonshine-ai/moonshine $repo
    if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)" }
} else {
    Write-Host 'already cloned'
}

Write-Step 'Done'
Write-Host 'Toolchain versions:'
refreshenv 2>&1 | Out-Null
foreach ($tool in 'git --version', 'cmake --version', 'gh --version', 'uv --version') {
    try {
        $parts = $tool.Split(' ')
        $out = & $parts[0] $parts[1] 2>&1 | Select-Object -First 1
        Write-Host ("  {0,-8} {1}" -f $parts[0], $out)
    } catch {
        Write-Host ("  {0,-8} NOT FOUND" -f $tool.Split(' ')[0])
    }
}
