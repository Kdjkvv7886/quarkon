# Quarkon Windows installer - end-to-end smoke test.
#
# Performs a real per-user installation of a built setup executable, verifies the
# installed files, starts the application in headless self-check mode, checks that
# the loopback interface answered, then uninstalls and verifies the folder is gone.
#
#   pwsh -File packaging/windows/smoke_test.ps1 -SetupPath dist/windows/QuarkonSetup-3.2.0-win64.exe
#
# CI runs this on a clean windows-latest image. The `-SandboxedEnv` switch redirects
# TEMP/LOCALAPPDATA/USERPROFILE/APPDATA into a scratch folder, which is useful on a
# machine where the assistant's process sandbox blocks writes to the real profile.
#
# Exit code 0 = all checks passed.

param(
    [Parameter(Mandatory = $true)]
    [string]$SetupPath,
    [string]$WorkDir = (Join-Path $env:TEMP ("quarkon-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))),
    [switch]$SandboxedEnv
)

$ErrorActionPreference = "Stop"
$Script:Failures = 0

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert([bool]$Condition, [string]$Message) {
    if ($Condition) {
        Write-Host "    ok   $Message" -ForegroundColor Green
    } else {
        Write-Host "    FAIL $Message" -ForegroundColor Red
        $Script:Failures++
    }
}

function Invoke-Exe {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [hashtable]$Environment = @{}
    )
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $FilePath
    $info.Arguments = ($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($key in $Environment.Keys) {
        $info.EnvironmentVariables[$key] = $Environment[$key]
    }
    $process = [System.Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if (-not $process.WaitForExit(180000)) {
        $process.Kill()
        return [pscustomobject]@{ ExitCode = -1; StdOut = $stdout; StdErr = $stderr }
    }
    return [pscustomobject]@{ ExitCode = $process.ExitCode; StdOut = $stdout; StdErr = $stderr }
}

$SetupPath = (Resolve-Path $SetupPath).Path
$Version = ($SetupPath | Split-Path -Leaf) -replace '^QuarkonSetup-', '' -replace '-win64\.exe$', ''
$InstallDir = Join-Path $WorkDir "app"
$Scratch = Join-Path $WorkDir "scratch"
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null

$environment = @{}
if ($SandboxedEnv) {
    $environment["TEMP"] = $Scratch
    $environment["TMP"] = $Scratch
    $environment["LOCALAPPDATA"] = Join-Path $Scratch "Local"
    $environment["APPDATA"] = Join-Path $Scratch "Roaming"
    $environment["USERPROFILE"] = Join-Path $Scratch "Home"
    New-Item -ItemType Directory -Force -Path $environment["LOCALAPPDATA"], $environment["APPDATA"], $environment["USERPROFILE"] | Out-Null
}

Write-Step "Setup $Version"
Write-Host "    work dir: $WorkDir"

Write-Step "Installing without a window"
$install = Invoke-Exe -FilePath $SetupPath -Arguments @("--silent", "--dir", $InstallDir, "--no-launch", "--no-shortcuts") -Environment $environment
Assert ($install.ExitCode -eq 0) "installer exit code 0 (got $($install.ExitCode))"
if ($install.StdErr) { Write-Host $install.StdErr -ForegroundColor DarkGray }

Assert (Test-Path (Join-Path $InstallDir "Quarkon.exe")) "Quarkon.exe installed"
Assert (Test-Path (Join-Path $InstallDir "install.json")) "install.json written"
Assert (Test-Path (Join-Path $InstallDir "uninstall.ps1")) "uninstall.ps1 written"
Assert (-not (Test-Path (Join-Path $InstallDir ".quarkon-setup"))) "installation marker removed"

$record = Get-Content (Join-Path $InstallDir "install.json") -Raw | ConvertFrom-Json
Assert ($record.version -eq $Version) "install record reports version $Version"
Assert ($record.files -gt 100) "payload contains the application ($($record.files) files)"

Write-Step "Rendering every wizard page (headless widget check)"
$uitest = Invoke-Exe -FilePath $SetupPath -Arguments @("--ui-test") -Environment $environment
Assert ($uitest.ExitCode -eq 0) "wizard pages render without Tk errors (got $($uitest.ExitCode))"
$uitestUninstall = Invoke-Exe -FilePath $SetupPath -Arguments @("--ui-test", "--uninstall") -Environment $environment
Assert ($uitestUninstall.ExitCode -eq 0) "uninstall wizard renders (got $($uitestUninstall.ExitCode))"

Write-Step "Starting the installed application in self-check mode"
$check = Invoke-Exe -FilePath (Join-Path $InstallDir "Quarkon.exe") -Arguments @("--check") -Environment $environment
Assert ($check.ExitCode -eq 0) "Quarkon.exe --check exit code 0 (got $($check.ExitCode))"
$localAppData = if ($environment.ContainsKey("LOCALAPPDATA")) { $environment["LOCALAPPDATA"] } else { $env:LOCALAPPDATA }
$logPath = Join-Path $localAppData "Quarkon\logs\app.log"
Assert (Test-Path $logPath) "app.log written"
if (Test-Path $logPath) {
    $lines = Get-Content $logPath | Where-Object { $_ -match '"check"' }
    $last = $lines | Select-Object -Last 1
    Assert ($last -and ($last -match '"ok": true')) "self-check logged ok=true"
    Write-Host "    $last" -ForegroundColor DarkGray
}

Write-Step "Uninstalling"
$uninstall = Invoke-Exe -FilePath $SetupPath -Arguments @("--silent", "--uninstall", "--dir", $InstallDir) -Environment $environment
Assert ($uninstall.ExitCode -eq 0) "uninstaller exit code 0 (got $($uninstall.ExitCode))"
Assert (-not (Test-Path $InstallDir)) "program folder removed"

if ($Script:Failures -gt 0) {
    Write-Host "$($Script:Failures) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "Smoke test passed" -ForegroundColor Green
exit 0
