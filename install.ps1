# Quarkon installer for Windows.
#
# Installs Quarkon into a private virtual environment, creates desktop and
# Start-menu shortcuts, and opens the graphical wizard. Run from a normal
# PowerShell window (no administrator rights required):
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# The script never downloads anything except Python package dependencies, and it
# writes only inside %LOCALAPPDATA%\Quarkon.

param(
    [switch]$NoLaunch,
    [switch]$NoShortcuts
)

$ErrorActionPreference = "Stop"
$AppDir = Join-Path $env:LOCALAPPDATA "Quarkon"
$VenvDir = Join-Path $AppDir ".venv"
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Write-Ok($message)   { Write-Host "    $message" -ForegroundColor Green }
function Fail($message) {
    Write-Host "Problem: $message" -ForegroundColor Red
    Write-Host "Run 'quarkon doctor' after fixing the problem, or ask your IT team." -ForegroundColor Yellow
    exit 1
}

Write-Step "Checking Python"
$python = $null
foreach ($candidate in @("python", "py")) {
    try {
        $version = & $candidate -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) {
            $parts = $version.Split(".")
            if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) { $python = $candidate; break }
        }
    } catch { }
}
if (-not $python) {
    Fail "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and tick 'Add Python to PATH'."
}
Write-Ok "Python found: $python"

Write-Step "Creating application folder"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
Write-Ok $AppDir

if (-not (Test-Path $VenvDir)) {
    Write-Step "Creating a private environment (this takes a minute)"
    & $python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
$venvQuarkon = Join-Path $VenvDir "Scripts\quarkon.exe"

Write-Step "Installing Quarkon (no administrator rights needed)"
& $venvPython -m pip install --upgrade pip --quiet
if (Test-Path (Join-Path $SourceDir "pyproject.toml")) {
    & $venvPython -m pip install --quiet $SourceDir
} else {
    & $venvPython -m pip install --quiet quarkon
}
if ($LASTEXITCODE -ne 0) { Fail "Installation failed. Check the internet connection or the company proxy." }
Write-Ok "Installed"

if (-not $NoShortcuts) {
    Write-Step "Creating shortcuts"
    $shell = New-Object -ComObject WScript.Shell
    $target = $venvQuarkon
    $arguments = "launcher"
    $desktop = [Environment]::GetFolderPath("Desktop")
    $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    foreach ($folder in @($desktop, $startMenu)) {
        if (-not (Test-Path $folder)) { continue }
        $link = Join-Path $folder "Quarkon.lnk"
        $shortcut = $shell.CreateShortcut($link)
        $shortcut.TargetPath = $target
        $shortcut.Arguments = $arguments
        $shortcut.WorkingDirectory = $AppDir
        $shortcut.Description = "Quarkon - turn a reasoning model into a local model"
        $shortcut.Save()
    }
    Write-Ok "Shortcuts created"
}

if (-not $NoLaunch) {
    Write-Step "Starting Quarkon"
    & $venvQuarkon launcher
} else {
    Write-Host ""
    Write-Host "Installation complete. Start Quarkon from the desktop shortcut," -ForegroundColor Green
    Write-Host "or run: $venvQuarkon launcher" -ForegroundColor Green
}
