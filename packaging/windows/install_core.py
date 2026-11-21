"""Installation engine for the Quarkon Windows build.

The wizard in :mod:`installer_entry` is only a view; every side effect lives here
so the same code path can be driven by:

* the graphical wizard (default),
* IT deployment - ``QuarkonSetup.exe --silent --dir D:\\Apps\\Quarkon``,
* the unit tests - pure helpers are platform independent.

What an install does (all per-user, no administrator rights):

1. pre-flight checks (target is writable, enough free space, no partially
   extracted previous attempt),
2. unpack the payload that the setup executable carries,
3. write ``install.json`` - the record ``quarkon doctor`` and the uninstaller read,
4. write a PowerShell uninstaller, register the app under
   ``HKCU\\...\\Uninstall\\Quarkon`` so it appears in *Apps & features*,
5. create desktop / Start-menu shortcuts that point at ``Quarkon.exe``.

User projects (``%USERPROFILE%\\Quarkon``) and logs
(``%LOCALAPPDATA%\\Quarkon\\logs``) are outside the installation directory, so an
upgrade or uninstall never destroys work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

__all__ = [
    "APP_EXE",
    "APP_NAME",
    "MIN_FREE_SPACE_BYTES",
    "PUBLISHER",
    "UNINSTALL_KEY",
    "InstallOptions",
    "InstallResult",
    "InstalledInfo",
    "Step",
    "create_shortcuts",
    "default_data_dir",
    "default_install_dir",
    "detect_installations",
    "format_bytes",
    "install",
    "installed_info",
    "is_app_running",
    "launch_app",
    "read_install_record",
    "registry_values",
    "run_powershell",
    "stop_app",
    "uninstall",
    "uninstall_script",
    "write_registry",
]

APP_NAME: Final[str] = "Quarkon"
APP_EXE: Final[str] = "Quarkon.exe"
PUBLISHER: Final[str] = "Quarkon Maintainers"
HOMEPAGE: Final[str] = "https://github.com/Kdjkvv7886/quarkon"
UNINSTALL_KEY: Final[str] = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Quarkon"
#: A little headroom so the payload always fits even with a nearly full disk.
MIN_FREE_SPACE_BYTES: Final[int] = 120 * 1024 * 1024
_CREATE_NO_WINDOW: Final[int] = 0x08000000 if sys.platform == "win32" else 0

ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class Step:
    """One user-visible installation step."""

    key: str
    title: str


STEPS: Final[tuple[Step, ...]] = (
    Step("check", "Checking this computer"),
    Step("files", "Copying program files"),
    Step("record", "Writing installation record"),
    Step("shortcuts", "Creating shortcuts"),
    Step("registry", "Registering with Windows"),
)


@dataclass
class InstallOptions:
    """Everything the wizard asks for; defaults target a normal desktop."""

    install_dir: Path
    version: str
    desktop_shortcut: bool = True
    start_menu_shortcut: bool = True
    launch_after_install: bool = True
    silent: bool = False
    log: LogCallback | None = None

    def emit(self, message: str, level: str = "info") -> None:
        if self.log is not None:
            self.log(message, level)


@dataclass
class InstallResult:
    """Outcome of :func:`install`; ``ok`` is the single thing the UI checks."""

    ok: bool
    install_dir: Path | None = None
    message: str = ""
    hint: str = ""
    details: dict[str, object] = field(default_factory=dict)
    #: Non-fatal problems (for example a shortcut blocked by policy). The install
    #: is still usable, so the wizard shows these on the finish page instead of
    #: failing.
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InstalledInfo:
    """A previous installation discovered on this machine."""

    install_dir: Path
    version: str
    installed_at: str = ""

    @property
    def exe(self) -> Path:
        return self.install_dir / APP_EXE


# --------------------------------------------------------------------- helpers
def format_bytes(size: int) -> str:
    """Human-readable byte count with decimal units (matches Windows Explorer)."""
    value = float(max(size, 0))
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            if unit == "bytes":
                return f"{int(value)} bytes"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} GB"


def default_install_dir() -> Path:
    """Per-user program folder; no admin rights and no UAC prompt."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "Programs" / APP_NAME


def default_data_dir() -> Path:
    """Logs and preferences; kept separate from the uninstallable program folder."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / APP_NAME


def projects_dir() -> Path:
    """Where finished projects live - never touched by upgrade or uninstall."""
    return Path.home() / APP_NAME


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _free_space(path: Path) -> int:
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return 0
        probe = parent
    try:
        return int(shutil.disk_usage(str(probe)).free)
    except OSError:
        return 0


def _run_no_window(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a helper process without flashing a console window."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
        check=False,
    )


def run_powershell(script: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Execute ``script`` with Windows PowerShell, hidden, and return the result."""
    executable = "powershell.exe"
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if candidate.is_file():
            executable = str(candidate)
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as handle:
        handle.write(script)
        script_path = Path(handle.name)
    try:
        return _run_no_window(
            [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-WindowStyle",
                "Hidden",
                "-File",
                str(script_path),
            ],
            timeout=timeout,
        )
    finally:
        script_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ extraction
def _payload_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [info for info in archive.infolist() if not info.is_dir()]


def payload_uncompressed_size(payload: Path) -> int:
    """Total size after extraction, used for the free-space pre-flight."""
    try:
        with zipfile.ZipFile(payload) as archive:
            return sum(info.file_size for info in _payload_members(archive))
    except (OSError, zipfile.BadZipFile):
        return 0


def _safe_extract(payload: Path, target: Path) -> int:
    """Extract ``payload`` into ``target``, rejecting path traversal. Returns file count."""
    target = target.resolve()
    count = 0
    with zipfile.ZipFile(payload) as archive:
        for info in _payload_members(archive):
            destination = (target / info.filename).resolve()
            if not str(destination).startswith(str(target)):
                raise ValueError(f"payload entry escapes the install folder: {info.filename}")
            _ensure_parent(destination)
            with archive.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            count += 1
    return count


# ---------------------------------------------------------------- registry/PS
def registry_values(version: str, install_dir: Path, size_bytes: int) -> dict[str, object]:
    """Values written under ``HKCU`` so the app shows in *Apps & features*."""
    powershell = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    uninstall_script = install_dir / "uninstall.ps1"
    return {
        "DisplayName": APP_NAME,
        "DisplayVersion": version,
        "Publisher": PUBLISHER,
        "DisplayIcon": str(install_dir / APP_EXE),
        "InstallLocation": str(install_dir),
        "UninstallString": f'"{powershell}" -NoProfile -ExecutionPolicy Bypass -File "{uninstall_script}"',
        "QuietUninstallString": f'"{powershell}" -NoProfile -ExecutionPolicy Bypass -File "{uninstall_script}" -Silent',
        "Comments": "Local distillation of reasoning models",
        "URLInfoAbout": HOMEPAGE,
        "EstimatedSize": max(1, size_bytes // 1024),
        "NoModify": 1,
        "NoRepair": 1,
    }


def write_registry(values: dict[str, object]) -> None:
    """Write the uninstall entry for the current user (no-op outside Windows)."""
    if sys.platform != "win32":
        return
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY, 0, winreg.KEY_WRITE) as key:
        for name, value in values.items():
            if isinstance(value, int):
                winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
            else:
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(value))


def delete_registry() -> None:
    """Remove the uninstall entry for the current user."""
    if sys.platform != "win32":
        return
    import winreg

    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except OSError:
        # Missing key or a policy that forbids the write - both are non-fatal.
        return


def shortcut_paths() -> dict[str, Path]:
    """Desktop and Start-menu locations for the shortcuts we may create."""
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    appdata = os.environ.get("APPDATA")
    programs = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" if appdata else Path.home()
    return {
        "desktop": desktop / f"{APP_NAME}.lnk",
        "start_menu_dir": programs / APP_NAME,
        "start_menu": programs / APP_NAME / f"{APP_NAME}.lnk",
        "uninstall": programs / APP_NAME / f"Uninstall {APP_NAME}.lnk",
    }


def _shortcut_script(install_dir: Path, *, desktop: bool, start_menu: bool) -> str:
    """PowerShell that creates the .lnk files through the Windows shell API."""
    paths = shortcut_paths()
    exe = install_dir / APP_EXE
    lines = ["$ErrorActionPreference = 'Stop'", "$shell = New-Object -ComObject WScript.Shell"]
    targets: list[tuple[str, str, str]] = []
    if desktop:
        targets.append((str(paths["desktop"]), "", "Open the Quarkon interface"))
    if start_menu:
        targets.append((str(paths["start_menu"]), "", "Open the Quarkon interface"))
        targets.append(
            (
                str(paths["uninstall"]),
                f'-NoProfile -ExecutionPolicy Bypass -File "{install_dir / "uninstall.ps1"}"',
                f"Remove {APP_NAME} from this computer",
            )
        )
    for link, arguments, description in targets:
        lines.extend(
            [
                f"$link = '{link}'",
                "$folder = Split-Path -Parent $link",
                "if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }",
                "$shortcut = $shell.CreateShortcut($link)",
                f"$shortcut.TargetPath = '{exe}'",
                f"$shortcut.Arguments = '{arguments}'",
                f"$shortcut.WorkingDirectory = '{install_dir}'",
                f"$shortcut.IconLocation = '{exe},0'",
                f"$shortcut.Description = '{description}'",
                "$shortcut.Save()",
            ]
        )
    return "\n".join(lines) + "\n"


def create_shortcuts(install_dir: Path, *, desktop: bool, start_menu: bool) -> list[Path]:
    """Create the requested shortcuts and return the files that exist afterwards."""
    if sys.platform != "win32" or not (desktop or start_menu):
        return []
    result = run_powershell(_shortcut_script(install_dir, desktop=desktop, start_menu=start_menu))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "PowerShell could not create shortcuts")
    paths = shortcut_paths()
    created: list[Path] = []
    for key in ("desktop", "start_menu", "uninstall"):
        path = paths[key]
        if path.is_file():
            created.append(path)
    if desktop and paths["desktop"] not in created:
        raise RuntimeError("the desktop shortcut was not created")
    if start_menu and paths["start_menu"] not in created:
        raise RuntimeError("the Start-menu shortcut was not created")
    return created


# ------------------------------------------------------------- install record
def _record_path(install_dir: Path) -> Path:
    return install_dir / "install.json"


def read_install_record(install_dir: Path) -> InstalledInfo | None:
    """Read ``install.json``; ``None`` when the folder is not a Quarkon install."""
    record = _record_path(install_dir)
    if not record.is_file():
        return None
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = str(payload.get("version") or "")
    if not version:
        version = "unknown"
    exe = install_dir / APP_EXE
    if not exe.is_file():
        return None
    return InstalledInfo(
        install_dir=install_dir,
        version=version,
        installed_at=str(payload.get("installed_at") or ""),
    )


def installed_info() -> InstalledInfo | None:
    """Return the default install if this machine has one."""
    return read_install_record(default_install_dir())


def detect_installations() -> list[InstalledInfo]:
    """Find existing installs in the default folder and in the registry."""
    candidates: list[Path] = [default_install_dir()]
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
                location, _ = winreg.QueryValueEx(key, "InstallLocation")
                if location:
                    candidates.append(Path(str(location)))
        except (FileNotFoundError, OSError, ImportError):
            pass
    found: dict[str, InstalledInfo] = {}
    for candidate in candidates:
        info = read_install_record(candidate)
        if info is not None:
            found[str(candidate).lower()] = info
    return list(found.values())


def _write_record(options: InstallOptions, files: int, size_bytes: int) -> None:
    payload = {
        "schema": 1,
        "product": APP_NAME,
        "version": options.version,
        "installed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "install_dir": str(options.install_dir),
        "files": files,
        "size_bytes": size_bytes,
        "options": {
            "desktop_shortcut": options.desktop_shortcut,
            "start_menu_shortcut": options.start_menu_shortcut,
        },
    }
    _record_path(options.install_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- uninstaller
def uninstall_script(install_dir: Path, version: str, *, desktop: bool, start_menu: bool) -> str:
    """PowerShell uninstaller registered under *Apps & features*.

    The script asks for confirmation (unless ``-Silent``), copies itself to
    ``%TEMP%`` so it can delete the folder it lives in, stops a running Quarkon,
    removes shortcuts and the registry entry, and - only when the user asks for
    it - the project folder. Logs and preferences under ``%LOCALAPPDATA%`` are
    removed with the program; user projects are kept by default.
    """
    paths = shortcut_paths()
    keep_projects = projects_dir()
    lines = [
        f"# {APP_NAME} {version} uninstaller. Generated by the setup; safe to delete.",
        "param([switch]$Silent, [switch]$FromTemp)",
        "$ErrorActionPreference = 'SilentlyContinue'",
        f"$installDir = '{install_dir}'",
        f"$desktopShortcut = '{paths['desktop']}'",
        f"$startMenuDir = '{paths['start_menu_dir']}'",
        f"$projectsDir = '{keep_projects}'",
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null",
        "if (-not $FromTemp) {",
        "    if (-not $Silent) {",
        "        $answer = [System.Windows.Forms.MessageBox]::Show(",
        f"            'Remove {APP_NAME} {version} from this computer?`n`nYour projects stay in ' + $projectsDir,",
        "            'Uninstall Quarkon', 'YesNo', 'Question')",
        "        if ($answer -ne 'Yes') { exit 0 }",
        "        $answer2 = [System.Windows.Forms.MessageBox]::Show(",
        "            'Also delete your Quarkon projects and logs?`n`nChoose No to keep your work.',",
        "            'Keep your projects?', 'YesNo', 'Warning')",
        "        $purgeProjects = ($answer2 -eq 'Yes')",
        "    } else {",
        "        $purgeProjects = $false",
        "    }",
        "    $self = $MyInvocation.MyCommand.Path",
        "    $copy = Join-Path $env:TEMP ('quarkon-uninstall-' + [guid]::NewGuid().ToString('N') + '.ps1')",
        "    Copy-Item -LiteralPath $self -Destination $copy -Force",
        "    $arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File',('\"' + $copy + '\"'),'-FromTemp')",
        "    if ($Silent) { $arguments += '-Silent' }",
        "    if ($purgeProjects) { $arguments += '-PurgeProjects' }",
        "    Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments | Out-Null",
        "    exit 0",
        "}",
        "Start-Sleep -Milliseconds 900",
        "Get-Process -Name 'Quarkon' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue",
        "Remove-Item -LiteralPath $desktopShortcut -Force",
        "Remove-Item -LiteralPath $startMenuDir -Recurse -Force",
        f"Remove-Item -LiteralPath 'HKCU:\\{UNINSTALL_KEY}' -Recurse -Force",
        f"Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA '{APP_NAME}') -Recurse -Force",
        "if ($PurgeProjects) {",
        "    Remove-Item -LiteralPath $projectsDir -Recurse -Force",
        "}",
        "Remove-Item -LiteralPath $installDir -Recurse -Force",
        "Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- install
def _clear_install_dir(target: Path, log: LogCallback | None) -> None:
    """Remove a previous Quarkon copy while refusing to delete foreign folders."""
    if not target.exists():
        return
    if not target.is_dir():
        raise RuntimeError(f"{target} exists and is not a folder")
    entries = list(target.iterdir())
    if not entries:
        return
    if (
        not (target / APP_EXE).exists()
        and not (target / "install.json").exists()
        and not (target / ".quarkon-setup").exists()
    ):
        raise RuntimeError(
            f"{target} already contains files that do not belong to {APP_NAME}; "
            "choose another folder"
        )
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=False)
        else:
            entry.unlink()
    if log is not None:
        log(f"Removed the previous version from {target}", "dim")


def install(
    options: InstallOptions,
    payload: Path,
    *,
    progress: ProgressCallback | None = None,
) -> InstallResult:
    """Install from ``payload`` (a zip produced by the build) into ``options.install_dir``."""
    target = options.install_dir.expanduser()
    log = options.log

    def report(fraction: float, message: str, level: str = "info") -> None:
        if progress is not None:
            progress(fraction, message)
        if log is not None:
            log(message, level)

    try:
        # ---------------------------------------------------------- pre-flight
        report(0.03, "Checking the target folder", "info")
        if payload.is_file() is False:
            raise FileNotFoundError(f"the setup payload is missing: {payload}")
        _ensure_parent(target)
        if target.exists() and not target.is_dir():
            raise RuntimeError(f"{target} is a file, not a folder")
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".quarkon-write-test"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(f"cannot write to {target}: {exc}") from exc
        required = payload_uncompressed_size(payload) + MIN_FREE_SPACE_BYTES
        free = _free_space(target)
        if free and free < required:
            raise RuntimeError(
                f"not enough free space: {format_bytes(free)} available, "
                f"{format_bytes(required)} needed"
            )
        report(0.10, f"Target folder is writable ({format_bytes(free)} free)", "ok")

        # -------------------------------------------------------------- files
        report(0.16, "Removing the previous version", "info")
        _clear_install_dir(target, log)
        # Marker: a failed extraction leaves it behind, so the retry is allowed to
        # clear the folder even though the app executable is not there yet.
        (target / ".quarkon-setup").write_text(f"installing {options.version}\n", encoding="utf-8")
        report(0.24, "Copying program files", "info")
        count = _safe_extract(payload, target)
        size_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        if not (target / APP_EXE).is_file():
            raise RuntimeError(f"{APP_EXE} is missing from the payload")
        report(0.62, f"{count} files copied ({format_bytes(size_bytes)})", "ok")

        # ------------------------------------------------------------- record
        report(0.66, "Writing the installation record", "info")
        _write_record(options, count, size_bytes)
        script = uninstall_script(
            target,
            options.version,
            desktop=options.desktop_shortcut,
            start_menu=options.start_menu_shortcut,
        )
        (target / "uninstall.ps1").write_text(script, encoding="utf-8-sig")
        report(0.72, "Uninstaller written", "ok")

        # ---------------------------------------------------------- shortcuts
        shortcuts: list[str] = []
        warnings: list[str] = []
        if sys.platform == "win32" and (options.desktop_shortcut or options.start_menu_shortcut):
            report(0.78, "Creating shortcuts", "info")
            try:
                shortcuts = [
                    str(path)
                    for path in create_shortcuts(
                        target, desktop=options.desktop_shortcut, start_menu=options.start_menu_shortcut
                    )
                ]
            except (RuntimeError, OSError) as exc:
                warnings.append(f"Windows did not allow the shortcuts to be created: {exc}")
                report(0.80, f"Shortcuts skipped: {exc}", "warn")
            for path in shortcuts:
                report(0.86, f"Shortcut: {Path(path).name}", "dim")

        # ----------------------------------------------------------- registry
        if sys.platform == "win32":
            report(0.92, "Registering with Windows", "info")
            try:
                write_registry(registry_values(options.version, target, size_bytes))
            except OSError as exc:
                warnings.append(
                    "Quarkon could not add itself to Windows Settings > Apps "
                    f"({exc}). The application works; remove it with uninstall.ps1."
                )
                report(0.94, f"Windows uninstall entry skipped: {exc}", "warn")

        (target / ".quarkon-setup").unlink(missing_ok=True)
        report(1.0, f"{APP_NAME} {options.version} is installed", "ok")
        return InstallResult(
            ok=True,
            install_dir=target,
            message=f"{APP_NAME} {options.version} is installed",
            details={"files": count, "size_bytes": size_bytes, "shortcuts": shortcuts},
            warnings=warnings,
        )
    except (RuntimeError, OSError, ValueError, zipfile.BadZipFile) as exc:
        message = str(exc)
        hint = "Close any open Quarkon window and try again, or pick another folder."
        if "free space" in message:
            hint = "Free some disk space, or choose a folder on another drive."
        if "belong to" in message:
            hint = "Choose an empty folder or the folder of a previous Quarkon install."
        report(0.0, f"Installation failed: {message}", "error")
        return InstallResult(ok=False, install_dir=target, message=message, hint=hint)


# ------------------------------------------------------------------- uninstall
def uninstall(*, install_dir: Path | None = None, silent: bool = False, log: LogCallback | None = None) -> InstallResult:
    """Remove shortcuts, registry entry and the program folder."""
    target = install_dir or default_install_dir()
    info = read_install_record(target)
    try:
        if info is None and not target.exists():
            return InstallResult(ok=True, message="Quarkon is not installed")

        def say(message: str, level: str = "info") -> None:
            if log is not None:
                log(message, level)

        say(f"Removing {target}", "info")
        paths = shortcut_paths()
        for key in ("desktop",):
            path = paths[key]
            if path.is_file():
                path.unlink(missing_ok=True)
                say(f"Removed {path.name}", "dim")
        if paths["start_menu_dir"].is_dir():
            shutil.rmtree(paths["start_menu_dir"], ignore_errors=True)
            say("Removed the Start-menu folder", "dim")
        delete_registry()
        data_dir = default_data_dir()
        if data_dir.is_dir():
            shutil.rmtree(data_dir, ignore_errors=True)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=False)
        say("Quarkon removed; projects in %USERPROFILE%\\Quarkon were kept", "ok")
        return InstallResult(ok=True, message="Quarkon was removed from this computer")
    except OSError as exc:
        return InstallResult(
            ok=False,
            message=str(exc),
            hint="Close Quarkon (and any open project) and run the uninstaller again.",
        )


# ---------------------------------------------------------------------- launch
def is_app_running() -> bool:
    """True when a Quarkon window/process is alive (the installer must close it)."""
    if sys.platform != "win32":
        return False
    result = _run_no_window(["tasklist", "/FI", "IMAGENAME eq " + APP_EXE, "/NH"])
    return APP_EXE.lower() in result.stdout.lower()


def stop_app() -> bool:
    """Ask Windows to close a running Quarkon; returns True when it is gone."""
    if sys.platform != "win32":
        return False
    result = _run_no_window(["taskkill", "/IM", APP_EXE, "/F"])
    return result.returncode == 0


def launch_app(install_dir: Path) -> bool:
    """Start ``Quarkon.exe`` detached; returns False when the file is missing."""
    exe = install_dir / APP_EXE
    if not exe.is_file():
        return False
    starter = getattr(os, "startfile", None)
    if starter is None:
        return False
    starter(str(exe))
    return True
