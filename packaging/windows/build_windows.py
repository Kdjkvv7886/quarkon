"""Build the Windows desktop application and the single-file setup wizard.

Usage (from the repository root, on Windows)::

    python packaging/windows/build_windows.py

Outputs (``dist/windows/`` unless ``--out`` is given)::

    QuarkonSetup-<version>-win64.exe        installer, the file users download
    Quarkon-<version>-win64-portable.zip    unpack-and-run build (no shortcuts)
    SHA256SUMS.txt                          checksums for both artefacts

The build has two PyInstaller stages:

1. **application** - ``--onedir`` bundle of ``app_entry.py`` that carries a private
   Python runtime, Pydantic, Click and the UI assets. No Python installation and no
   administrator rights are needed on the target machine.
2. **setup** - ``--onefile`` bundle of ``installer_entry.py`` that embeds the zipped
   application as ``payload.zip``. One file to download, and the wizard/library
   code stays editable and testable.

Requires ``pyinstaller`` (``pip install -r packaging/windows/requirements.txt``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PACKAGING_DIR: Final[Path] = Path(__file__).resolve().parent
ASSETS_DIR: Final[Path] = PACKAGING_DIR / "assets"

#: Modules that belong to the optional training stack. The desktop app never
#: imports them at start-up; excluding them keeps the download small.
_EXCLUDED_MODULES: Final[tuple[str, ...]] = (
    "torch",
    "torchvision",
    "transformers",
    "unsloth",
    "trl",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "pandas",
    "matplotlib",
    "IPython",
    "pytest",
    "sphinx",
)


def read_version() -> str:
    """Single source of truth: ``quarkon/core/version.py``."""
    text = (REPO_ROOT / "quarkon" / "core" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__[^=]*=\s*"([^"]+)"', text)
    if match is None:
        raise SystemExit("error: could not read the version from quarkon/core/version.py")
    return match.group(1)


def _pyinstaller() -> Any:
    try:
        import PyInstaller.__main__ as entry
    except ImportError:
        raise SystemExit(
            "error: PyInstaller is not installed.\n"
            "       Run: pip install -r packaging/windows/requirements.txt"
        ) from None
    return entry.run


def _add_data(source: Path, target: str) -> list[str]:
    """PyInstaller ``--add-data`` argument (``;`` separator on Windows)."""
    return ["--add-data", f"{source}{os.pathsep}{target}"]


def _run_pyinstaller(args: list[str]) -> None:
    runner = _pyinstaller()
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        runner(args)
    finally:
        os.chdir(previous)


def build_app(*, distpath: Path, workpath: Path, specpath: Path, keep_work: bool) -> Path:
    """Stage 1: the onedir desktop application."""
    if distpath.exists() and not keep_work:
        shutil.rmtree(distpath, ignore_errors=True)
    args: list[str] = [
        "--noconfirm",
        "--clean",
        "--noupx",
        "--windowed",
        "--onedir",
        "--name",
        "Quarkon",
        f"--distpath={distpath}",
        f"--workpath={workpath}",
        f"--specpath={specpath}",
        f"--icon={ASSETS_DIR / 'quarkon.ico'}",
        f"--paths={PACKAGING_DIR}",
        # The application imports the ``quarkon`` package from the checkout, and
        # the build script runs from packaging/windows - add the repository root
        # so the analysis can follow the imports (the app is built in-process).
        f"--paths={REPO_ROOT}",
        "--hidden-import=quarkon.ui.server",
        "--hidden-import=quarkon.core.version",
        "--log-level=WARN",
        *_add_data(ASSETS_DIR, "assets"),
        *_add_data(REPO_ROOT / "quarkon" / "ui" / "assets", "quarkon/ui/assets"),
        *_add_data(REPO_ROOT / "quarkon" / "i18n" / "locales", "quarkon/i18n/locales"),
    ]
    for module in _EXCLUDED_MODULES:
        args.extend(["--exclude-module", module])
    args.append(str(PACKAGING_DIR / "app_entry.py"))
    _run_pyinstaller(args)
    app_dir = distpath / "Quarkon"
    exe = app_dir / "Quarkon.exe"
    if not exe.is_file():
        raise SystemExit(f"error: PyInstaller did not produce {exe}")
    return app_dir


def zip_app(app_dir: Path, payload: Path) -> Path:
    """Zip the application folder (contents at the archive root)."""
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.unlink(missing_ok=True)
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(app_dir).as_posix())
    return payload


def write_build_info(build_dir: Path, version: str) -> Path:
    """Stamp the version into the installer bundle."""
    build_info = build_dir / "build_info.py"
    build_info.write_text(
        '"""Generated by packaging/windows/build_windows.py - do not edit."""\n'
        f'VERSION = "{version}"\n'
        f'BUILD_DATE = "{time.strftime("%Y-%m-%d")}"\n',
        encoding="utf-8",
    )
    return build_info


def build_setup(*, version: str, payload: Path, build_dir: Path, distpath: Path, workpath: Path, specpath: Path) -> Path:
    """Stage 2: the single-file wizard that carries the payload."""
    if distpath.exists():
        shutil.rmtree(distpath, ignore_errors=True)
    args: list[str] = [
        "--noconfirm",
        "--clean",
        "--noupx",
        "--windowed",
        "--onefile",
        "--name",
        f"QuarkonSetup-{version}",
        f"--distpath={distpath}",
        f"--workpath={workpath}",
        f"--specpath={specpath}",
        f"--icon={ASSETS_DIR / 'quarkon.ico'}",
        f"--paths={PACKAGING_DIR}",
        f"--paths={build_dir}",
        "--log-level=WARN",
        *_add_data(payload, "."),
        *_add_data(ASSETS_DIR, "assets"),
        "--exclude-module",
        "quarkon",
        "--exclude-module",
        "pydantic",
        "--exclude-module",
        "click",
        "--exclude-module",
        "httpx",
        "--exclude-module",
        "yaml",
    ]
    for module in _EXCLUDED_MODULES:
        args.extend(["--exclude-module", module])
    args.append(str(PACKAGING_DIR / "installer_entry.py"))
    _run_pyinstaller(args)
    exe = distpath / f"QuarkonSetup-{version}.exe"
    if not exe.is_file():
        raise SystemExit(f"error: PyInstaller did not produce {exe}")
    return exe


def portable_zip(app_dir: Path, target: Path) -> Path:
    """A no-install zip for users who cannot run a setup executable."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                archive.write(path, ("Quarkon/" + path.relative_to(app_dir).as_posix()))
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Quarkon Windows installer")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist" / "windows"), help="artefact folder")
    parser.add_argument("--version", default="", help="override the version stamp")
    parser.add_argument("--skip-app", action="store_true", help="reuse the previous application build")
    parser.add_argument("--skip-setup", action="store_true", help="build only the application")
    parser.add_argument("--keep-work", action="store_true", help="keep PyInstaller work folders")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        raise SystemExit("error: the Windows artefacts can only be built on Windows")

    version = args.version or read_version()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "windows"
    build_dir.mkdir(parents=True, exist_ok=True)
    write_build_info(build_dir, version)
    print(f"Quarkon {version} - building Windows artefacts")

    app_dir = build_dir / "app" / "Quarkon"
    payload = build_dir / "payload.zip"
    if not args.skip_app:
        started = time.time()
        app_dir = build_app(
            distpath=build_dir / "app",
            workpath=build_dir / "work-app",
            specpath=build_dir / "spec-app",
            keep_work=args.keep_work,
        )
        print(f"  application built in {time.time() - started:.0f}s: {app_dir}")
        zip_app(app_dir, payload)
        print(f"  payload: {payload} ({_human(payload.stat().st_size)})")
    elif not payload.is_file():
        raise SystemExit("error: --skip-app needs an existing payload.zip")

    artefacts: list[Path] = []
    if not args.skip_setup:
        started = time.time()
        setup = build_setup(
            version=version,
            payload=payload,
            build_dir=build_dir,
            distpath=build_dir / "setup",
            workpath=build_dir / "work-setup",
            specpath=build_dir / "spec-setup",
        )
        print(f"  setup built in {time.time() - started:.0f}s: {setup} ({_human(setup.stat().st_size)})")
        target = out_dir / f"QuarkonSetup-{version}-win64.exe"
        shutil.copy2(setup, target)
        artefacts.append(target)

    portable = portable_zip(app_dir, out_dir / f"Quarkon-{version}-win64-portable.zip")
    artefacts.append(portable)
    print(f"  portable: {portable} ({_human(portable.stat().st_size)})")

    lines = [f"{sha256(path)}  {path.name}" for path in artefacts]
    sums = out_dir / "SHA256SUMS.txt"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("artefacts:")
    for line in lines:
        print(f"  {line}")
    print(f"  {sums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
