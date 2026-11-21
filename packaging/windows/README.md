# Windows installer and desktop application

Two artefacts are built from this folder:

| Artefact | What it is |
|---|---|
| `QuarkonSetup-<version>-win64.exe` | Single-file graphical installer. Per-user install, shortcuts, uninstall entry, opens the interface. |
| `Quarkon-<version>-win64-portable.zip` | The same application folder zipped; unpack and run `Quarkon\Quarkon.exe` when running a setup executable is not allowed. |

## Build

On Windows, with Python 3.10-3.12:

```powershell
pip install -r packaging/windows/requirements.txt
pip install -e .
python packaging/windows/build_windows.py
```

The script runs PyInstaller twice:

1. **application** — `app_entry.py` → `Quarkon.exe` (onedir): private Python
   runtime, Pydantic, Click, HTTPX, the web UI assets and all locale catalogs.
2. **setup** — `installer_entry.py` → `QuarkonSetup-<version>.exe` (onefile) with
   the application folder zipped inside as `payload.zip`.

Outputs land in `dist/windows/` together with `SHA256SUMS.txt`. Options:
`--skip-app` reuses the previous application build, `--skip-setup` builds only the
application, `--version X.Y.Z` overrides the stamp, `--keep-work` keeps the
PyInstaller work folders for debugging.

## Test the build

```powershell
# 1. install silently into a throwaway folder
dist\windows\QuarkonSetup-3.2.0-win64.exe --silent --dir "$env:TEMP\qtest" --no-launch
# 2. the installed app answers on the loopback interface
& "$env:TEMP\qtest\Quarkon.exe" --check
#    -> exit code 0; last line of %LOCALAPPDATA%\Quarkon\logs\app.log is
#       {"check": "ui", "ok": true, "url": "http://127.0.0.1:...", ...}
# 3. uninstall
dist\windows\QuarkonSetup-3.2.0-win64.exe --silent --uninstall --dir "$env:TEMP\qtest"
```

The setup also accepts `--no-shortcuts`, `--no-desktop`, `--no-start-menu`,
`--no-launch`, `--uninstall`, `--version` and `--ui-test` (renders every wizard
page without showing the window; used by the smoke test and CI). Exit codes: `0`
success, `1` install failure, `2` uninstall failure.

`packaging/windows/smoke_test.ps1` automates the whole sequence above (install,
wizard render check, application self-check, uninstall) and is what CI runs:

```powershell
pwsh -File packaging/windows/smoke_test.ps1 -SetupPath dist/windows/QuarkonSetup-3.2.0-win64.exe
```

## What an install writes

| Path | Purpose |
|---|---|
| `%LOCALAPPDATA%\Programs\Quarkon\` | Program files, `Quarkon.exe`, `install.json`, `uninstall.ps1` |
| `%USERPROFILE%\Desktop\Quarkon.lnk` | Optional desktop shortcut |
| `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Quarkon\` | Optional Start-menu shortcuts (`Quarkon`, `Uninstall Quarkon`) |
| `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Quarkon` | Entry in *Windows Settings → Apps* |
| `%USERPROFILE%\Quarkon\` | Projects — never touched by upgrade or uninstall |
| `%LOCALAPPDATA%\Quarkon\logs\app.log` | Interface log (a windowed build has no console) |

Nothing is written to `Program Files`, nothing needs elevation, and the installer
never touches machine-wide registry keys.

## Source map

| File | Responsibility |
|---|---|
| `install_core.py` | Installation engine: pre-flight, payload extraction, install record, shortcuts, registry, uninstaller. Imported by both the wizard and the tests. |
| `installer_entry.py` | Five-page wizard (Welcome → Folder → Options → Progress → Finish), uninstall flow and the `--silent` path. |
| `app_entry.py` | `Quarkon.exe`: starts the loopback interface, opens the browser, keeps a control window, `--check` self-test. |
| `ui_kit.py` | Shared Tk design system (palette, gradient hero, rounded buttons, progress strip, log view, brand mark). |
| `make_icon.py` | Regenerates `assets/quarkon.ico` (7 sizes) and `assets/quarkon-256.png` from code. |
| `build_windows.py` | PyInstaller orchestration, payload zipping, portable zip and checksums. |

## Code signing

Both executables carry publisher metadata (`Kdjkvv7886`) and, when a certificate is
configured, an Authenticode signature. The build signs `Quarkon.exe` **before** it is
zipped into the payload and signs the setup executable after it is produced; without
a certificate the build still succeeds and prints `signing: skipped`.

```powershell
$env:QUARKON_SIGN_PFX = "$PWD\build\signing\quarkon-signing.pfx"
$env:QUARKON_SIGN_PFX_PASSWORD = "…"
python packaging/windows/build_windows.py

Get-AuthenticodeSignature dist\windows\QuarkonSetup-3.2.1-win64.exe |
    Format-List Status, SignerCertificate
```

CI uses repository secrets `WINDOWS_SIGN_PFX_BASE64` (base64 of the PFX) and
`WINDOWS_SIGN_PFX_PASSWORD`; the workflow imports the PFX into the runner's user
store and the same code path signs the artefacts.

The project certificate is **self-signed** (`CN=Kdjkvv7886, O=Kdjkvv7886, C=UA`,
RSA-3072, code-signing EKU, valid until 2029-11-21). `Get-AuthenticodeSignature`
reports `UnknownError` — that means "the signature is valid but the root is not
trusted", not "unsigned". SmartScreen keeps warning until the certificate is placed
in *Trusted Root Certification Authorities* on the target machine, or until a
CA-issued certificate replaces it. `packaging/windows/assets/Kdjkvv7886-codesign.cer`
is the public part and is published with the release.

## Design notes

- **Per-user by design.** Corporate laptops often forbid elevation; everything
  here works from an ordinary account, and an uninstall removes exactly what was
  added.
- **The wizard states consequences.** The welcome page says what will be added,
  the folder page shows free space, the options page lists the shortcuts, and the
  uninstall page says explicitly that projects are kept unless asked otherwise.
- **No silent data loss.** A failed extraction leaves `.quarkon-setup`, so the
  next run is allowed to clean the folder; a folder with foreign files is refused
  rather than wiped.
- **The application cannot be orphaned.** Closing the `Quarkon.exe` window stops
  the local server; the interface URL contains a per-session token, so no other
  browser page can drive the pipeline.
- **PyInstaller specifics:** the installer embeds `payload.zip` and extracts it to
  the install folder; Qt/webview are not used, so the download stays small and the
  interface is the same HTML the SDK ships.

## Continuous integration

`.github/workflows/release-windows.yml` builds both artefacts on `windows-latest`
for every manual run (kept as a workflow artifact) and attaches them to the
GitHub Release when a `v*` tag is pushed.
