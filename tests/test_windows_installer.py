"""Tests for the Windows installer engine.

The graphical wizard is a thin view over :mod:`install_core`, so the logic that
matters - what gets written where, how the uninstaller behaves, how a hostile
payload is rejected - is tested here on every platform. Anything that needs a real
Windows shell (shortcut COM objects, the registry hive) is covered by the smoke
test in ``packaging/windows/README.md`` instead, because GitHub's Linux runners
cannot execute it.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

WINDOWS_DIR = Path(__file__).resolve().parents[1] / "packaging" / "windows"
if str(WINDOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WINDOWS_DIR))

import install_core as core  # noqa: E402


def _fake_payload(tmp_path: Path, *, extras: dict[str, str] | None = None) -> Path:
    """Minimal payload zip that looks like the real application bundle."""
    payload = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(core.APP_EXE, "MZ fake executable")
        archive.writestr("_internal/base_library.zip", "data")
        archive.writestr("_internal/fonts/seguiemj.ttf", "data")
        for name, content in (extras or {}).items():
            archive.writestr(name, content)
    return payload


# ------------------------------------------------------------------ constants
def test_product_constants_are_pinned() -> None:
    assert core.APP_NAME == "Quarkon"
    assert core.APP_EXE == "Quarkon.exe"
    assert core.PUBLISHER == "Quarkon Maintainers"
    assert core.UNINSTALL_KEY.endswith("Uninstall\\Quarkon")
    # The pre-flight must leave room for the payload plus working space.
    assert core.MIN_FREE_SPACE_BYTES >= 100 * 1024 * 1024


def test_format_bytes_matches_windows_conventions() -> None:
    assert core.format_bytes(0) == "0 bytes"
    assert core.format_bytes(999) == "999 bytes"
    assert core.format_bytes(1_000) == "1.0 KB"
    assert core.format_bytes(18_500_000) == "18.5 MB"
    assert core.format_bytes(4_700_000_000) == "4.7 GB"


def test_default_folders_are_per_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert core.default_install_dir() == tmp_path / "Programs" / "Quarkon"
    assert core.default_data_dir() == tmp_path / "Quarkon"
    projects = core.projects_dir()
    assert projects.name == "Quarkon"
    assert projects.parent != tmp_path


# ------------------------------------------------------------------- registry
def test_registry_values_describe_the_install(tmp_path: Path) -> None:
    values = core.registry_values("9.9.9", tmp_path / "Quarkon", 123_456_000)
    assert values["DisplayName"] == "Quarkon"
    assert values["DisplayVersion"] == "9.9.9"
    assert values["Publisher"] == core.PUBLISHER
    assert values["InstallLocation"] == str(tmp_path / "Quarkon")
    assert str(values["DisplayIcon"]).endswith("Quarkon.exe")
    assert "uninstall.ps1" in str(values["UninstallString"])
    assert "-Silent" in str(values["QuietUninstallString"])
    assert values["EstimatedSize"] == 123_456_000 // 1024  # KB, for Programs and Features
    assert values["NoModify"] == 1
    assert values["NoRepair"] == 1


def test_uninstall_script_is_safe_and_complete(tmp_path: Path) -> None:
    install_dir = tmp_path / "Programs" / "Quarkon"
    script = core.uninstall_script(install_dir, "3.2.0", desktop=True, start_menu=True)
    assert script.startswith("# Quarkon 3.2.0 uninstaller")
    assert "param([switch]$Silent, [switch]$FromTemp)" in script
    assert str(install_dir) in script
    assert str(core.projects_dir()) in script
    assert "Uninstall\\Quarkon" in script
    assert "$purgeProjects" in script
    assert "Copy-Item -LiteralPath $self" in script
    # The dialog text must tell the user what is kept.
    assert "projects stay" in script
    assert script.endswith("\n")


# -------------------------------------------------------------------- payload
def test_payload_size_counts_extracted_bytes(tmp_path: Path) -> None:
    payload = _fake_payload(tmp_path)
    expected = len("MZ fake executable") + len("data") + len("data")
    assert core.payload_uncompressed_size(payload) == expected


def test_safe_extract_writes_files_and_counts_them(tmp_path: Path) -> None:
    payload = _fake_payload(tmp_path)
    target = tmp_path / "install"
    target.mkdir()
    count = core._safe_extract(payload, target)
    assert count == 3
    assert (target / core.APP_EXE).read_text() == "MZ fake executable"
    assert (target / "_internal" / "fonts" / "seguiemj.ttf").is_file()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _fake_payload(tmp_path, extras={"../escape.txt": "nope"})
    target = tmp_path / "install"
    target.mkdir()
    with pytest.raises(ValueError, match="escapes the install folder"):
        core._safe_extract(payload, target)
    assert not (tmp_path / "escape.txt").exists()


# --------------------------------------------------------------------- record
def test_read_install_record_requires_executable(tmp_path: Path) -> None:
    assert core.read_install_record(tmp_path) is None
    (tmp_path / "install.json").write_text('{"version": "3.2.0"}', encoding="utf-8")
    assert core.read_install_record(tmp_path) is None  # no executable yet
    (tmp_path / core.APP_EXE).write_bytes(b"MZ")
    info = core.read_install_record(tmp_path)
    assert info is not None
    assert info.version == "3.2.0"
    assert info.install_dir == tmp_path


def test_read_install_record_survives_broken_json(tmp_path: Path) -> None:
    (tmp_path / core.APP_EXE).write_bytes(b"MZ")
    (tmp_path / "install.json").write_text("{not json", encoding="utf-8")
    assert core.read_install_record(tmp_path) is None


# -------------------------------------------------------------------- install
def test_install_extracts_payload_and_records_metadata(tmp_path: Path) -> None:
    payload = _fake_payload(tmp_path)
    target = tmp_path / "Programs" / "Quarkon"
    events: list[tuple[float, str]] = []
    options = core.InstallOptions(
        install_dir=target,
        version="9.9.9",
        desktop_shortcut=False,
        start_menu_shortcut=False,
        launch_after_install=False,
    )
    result = core.install(options, payload, progress=lambda fraction, message: events.append((fraction, message)))
    assert result.ok, result.message
    assert (target / core.APP_EXE).is_file()
    record = core.read_install_record(target)
    assert record is not None and record.version == "9.9.9"
    assert (target / "uninstall.ps1").is_file()
    assert not (target / ".quarkon-setup").exists()
    assert events and events[-1][0] == 1.0

    second = core.install(options, payload)
    assert second.ok, second.message  # upgrade over the same folder


def test_install_refuses_foreign_folder(tmp_path: Path) -> None:
    payload = _fake_payload(tmp_path)
    target = tmp_path / "Documents"
    target.mkdir()
    (target / "thesis.docx").write_text("important", encoding="utf-8")
    options = core.InstallOptions(
        install_dir=target, version="9.9.9", desktop_shortcut=False, start_menu_shortcut=False
    )
    result = core.install(options, payload)
    assert not result.ok
    assert "do not belong" in result.message
    assert "Choose an empty folder" in result.hint


def test_install_reports_missing_payload(tmp_path: Path) -> None:
    options = core.InstallOptions(
        install_dir=tmp_path / "Quarkon", version="9.9.9", desktop_shortcut=False, start_menu_shortcut=False
    )
    result = core.install(options, tmp_path / "absent.zip")
    assert not result.ok
    assert "missing" in result.message


def test_failed_extraction_leaves_marker_for_retry(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"definitely not a zip")
    target = tmp_path / "Quarkon"
    options = core.InstallOptions(
        install_dir=target, version="9.9.9", desktop_shortcut=False, start_menu_shortcut=False
    )
    first = core.install(options, broken)
    assert not first.ok
    # A corrupt file is not a zip at all: nothing was cleared, and the folder is
    # either absent or marked as ours so the next attempt may clean it.
    if target.exists():
        assert (target / ".quarkon-setup").exists()


# ------------------------------------------------------------------ uninstall
def test_uninstall_removes_the_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _fake_payload(tmp_path)
    target = tmp_path / "Programs" / "Quarkon"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "home" / "AppData" / "Roaming"))
    options = core.InstallOptions(
        install_dir=target, version="9.9.9", desktop_shortcut=False, start_menu_shortcut=False
    )
    assert core.install(options, payload).ok
    result = core.uninstall(install_dir=target)
    assert result.ok, result.message
    assert not target.exists()


# ------------------------------------------------------------------- shortcuts
def test_shortcut_targets_are_under_the_install_folder(tmp_path: Path) -> None:
    script = core._shortcut_script(tmp_path / "Quarkon", desktop=True, start_menu=True)
    assert str(tmp_path / "Quarkon" / core.APP_EXE) in script
    assert "WScript.Shell" in script
    assert "uninstall.ps1" in script
    assert "IconLocation" in script


# ------------------------------------------------------------- wizard helpers
def test_wizard_helpers_without_display() -> None:
    pytest.importorskip("tkinter")
    import installer_entry

    assert installer_entry.installer_version().count(".") >= 1
    missing = installer_entry._validate_target("")
    assert missing[0] is None and "Choose a folder" in missing[1]
    relative = installer_entry._validate_target("relative\\folder")
    assert relative[0] is None and "full path" in relative[1]
    root_only = installer_entry._validate_target(str(Path(Path.cwd().anchor)))
    assert root_only[0] is None and "drive root" in root_only[1]


def test_payload_path_honours_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("tkinter")
    import installer_entry

    custom = tmp_path / "custom-payload.zip"
    monkeypatch.setenv("QUARKON_SETUP_PAYLOAD", str(custom))
    assert installer_entry.payload_path() == custom
