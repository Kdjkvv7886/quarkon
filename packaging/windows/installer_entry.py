"""Quarkon graphical setup wizard (``QuarkonSetup-<version>.exe``).

The executable is a single file that carries the application payload. On the
first run it shows a five-step wizard; on a corporate network the same binary is
driven unattended:

    QuarkonSetup-3.2.0.exe --silent --dir D:\\Apps\\Quarkon --no-launch

Design rules
------------
* **No admin rights.** Everything is per-user: ``%LOCALAPPDATA%\\Programs\\Quarkon``,
  shortcuts in the user profile, one ``HKCU`` uninstall entry.
* **No hidden steps.** Each page says what will happen; the progress page streams
  the real operations; failures show the cause plus the next action.
* **One code path.** The wizard, the silent switch and the tests all call
  :mod:`install_core`; the UI only renders.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

_SETUP_LOG = Path(os.environ.get("TEMP", ".")) / "quarkon-setup.log"


def _log_event(message: str) -> None:
    """Append a diagnostic line; the wizard is windowed, so there is no console."""
    try:
        _SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _SETUP_LOG.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except OSError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install_core as core  # noqa: E402
import ui_kit as ui  # noqa: E402

try:  # written by the build script next to this file
    from build_info import BUILD_DATE as _BUILD_DATE
    from build_info import VERSION as _VERSION

    VERSION = str(_VERSION)
    BUILD_DATE = str(_BUILD_DATE)
except ImportError:  # pragma: no cover - running straight from the source tree
    BUILD_DATE = ""
    VERSION = ""


WINDOW_TITLE = "Quarkon Setup"
_STEPS = ("Welcome", "Folder", "Options", "Install", "Finish")
_STEPS_UNINSTALL = ("Uninstall", "Remove", "Done")


# --------------------------------------------------------------------- helpers
def installer_version() -> str:
    """Version shown in the wizard; prefers the generated build stamp."""
    if VERSION and VERSION != "dev":
        return VERSION
    version_file = Path(__file__).resolve().parents[2] / "quarkon" / "core" / "version.py"
    try:
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    except OSError:
        return "0.0.0"
    return VERSION or "0.0.0"


def payload_path() -> Path:
    """Payload zip bundled next to the executable / extracted by PyInstaller."""
    override = __import__("os").environ.get("QUARKON_SETUP_PAYLOAD")
    if override:
        return Path(override)
    bundle = getattr(sys, "_MEIPASS", None)
    base = Path(bundle) if bundle else Path(__file__).resolve().parent
    return base / "payload.zip"


def _human_free_space(path: Path) -> str:
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return "unknown"
        probe = parent
    try:
        import shutil

        return core.format_bytes(int(shutil.disk_usage(str(probe)).free))
    except OSError:
        return "unknown"


def _validate_target(raw: str) -> tuple[Path | None, str]:
    text = raw.strip().strip('"')
    if not text:
        return None, "Choose a folder for the program files."
    path = Path(text).expanduser()
    if not path.is_absolute():
        return None, "Use a full path such as C:\\Users\\you\\AppData\\Local\\Programs\\Quarkon."
    if path.parent == path:
        return None, "A drive root cannot be the install folder; add a folder name."
    info = core.read_install_record(path)
    if path.exists() and not info and any(path.iterdir()) and not (path / ".quarkon-setup").exists():
        return None, "That folder already contains other files. Choose an empty folder or the default."
    return path, ""


# ---------------------------------------------------------------------- wizard
@dataclass
class _Defaults:
    """Values shared by the wizard and the silent path."""

    version: str
    install_dir: Path
    desktop: bool = True
    start_menu: bool = True
    launch: bool = True


class Wizard:
    """The five-page setup wizard."""

    def __init__(self, root: tk.Tk, defaults: _Defaults, *, mode: str = "install") -> None:
        self.root = root
        self.mode = mode
        self.defaults = defaults
        self.version = defaults.version
        self.install_dir_var = tk.StringVar(value=str(defaults.install_dir))
        self.desktop_var = tk.BooleanVar(value=defaults.desktop)
        self.start_menu_var = tk.BooleanVar(value=defaults.start_menu)
        self.launch_var = tk.BooleanVar(value=defaults.launch)
        self.purge_projects_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Preparing…")
        self.error_var = tk.StringVar(value="")
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._installing = False
        self._installed_path: Path | None = None
        self._warnings: list[str] = []
        self._payload = payload_path()
        self.pages: dict[str, tk.Frame] = {}
        self.hero = ui.HeroPanel(root, steps=_STEPS_UNINSTALL if mode == "uninstall" else _STEPS)
        self.hero.pack(side="left", fill="y")
        self._build_shell()
        self._pump()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if mode == "uninstall":
            self.go_uninstall()
        else:
            self.go_welcome()

    # ------------------------------------------------------------------ shell
    def _build_shell(self) -> None:
        self.right = tk.Frame(self.root, bg=ui.BG)
        self.right.pack(side="left", fill="both", expand=True)
        self.page_holder = tk.Frame(self.right, bg=ui.BG)
        self.page_holder.pack(fill="both", expand=True, padx=34, pady=(28, 0))
        self.bottom = tk.Frame(self.right, bg=ui.BG)
        self.bottom.pack(fill="x", padx=34, pady=(0, 24))

    def _page(self, name: str) -> tk.Frame:
        frame = self.pages.get(name)
        if frame is None:
            frame = tk.Frame(self.page_holder, bg=ui.BG)
            self.pages[name] = frame
        return frame

    def _title(self, parent: tk.Misc, text: str, subtitle: str = "") -> None:
        tk.Label(
            parent, text=text, bg=ui.BG, fg=ui.TEXT, font=ui.fonts()["display"],
            anchor="w", justify="left",
        ).pack(fill="x")
        if subtitle:
            tk.Label(
                parent, text=subtitle, bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["subtitle"],
                anchor="w", justify="left", wraplength=560, pady=8,
            ).pack(fill="x")

    def _bullets(self, parent: tk.Misc, items: tuple[tuple[str, str], ...]) -> None:
        holder = tk.Frame(parent, bg=ui.BG)
        holder.pack(fill="x", pady=(6, 0))
        for index, (headline, detail) in enumerate(items):
            row = tk.Frame(holder, bg=ui.BG)
            row.pack(fill="x", pady=(0, 10))
            badge = tk.Canvas(row, width=18, height=18, bg=ui.BG, highlightthickness=0, bd=0)
            badge.pack(side="left", anchor="n", padx=(0, 10))
            badge.create_oval(1, 1, 17, 17, fill="#142033", outline=ui.ACCENT, width=1)
            badge.create_text(9, 9, text="✓", fill=ui.ACCENT, font=ui.fonts()["tiny"])
            labels = tk.Frame(row, bg=ui.BG)
            labels.pack(side="left", fill="x", expand=True)
            tk.Label(
                labels, text=headline, bg=ui.BG, fg=ui.TEXT, font=ui.fonts()["body"], anchor="w",
            ).pack(fill="x")
            tk.Label(
                labels, text=detail, bg=ui.BG, fg=ui.MUTED_DIM, font=ui.fonts()["small"],
                anchor="w", justify="left", wraplength=520,
            ).pack(fill="x")
            _ = index

    def _clear_bottom(self) -> None:
        for child in self.bottom.winfo_children():
            child.destroy()

    def _button(self, text: str, command: Any, *, kind: str = "primary", width: int = 150) -> Any:
        button = ui.AccentButton(self.bottom, text=text, command=command, kind=kind, width=width)
        return button

    def _show(self, name: str, *, step: int) -> None:
        for frame in self.pages.values():
            frame.pack_forget()
        page = self._page(name)
        page.pack(fill="both", expand=True)
        self.hero.set_active(step)
        self._clear_bottom()
        self._page_bottom(name)

    def _page_bottom(self, name: str) -> None:
        if self.mode == "uninstall":
            if name == "uninstall":
                self._button("Remove Quarkon", self.start_uninstall, kind="danger", width=170).pack(side="right")
                self._button("Cancel", self._on_close, kind="ghost", width=110).pack(side="right", padx=(0, 10))
                return
            if name == "removing":
                return
            if name == "uninstall-done":
                self._button("Close", self._on_close, width=140).pack(side="right")
                return
        if name == "welcome":
            self._button("Continue", self.go_folder, width=140).pack(side="right")
            self._button("Cancel", self._on_close, kind="ghost", width=110).pack(side="right", padx=(0, 10))
        elif name == "folder":
            self._button("Continue", self.go_options, width=140).pack(side="right")
            self._button("Back", self.go_welcome, kind="secondary", width=110).pack(side="right", padx=(0, 10))
            self._button("Cancel", self._on_close, kind="ghost", width=110).pack(side="left")
        elif name == "options":
            self._button(f"Install {self.version}", self.start_install, width=170).pack(side="right")
            self._button("Back", self.go_folder, kind="secondary", width=110).pack(side="right", padx=(0, 10))
            self._button("Cancel", self._on_close, kind="ghost", width=110).pack(side="left")
        elif name == "progress":
            self._button("Cancel", self._on_close, kind="ghost", width=110).pack(side="right")
        elif name == "finish":
            if self._installed_dir is not None:
                self._button("Open Quarkon", lambda: core.launch_app(self._installed_dir or self.defaults.install_dir), width=160).pack(side="right")
                self._button(
                    "Projects folder",
                    self._open_projects,
                    kind="secondary",
                    width=150,
                ).pack(side="right", padx=(0, 10))
            self._button("Close", self._on_close, kind="ghost", width=110).pack(side="left")
        elif name in {"error", "uninstall_error"}:
            self._button("Try again", self.go_options, kind="secondary", width=130).pack(side="right")
            self._button("Close", self._on_close, kind="ghost", width=110).pack(side="right", padx=(0, 10))

    # ------------------------------------------------------------------ pages
    @property
    def _installed_dir(self) -> Path | None:
        return getattr(self, "_installed_path", None)

    def go_welcome(self) -> None:
        page = self._page("welcome")
        for child in page.winfo_children():
            child.destroy()
        self._title(
            page,
            f"Install Quarkon {self.version}",
            "A local studio that turns a reasoning model into a smaller model your team "
            "can run on a laptop or a server. Everything stays on this computer.",
        )
        tk.Frame(page, bg=ui.BORDER, height=1).pack(fill="x", pady=(6, 16))
        self._bullets(
            page,
            (
                ("Installs without administrator rights", "The program goes into your own profile; Windows never asks for a password."),
                ("Adds shortcuts you control", "Desktop and Start-menu entries; delete them any time without breaking the app."),
                ("Registers a proper uninstaller", "Quarkon appears in Settings → Apps, with the version and publisher."),
                ("Keeps your work separate", r"Projects live in %USERPROFILE%\Quarkon and survive every upgrade."),
            ),
        )
        existing = core.detect_installations()
        if existing:
            newest = existing[0]
            tk.Label(
                page,
                text=f"Version {newest.version} is already installed in {newest.install_dir}. "
                "This setup replaces the program files and keeps your projects.",
                bg="#1d1a12", fg=ui.WARN, font=ui.fonts()["small"], anchor="w",
                justify="left", wraplength=560, padx=12, pady=8,
            ).pack(fill="x", pady=(10, 0))
        tk.Label(
            page,
            text=f"Apache 2.0  ·  version {self.version}" + (f"  ·  built {BUILD_DATE}" if BUILD_DATE else ""),
            bg=ui.BG, fg=ui.MUTED_DIM, font=ui.fonts()["tiny"], anchor="w",
        ).pack(side="bottom", fill="x")
        self._show("welcome", step=0)

    def go_folder(self) -> None:
        page = self._page("folder")
        for child in page.winfo_children():
            child.destroy()
        self._title(
            page,
            "Where should the program live?",
            "The default folder is inside your user profile and needs no special permissions.",
        )
        row = tk.Frame(page, bg=ui.BG)
        row.pack(fill="x", pady=(18, 6))
        entry = tk.Entry(
            row, textvariable=self.install_dir_var, bg=ui.SURFACE, fg=ui.TEXT,
            insertbackground=ui.TEXT, bd=0, font=ui.fonts()["body"],
            highlightthickness=1, highlightbackground=ui.BORDER, highlightcolor=ui.ACCENT,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 10))
        ui.AccentButton(row, text="Browse…", command=self._browse, kind="secondary", width=110).pack(side="left")

        self._folder_note = tk.Label(
            page, text="", bg=ui.BG, fg=ui.MUTED_DIM, font=ui.fonts()["small"],
            anchor="w", justify="left", wraplength=560,
        )
        self._folder_note.pack(fill="x", pady=(4, 0))
        self._folder_error = tk.Label(
            page, text="", bg=ui.BG, fg=ui.DANGER, font=ui.fonts()["small"],
            anchor="w", justify="left", wraplength=560,
        )
        self._folder_error.pack(fill="x", pady=(2, 0))
        self._refresh_space()
        entry.bind("<Return>", lambda _event: self.go_options())
        self._show("folder", step=1)

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self.root, title="Choose the Quarkon program folder",
            initialdir=str(Path(self.install_dir_var.get()).parent),
            mustexist=False,
        )
        if chosen:
            self.install_dir_var.set(str(Path(chosen) / core.APP_NAME))
            self._refresh_space()

    def _refresh_space(self) -> None:
        target = Path(self.install_dir_var.get())
        free = _human_free_space(target)
        payload_size = 0
        if self._payload.is_file():
            payload_size = core.payload_uncompressed_size(self._payload)
        needed = core.format_bytes(payload_size + core.MIN_FREE_SPACE_BYTES)
        text = f"Free space at this location: {free}  ·  Quarkon needs about {needed}."
        if not self._payload.is_file():
            text = "Free space at this location: " + free + "."
        if hasattr(self, "_folder_note"):
            self._folder_note.configure(text=text)
        if hasattr(self, "_folder_error"):
            self._folder_error.configure(text="")

    def go_options(self) -> None:
        target, error = _validate_target(self.install_dir_var.get())
        if target is None:
            self.go_folder()
            self._folder_error.configure(text=error)
            return
        self.defaults.install_dir = target
        page = self._page("options")
        for child in page.winfo_children():
            child.destroy()
        self._title(
            page,
            "Shortcuts and first start",
            "Choose what the setup adds. Everything can be changed later in Windows Settings.",
        )
        card = tk.Frame(page, bg=ui.SURFACE, highlightthickness=1, highlightbackground=ui.BORDER)
        card.pack(fill="x", pady=(16, 12), ipady=8)
        inner = tk.Frame(card, bg=ui.SURFACE)
        inner.pack(fill="x", padx=16, pady=10)
        ui.CheckRow(
            inner, text="Desktop shortcut", hint="A Quarkon icon on your desktop.",
            variable=self.desktop_var,
        ).pack(fill="x", pady=(0, 10))
        ui.CheckRow(
            inner, text="Start-menu shortcut", hint="Adds Quarkon and its uninstaller to the Start menu.",
            variable=self.start_menu_var,
        ).pack(fill="x", pady=(0, 10))
        ui.CheckRow(
            inner, text="Launch Quarkon when the setup finishes",
            hint="The interface opens in your browser straight away.", variable=self.launch_var,
        ).pack(fill="x")

        info = tk.Frame(page, bg=ui.SURFACE_ALT, highlightthickness=1, highlightbackground=ui.BORDER_SOFT)
        info.pack(fill="x", pady=(6, 0), ipady=6)
        tk.Label(
            info,
            text=f"Program files: {target}\nProjects: {core.projects_dir()}",
            bg=ui.SURFACE_ALT, fg=ui.MUTED, font=ui.fonts()["small"], anchor="w",
            justify="left", padx=14, pady=8,
        ).pack(fill="x")
        self.hero.set_steps(_STEPS)
        self._show("options", step=2)

    def go_progress(self) -> None:
        page = self._page("progress")
        for child in page.winfo_children():
            child.destroy()
        self._title(page, f"Installing Quarkon {self.version}", "This usually takes under a minute.")
        self.status_var.set("Starting…")
        tk.Label(
            page, textvariable=self.status_var, bg=ui.BG, fg=ui.TEXT,
            font=ui.fonts()["body"], anchor="w",
        ).pack(fill="x", pady=(18, 8))
        self._progress = ui.ProgressStrip(page, height=10)
        self._progress.pack(fill="x", pady=(0, 16))
        self._log = ui.LogView(page, height=9)
        self._log.pack(fill="both", expand=True, pady=(0, 6))
        self._log.append(f"Installing into {self.defaults.install_dir}", "dim")
        self._show("progress", step=3)

    def go_finish(self) -> None:
        page = self._page("finish")
        for child in page.winfo_children():
            child.destroy()
        badge = tk.Canvas(page, width=64, height=64, bg=ui.BG, highlightthickness=0, bd=0)
        badge.pack(anchor="w", pady=(20, 6))
        badge.create_oval(2, 2, 62, 62, fill="#10261a", outline=ui.OK, width=2)
        badge.create_line([18, 32, 28, 43, 47, 21], fill=ui.OK, width=4, capstyle="round", joinstyle="round")
        tk.Label(
            page, text="Quarkon is installed", bg=ui.BG, fg=ui.TEXT,
            font=ui.fonts()["display"], anchor="w",
        ).pack(fill="x")
        tk.Label(
            page,
            text="The interface opens in your browser. Next time, start Quarkon from the "
            "desktop or Start-menu shortcut.",
            bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["subtitle"], anchor="w", justify="left",
            wraplength=560,
        ).pack(fill="x", pady=(8, 16))
        info = tk.Frame(page, bg=ui.SURFACE, highlightthickness=1, highlightbackground=ui.BORDER)
        info.pack(fill="x", ipady=4)
        tk.Label(
            info,
            text=f"Installed in: {self._installed_dir}\nProjects: {core.projects_dir()}",
            bg=ui.SURFACE, fg=ui.MUTED, font=ui.fonts()["small"], anchor="w",
            justify="left", padx=14, pady=10,
        ).pack(fill="x")
        for warning in self._warnings:
            tk.Label(
                page, text=warning, bg="#1d1a12", fg=ui.WARN, font=ui.fonts()["small"],
                anchor="w", justify="left", wraplength=560, padx=12, pady=8,
            ).pack(fill="x", pady=(8, 0))
        self._show("finish", step=4)

    def go_error(self, message: str, hint: str, *, page_name: str = "error") -> None:
        page = self._page(page_name)
        for child in page.winfo_children():
            child.destroy()
        self._title(page, "The installation did not finish", "")
        tk.Label(
            page, text=message, bg=ui.BG, fg=ui.DANGER, font=ui.fonts()["body"],
            anchor="w", justify="left", wraplength=560,
        ).pack(fill="x", pady=(10, 8))
        tk.Label(
            page, text=hint, bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["small"],
            anchor="w", justify="left", wraplength=560,
        ).pack(fill="x")
        self._show(page_name, step=3)

    # ----------------------------------------------------------------- uninstall
    def go_uninstall(self) -> None:
        info = core.installed_info()
        page = self._page("uninstall")
        for child in page.winfo_children():
            child.destroy()
        if info is None:
            self.go_uninstall_missing()
            return
        self._installed_path = info.install_dir
        self._title(
            page,
            "Uninstall Quarkon",
            f"Quarkon {info.version} is installed in {info.install_dir}.",
        )
        tk.Label(
            page,
            text=f"Your projects live in {core.projects_dir()} and are kept unless you ask "
            "to delete them. Shortcuts and the Windows uninstall entry are removed either way.",
            bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["small"], anchor="w", justify="left",
            wraplength=560,
        ).pack(fill="x", pady=(12, 14))
        ui.CheckRow(
            page, text="Also delete my Quarkon projects",
            hint="Deletes samples, adapters and models in the projects folder. This cannot be undone.",
            variable=self.purge_projects_var,
        ).pack(fill="x", pady=(4, 0))
        self._show("uninstall", step=0)

    def go_uninstall_missing(self) -> None:
        page = self._page("uninstall-done")
        for child in page.winfo_children():
            child.destroy()
        self._title(page, "Quarkon is not installed", "")
        tk.Label(
            page,
            text="There is no Quarkon installation in the default user profile on this computer.",
            bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["body"], anchor="w", wraplength=560,
        ).pack(fill="x", pady=(10, 0))
        self._show("uninstall-done", step=2)

    # ------------------------------------------------------------------- actions
    def _open_projects(self) -> None:
        starter = getattr(__import__("os"), "startfile", None)
        if starter is not None:
            starter(str(core.projects_dir()))

    def start_install(self) -> None:
        if self._installing:
            return
        target, error = _validate_target(self.install_dir_var.get())
        if target is None:
            self._folder_error.configure(text=error)
            return
        if core.is_app_running():
            answer = messagebox.askyesno(
                WINDOW_TITLE,
                "Quarkon is running. Close it so the program files can be replaced?\n\n"
                "Choose Yes to close it now.",
                parent=self.root,
            )
            if not answer:
                return
            core.stop_app()
        options = core.InstallOptions(
            install_dir=target,
            version=self.version,
            desktop_shortcut=self.desktop_var.get(),
            start_menu_shortcut=self.start_menu_var.get(),
            launch_after_install=self.launch_var.get(),
            log=self._thread_log,
        )
        self._installing = True
        self.go_progress()
        self._log.append("Closing any running copy of Quarkon", "dim")
        worker = threading.Thread(target=self._install_worker, args=(options,), daemon=True)
        worker.start()

    def _install_worker(self, options: core.InstallOptions) -> None:
        result = core.install(
            options,
            self._payload,
            progress=lambda fraction, message: self._events.put(("progress", (fraction, message))),
        )
        self._events.put(("installed", result))

    def _thread_log(self, message: str, level: str = "info") -> None:
        self._events.put(("log", (message, level)))

    def start_uninstall(self) -> None:
        if self._installing:
            return
        target = self._installed_dir or core.default_install_dir()
        self._installing = True
        page = self._page("removing")
        for child in page.winfo_children():
            child.destroy()
        self._title(page, "Removing Quarkon", "")
        self.status_var.set("Removing program files…")
        tk.Label(
            page, textvariable=self.status_var, bg=ui.BG, fg=ui.MUTED,
            font=ui.fonts()["body"], anchor="w",
        ).pack(fill="x", pady=(12, 0))
        self._show("removing", step=1)
        worker = threading.Thread(
            target=self._uninstall_worker,
            args=(target, self.purge_projects_var.get()),
            daemon=True,
        )
        worker.start()

    def _uninstall_worker(self, target: Path, purge: bool) -> None:
        result = core.uninstall(install_dir=target, log=self._thread_log)
        if result.ok and purge:
            import shutil

            shutil.rmtree(core.projects_dir(), ignore_errors=True)
        self._events.put(("uninstalled", result))

    # ---------------------------------------------------------------- event pump
    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "progress":
                    fraction, message = payload
                    if hasattr(self, "_progress"):
                        self._progress.set(fraction)
                    self.status_var.set(message)
                elif kind == "log":
                    message, level = payload
                    if hasattr(self, "_log"):
                        self._log.append(message, level)
                elif kind == "installed":
                    self._on_installed(payload)
                elif kind == "uninstalled":
                    self._on_uninstalled(payload)
        except queue.Empty:
            pass
        self.root.after(60, self._pump)

    def _on_installed(self, result: core.InstallResult) -> None:
        self._installing = False
        if not result.ok:
            self.go_error(result.message, result.hint or "Check the log above and try again.")
            return
        self._installed_path = result.install_dir
        self._warnings = list(result.warnings)
        self._progress.set(1.0)
        self.go_finish()
        if self.launch_var.get() and result.install_dir is not None:
            self.root.after(200, lambda: core.launch_app(result.install_dir or core.default_install_dir()))

    def _on_uninstalled(self, result: core.InstallResult) -> None:
        self._installing = False
        page = self._page("uninstall-done")
        for child in page.winfo_children():
            child.destroy()
        if result.ok:
            self._title(page, "Quarkon was removed", "Its shortcuts and the Windows uninstall entry are gone.")
            tk.Label(
                page,
                text=f"Your projects are still in {core.projects_dir()} unless you asked to delete them.",
                bg=ui.BG, fg=ui.MUTED, font=ui.fonts()["body"], anchor="w", wraplength=560,
            ).pack(fill="x", pady=(10, 0))
            self._show("uninstall-done", step=2)
        else:
            self.go_error(result.message, result.hint or "Close Quarkon and try again.", page_name="uninstall_error")

    # ------------------------------------------------------------------- closing
    def _on_close(self) -> None:
        if self._installing and self.mode == "install":
            answer = messagebox.askyesno(
                WINDOW_TITLE,
                "The installation is still running. Stop it and close?\n\n"
                "A later run continues from a clean folder.",
                parent=self.root,
            )
            if not answer:
                return
        self.root.destroy()


# ---------------------------------------------------------------- silent mode
def run_silent(args: argparse.Namespace) -> int:
    """Unattended install / uninstall used by IT deployment and by the tests."""
    payload = payload_path()
    logger = _silent_logger()
    if args.uninstall:
        result = core.uninstall(install_dir=Path(args.dir) if args.dir else None, silent=True, log=logger)
        return 0 if result.ok else 2
    target = Path(args.dir) if args.dir else core.default_install_dir()
    options = core.InstallOptions(
        install_dir=target,
        version=installer_version(),
        desktop_shortcut=not args.no_shortcuts and not args.no_desktop,
        start_menu_shortcut=not args.no_shortcuts and not args.no_start_menu,
        launch_after_install=not args.no_launch,
        silent=True,
        log=logger,
    )
    result = core.install(options, payload)
    if not result.ok:
        logger(f"Installation failed: {result.message}", "error")
        if result.hint:
            logger(result.hint, "dim")
        return 1
    for warning in result.warnings:
        logger(warning, "warn")
    if options.launch_after_install and result.install_dir is not None:
        core.launch_app(result.install_dir)
    return 0


def _silent_logger() -> Any:
    def log(message: str, level: str = "info") -> None:
        line = f"[{level}] {message}"
        _log_event(line)
        if sys.stdout is not None:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
    return log


# ------------------------------------------------------------------ ui self-test
def run_ui_test(mode: str) -> int:
    """Build every wizard page without showing the window.

    This catches Tcl/Tk errors (bad widget options, destroyed parents, attribute
    name collisions with Tk internals) on CI and during local builds; it never
    touches the filesystem.
    """
    ui.enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    root.configure(bg=ui.BG)
    ui.apply_theme(root)
    defaults = _Defaults(version=installer_version(), install_dir=core.default_install_dir())
    wizard = Wizard(root, defaults, mode="uninstall" if mode == "uninstall" else "install")
    try:
        if mode == "uninstall":
            wizard.go_uninstall()
            root.update_idletasks()
            wizard.go_uninstall_missing()
            root.update_idletasks()
        else:
            wizard.go_welcome()
            root.update_idletasks()
            wizard.go_folder()
            root.update_idletasks()
            wizard.go_options()
            root.update_idletasks()
            wizard._installed_path = core.default_install_dir()
            wizard._warnings = ["Example non-fatal warning"]
            wizard.go_finish()
            root.update_idletasks()
            wizard.go_error("Example failure", "Example hint")
            root.update_idletasks()
    finally:
        root.destroy()
    _log_event(f"--- ui-test ok mode={mode}")
    return 0


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    _log_event(f"--- setup start argv={list(argv) if argv is not None else sys.argv[1:]}")
    try:
        return _main(argv)
    except SystemExit:
        raise
    except BaseException:
        _log_event("fatal error:\n" + traceback.format_exc())
        raise


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="QuarkonSetup", description="Quarkon setup")
    parser.add_argument("--silent", action="store_true", help="install without a window")
    parser.add_argument("--dir", default="", help="install folder")
    parser.add_argument("--no-shortcuts", action="store_true", help="create no shortcuts")
    parser.add_argument("--no-desktop", action="store_true", help="skip the desktop shortcut")
    parser.add_argument("--no-start-menu", action="store_true", help="skip the Start-menu shortcut")
    parser.add_argument("--no-launch", action="store_true", help="do not start Quarkon afterwards")
    parser.add_argument("--uninstall", action="store_true", help="remove Quarkon")
    parser.add_argument("--version", action="store_true", help="print the setup version")
    parser.add_argument("--ui-test", action="store_true", help="render every wizard page and exit")
    args = parser.parse_args(argv)

    if args.version:
        version = installer_version()
        if sys.stdout is not None:
            sys.stdout.write(f"Quarkon Setup {version}\n")
        return 0
    if args.ui_test:
        code = run_ui_test("uninstall" if args.uninstall else "install")
        _log_event(f"--- setup done exit={code}")
        return code
    if args.silent:
        code = run_silent(args)
        _log_event(f"--- setup done exit={code}")
        return code

    ui.enable_dpi_awareness()
    root = tk.Tk()
    root.title(f"{WINDOW_TITLE} — Quarkon {installer_version()}")
    root.configure(bg=ui.BG)
    root.resizable(False, False)
    ui.apply_theme(root)
    ui.set_window_icon(root)
    ui.style_titlebar(root)
    width, height = 900, 600
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 3)
    root.geometry(f"{width}x{height}+{x}+{y}")
    defaults = _Defaults(version=installer_version(), install_dir=Path(args.dir) if args.dir else core.default_install_dir())
    Wizard(root, defaults, mode="uninstall" if args.uninstall else "install")
    root.mainloop()
    _log_event("--- setup done exit=0 (wizard closed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
