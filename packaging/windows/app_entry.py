"""Quarkon desktop entry point (``Quarkon.exe``).

The application interface is the local web UI the pipeline ships; this executable
is the double-clickable shell around it:

* starts the loopback server on a free port and opens the default browser once it
  is listening (the per-session token is already in the URL, so there is no login),
* keeps a small window with the address, a copy button and a Stop button -
  closing the window stops the server, so the app never leaves a hidden daemon,
* redirects ``stdout``/``stderr`` to ``%LOCALAPPDATA%\\Quarkon\\logs\\app.log``
  because a windowed executable has no console,
* offers ``--check``: a headless self-test used by the installer smoke test and by
  support ("does the interface answer on this computer?").

Only one instance runs per user; a second double-click focuses the running
interface instead of starting a second server.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from quarkon.ui.server import UIServer

APP_TITLE = "Quarkon"
DEFAULT_PORT = 8765
_HTTP_TIMEOUT = 3.0


# Tk drops images that have no Python reference; the brand mark is kept here for
# the lifetime of the process.
_IMAGE_KEEPALIVE: list[Any] = []


# ------------------------------------------------------------------ bootstrap
def _log_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / "Quarkon" / "logs" / "app.log"


def _redirect_streams() -> Path:
    """Windowed apps have no console: point output at a log file instead."""
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.stdout is not None and sys.stderr is not None:
        return path
    try:
        handle = path.open("a", encoding="utf-8", buffering=1)
    except OSError:
        return path
    sys.stdout = sys.stdout or handle
    sys.stderr = sys.stderr or handle
    return path


def _load_ui_kit() -> Any:
    """Import the shared design system from the build folder or the source tree."""
    try:
        import ui_kit

        return ui_kit
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ui_kit

        return ui_kit


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _single_instance_lock() -> Any | None:
    """Named mutex so a second double-click does not start a second server."""
    if sys.platform != "win32":
        return threading.Lock()
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        kernel32 = windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, 0, "Local\\QuarkonDesktopInstance")
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            return None
        return handle or None
    except Exception:
        return None


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:  # nosec B310 - loopback only
            return int(response.status) == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _http_body(url: str) -> str:
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:  # nosec B310 - loopback only
        return str(response.read().decode("utf-8", errors="replace"))


class ServerHost:
    """Owns the interface server thread and reports its address."""

    def __init__(self, project_dir: Path, port: int) -> None:
        self.project_dir = project_dir
        self.requested_port = port
        self.server: UIServer | None = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="quarkon-ui", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            from quarkon.ui.server import UIServer

            server = UIServer(
                project_dir=self.project_dir,
                host="127.0.0.1",
                port=self.requested_port,
                open_browser=False,
                locale=None,
            )
            self.server = server
            server.serve_forever()
        except Exception as exc:  # pragma: no cover - depends on the live host
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                with _log_path().open("a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
            except OSError:
                pass

    def url(self) -> str | None:
        server = self.server
        if server is None or not server.port:
            return None
        return f"http://127.0.0.1:{server.port}/?token={server.token}"

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                return False
            server = self.server
            if server is not None and server.port and _http_ok(f"http://127.0.0.1:{server.port}/"):
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()


# ---------------------------------------------------------------- self-test
def _append_log(payload: dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_check(port: int = 0) -> int:
    """Start the server, fetch the page from the loopback and report the result."""
    project = Path.home() / APP_TITLE
    try:
        project.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _append_log({"check": "ui", "ok": False, "error": f"project folder not writable: {exc}"})
        return 1
    host = ServerHost(project, port or _free_port())
    host.start()
    ready = host.wait_until_ready()
    url = host.url() or ""
    page_ok = False
    if ready and host.server is not None:
        try:
            page_ok = "Quarkon" in _http_body(f"http://127.0.0.1:{host.server.port}/")
        except (urllib.error.URLError, OSError, ValueError):
            page_ok = False
    code = 0 if (ready and page_ok) else 1
    _append_log(
        {"check": "ui", "ok": code == 0, "url": url, "page_ok": page_ok, "error": host.error}
    )
    if sys.stdout is not None:
        sys.stdout.write(("CHECK OK " if code == 0 else "CHECK FAILED ") + url + "\n")
        sys.stdout.flush()
    host.stop()
    return code


# -------------------------------------------------------------------- desktop
def _run_desktop(port: int, open_browser: bool) -> int:
    import tkinter as tk
    from tkinter import messagebox

    ui = _load_ui_kit()
    ui.enable_dpi_awareness()

    lock = _single_instance_lock()
    if lock is None:
        existing = f"http://127.0.0.1:{DEFAULT_PORT}/"
        if _http_ok(existing):
            webbrowser.open(existing)
            return 0
        messagebox.showinfo(
            APP_TITLE,
            "Quarkon is already running.\n\n"
            f"If its window is not visible, open {existing} in your browser.",
        )
        return 0

    project = Path.home() / APP_TITLE
    try:
        project.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        messagebox.showerror(
            APP_TITLE,
            f"Quarkon cannot create its project folder at {project}.\n\n{exc}\n\n"
            "Ask IT whether the folder is redirected or read-only.",
        )
        return 1
    chosen_port = port or (DEFAULT_PORT if _port_is_free("127.0.0.1", DEFAULT_PORT) else _free_port())
    host = ServerHost(project, chosen_port)
    host.start()

    root = tk.Tk()
    root.title(f"{APP_TITLE} — local model studio")
    root.configure(bg=ui.BG)
    root.resizable(False, False)
    ui.apply_theme(root)
    ui.set_window_icon(root)
    ui.style_titlebar(root)
    width, height = 760, 540
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - width) // 2)
    y = max(0, (root.winfo_screenheight() - height) // 3)
    root.geometry(f"{width}x{height}+{x}+{y}")

    header = tk.Canvas(root, height=170, bg=ui.BG, highlightthickness=0, bd=0)
    header.pack(fill="x")
    ui.gradient(header, width, 170, "#132033", "#0b0f14")
    photo = ui.brand_image(header, 64)
    if photo is not None:
        _IMAGE_KEEPALIVE.append(photo)
        header.create_image(30, 30, image=photo, anchor="nw")
    header.create_text(112, 42, text=APP_TITLE, anchor="nw", fill=ui.TEXT, font=ui.fonts()["display"])
    header.create_text(
        112, 84,
        text="Distill a reasoning model into a compact model on this computer",
        anchor="nw", fill=ui.MUTED, font=ui.fonts()["subtitle"],
    )
    header.create_text(
        30, 140, text="Runs on 127.0.0.1 only  ·  token-protected  ·  nothing leaves this PC",
        anchor="nw", fill=ui.MUTED_DIM, font=ui.fonts()["small"],
    )

    body = tk.Frame(root, bg=ui.BG)
    body.pack(fill="both", expand=True, padx=30, pady=(18, 0))

    status_var = tk.StringVar(value="Starting the interface…")
    tk.Label(
        body, textvariable=status_var, bg=ui.BG, fg=ui.MUTED,
        font=ui.fonts()["subtitle"], anchor="w",
    ).pack(fill="x")

    url_var = tk.StringVar(value="")
    url_row = tk.Frame(body, bg=ui.SURFACE, highlightthickness=1, highlightbackground=ui.BORDER)
    url_row.pack(fill="x", pady=(10, 16))
    tk.Entry(
        url_row, textvariable=url_var, bg=ui.SURFACE, fg=ui.TEXT, bd=0,
        readonlybackground=ui.SURFACE, insertbackground=ui.TEXT,
        font=ui.fonts()["body"], state="readonly",
    ).pack(side="left", fill="x", expand=True, padx=14, pady=12)

    actions = tk.Frame(body, bg=ui.BG)
    actions.pack(fill="x", pady=(0, 10))

    def open_ui() -> None:
        if host.url() is not None:
            webbrowser.open(host.url() or "")

    def copy_url() -> None:
        root.clipboard_clear()
        root.clipboard_append(host.url() or url_var.get())

    def open_projects() -> None:
        starter = getattr(os, "startfile", None)
        if starter is not None:
            starter(str(project))

    def quit_app() -> None:
        status_var.set("Stopping the interface…")
        root.update_idletasks()
        host.stop()
        root.after(150, root.destroy)

    ui.AccentButton(actions, text="Open interface", command=open_ui, width=176).pack(side="left")
    if sys.platform == "win32":
        ui.AccentButton(actions, text="Copy address", command=copy_url, kind="secondary", width=132).pack(
            side="left", padx=(10, 0)
        )
        ui.AccentButton(actions, text="Projects folder", command=open_projects, kind="secondary", width=148).pack(
            side="left", padx=(10, 0)
        )
    ui.AccentButton(actions, text="Stop Quarkon", command=quit_app, kind="ghost", width=140).pack(side="right")

    hint_var = tk.StringVar(
        value="The interface opens in your browser. Keep this window open while you work."
    )
    tk.Label(
        body, textvariable=hint_var, bg=ui.BG, fg=ui.MUTED_DIM,
        font=ui.fonts()["small"], anchor="w", justify="left", wraplength=690,
    ).pack(fill="x", pady=(6, 0))

    tk.Label(
        root, text=f"{APP_TITLE} · projects in {project}", bg=ui.BG, fg=ui.MUTED_DIM,
        font=ui.fonts()["tiny"], anchor="w",
    ).pack(fill="x", padx=30, pady=(4, 12))

    state = {"opened": False}

    def apply_ready() -> None:
        url_var.set(host.url() or "")
        status_var.set("Ready — the interface is running")
        hint_var.set(
            "The address above is personal to this session. Closing this window stops the interface."
        )
        if open_browser and not state["opened"]:
            state["opened"] = True
            webbrowser.open(host.url() or "")

    def apply_error() -> None:
        status_var.set("The interface could not start")
        hint_var.set(
            f"{host.error or 'unknown error'}. Close Quarkon and start it again; "
            r"the log is in %LOCALAPPDATA%\Quarkon\logs\app.log."
        )

    def wait_worker() -> None:
        if host.wait_until_ready():
            root.after(0, apply_ready)
        else:
            root.after(0, apply_error)

    threading.Thread(target=wait_worker, name="quarkon-ready", daemon=True).start()
    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()
    _ = lock  # the mutex handle is intentionally held for the process lifetime
    return 0


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    """Entry point used by PyInstaller and by the tests."""
    parser = argparse.ArgumentParser(prog="Quarkon", description="Quarkon desktop application")
    parser.add_argument("--port", type=int, default=0, help="interface port (0 = choose a free one)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--check", action="store_true", help="headless self-test of the interface")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)

    _redirect_streams()
    if args.version:
        from quarkon.core.version import __version__ as version

        if sys.stdout is not None:
            sys.stdout.write(f"Quarkon {version}\n")
        return 0
    if args.check:
        return run_check(port=args.port)
    return _run_desktop(port=args.port, open_browser=not args.no_browser)


def _guarded_main() -> int:
    """Last-resort handler: a windowed build has nowhere to print a traceback."""
    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        try:
            crash = Path(os.environ.get("TEMP", ".")) / "quarkon-app-crash.log"
            crash.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(_guarded_main())
