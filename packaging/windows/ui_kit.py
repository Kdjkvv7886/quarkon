"""Shared visual language for the Quarkon Windows installer and app window.

Both executables are Tkinter programs, so the design system lives here instead of
being duplicated: one palette, one set of widgets, one brand mark. The goal is a
calm, dark, high-contrast interface that looks intentional on a corporate desktop
and stays readable on a projector.

Everything is drawn with the Tk canvas (no external theme engine, no images that
the build could forget to ship): rounded panels, gradient hero, focus rings and
hover states are all vector primitives.
"""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
from typing import Final

__all__ = [
    "ACCENT",
    "ACCENT_2",
    "BG",
    "BORDER",
    "DANGER",
    "MUTED",
    "OK",
    "SURFACE",
    "SURFACE_ALT",
    "TEXT",
    "WARN",
    "AccentButton",
    "CheckRow",
    "HeroPanel",
    "IconButton",
    "LogView",
    "ProgressStrip",
    "TextLink",
    "apply_theme",
    "brand_image",
    "darken",
    "enable_dpi_awareness",
    "fonts",
    "gradient",
    "round_rect",
    "set_window_icon",
    "style_titlebar",
]

# --------------------------------------------------------------------- palette
BG: Final[str] = "#0b0f14"
SURFACE: Final[str] = "#111821"
SURFACE_ALT: Final[str] = "#0d141c"
BORDER: Final[str] = "#1e2a38"
BORDER_SOFT: Final[str] = "#17222f"
TEXT: Final[str] = "#e8eef7"
MUTED: Final[str] = "#93a4b8"
MUTED_DIM: Final[str] = "#61738a"
ACCENT: Final[str] = "#3b82f6"
ACCENT_2: Final[str] = "#8b5cf6"
ACCENT_HOVER: Final[str] = "#4f8cff"
OK: Final[str] = "#22c55e"
WARN: Final[str] = "#f59e0b"
DANGER: Final[str] = "#ef4444"

def _asset_dir() -> Path:
    """Locate the bundled assets in the source tree and inside a PyInstaller bundle."""
    candidates = [Path(__file__).resolve().parent / "assets"]
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.append(Path(bundle) / "assets")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def enable_dpi_awareness() -> None:
    """Ask Windows for per-monitor DPI scaling so the window is not blurry."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            windll.user32.SetProcessDPIAware()
    except Exception:
        # A missing shcore/DPI API only means slightly softer text; never fatal.
        pass


def _scale() -> float:
    """Best-effort DPI scale factor used to size fonts and paddings."""
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return 1.0
        hdc = windll.user32.GetDC(0)
        dpi = windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        windll.user32.ReleaseDC(0, hdc)
        return float(max(1.0, min(2.0, dpi / 96.0)))
    except Exception:
        return 1.0


_SCALE: float = 1.0


def px(value: int) -> int:
    """Scale a pixel value for the current DPI."""
    return max(1, round(value * _SCALE))


def _family(candidates: tuple[str, ...], fallback: str) -> str:
    try:
        available = {name.lower() for name in tkfont.families()}
    except tk.TclError:
        return fallback
    for candidate in candidates:
        if candidate.lower() in available:
            return candidate
    return fallback


_FONT_CACHE: dict[str, tkfont.Font] = {}


def fonts() -> dict[str, tkfont.Font]:
    """Create the type scale once the Tk root exists, then reuse it."""
    if _FONT_CACHE:
        return _FONT_CACHE
    body = _family(("Segoe UI Variable Text", "Segoe UI", "Inter"), "TkDefaultFont")
    display = _family(("Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI"), body)
    scale = _SCALE
    created = {
        "display": tkfont.Font(family=display, size=max(16, round(21 * scale)), weight="bold"),
        "title": tkfont.Font(family=display, size=max(13, round(15 * scale)), weight="bold"),
        "subtitle": tkfont.Font(family=body, size=max(10, round(11 * scale))),
        "body": tkfont.Font(family=body, size=max(9, round(10 * scale))),
        "small": tkfont.Font(family=body, size=max(8, round(9 * scale))),
        "tiny": tkfont.Font(family=body, size=max(7, round(8 * scale))),
        "mono": tkfont.Font(family=_family(("Cascadia Mono", "Consolas", "Courier New"), "TkFixedFont"), size=max(8, round(9 * scale))),
        "button": tkfont.Font(family=body, size=max(9, round(10 * scale)), weight="bold"),
    }
    _FONT_CACHE.update(created)
    return _FONT_CACHE


def apply_theme(root: tk.Misc) -> None:
    """Apply base colours to classic Tk widgets and ttk controls."""
    global _SCALE
    _SCALE = _scale()
    root.tk_setPalette(
        background=BG,
        foreground=TEXT,
        activeBackground=SURFACE,
        activeForeground=TEXT,
        highlightBackground=BORDER,
        highlightColor=ACCENT,
        selectBackground=ACCENT,
        selectForeground="#ffffff",
    )
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("TSeparator", background=BORDER)


def style_titlebar(window: tk.Tk) -> None:
    """Switch the native title bar to dark mode (Windows 10 1809+ / 11)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        window.update_idletasks()
        hwnd = windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE, then pre-20H1 id
            result = windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
            if result == 0:
                break
    except Exception:
        # Cosmetic only: an old build keeps the light title bar and still works.
        pass


def set_window_icon(window: tk.Tk, name: str = "quarkon.ico") -> None:
    """Set the task-bar icon from the bundled assets, if present."""
    path = _asset_dir() / name
    if not path.is_file():
        return
    try:
        window.iconbitmap(default=str(path))  # type: ignore[no-untyped-call]
    except tk.TclError:
        pass


def brand_image(master: tk.Misc, size: int = 64) -> tk.PhotoImage | None:
    """Load the brand PNG at ``size`` px using integer sub-sampling (crisp, no blur)."""
    path = _asset_dir() / "quarkon-256.png"
    if not path.is_file():
        return None
    try:
        image = tk.PhotoImage(master=master, file=str(path))
    except tk.TclError:
        return None
    factor = max(1, 256 // max(size, 1))
    if factor > 1:
        image = image.subsample(factor, factor)
    return image


# ------------------------------------------------------------------- primitives
def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mix(color_a: str, color_b: str, t: float) -> str:
    """Linear blend between two hex colours (``t`` = 0 -> a, 1 -> b)."""
    a = _hex_to_rgb(color_a)
    b = _hex_to_rgb(color_b)
    return _rgb_to_hex(tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))  # type: ignore[arg-type]


def darken(color: str, factor: float = 0.18) -> str:
    """Return ``color`` blended towards black."""
    return mix(color, "#000000", factor)


def gradient(
    canvas: tk.Canvas,
    width: int,
    height: int,
    top: str,
    bottom: str,
    *,
    horizontal: bool = False,
    steps: int = 220,
) -> None:
    """Paint a smooth two-stop gradient by drawing thin lines."""
    canvas.delete("gradient")
    span = width if horizontal else height
    count = max(2, min(steps, span))
    for index in range(count):
        t = index / (count - 1)
        color = mix(top, bottom, t)
        if horizontal:
            x0 = round(width * index / count)
            x1 = round(width * (index + 1) / count) + 1
            canvas.create_rectangle(x0, 0, x1, height, fill=color, outline=color, tags="gradient")
        else:
            y0 = round(height * index / count)
            y1 = round(height * (index + 1) / count) + 1
            canvas.create_rectangle(0, y0, width, y1, fill=color, outline=color, tags="gradient")
    canvas.tag_lower("gradient")


def round_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    *,
    fill: str = "",
    outline: str = "",
    width: float = 1.0,
) -> int:
    """Draw a rounded rectangle; returns the canvas item id."""
    radius = max(0.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return int(
        canvas.create_polygon(
            points, smooth=True, splinesteps=24, fill=fill, outline=outline, width=width
        )
    )


# --------------------------------------------------------------------- widgets
class HeroPanel(tk.Frame):
    """Left rail of the wizard: gradient background, brand mark, step list."""

    def __init__(self, master: tk.Misc, *, width: int = 268, steps: tuple[str, ...] = ()) -> None:
        super().__init__(master, bg=BG, width=width)
        self.pack_propagate(False)
        self._canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=BG)
        self._canvas.pack(fill="both", expand=True)
        self._steps = steps
        self._active = 0
        self._width = width
        self._photo: tk.PhotoImage | None = None
        self._canvas.bind("<Configure>", lambda _event: self._redraw())

    def set_steps(self, steps: tuple[str, ...]) -> None:
        self._steps = steps
        self._redraw()

    def set_active(self, index: int) -> None:
        self._active = index
        self._redraw()

    def _redraw(self) -> None:
        canvas = self._canvas
        width = self.winfo_width() if self.winfo_width() > 10 else self._width
        height = self.winfo_height() if self.winfo_height() > 10 else 560
        canvas.delete("all")
        gradient(canvas, width, height, "#101a2a", "#070a0f", steps=160)
        # Accent glows in the upper corner.
        for index in range(9):
            colour = mix(ACCENT, "#070a0f", 0.55 + index * 0.045)
            canvas.create_oval(
                -width * 0.45 + index * 6, -height * 0.30 + index * 6,
                width * 0.55 - index * 4, height * 0.28 - index * 5,
                outline="", fill=colour, tags="glow",
            )
        canvas.tag_lower("glow")
        # Hairline on the right edge.
        canvas.create_line(width - 1, 0, width - 1, height, fill="#182534")

        self._photo = self._photo or brand_image(self, 64)
        if self._photo is not None:
            canvas.create_image(34, 40, image=self._photo, anchor="nw")
        canvas.create_text(
            34, 124, text="Quarkon", anchor="nw",
            fill=TEXT, font=fonts()["display"],
        )
        canvas.create_text(
            34, 158, text="Local reasoning models,\nready for your team", anchor="nw",
            fill=MUTED, font=fonts()["small"],
        )

        y = 214
        for index, label in enumerate(self._steps):
            active = index == self._active
            done = index < self._active
            colour = TEXT if active else (MUTED if not done else MUTED_DIM)
            badge = ACCENT if active else ("#1b2b3f" if not done else "#16241d")
            badge_text = f"{index + 1}" if not done else "✓"
            canvas.create_oval(34, y - 11, 56, y + 11, fill=badge, outline="")
            canvas.create_text(
                45, y, text=badge_text, fill=TEXT if active or done else MUTED_DIM,
                font=fonts()["tiny"],
            )
            canvas.create_text(68, y, text=label, anchor="w", fill=colour, font=fonts()["body"])
            if index < len(self._steps) - 1:
                canvas.create_line(45, y + 13, 45, y + 27, fill=BORDER, width=1)
            y += 40

        canvas.create_text(
            34, height - 30, text="No administrator rights required", anchor="w",
            fill=MUTED_DIM, font=fonts()["tiny"],
        )


class AccentButton(tk.Canvas):
    """Rounded, animated (hover/press/focus) button drawn on a canvas."""

    _KINDS: Final[dict[str, tuple[str, str, str]]] = {
        "primary": (ACCENT, ACCENT_HOVER, "#ffffff"),
        "secondary": (SURFACE, "#16202c", TEXT),
        "ghost": (BG, "#141d28", MUTED),
        "danger": ("#7f1d1d", DANGER, "#ffffff"),
    }

    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str,
        command: Callable[[], None],
        kind: str = "primary",
        width: int = 148,
        height: int = 42,
        font_key: str = "button",
    ) -> None:
        super().__init__(
            master, width=px(width), height=px(height), bg=BG, highlightthickness=0, bd=0, cursor="hand2"
        )
        self._text = text
        self._command = command
        self._kind = kind
        self._width_px = px(width)
        self._height_px = px(height)
        self._font_key = font_key
        self._state = "normal"
        self._hover = False
        self._focus = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Return>", lambda _event: self._invoke())
        self.bind("<space>", lambda _event: self._invoke())
        self.configure(takefocus=True)
        self._redraw()

    # -- api
    def set_text(self, text: str) -> None:
        self._text = text
        self._redraw()

    def set_state(self, state: str) -> None:
        """``normal``, ``disabled`` or ``busy`` (busy disables and shows a hint)."""
        self._state = state
        self.configure(cursor="hand2" if state == "normal" else "arrow")
        self._redraw()

    # -- events
    def _on_enter(self, _event: tk.Event[tk.Misc]) -> None:
        self._hover = True
        self._redraw()

    def _on_leave(self, _event: tk.Event[tk.Misc]) -> None:
        self._hover = False
        self._redraw()

    def _on_press(self, _event: tk.Event[tk.Misc]) -> None:
        if self._state != "normal":
            return
        self._focus = True
        self.focus_set()
        self._redraw()

    def _on_release(self, event: tk.Event[tk.Misc]) -> None:
        if self._state != "normal":
            return
        inside = 0 <= event.x <= self._width_px and 0 <= event.y <= self._height_px
        self._redraw()
        if inside:
            self._invoke()

    def _on_focus_in(self, _event: tk.Event[tk.Misc]) -> None:
        self._focus = True
        self._redraw()

    def _on_focus_out(self, _event: tk.Event[tk.Misc]) -> None:
        self._focus = False
        self._redraw()

    def _invoke(self) -> None:
        if self._state == "normal":
            self._command()

    # -- drawing
    def _redraw(self) -> None:
        self.delete("all")
        base, hover, fg = self._KINDS[self._kind]
        fill = hover if (self._hover and self._state == "normal") else base
        if self._state == "disabled":
            fill = SURFACE_ALT
            fg = MUTED_DIM
        radius = self._height_px / 2 if self._kind == "primary" else px(10)
        round_rect(self, 1, 1, self._width_px - 1, self._height_px - 1, radius, fill=fill, outline="")
        if self._kind in {"secondary", "ghost"}:
            round_rect(self, 1, 1, self._width_px - 1, self._height_px - 1, radius, fill="", outline=BORDER)
        if self._kind == "primary" and self._state == "normal":
            # A one-pixel inner highlight keeps the pill from looking flat.
            round_rect(
                self, 2, 2, self._width_px - 2, self._height_px - 2, radius - 1,
                fill="", outline=mix(ACCENT_HOVER, "#ffffff", 0.25),
            )
        if self._focus:
            round_rect(
                self, 0, 0, self._width_px, self._height_px, radius + 1,
                fill="", outline=mix(ACCENT, "#ffffff", 0.35),
            )
        self.create_text(
            self._width_px / 2, self._height_px / 2 + 1, text=self._text, fill=fg, font=fonts()[self._font_key]
        )
        if self._state == "busy":
            self.create_text(
                self._width_px / 2, self._height_px - px(6), text="working…", fill=MUTED_DIM, font=fonts()["tiny"]
            )


class IconButton(tk.Canvas):
    """Square, tooltip-less icon button used for the window actions."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        glyph: str,
        command: Callable[[], None],
        size: int = 30,
        tooltip: str = "",
    ) -> None:
        super().__init__(master, width=px(size), height=px(size), bg=SURFACE, highlightthickness=0, bd=0, cursor="hand2")
        self._glyph = glyph
        self._command = command
        self._size = px(size)
        self._tooltip = tooltip
        self._hover = False
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", lambda _event: self._command())
        self._redraw()

    def _enter(self, _event: tk.Event[tk.Misc]) -> None:
        self._hover = True
        self._redraw()

    def _leave(self, _event: tk.Event[tk.Misc]) -> None:
        self._hover = False
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        fill = "#1a2532" if self._hover else SURFACE
        round_rect(self, 0, 0, self._size, self._size, px(8), fill=fill, outline="")
        self.create_text(
            self._size / 2, self._size / 2, text=self._glyph, fill=TEXT if self._hover else MUTED,
            font=fonts()["title"],
        )


class CheckRow(tk.Frame):
    """Checkbox row with a label and an explanatory second line."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str,
        hint: str,
        variable: tk.BooleanVar,
        on_toggle: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, bg=SURFACE)
        self._variable = variable
        self._on_toggle = on_toggle
        self._canvas = tk.Canvas(self, width=px(20), height=px(20), bg=SURFACE, highlightthickness=0, bd=0, cursor="hand2")
        self._canvas.pack(side="left", anchor="n")
        self._labels = tk.Frame(self, bg=SURFACE)
        self._labels.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self._title = tk.Label(
            self._labels, text=text, bg=SURFACE, fg=TEXT, font=fonts()["body"], anchor="w",
        )
        self._title.pack(fill="x")
        self._height_pxint = tk.Label(
            self._labels, text=hint, bg=SURFACE, fg=MUTED_DIM, font=fonts()["small"],
            anchor="w", justify="left",
        )
        self._height_pxint.pack(fill="x")
        for widget in (self, self._canvas, self._labels, self._title, self._height_pxint):
            widget.bind("<Button-1>", self._toggle)
        self._paint()

    def _toggle(self, _event: tk.Event[tk.Misc]) -> None:
        self._variable.set(not self._variable.get())
        self._paint()
        if self._on_toggle is not None:
            self._on_toggle()

    def _paint(self) -> None:
        canvas = self._canvas
        canvas.delete("all")
        checked = self._variable.get()
        round_rect(canvas, 1, 1, px(19), px(19), px(5), fill=ACCENT if checked else SURFACE_ALT, outline=ACCENT if checked else BORDER)
        if checked:
            canvas.create_line(
                [px(5), px(10), px(9), px(14), px(15), px(6)],
                fill="#ffffff", width=max(2, px(2)), capstyle="round", joinstyle="round",
            )


class ProgressStrip(tk.Canvas):
    """Thin gradient progress bar with an optional percentage label."""

    def __init__(self, master: tk.Misc, *, height: int = 8) -> None:
        super().__init__(master, height=px(height), bg=SURFACE, highlightthickness=0, bd=0)
        self._value = 0.0
        self._height_px = px(height)
        self.bind("<Configure>", lambda _event: self._redraw())

    def set(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = self._height_px
        round_rect(self, 0, 0, width, height, height / 2, fill=SURFACE_ALT, outline=BORDER_SOFT)
        filled = width * self._value
        if filled > 2:
            round_rect(self, 0, 0, filled, height, height / 2, fill=ACCENT, outline="")
            if filled > 10:
                for index in range(24):
                    t0 = index / 24
                    t1 = (index + 1) / 24
                    self.create_rectangle(
                        filled * t0, 0, filled * t1 + 1, height,
                        fill=mix(ACCENT, ACCENT_2, t0), outline="", tags="fill",
                    )
                self.tag_lower("fill")


class LogView(tk.Frame):
    """Read-only monospace log with severity colours."""

    def __init__(self, master: tk.Misc, *, height: int = 9) -> None:
        super().__init__(master, bg=SURFACE_ALT, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self._text = tk.Text(
            self, height=height, bg=SURFACE_ALT, fg=MUTED, insertbackground=TEXT, bd=0,
            highlightthickness=0, font=fonts()["mono"], wrap="word", padx=10, pady=8,
            state="disabled", cursor="arrow",
        )
        scroll = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)
        self._text.tag_configure("ok", foreground=OK)
        self._text.tag_configure("warn", foreground=WARN)
        self._text.tag_configure("error", foreground=DANGER)
        self._text.tag_configure("dim", foreground=MUTED_DIM)

    def append(self, line: str, level: str = "info") -> None:
        tag = "" if level == "info" else level
        self._text.configure(state="normal")
        self._text.insert("end", line + "\n", tag)
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


class TextLink(tk.Label):
    """Underlined, hand-cursor label for secondary actions."""

    def __init__(self, master: tk.Misc, *, text: str, command: Callable[[], None]) -> None:
        super().__init__(
            master, text=text, bg=BG, fg=MUTED, font=fonts()["small"], cursor="hand2",
        )
        self._command = command
        self.bind("<Button-1>", lambda _event: self._command())
        self.bind("<Enter>", lambda _event: self.configure(fg=ACCENT))
        self.bind("<Leave>", lambda _event: self.configure(fg=MUTED))
