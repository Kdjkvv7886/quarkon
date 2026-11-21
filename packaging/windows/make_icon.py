"""Generate the Quarkon application icon.

The icon is drawn from scratch (no stock art) so the brand is reproducible and
the repository carries no binary that cannot be regenerated:

    python packaging/windows/make_icon.py

Outputs
-------
``packaging/windows/assets/quarkon.ico``
    Multi-resolution Windows icon (16-256 px) used by the executable, the
    shortcuts and the Add/Remove Programs entry.
``packaging/windows/assets/quarkon-256.png``
    Same artwork at 256 px for documentation and the installer preview.

The mark is a three-armed orbital (a quark trefoil) on a rounded dark tile. It is
rendered at 4x and downsampled so the edges stay smooth on every size.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

#: Canvas size before downsampling. Everything is drawn on this grid and then
#: scaled down, which gives clean antialiasing without a raster editor.
_RENDER: int = 1024
_ICON_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)

_BACKGROUND_TOP: tuple[int, int, int] = (23, 33, 49)
_BACKGROUND_BOTTOM: tuple[int, int, int] = (8, 11, 16)
_ACCENT_BLUE: tuple[int, int, int] = (59, 130, 246)
_ACCENT_VIOLET: tuple[int, int, int] = (139, 92, 246)
_ACCENT_CYAN: tuple[int, int, int] = (34, 211, 238)

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"


def _vertical_gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    """Return a top-to-bottom RGB gradient of ``size`` x ``size`` pixels."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    assert pixels is not None
    for y in range(size):
        blend = y / max(size - 1, 1)
        red = round(top[0] + (bottom[0] - top[0]) * blend)
        green = round(top[1] + (bottom[1] - top[1]) * blend)
        blue = round(top[2] + (bottom[2] - top[2]) * blend)
        for x in range(size):
            pixels[x, y] = (red, green, blue)
    return image


def _rounded_mask(size: int, radius: int) -> Image.Image:
    """Rounded-square alpha mask used to clip the background tile."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _ring_layer(
    size: int,
    center: tuple[float, float],
    radii: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
) -> Image.Image:
    """A single elliptical orbit drawn on a transparent layer."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    rx, ry = radii
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=width)
    return layer


def _arc_layer(
    size: int,
    center: tuple[float, float],
    radii: tuple[float, float],
    start: float,
    end: float,
    color: tuple[int, int, int, int],
    width: int,
) -> Image.Image:
    """A partial orbital arc, used to keep the trefoil from looking like a logo grid."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    rx, ry = radii
    draw.arc((cx - rx, cy - ry, cx + rx, cy + ry), start=start, end=end, fill=color, width=width)
    return layer


def _glow(layer: Image.Image, radius: int = 26) -> Image.Image:
    """Soft halo under the mark so the icon reads on both dark and light taskbars."""
    blurred = layer.filter(ImageFilter.GaussianBlur(radius))
    alpha = blurred.getchannel("A").point(lambda value: min(255, round(value * 0.85)))
    blurred.putalpha(alpha)
    return blurred


def render_mark(size: int = _RENDER) -> Image.Image:
    """Render the full-brand icon (tile + trefoil) at ``size`` pixels."""
    canvas = _vertical_gradient(size, _BACKGROUND_TOP, _BACKGROUND_BOTTOM).convert("RGBA")
    canvas.putalpha(_rounded_mask(size, radius=round(size * 0.225)))

    center = (size / 2, size / 2)
    ring_width = max(round(size * 0.034), 2)
    radius_long = size * 0.252
    radius_short = size * 0.166

    orbits = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    palette = (
        (*_ACCENT_BLUE, 255),
        (*_ACCENT_VIOLET, 255),
        (*_ACCENT_CYAN, 235),
    )
    # Three ellipses rotated by 60 degrees form the trefoil. Each one is drawn as
    # a full ring plus a second, brighter arc on top for a sense of depth.
    for index, angle in enumerate((0, 60, 120)):
        ring = _ring_layer(size, center, (radius_long, radius_short), palette[index], ring_width)
        arc = _arc_layer(
            size,
            center,
            (radius_long - ring_width * 0.35, radius_short - ring_width * 0.35),
            start=angle + 200,
            end=angle + 330,
            color=(255, 255, 255, 190),
            width=max(round(ring_width * 0.45), 2),
        )
        rotated = Image.alpha_composite(ring, arc).rotate(angle, center=center, resample=Image.BICUBIC)
        orbits = Image.alpha_composite(orbits, rotated)

    # Nucleus: a bright dot at the visual centre with a vignette of its own.
    nucleus = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    nucleus_draw = ImageDraw.Draw(nucleus)
    dot_radius = size * 0.047
    halo_radius = dot_radius * 2.6
    # Halo first, dot second: drawing the translucent halo afterwards would
    # overwrite the opaque dot on the same layer.
    nucleus_draw.ellipse(
        (
            center[0] - halo_radius,
            center[1] - halo_radius,
            center[0] + halo_radius,
            center[1] + halo_radius,
        ),
        fill=(120, 190, 255, 70),
    )
    nucleus_draw.ellipse(
        (
            center[0] - dot_radius,
            center[1] - dot_radius,
            center[0] + dot_radius,
            center[1] + dot_radius,
        ),
        fill=(255, 255, 255, 255),
    )

    canvas = Image.alpha_composite(canvas, _glow(orbits, radius=round(size * 0.018)))
    canvas = Image.alpha_composite(canvas, orbits)
    canvas = Image.alpha_composite(canvas, nucleus)
    return canvas


def _downsample(image: Image.Image, size: int) -> Image.Image:
    """High-quality resize that keeps thin arcs visible at 16 px."""
    return image.resize((size, size), Image.LANCZOS)


def main() -> int:
    """Write the icon and the preview PNG; return a process exit code."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    master = render_mark()
    icon_path = ASSETS_DIR / "quarkon.ico"
    png_path = ASSETS_DIR / "quarkon-256.png"

    frames = [_downsample(master, size) for size in _ICON_SIZES]
    frames[-1].save(icon_path, format="ICO", sizes=[(size, size) for size in _ICON_SIZES])
    _downsample(master, 256).save(png_path, format="PNG")

    # Sanity check: every expected frame must survive the ICO round-trip.
    with Image.open(icon_path) as check:
        available = sorted(check.ico.sizes()) if check.format == "ICO" else []
    expected = [(size, size) for size in _ICON_SIZES]
    missing = [frame for frame in expected if frame not in available]
    if missing:
        print(f"error: icon is missing frames: {missing}", file=sys.stderr)
        return 1
    print(f"wrote {icon_path} ({icon_path.stat().st_size} bytes, {len(available)} frames)")
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
    _ = ImageChops  # imported for completeness of the drawing toolkit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
