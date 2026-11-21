"""One-off: update RELEASE.md for the Quarkon 2.0.0 release."""

from __future__ import annotations

from pathlib import Path

path = Path("RELEASE.md")
src = path.read_text(encoding="utf-8")

pairs = [
    (
        "Status of the **v1.1.1** release (security-hardened).",
        "Status of the **v2.0.0** release (renamed to Quarkon; security-hardened).",
    ),
    (
        "| Version | `1.1.1` (defined once in `quarkon/core/version.py`) |",
        "| Version | `2.0.0` (defined once in `quarkon/core/version.py`) |",
    ),
    (
        "| Repository | <https://github.com/Kdjkvv7886/quarkon> (private) |",
        "| Repository | <https://github.com/Kdjkvv7886/quarkon> (private) |",
    ),
    (
        "| Artefacts | `quarkon-1.1.1-py3-none-any.whl`, `quarkon-1.1.1.tar.gz` |",
        "| Artefacts | `quarkon-2.0.0-py3-none-any.whl`, `quarkon-2.0.0.tar.gz` |",
    ),
    ("| Tags | `v1.1.0`, `v1.1.1` |", "| Tags | `v1.1.0`, `v1.1.1`, `v2.0.0` |"),
    ("`quarkon --version` -> 1.1.1", "`quarkon --version` -> 2.0.0"),
    ("`quarkon lint`", "`quarkon lint`"),
]

for old, new in pairs:
    if old in src:
        src = src.replace(old, new, 1)

# Record the rename migration next to the inventory table.
marker = "## Verified locally"
if "## Rename (2.0.0)" not in src:
    rename_section = """## Rename (2.0.0)

| Before | After |
|---|---|
| `pip install opendistill` | `pip install quarkon` |
| `from opendistill import DistillationEngine` | `from quarkon import DistillationEngine` |
| `opendistill --config ... run` | `quarkon --config ... run` |
| `OPENDISTILL_DEBUG` | `QUARKON_DEBUG` |
| `logs/opendistill.log` | `logs/quarkon.log` |
| `OpenDistillError` | `QuarkonError` |

Unchanged: `DEEPSEEK_API_KEY`, the YAML schema, the workspace layout, the
manifest format and every security control described below.

"""
    src = src.replace(marker, rename_section + marker, 1)

path.write_text(src, encoding="utf-8", newline="\n")
print("RELEASE.md: Quarkon 2.0.0 facts recorded")
