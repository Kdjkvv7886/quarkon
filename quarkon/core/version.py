"""Single source of truth for the package version (Quarkon).

The version appears in three places that must agree: this module, the project
metadata in ``pyproject.toml`` (which builds the wheel) and the changelog. A
packaging test asserts all three match, so a release cannot drift silently.
"""

from __future__ import annotations

from typing import Final

__all__ = ["__version__"]

__version__: Final[str] = "3.2.1"
