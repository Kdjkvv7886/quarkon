"""Localisation for the Quarkon interface.

Design goals:

* **Stdlib only.** Enterprise environments often block package installs; the
  translation layer must not add a dependency.
* **Graceful fallback.** A missing key falls back to the base language and then
  to a humanised key, so a partially translated catalog never crashes the UI.
* **Honest coverage.** Each locale declares what it covers; the UI shows that
  instead of silently mixing languages.

Locales live as flat JSON dictionaries in ``locales/<code>.json``. Adding a
language is a data-only change: drop in the file, register it in
:data:`SUPPORTED_LOCALES`.

Russian is intentionally not shipped (product decision, not an oversight).
"""

from __future__ import annotations

import json
import locale as _system_locale
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

__all__ = [
    "BASE_LOCALE",
    "SUPPORTED_LOCALES",
    "Coverage",
    "LocaleInfo",
    "Translator",
    "available_locales",
    "detect_locale",
    "get_translator",
    "load_catalog",
    "save_preferred_locale",
    "t",
]

BASE_LOCALE: Final[str] = "en"
_LOCALES_DIR: Final[Path] = Path(__file__).parent / "locales"
_PREFERENCES: Final[Path] = Path.home() / ".quarkon" / "preferences.json"


class Coverage(str):
    """How complete a locale is (kept as a string for JSON friendliness)."""


@dataclass(frozen=True)
class LocaleInfo:
    """Registry entry for one language."""

    code: str
    english_name: str
    native_name: str
    rtl: bool = False
    #: ``"full"`` for interface + guidance, ``"partial"`` when some keys fall
    #: back to English (the UI surfaces this instead of hiding it).
    coverage: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "english_name": self.english_name,
            "native_name": self.native_name,
            "rtl": self.rtl,
            "coverage": self.coverage,
        }


#: Languages shipped with this release. European, Middle Eastern and Asian
#: locales are listed here; each catalog carries the same key set as English.
SUPPORTED_LOCALES: Final[tuple[LocaleInfo, ...]] = (
    LocaleInfo("en", "English", "English"),
    LocaleInfo("uk", "Ukrainian", "Українська"),
    LocaleInfo("de", "German", "Deutsch"),
    LocaleInfo("fr", "French", "Français"),
    LocaleInfo("es", "Spanish", "Español"),
    LocaleInfo("it", "Italian", "Italiano"),
    LocaleInfo("pt", "Portuguese", "Português"),
    LocaleInfo("nl", "Dutch", "Nederlands"),
    LocaleInfo("pl", "Polish", "Polski"),
    LocaleInfo("cs", "Czech", "Čeština"),
    LocaleInfo("ro", "Romanian", "Română"),
    LocaleInfo("el", "Greek", "Ελληνικά"),
    LocaleInfo("sv", "Swedish", "Svenska"),
    LocaleInfo("fi", "Finnish", "Suomi"),
    LocaleInfo("tr", "Turkish", "Türkçe"),
    LocaleInfo("ar", "Arabic", "العربية", rtl=True),
    LocaleInfo("he", "Hebrew", "עברית", rtl=True),
    LocaleInfo("hi", "Hindi", "हिन्दी"),
    LocaleInfo("zh-Hans", "Chinese (Simplified)", "简体中文"),
    LocaleInfo("ja", "Japanese", "日本語"),
    LocaleInfo("ko", "Korean", "한국어"),
)

#: Every language the product intends to ship; the UI shows progress honestly so
#: enterprise buyers know exactly what is translated today.
LOCALE_ROADMAP: Final[tuple[tuple[str, str], ...]] = (
    ("da", "Danish"),
    ("no", "Norwegian"),
    ("is", "Icelandic"),
    ("et", "Estonian"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("hr", "Croatian"),
    ("sr", "Serbian"),
    ("bg", "Bulgarian"),
    ("hu", "Hungarian"),
    ("ca", "Catalan"),
    ("eu", "Basque"),
    ("gl", "Galician"),
    ("ga", "Irish"),
    ("mt", "Maltese"),
    ("zh-Hant", "Chinese (Traditional)"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("fil", "Filipino"),
    ("bn", "Bengali"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("mr", "Marathi"),
    ("gu", "Gujarati"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("ne", "Nepali"),
    ("si", "Sinhala"),
    ("km", "Khmer"),
    ("my", "Burmese"),
    ("lo", "Lao"),
    ("mn", "Mongolian"),
    ("kk", "Kazakh"),
)

_LOCALES_BY_CODE: Final[dict[str, LocaleInfo]] = {info.code: info for info in SUPPORTED_LOCALES}
_ALIASES: Final[dict[str, str]] = {
    "ua": "uk",
    "zh": "zh-Hans",
    "zh_cn": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh_hans": "zh-Hans",
    "he-il": "he",
    "iw": "he",
}


def available_locales() -> list[LocaleInfo]:
    """Locales that can be selected today."""
    return list(SUPPORTED_LOCALES)


def _normalise(code: str) -> str:
    cleaned = code.strip().replace("_", "-")
    lowered = cleaned.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    for info in SUPPORTED_LOCALES:
        if info.code.lower() == lowered or info.code.lower().split("-")[0] == lowered:
            return info.code
    return BASE_LOCALE


def detect_locale(explicit: str | None = None) -> str:
    """Resolve the interface language: explicit flag > env > OS locale > base."""
    if explicit:
        return _normalise(explicit)
    env_value = os.environ.get("QUARKON_LANG", "").strip()
    if env_value:
        return _normalise(env_value)
    try:
        system = _system_locale.getlocale()[0] or ""
    except (ValueError, TypeError):  # pragma: no cover - platform dependent
        system = ""
    return _normalise(system) if system else BASE_LOCALE


def save_preferred_locale(code: str) -> str:
    """Persist the user's language choice for the GUI and CLI."""
    resolved = _normalise(code)
    _PREFERENCES.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if _PREFERENCES.is_file():
        try:
            payload = json.loads(_PREFERENCES.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["locale"] = resolved
    _PREFERENCES.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    get_translator.cache_clear()
    return resolved


def preferred_locale() -> str | None:
    """Read the saved preference, if any."""
    if not _PREFERENCES.is_file():
        return None
    try:
        payload = json.loads(_PREFERENCES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("locale")
    return str(value) if value else None


@lru_cache(maxsize=64)
def load_catalog(code: str) -> dict[str, str]:
    """Load one catalog (cached). Missing files yield an empty mapping."""
    path = _LOCALES_DIR / f"{code}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt install
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _humanise(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_", " ").strip().capitalize()


class Translator:
    """Message lookup with a two-step fallback chain."""

    def __init__(self, locale: str = BASE_LOCALE) -> None:
        self.locale = _normalise(locale)
        self.info = _LOCALES_BY_CODE.get(self.locale, _LOCALES_BY_CODE[BASE_LOCALE])
        self._catalog = load_catalog(self.locale)
        self._base = load_catalog(BASE_LOCALE)

    def t(self, key: str, /, **kwargs: object) -> str:
        template = self._catalog.get(key) or self._base.get(key) or _humanise(key)
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError, ValueError):  # pragma: no cover - bad placeholder
                return template
        return template

    # Alias so UI code reads naturally.
    __call__ = t

    def has(self, key: str) -> bool:
        return key in self._catalog

    def missing_keys(self) -> list[str]:
        return sorted(set(self._base) - set(self._catalog))

    def to_dict(self) -> dict[str, Any]:
        payload = self.info.to_dict()
        payload["keys"] = len(self._catalog)
        payload["missing"] = len(self.missing_keys())
        return payload


@lru_cache(maxsize=64)
def get_translator(locale: str | None = None) -> Translator:
    """Translator for the requested locale, honouring the saved preference."""
    requested = locale or preferred_locale() or detect_locale()
    return Translator(requested)


def t(key: str, /, *, locale: str | None = None, **kwargs: object) -> str:
    """Module-level shortcut used by CLI commands and the web UI."""
    return get_translator(locale).t(key, **kwargs)
