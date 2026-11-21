"""Clean-Room Protocol: format-preserving surrogate generation for PII.

Privacy module for data that leaves the machine on its way to a teacher API. It
replaces personally identifiable values with **realistic, format-preserving
surrogates** (FPSP - Format-Preserving Natural Entity Surrogacy) instead of
static masks such as ``[PERSON_1]`` or ``[IBAN]``:

* static masks destroy the token statistics the student is trained on (an LLM
  never sees ``[PERSON_1]`` in production text) and they leak *structure*: the
  same mask token appearing twice tells the model that two different strings are
  the same entity, while different masks for the same person across two requests
  destroy that identity;
* surrogates keep the lexical shape (``first.last@domain.tld``, ``+380 (44)
  123-45-67``, a Luhn-valid 16-digit card, a mod-97 valid IBAN), so the teacher's
  tokenizer sees natural language and the resulting traces are trainable.

The mapping lives **only in RAM**, inside :class:`EphemeralVault`, and is
destroyed by :meth:`CleanRoomSanitizer.purge` when a generation run finishes. The
vault is the only place where the original values exist after sanitisation.

Surrogates are derived deterministically from ``(kind, original, salt)`` via
SHA-256, so two runs over the same data produce the same surrogate text without
persisting any mapping - which keeps datasets reproducible for the audit trail.

Honest limitations (documented, not hidden):

* Person/address detection is heuristic (honorifics, explicit patterns and a
  configurable name list), because this module intentionally has no ML
  dependency. Supply ``PrivacyConfig.custom_names`` for domain-specific names.
* Python strings are immutable, so :meth:`purge` can only drop references and
  overwrite the vault containers; it cannot zero the bytes of strings the
  interpreter already created. Purge still removes the mapping (the only place
  that links a surrogate back to a real value) and is what the pipeline calls.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

__all__ = [
    "DEFAULT_ENTITIES",
    "CleanRoomSanitizer",
    "EntityKind",
    "EphemeralVault",
    "Finding",
    "SanitizationResult",
    "SurrogateFactory",
    "VaultStats",
    "iban_valid",
    "luhn_valid",
    "sanitize_text",
]

_ALPHANUM: Final[str] = "abcdefghijklmnopqrstuvwxyz0123456789"
_UPPER_ALNUM: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_DIGITS: Final[str] = "0123456789"

_FIRST_NAMES: Final[tuple[str, ...]] = (
    "Amelia", "Noah", "Sofia", "Liam", "Marta", "Oliver", "Iryna", "Ethan",
    "Clara", "Mateo", "Nadia", "Tomas", "Elena", "Marcus", "Priya", "Jonas",
    "Leila", "Andrii", "Hanna", "Rafael", "Yuki", "Oscar", "Inna", "Diego",
)
_LAST_NAMES: Final[tuple[str, ...]] = (
    "Bennett", "Kovalenko", "Reyes", "Muller", "Shevchenko", "Novak", "Okafor",
    "Lindqvist", "Moreau", "Petrenko", "Hughes", "Berg", "Tanaka", "Rossi",
    "Silva", "Kaur", "Ivanenko", "Fischer", "Almeida", "Bondar", "Larsen",
)
_STREET_NAMES: Final[tuple[str, ...]] = (
    "Klenova", "Redwood", "Chestnut", "Maple", "Harbour", "Orchard", "Linden",
    "Willow", "Granite", "Saffron", "Copper", "Juniper", "Vesela", "Zelena",
)
_STREET_SUFFIXES: Final[tuple[str, ...]] = (
    "Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd",
    "Lane", "Ln", "Drive", "Dr", "Square", "Sq",
)
_CITIES: Final[tuple[str, ...]] = (
    "Northfield", "Lviv", "Riverton", "Ostrava", "Kyiv", "Eastport",
    "Bratislava", "Lakeside", "Gdansk", "Fairview",
)
_ORG_LABELS: Final[tuple[str, ...]] = (
    "northwind", "redwood", "lakeside", "stonebridge", "clearpath",
    "harborline", "silveroak", "bluecrest", "mapleworks", "ironvale",
)
#: Reserved / documentation domains, so surrogates can never resolve to a real
#: mailbox. ``.example`` is reserved by RFC 2606; ``example.*`` is reserved too.
_SAFE_DOMAINS: Final[tuple[str, ...]] = ("example.com", "example.org", "example.net")


class EntityKind(str, Enum):
    """Kinds of identifiers the sanitizer recognises."""

    PERSON = "person"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CARD = "card"
    IBAN = "iban"
    IP = "ip"
    # Enum label, not a credential.
    SECRET = "secret"  # nosec B105


DEFAULT_ENTITIES: Final[tuple[EntityKind, ...]] = (
    EntityKind.EMAIL,
    EntityKind.SECRET,
    EntityKind.CARD,
    EntityKind.IBAN,
    EntityKind.PHONE,
    EntityKind.IP,
    EntityKind.ADDRESS,
    EntityKind.PERSON,
)

#: Detection priority when two matches overlap: higher wins.
_PRIORITY: Final[dict[EntityKind, int]] = {
    EntityKind.EMAIL: 100,
    EntityKind.SECRET: 95,
    EntityKind.IBAN: 90,
    EntityKind.CARD: 85,
    EntityKind.IP: 80,
    EntityKind.PHONE: 70,
    EntityKind.ADDRESS: 60,
    EntityKind.PERSON: 50,
}

_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)",
)
_IBAN_RE: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{2}\d{2})((?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?)\b")
_CARD_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
#: "card ... 4539 5789 1234 5678" - explicit context, checksum not authoritative.
_CARD_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:card|cc|credit\s+card|debit\s+card|pan)\b[^\n]{0,24}?"
    r"((?:\d[ -]?){12,18}\d)"
)
_PHONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w+])(\+?\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d(?:[\d\s.-]{7,15}\d)"
)
_IP_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b")
_SECRET_RES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("sk", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("gh", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
    ("password", re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S+")),
)
_ADDRESS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,5})\s+((?:[A-Z][\w'-]+\s+){0,2}[A-Z][\w'-]+)\s+(" + "|".join(_STREET_SUFFIXES) + r")\b\.?",
)
_CITY_STATE_ZIP_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][a-z][A-Za-z'-]{2,}),\s([A-Z]{2})\s(\d{5})\b"
)
_HONORIFIC_PERSON_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Mx)\.?\s+([A-Z][a-z][A-Za-z'-]+(?:\s+[A-Z][a-z][A-Za-z'-]+)?)"
)
#: The case-insensitive flag is scoped to the trigger word ONLY. Applying ``(?i)``
#: to the whole pattern makes the name group match ordinary lowercase pairs such
#: as "user request for", which silently corrupted prompts and training data.
_NAME_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i:\b(?:name|customer|client|user|employee|contact|patient|student)\b)"
    r"(?:\s+(?i:is|of|for))?:?\s+"
    r"([A-Z][a-z][A-Za-z'-]+\s+[A-Z][a-z][A-Za-z'-]+)"
)


def _digest(kind: str, value: str, salt: str) -> bytes:
    """Deterministic 32-byte digest binding a value to a surrogate space."""
    return hashlib.sha256(f"{salt}\x00{kind}\x00{value}".encode()).digest()


def _pick(pool: Sequence[str], digest: bytes, offset: int = 0) -> str:
    """Pick a stable element of ``pool`` for a digest."""
    index = int.from_bytes(digest[offset : offset + 4], "big") % len(pool)
    return pool[index]


def _digits(digest: bytes, count: int, *, offset: int = 0, avoid_leading_zero: bool = False) -> str:
    """Generate ``count`` deterministic digits from a digest."""
    out: list[str] = []
    cursor = offset
    while len(out) < count:
        byte = digest[cursor % len(digest)]
        cursor += 1
        out.append(_DIGITS[byte % 10])
    if avoid_leading_zero and out and out[0] == "0":
        out[0] = _DIGITS[(digest[0] % 9) + 1]
    return "".join(out)


def _random_body(digest: bytes, length: int, alphabet: str) -> str:
    return "".join(alphabet[digest[index % len(digest)] % len(alphabet)] for index in range(length))


def _apply_case(template: str, replacement: str) -> str:
    """Match the capitalisation style of ``template`` on ``replacement``."""
    if template.isupper():
        return replacement.upper()
    if template.islower():
        return replacement.lower()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def luhn_check_digit(prefix: str) -> str:
    """Return the Luhn check digit for ``prefix`` (used for card surrogates)."""
    digits = [int(char) for char in prefix]
    parity = (len(digits) + 1) % 2
    total = 0
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    """Whether a digit string passes the Luhn checksum."""
    digits = [int(char) for char in number if char.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


_IBAN_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def iban_check_digits(country: str, bban: str) -> str:
    """Compute ISO 13616 check digits for ``country`` + ``bban``."""
    rearranged = bban + country + "00"
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    remainder = int(numeric) % 97
    return f"{98 - remainder:02d}"


def iban_valid(iban: str) -> bool:
    """Validate an IBAN's country code, length and mod-97 checksum."""
    compact = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:  # pragma: no cover - defensive
        return False


@dataclass(frozen=True)
class Finding:
    """One recognised entity in the source text."""

    kind: EntityKind
    start: int
    end: int
    original: str
    surrogate: str

    def to_dict(self) -> dict[str, object]:
        # The original value is intentionally excluded from serialisable output.
        return {
            "kind": self.kind.value,
            "start": self.start,
            "end": self.end,
            "surrogate": self.surrogate,
            "length": len(self.original),
        }


@dataclass(frozen=True)
class SanitizationResult:
    """Sanitised text plus non-identifying statistics."""

    text: str
    findings: tuple[Finding, ...]
    counts: Mapping[EntityKind, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.findings)

    @property
    def total(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, object]:
        """Serialisable summary - never contains original values."""
        return {
            "changed": self.changed,
            "total": self.total,
            "counts": {kind.value: count for kind, count in self.counts.items()},
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class VaultStats:
    """Aggregate counters for one vault (no originals, no surrogates)."""

    entries: int
    by_kind: Mapping[str, int]
    purged: bool


class EphemeralVault:
    """In-memory, bidirectional mapping between real values and surrogates.

    Deliberately not serialisable: there is no ``to_dict``/``save`` method, so an
    accidental ``json.dump`` of run state cannot leak the mapping. The vault also
    refuses to be logged: :meth:`__repr__` exposes counts only.

    The mapping is split per :class:`EntityKind`, so the same literal string can
    be a person in one field and a city in another without collisions.
    """

    __slots__ = ("_counts", "_forward", "_purged", "_reverse")

    def __init__(self) -> None:
        self._forward: dict[tuple[EntityKind, str], str] = {}
        self._reverse: dict[EntityKind, dict[str, str]] = {kind: {} for kind in EntityKind}
        self._counts: dict[EntityKind, int] = {kind: 0 for kind in EntityKind}
        self._purged = False

    # ------------------------------------------------------------- mapping
    def get_or_create(self, kind: EntityKind, original: str, factory: SurrogateFactory) -> str:
        """Return the surrogate for ``original``, creating it once."""
        if self._purged:
            raise RuntimeError("EphemeralVault has been purged; create a new sanitizer")
        key = (kind, original)
        existing = self._forward.get(key)
        if existing is not None:
            self._counts[kind] += 1
            return existing
        surrogate = factory(kind, original)
        # Guard against a surrogate colliding with another real value.
        while surrogate in self._reverse[kind] and self._reverse[kind][surrogate] != original:
            surrogate = factory(kind, original + "\x00")
        self._forward[key] = surrogate
        self._reverse[kind][surrogate] = original
        self._counts[kind] += 1
        return surrogate

    def surrogate(self, kind: EntityKind, original: str) -> str | None:
        return self._forward.get((kind, original))

    def original(self, kind: EntityKind, surrogate: str) -> str | None:
        return self._reverse.get(kind, {}).get(surrogate)

    def surrogates(self, kind: EntityKind) -> tuple[str, ...]:
        """Surrogates issued for ``kind`` (never the original values)."""
        return tuple(self._reverse.get(kind, {}))

    # --------------------------------------------------------------- stats
    @property
    def size(self) -> int:
        return len(self._forward)

    def stats(self) -> VaultStats:
        return VaultStats(
            entries=len(self._forward),
            by_kind={kind.value: count for kind, count in self._counts.items() if count},
            purged=self._purged,
        )

    @property
    def purged(self) -> bool:
        return self._purged

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<EphemeralVault entries={len(self._forward)} purged={self._purged}>"

    __str__ = __repr__

    def __iter__(self) -> Iterator[tuple[EntityKind, str]]:  # pragma: no cover - defensive
        """Iterating a vault is refused: it would expose real values."""
        raise TypeError("EphemeralVault is not iterable")

    # --------------------------------------------------------------- purge
    def purge(self) -> int:
        """Overwrite and drop every stored value; return the number of entries.

        Both maps are overwritten entry by entry before being cleared so that the
        process does not keep the last references alive longer than necessary.
        Strings themselves are immutable (see the module docstring), but the
        *linkage* between a real value and its surrogate is destroyed here, which
        is what the clean-room protocol requires.
        """
        entries = len(self._forward)
        for key in list(self._forward):
            self._forward[key] = ""
        self._forward.clear()
        for kind, reverse in self._reverse.items():
            for surrogate in list(reverse):
                reverse[surrogate] = ""
            reverse.clear()
            self._reverse[kind] = {}
        for kind in self._counts:
            self._counts[kind] = 0
        self._purged = True
        return entries


SurrogateFactory = Callable[[EntityKind, str], str]


class CleanRoomSanitizer:
    """Replace PII with realistic, format-preserving surrogates.

    Parameters
    ----------
    entities:
        Kinds to detect. Defaults to :data:`DEFAULT_ENTITIES` (everything).
    custom_names:
        Extra literal person names (e.g. domain-specific customer names) that the
        heuristic detector would miss. Matched case-insensitively on word
        boundaries.
    salt:
        Salt for surrogate derivation. Changing it yields a different (still
        stable) surrogate set; keep it empty unless two tenants must not share
        surrogate spaces.
    vault:
        Reuse an existing vault (e.g. across phases of one run).
    """

    def __init__(
        self,
        *,
        entities: Sequence[EntityKind | str] | None = None,
        custom_names: Sequence[str] = (),
        salt: str = "",
        vault: EphemeralVault | None = None,
    ) -> None:
        selected = entities or DEFAULT_ENTITIES
        self._entities: tuple[EntityKind, ...] = tuple(
            kind if isinstance(kind, EntityKind) else EntityKind(str(kind))
            for kind in selected
        )
        self._salt = salt
        self.vault = vault or EphemeralVault()
        self._custom_names = tuple(
            name.strip() for name in custom_names if name and name.strip()
        )
        self._custom_patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            for name in self._custom_names
        )

    # ------------------------------------------------------------ public API
    @property
    def entities(self) -> tuple[EntityKind, ...]:
        return self._entities

    def sanitize(self, text: str) -> SanitizationResult:
        """Return ``text`` with every recognised entity replaced by a surrogate."""
        if not text:
            return SanitizationResult(text=text, findings=())
        if self.vault.purged:
            raise RuntimeError("this sanitizer was purged; create a new CleanRoomSanitizer")
        candidates: list[_RawMatch] = []
        candidates.extend(self._match_emails(text))
        candidates.extend(self._match_secrets(text))
        candidates.extend(self._match_ibans(text))
        candidates.extend(self._match_cards(text))
        candidates.extend(self._match_ips(text))
        candidates.extend(self._match_phones(text))
        candidates.extend(self._match_addresses(text))
        candidates.extend(self._match_persons(text))
        selected = _resolve_overlaps(candidates)
        if not selected:
            return SanitizationResult(text=text, findings=())

        pieces: list[str] = []
        findings: list[Finding] = []
        cursor = 0
        counts: dict[EntityKind, int] = {}
        for match in selected:
            surrogate = self.vault.get_or_create(match.kind, match.text, self._factory)
            pieces.append(text[cursor : match.start])
            pieces.append(surrogate)
            cursor = match.end
            findings.append(
                Finding(
                    kind=match.kind,
                    start=match.start,
                    end=match.end,
                    original=match.text,
                    surrogate=surrogate,
                )
            )
            counts[match.kind] = counts.get(match.kind, 0) + 1
        pieces.append(text[cursor:])
        return SanitizationResult(text="".join(pieces), findings=tuple(findings), counts=counts)

    def restore(self, text: str) -> str:
        """Undo sanitisation using the live vault (fails after :meth:`purge`)."""
        if self.vault.purged:
            raise RuntimeError("cannot restore: the vault was purged")
        result = text
        # Longest surrogates first, so a surrogate contained in another is not
        # replaced piecemeal.
        for kind in self._entities:
            surrogates = sorted(self.vault.surrogates(kind), key=len, reverse=True)
            for surrogate in surrogates:
                original = self.vault.original(kind, surrogate)
                if original:
                    result = result.replace(surrogate, original)
        return result

    def purge(self) -> int:
        """Destroy the mapping; the sanitizer is unusable afterwards."""
        return self.vault.purge()

    @property
    def stats(self) -> VaultStats:
        return self.vault.stats()

    # ----------------------------------------------------------- surrogates
    def _factory(self, kind: EntityKind, original: str) -> str:
        digest = _digest(kind.value, original, self._salt)
        builders = {
            EntityKind.EMAIL: self._surrogate_email,
            EntityKind.SECRET: self._surrogate_secret,
            EntityKind.IBAN: self._surrogate_iban,
            EntityKind.CARD: self._surrogate_card,
            EntityKind.IP: self._surrogate_ip,
            EntityKind.PHONE: self._surrogate_phone,
            EntityKind.ADDRESS: self._surrogate_address,
            EntityKind.PERSON: self._surrogate_person,
        }
        return builders[kind](original, digest)

    def _surrogate_email(self, original: str, digest: bytes) -> str:
        match = _EMAIL_RE.fullmatch(original)
        if match is None:
            return self._surrogate_generic(original, digest, _ALPHANUM)
        local, domain = match.group(1), match.group(2)
        first = _pick(_FIRST_NAMES, digest, 0)
        last = _pick(_LAST_NAMES, digest, 4)
        if "+" in local:
            base, _, tag = local.partition("+")
            new_local = f"{_apply_case(base, first.lower())}.{_apply_case(base, last.lower())}+{_random_body(digest, len(tag) or 4, _ALPHANUM)}"
        elif "." in local:
            new_local = _apply_case(local, f"{first}.{last}")
        elif "_" in local:
            new_local = _apply_case(local, f"{first}_{last}")
        elif "-" in local:
            new_local = _apply_case(local, f"{first}-{last}")
        else:
            new_local = _apply_case(local, first)
        # Preserve the original TLD length so the token shape is unchanged.
        tld_len = len(domain.rsplit(".", 1)[-1])
        safe = next((item for item in _SAFE_DOMAINS if len(item.rsplit(".", 1)[-1]) == tld_len), None)
        new_domain = safe or _SAFE_DOMAINS[0]
        return f"{new_local}@{new_domain}"

    def _surrogate_secret(self, original: str, digest: bytes) -> str:
        prefix = ""
        body = original
        for candidate in ("sk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "AKIA", "xoxb-", "xoxp-", "Bearer "):
            if original.startswith(candidate):
                prefix, body = candidate, original[len(candidate) :]
                break
        if not prefix and "=" in original:
            head, _, value = original.partition("=")
            return f"{head}={_random_body(digest, max(len(value.strip()), 4), _ALPHANUM)}"
        alphabet = _UPPER_ALNUM if body.isupper() else _ALPHANUM
        return prefix + _random_body(digest, max(len(body), 8), alphabet)

    def _surrogate_iban(self, original: str, digest: bytes) -> str:
        compact = original.replace(" ", "")
        country = compact[:2]
        length = len(compact)
        bban_len = max(4, length - 4)
        bban = _digits(digest, bban_len, offset=2, avoid_leading_zero=True)
        check = iban_check_digits(country, bban)
        surrogate = f"{country}{check}{bban}"
        # Re-apply the original grouping (spaces every 4 characters).
        if " " in original:
            groups: list[str] = []
            for index in range(0, len(surrogate), 4):
                groups.append(surrogate[index : index + 4])
            return " ".join(groups)
        return surrogate

    def _surrogate_card(self, original: str, digest: bytes) -> str:
        digits = "".join(char for char in original if char.isdigit())
        keep = min(6, max(1, len(digits) // 3))
        prefix = digits[:keep]
        body_len = max(1, len(digits) - keep - 1)
        body = _digits(digest, body_len, offset=1)
        surrogate = prefix + body
        surrogate += luhn_check_digit(surrogate)
        if not luhn_valid(surrogate):  # pragma: no cover - defensive
            surrogate = surrogate[:-1] + luhn_check_digit(surrogate[:-1])
        # Preserve the separator skeleton of the original.
        separators = [char for char in original if not char.isdigit()]
        if not separators:
            return surrogate
        out: list[str] = []
        cursor = 0
        for char in original:
            if char.isdigit():
                out.append(surrogate[cursor] if cursor < len(surrogate) else "")
                cursor += 1
            else:
                out.append(char)
        return "".join(out)

    def _surrogate_ip(self, original: str, digest: bytes) -> str:
        # Documentation ranges (RFC 5737) can never point at a real host.
        prefixes = (("192", "0", "2"), ("198", "51", "100"), ("203", "0", "113"))
        prefix = prefixes[digest[0] % len(prefixes)]
        last = str(int.from_bytes(digest[1:3], "big") % 254 + 1)
        return ".".join((*prefix, last))

    def _surrogate_phone(self, original: str, digest: bytes) -> str:
        digits = [char for char in original if char.isdigit()]
        if len(digits) < 7:
            return self._surrogate_generic(original, digest, _DIGITS)
        # Keep the leading group verbatim: for "+380 (44) ..." that is the
        # country code, for "+1 415 ..." the trunk prefix. Randomise the rest.
        keep = 0
        stripped = original.lstrip()
        if stripped.startswith("+") and len(digits) >= 9:
            first_separator = next(
                (index for index, char in enumerate(stripped[1:], start=1) if not char.isdigit()),
                len(stripped),
            )
            keep = sum(1 for char in stripped[1:first_separator] if char.isdigit())
            keep = max(1, min(keep, len(digits) - 4))
        surrogate_digits = "".join(digits[:keep]) + _digits(
            digest, len(digits) - keep, offset=1, avoid_leading_zero=True
        )
        out: list[str] = []
        cursor = 0
        for char in original:
            if char.isdigit():
                out.append(surrogate_digits[cursor] if cursor < len(surrogate_digits) else "")
                cursor += 1
            else:
                out.append(char)
        return "".join(out)

    def _surrogate_address(self, original: str, digest: bytes) -> str:
        address = _ADDRESS_RE.fullmatch(original)
        if address is not None:
            number, street, suffix = address.group(1), address.group(2), address.group(3)
            new_number = _digits(digest, len(number), offset=0, avoid_leading_zero=True)
            words = len(street.split())
            picked = [_pick(_STREET_NAMES, digest, 4 + index * 4) for index in range(max(1, words))]
            return f"{new_number} {' '.join(picked)} {suffix}"
        city = _CITY_STATE_ZIP_RE.fullmatch(original)
        if city is not None:
            new_city = _pick(_CITIES, digest, 2)
            new_zip = _digits(digest, len(city.group(3)), offset=6, avoid_leading_zero=True)
            return f"{new_city}, {city.group(2)} {new_zip}"
        return self._surrogate_generic(original, digest, _ALPHANUM)

    def _surrogate_person(self, original: str, digest: bytes) -> str:
        tokens = original.split()
        if len(tokens) >= 2:
            first = _apply_case(tokens[0], _pick(_FIRST_NAMES, digest, 0))
            last = _apply_case(tokens[-1], _pick(_LAST_NAMES, digest, 4))
            middle = tokens[1:-1]
            return " ".join([first, *middle, last])
        return _apply_case(original, _pick(_FIRST_NAMES, digest, 0))

    def _surrogate_generic(self, original: str, digest: bytes, alphabet: str) -> str:
        return _random_body(digest, max(len(original), 4), alphabet)

    # ------------------------------------------------------------- matching
    def _match_emails(self, text: str) -> list[_RawMatch]:
        if EntityKind.EMAIL not in self._entities:
            return []
        return [
            _RawMatch(EntityKind.EMAIL, match.start(), match.end(), match.group(0))
            for match in _EMAIL_RE.finditer(text)
        ]

    def _match_secrets(self, text: str) -> list[_RawMatch]:
        if EntityKind.SECRET not in self._entities:
            return []
        matches: list[_RawMatch] = []
        for _vendor, pattern in _SECRET_RES:
            for match in pattern.finditer(text):
                matches.append(
                    _RawMatch(EntityKind.SECRET, match.start(), match.end(), match.group(0))
                )
        return matches

    def _match_ibans(self, text: str) -> list[_RawMatch]:
        if EntityKind.IBAN not in self._entities:
            return []
        matches: list[_RawMatch] = []
        for match in _IBAN_RE.finditer(text):
            if iban_valid(match.group(0)):
                matches.append(
                    _RawMatch(EntityKind.IBAN, match.start(), match.end(), match.group(0))
                )
        return matches

    def _match_cards(self, text: str) -> list[_RawMatch]:
        if EntityKind.CARD not in self._entities:
            return []
        matches: list[_RawMatch] = []
        seen: set[tuple[int, int]] = set()
        for match in _CARD_RE.finditer(text):
            candidate = match.group(0)
            if luhn_valid(candidate) or _looks_like_pan(candidate):
                matches.append(
                    _RawMatch(EntityKind.CARD, match.start(), match.end(), candidate)
                )
                seen.add((match.start(), match.end()))
        for match in _CARD_CONTEXT_RE.finditer(text):
            start, end = match.span(1)
            if (start, end) not in seen:
                matches.append(_RawMatch(EntityKind.CARD, start, end, match.group(1)))
        return matches

    def _match_ips(self, text: str) -> list[_RawMatch]:
        if EntityKind.IP not in self._entities:
            return []
        return [
            _RawMatch(EntityKind.IP, match.start(), match.end(), match.group(0))
            for match in _IP_RE.finditer(text)
        ]

    def _match_phones(self, text: str) -> list[_RawMatch]:
        if EntityKind.PHONE not in self._entities:
            return []
        matches: list[_RawMatch] = []
        for match in _PHONE_RE.finditer(text):
            candidate = match.group(0)
            digits = sum(1 for char in candidate if char.isdigit())
            if digits < 9:
                continue
            # Require a phone-ish signal: a leading '+/-()' or a separation char.
            if not (candidate.lstrip().startswith(("+", "(")) or re.search(r"[\s.-]", candidate[1:])):
                continue
            matches.append(_RawMatch(EntityKind.PHONE, match.start(), match.end(), candidate))
        return matches

    def _match_addresses(self, text: str) -> list[_RawMatch]:
        if EntityKind.ADDRESS not in self._entities:
            return []
        matches: list[_RawMatch] = [
            _RawMatch(EntityKind.ADDRESS, match.start(), match.end(), match.group(0))
            for match in _ADDRESS_RE.finditer(text)
        ]
        matches.extend(
            _RawMatch(EntityKind.ADDRESS, match.start(), match.end(), match.group(0))
            for match in _CITY_STATE_ZIP_RE.finditer(text)
        )
        return matches

    def _match_persons(self, text: str) -> list[_RawMatch]:
        if EntityKind.PERSON not in self._entities:
            return []
        matches: list[_RawMatch] = []
        for pattern in (_HONORIFIC_PERSON_RE, _NAME_CONTEXT_RE):
            for match in pattern.finditer(text):
                start, end = match.span(1)
                matches.append(_RawMatch(EntityKind.PERSON, start, end, match.group(1)))
        for pattern in self._custom_patterns:
            for match in pattern.finditer(text):
                matches.append(
                    _RawMatch(EntityKind.PERSON, match.start(), match.end(), match.group(0))
                )
        return matches


def _looks_like_pan(candidate: str) -> bool:
    """Whether a digit run has the shape of a payment card number.

    ``luhn_valid`` alone under-detects: test cards, obfuscated PANs and numbers
    with typos are exactly the values that must not leak. The shape check (15-19
    digits, issuer prefix 3/4/5/6) is the standard heuristic, and over-masking an
    order number is the safe direction for a clean-room protocol.
    """
    digits = "".join(char for char in candidate if char.isdigit())
    if not 15 <= len(digits) <= 19:
        return False
    return digits[0] in "3456"


@dataclass(frozen=True)
class _RawMatch:
    """Internal match record (may be dropped as an overlap)."""

    kind: EntityKind
    start: int
    end: int
    text: str


def _resolve_overlaps(matches: Iterable[_RawMatch]) -> list[_RawMatch]:
    """Select non-overlapping matches, preferring longer and higher-priority ones."""
    ordered = sorted(
        matches,
        key=lambda item: (-(item.end - item.start), -_PRIORITY[item.kind], item.start),
    )
    taken: list[_RawMatch] = []
    for candidate in ordered:
        if any(candidate.start < other.end and other.start < candidate.end for other in taken):
            continue
        taken.append(candidate)
    return sorted(taken, key=lambda item: item.start)


def sanitize_text(
    text: str,
    *,
    entities: Sequence[EntityKind | str] | None = None,
    custom_names: Sequence[str] = (),
    salt: str = "",
) -> SanitizationResult:
    """One-shot helper: sanitise ``text`` with a throw-away vault."""
    sanitizer = CleanRoomSanitizer(entities=entities, custom_names=custom_names, salt=salt)
    try:
        return sanitizer.sanitize(text)
    finally:
        sanitizer.purge()
