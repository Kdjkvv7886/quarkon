"""Three-tier SLRU cache for Punica multi-tenant adapters (ADR-008).

Sizing comes straight from the audit:

* Adapter size: **18.5 MB** for ``DeepSeek-R1-Distill-Qwen-1.5B`` at ``r=16``,
  ``alpha=16`` over all seven linear projections.
* Tier 1 (VRAM): **32** adapters -> ~**592 MB** of device memory.
* Tier 2 (host RAM): **256** adapters -> ~**4.7 GB**, pinned with ``mlock`` in
  production (see :mod:`quark.security.memory`).
* Tier 3 (local NVMe): the remaining cold adapters.
* DMA swap latency RAM -> VRAM over pinned memory: **<= 0.65 ms**.

Eviction is SLRU *per tier*: a new adapter enters the probation segment; a hit
promotes it to protected; protected overflow demotes back to probation; tier
overflow demotes the LRU probation entry to the next tier. New adapters therefore
always fill a tier to capacity, while frequently used tenants survive churn.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ADAPTER_SIZE_BYTES",
    "NVME_READ_MS_MAX",
    "RAM_TIER_LIMIT",
    "SLRU_PROTECTED_RATIO",
    "SWAP_LATENCY_MS_MAX",
    "VRAM_TIER_LIMIT",
    "AdapterHierarchyCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheTier",
]

#: Adapter artefact size from ADR-008 (decimal MB: 32 * 18.5 = 592 MB).
ADAPTER_SIZE_BYTES: Final[int] = 18_500_000

#: VRAM tier capacity (adapters).
VRAM_TIER_LIMIT: Final[int] = 32

#: Pinned host-RAM tier capacity (adapters).
RAM_TIER_LIMIT: Final[int] = 256

#: Default share of each tier reserved for the SLRU protected segment.
SLRU_PROTECTED_RATIO: Final[float] = 0.75

#: RAM -> VRAM swap over pinned memory must stay under this latency.
SWAP_LATENCY_MS_MAX: Final[float] = 0.65

#: Cold path: an NVMe read of one 18.5 MB adapter on a datacentre drive.
NVME_READ_MS_MAX: Final[float] = 4.0


class CacheTier(str, Enum):
    """Residency tiers, ordered from fastest/least plentiful to slowest."""

    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"

    @property
    def rank(self) -> int:
        return {"vram": 0, "ram": 1, "nvme": 2}[self.value]


class CacheConfig(BaseModel):
    """Validated cache sizing; defaults are the audited values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vram_limit: int = Field(default=VRAM_TIER_LIMIT, ge=1, le=1024)
    ram_limit: int = Field(default=RAM_TIER_LIMIT, ge=1, le=8192)
    adapter_size_bytes: int = Field(default=ADAPTER_SIZE_BYTES, ge=1_000_000)
    protected_ratio: float = Field(
        default=SLRU_PROTECTED_RATIO,
        gt=0.0,
        lt=1.0,
        description="Share of each tier reserved for the SLRU protected segment.",
    )

    @model_validator(mode="after")
    def _ram_must_exceed_vram(self) -> CacheConfig:
        """A smaller RAM tier than VRAM would make demotion impossible."""
        if self.ram_limit <= self.vram_limit:
            raise ValueError("ram_limit must exceed vram_limit so demotion has room")
        return self


@dataclass
class CacheEntry:
    """Bookkeeping for one adapter; payload loading is the caller's concern."""

    adapter_id: str
    tier: CacheTier
    size_bytes: int
    hits: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    protected: bool = False

    def touch(self) -> None:
        self.hits += 1
        self.last_access = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "tier": self.tier.value,
            "size_bytes": self.size_bytes,
            "hits": self.hits,
            "protected": self.protected,
        }


@dataclass
class CacheStats:
    """Counters exposed to the metrics endpoint."""

    hits: dict[str, int] = field(default_factory=lambda: {"vram": 0, "ram": 0, "nvme": 0})
    misses: int = 0
    insertions: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions_to_nvme: int = 0
    removals: int = 0

    @property
    def total_hits(self) -> int:
        return sum(self.hits.values())

    @property
    def lookups(self) -> int:
        return self.total_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": dict(self.hits),
            "misses": self.misses,
            "insertions": self.insertions,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "evictions_to_nvme": self.evictions_to_nvme,
            "removals": self.removals,
            "hit_rate": round(self.hit_rate, 4),
        }


class _Segment:
    """SLRU pair (probation + protected) for a single tier.

    Capacity is the tier limit: a new entry always fits while the tier is not
    full. The protected segment is a *promotion* target for entries that were hit
    again, sized by ``protected_ratio``; its overflow returns entries to probation
    instead of dropping them, so a hot tenant is never lost to a promotion.
    """

    def __init__(self, capacity: int, protected_ratio: float) -> None:
        self.capacity = capacity
        self.protected_capacity = max(1, int(capacity * protected_ratio))
        self.probation: OrderedDict[str, CacheEntry] = OrderedDict()
        self.protected: OrderedDict[str, CacheEntry] = OrderedDict()

    def __len__(self) -> int:
        return len(self.probation) + len(self.protected)

    def find(self, adapter_id: str) -> CacheEntry | None:
        entry = self.protected.get(adapter_id)
        if entry is not None:
            self.protected.move_to_end(adapter_id)
            return entry
        return self.probation.get(adapter_id)

    def insert(self, entry: CacheEntry) -> CacheEntry | None:
        """Insert into probation; return the LRU entry to demote when full."""
        self.probation[entry.adapter_id] = entry
        entry.protected = False
        if len(self) <= self.capacity:
            return None
        if self.probation:
            _victim_id, victim = self.probation.popitem(last=False)
            return victim
        return self.protected.popitem(last=False)[1]  # pragma: no cover - defensive

    def promote(self, entry: CacheEntry) -> None:
        """Move a hit entry into protected, demoting the protected LRU to probation."""
        self.probation.pop(entry.adapter_id, None)
        if self.protected.get(entry.adapter_id) is None:
            self.protected[entry.adapter_id] = entry
        else:  # pragma: no cover - already protected, just refresh recency
            self.protected.move_to_end(entry.adapter_id)
        entry.protected = True
        while len(self.protected) > self.protected_capacity:
            victim_id, victim = self.protected.popitem(last=False)
            victim.protected = False
            self.probation[victim_id] = victim

    def pop(self, adapter_id: str) -> CacheEntry | None:
        entry = self.probation.pop(adapter_id, None)
        if entry is not None:
            return entry
        return self.protected.pop(adapter_id, None)

    def clear(self) -> None:
        self.probation.clear()
        self.protected.clear()


class AdapterHierarchyCache:
    """VRAM <-> pinned RAM <-> NVMe adapter cache with per-tier SLRU eviction.

    The cache stores metadata only; moving adapter tensors is the serving layer's
    responsibility. That separation keeps the eviction policy unit-testable without
    CUDA and lets a real deployment stream the payloads asynchronously while the
    policy decides residency.

    Thread-safe: every public operation takes a re-entrant lock, matching a
    multi-threaded serving front end.
    """

    def __init__(
        self,
        vram_limit: int = VRAM_TIER_LIMIT,
        ram_limit: int = RAM_TIER_LIMIT,
        *,
        adapter_size_bytes: int = ADAPTER_SIZE_BYTES,
        protected_ratio: float = SLRU_PROTECTED_RATIO,
        config: CacheConfig | None = None,
    ) -> None:
        resolved = config or CacheConfig(
            vram_limit=vram_limit,
            ram_limit=ram_limit,
            adapter_size_bytes=adapter_size_bytes,
            protected_ratio=protected_ratio,
        )
        self.config = resolved
        self.vram_limit = resolved.vram_limit
        self.ram_limit = resolved.ram_limit
        self.adapter_size_bytes = resolved.adapter_size_bytes
        self.swap_latency_ms = SWAP_LATENCY_MS_MAX
        self._lock = threading.RLock()
        self._segments: dict[CacheTier, _Segment] = {
            CacheTier.VRAM: _Segment(resolved.vram_limit, resolved.protected_ratio),
            CacheTier.RAM: _Segment(resolved.ram_limit, resolved.protected_ratio),
        }
        self._nvme: OrderedDict[str, CacheEntry] = OrderedDict()
        self._locations: dict[str, CacheTier] = {}
        self.stats = CacheStats()

    # ------------------------------------------------------------- introspection
    def __len__(self) -> int:
        with self._lock:
            return len(self._locations)

    def __contains__(self, adapter_id: str) -> bool:
        with self._lock:
            return adapter_id in self._locations

    def tier_of(self, adapter_id: str) -> CacheTier | None:
        """Current residency, or ``None`` for an unknown adapter."""
        with self._lock:
            return self._locations.get(adapter_id)

    def get(self, adapter_id: str) -> CacheTier | None:
        """Look an adapter up, promote it, and return the tier it now resides in."""
        with self._lock:
            location = self._locations.get(adapter_id)
            if location is None:
                self.stats.misses += 1
                return None
            if location is CacheTier.NVME:
                return self._promote_from_nvme(adapter_id)
            segment = self._segments[location]
            entry = segment.find(adapter_id)
            if entry is None:  # pragma: no cover - location map is authoritative
                self.stats.misses += 1
                self._locations.pop(adapter_id, None)
                return None
            entry.touch()
            self.stats.hits[location.value] += 1
            if not entry.protected:
                segment.promote(entry)
                self.stats.promotions += 1
            if location is CacheTier.RAM:
                # A RAM hit is the strongest "warm tenant" signal: move it into
                # VRAM so the next call is served from device memory (ADR-008).
                return self._relocate(adapter_id, CacheTier.VRAM)
            return location

    def put(self, adapter_id: str) -> CacheTier:
        """Insert (or touch) an adapter and return its residency tier."""
        with self._lock:
            existing = self._locations.get(adapter_id)
            if existing is not None:
                found = self.get(adapter_id)
                return found if found is not None else CacheTier.VRAM
            self.stats.insertions += 1
            return self._insert(CacheTier.VRAM, adapter_id)

    def remove(self, adapter_id: str) -> bool:
        """Drop an adapter from every tier (tenant offboarding)."""
        with self._lock:
            location = self._locations.pop(adapter_id, None)
            if location is None:
                return False
            if location is CacheTier.NVME:
                self._nvme.pop(adapter_id, None)
            else:
                self._segments[location].pop(adapter_id)
            self.stats.removals += 1
            return True

    def clear(self) -> None:
        """Empty the whole hierarchy (tests and graceful shutdown)."""
        with self._lock:
            for segment in self._segments.values():
                segment.clear()
            self._nvme.clear()
            self._locations.clear()

    # ------------------------------------------------------------- placements
    def _insert(self, tier: CacheTier, adapter_id: str) -> CacheTier:
        """Place a fresh entry in ``tier`` and cascade overflow downwards."""
        existing = self._locations.get(adapter_id)
        if existing is not None and existing is not tier:
            self._detach(adapter_id, existing)
        entry = CacheEntry(adapter_id=adapter_id, tier=tier, size_bytes=self.adapter_size_bytes)
        self._locations[adapter_id] = tier
        overflow = self._segments[tier].insert(entry)
        if overflow is not None:
            self._demote(overflow, tier)
        return tier

    def _relocate(self, adapter_id: str, target: CacheTier) -> CacheTier:
        """Move an entry between tiers, preserving its statistics."""
        source = self._locations.get(adapter_id)
        if source is None:
            return target
        entry: CacheEntry | None
        if source is CacheTier.NVME:
            entry = self._nvme.pop(adapter_id, None)
        else:
            entry = self._segments[source].pop(adapter_id)
        if entry is None:  # pragma: no cover - defensive
            entry = CacheEntry(
                adapter_id=adapter_id, tier=target, size_bytes=self.adapter_size_bytes
            )
        entry.tier = target
        entry.protected = False
        self._locations[adapter_id] = target
        overflow = self._segments[target].insert(entry)
        if overflow is not None:
            self._demote(overflow, target)
        return target

    def _demote(self, entry: CacheEntry, from_tier: CacheTier) -> None:
        """Move one overflow entry to the next tier (or NVMe for cold storage)."""
        self.stats.demotions += 1
        self._segments[from_tier].pop(entry.adapter_id)
        if from_tier is CacheTier.VRAM:
            entry.tier = CacheTier.RAM
            entry.protected = False
            self._locations[entry.adapter_id] = CacheTier.RAM
            overflow = self._segments[CacheTier.RAM].insert(entry)
            if overflow is not None:
                self._demote(overflow, CacheTier.RAM)
            return
        # RAM overflow: the adapter becomes cold and lives on NVMe.
        entry.tier = CacheTier.NVME
        entry.protected = False
        self._locations[entry.adapter_id] = CacheTier.NVME
        self._nvme[entry.adapter_id] = entry
        self.stats.evictions_to_nvme += 1

    def _detach(self, adapter_id: str, tier: CacheTier) -> None:
        if tier is CacheTier.NVME:
            self._nvme.pop(adapter_id, None)
        else:
            self._segments[tier].pop(adapter_id)

    def _promote_from_nvme(self, adapter_id: str) -> CacheTier:
        """NVMe hit: copy into RAM for durability, then warm VRAM for the caller."""
        self.stats.hits[CacheTier.NVME.value] += 1
        self.stats.promotions += 1
        # RAM first so an immediate VRAM eviction does not need another NVMe read
        # (ADR-008: t_swap <= 0.65 ms over pinned memory).
        self._relocate(adapter_id, CacheTier.RAM)
        return self._relocate(adapter_id, CacheTier.VRAM)

    # ------------------------------------------------------------- reporting
    def footprint(self) -> dict[str, Any]:
        """Memory actually held per tier, in the units the audit uses."""
        with self._lock:
            vram_entries = len(self._segments[CacheTier.VRAM])
            ram_entries = len(self._segments[CacheTier.RAM])
            nvme_entries = len(self._nvme)
            vram_bytes = vram_entries * self.adapter_size_bytes
            ram_bytes = ram_entries * self.adapter_size_bytes
            return {
                "vram_entries": vram_entries,
                "ram_entries": ram_entries,
                "nvme_entries": nvme_entries,
                "vram_bytes": vram_bytes,
                "ram_bytes": ram_bytes,
                "vram_mb": round(vram_bytes / 1_000_000, 1),
                "ram_gb": round(ram_bytes / 1_000_000_000, 3),
                "vram_limit_mb": round(self.vram_limit * self.adapter_size_bytes / 1_000_000, 1),
                "ram_limit_gb": round(self.ram_limit * self.adapter_size_bytes / 1_000_000_000, 2),
                "swap_latency_ms_max": self.swap_latency_ms,
                "nvme_read_ms_max": NVME_READ_MS_MAX,
            }

    def entries(self) -> list[dict[str, Any]]:
        """Snapshot of the hierarchy, hottest tier first (debug and tests)."""
        with self._lock:
            rows: list[dict[str, Any]] = []
            for tier in (CacheTier.VRAM, CacheTier.RAM):
                segment = self._segments[tier]
                for entry in list(segment.protected.values()) + list(segment.probation.values()):
                    rows.append(entry.to_dict())
            for entry in self._nvme.values():
                rows.append(entry.to_dict())
            return rows

    def describe(self) -> dict[str, Any]:
        """Configuration, footprint and counters for the metrics endpoint."""
        with self._lock:
            return {
                "adr": "ADR-008",
                "config": self.config.model_dump(),
                "footprint": self.footprint(),
                "stats": self.stats.to_dict(),
                "segments": {
                    tier.value: {
                        "probation": len(segment.probation),
                        "protected": len(segment.protected),
                        "capacity": segment.capacity,
                    }
                    for tier, segment in self._segments.items()
                },
            }
