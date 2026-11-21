"""Transactional, resumable storage for synthesised samples.

Two artifacts are maintained for every synthesis run:

1. ``samples.jsonl`` - append-only, newline-delimited JSON. This is the
   *dataset* (ChatML records, schema version :data:`DATASET_FORMAT_VERSION`).
   Each record is written with a single append followed by ``fsync``, so a
   process kill can lose at most the record being written and never corrupts
   earlier records.

2. ``samples.sqlite3`` - transactional ledger (WAL mode, ``synchronous=FULL``).
   One row per unit of work with its validation status
   (``pending``/``accepted``/``rejected``/``failed``), the attempt counter and
   the rejection reason.

Crash safety
------------
Per unit of work the sequence is fixed::

    BEGIN IMMEDIATE
      reserve the unit row (status=pending)
    COMMIT                          <- reservation is durable
      ... teacher request happens outside any transaction ...
    BEGIN IMMEDIATE
      append record to JSONL + fsync
      update the row to accepted / rejected
    COMMIT

If the process dies at any point the row stays ``pending``; the next run sees
it in :meth:`SampleStore.pending_items` and regenerates with the *same* sample
id. I/O errors during the commit phase trigger a compensating repair
(:meth:`SampleStore._compensate`) so a retry never duplicates a record.

``samples.jsonl`` is authoritative for the dataset content; the ledger is
authoritative for progress and attempts. :meth:`SampleStore.integrity_check`
cross-checks both.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from opendistill.core.exceptions import CheckpointError
from opendistill.core.logging import get_logger

__all__ = [
    "DATASET_FORMAT_VERSION",
    "SAMPLE_STATUSES",
    "PendingItem",
    "Sample",
    "SampleStore",
    "StoreStats",
    "derive_unit_id",
    "new_sample_id",
    "utc_now",
]

#: Version of the sample JSONL schema. Bumped only on breaking changes.
DATASET_FORMAT_VERSION: Final[int] = 1
SAMPLE_STATUSES: Final[tuple[str, ...]] = ("pending", "accepted", "rejected", "failed")
Status = Literal["pending", "accepted", "rejected", "failed"]
TerminalStatus = Literal["accepted", "rejected", "failed"]

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id            TEXT PRIMARY KEY,
    unit_id       TEXT NOT NULL,
    attempt       INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL CHECK (status IN ('pending','accepted','rejected','failed')),
    rule          TEXT,
    reason        TEXT,
    sample_sha256 TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_unit ON samples(unit_id);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
"""

_log = get_logger("synthesis.storage")


def utc_now() -> str:
    """Millisecond-resolution UTC timestamp, ISO-8601 with ``Z`` suffix."""
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def derive_unit_id(seed_index: int, generation: int) -> str:
    """Deterministic unit id for (seed, generation): ``d0004-g02``."""
    return f"d{seed_index:04d}-g{generation:02d}"


def new_sample_id() -> str:
    """UUIDv4 identifier used for every generated sample."""
    return str(uuid.uuid4())


def monotonic_ms() -> int:
    """Milliseconds on the monotonic clock (latency measurement)."""
    return int(time.monotonic() * 1000)


@dataclass(frozen=True)
class Sample:
    """One fully generated, validated, training-ready sample (ChatML record)."""

    id: str
    timestamp: str
    messages: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSONL representation (schema version 1)."""
        return {
            "format_version": DATASET_FORMAT_VERSION,
            "id": self.id,
            "timestamp": self.timestamp,
            "messages": [dict(message) for message in self.messages],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Sample:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("sample record has no messages")
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("sample message must be an object with role and content")
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("sample metadata must be an object")
        return cls(
            id=str(record["id"]),
            timestamp=str(record.get("timestamp") or utc_now()),
            messages=tuple(messages),
            metadata=metadata,
        )

    @property
    def assistant_content(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
        return ""

    @property
    def user_content(self) -> str:
        for message in self.messages:
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""


@dataclass(frozen=True)
class PendingItem:
    """A unit of work left ``pending`` by an interrupted attempt."""

    unit_id: str
    sample_id: str
    attempt: int

    @property
    def is_retry(self) -> bool:
        return self.attempt > 0


@dataclass
class StoreStats:
    """Aggregate counters read from the ledger."""

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    pending: int = 0
    attempts: int = 0

    @property
    def processed(self) -> int:
        return self.accepted + self.rejected + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "failed": self.failed,
            "pending": self.pending,
            "attempts": self.attempts,
            "processed": self.processed,
        }


class _AtomicJsonl:
    """Append-with-fsync helper and safe in-place repair for the dataset file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: Mapping[str, Any]) -> str:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return line

    def lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle if line.strip()]

    def rewrite(self, lines: list[str]) -> None:
        tmp = self.path.with_name(self.path.name + ".rewrite.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def remove_last(self, record_id: str) -> bool:
        """Remove the trailing record with ``record_id`` (cheap, repair only)."""
        lines = self.lines()
        for index in range(len(lines) - 1, -1, -1):
            try:
                if str(json.loads(lines[index]).get("id")) == record_id:
                    del lines[index]
                    self.rewrite(lines)
                    return True
            except json.JSONDecodeError:
                continue
        return False


class SampleStore:
    """Transactional sample store with a crash-safe JSONL dataset mirror.

    Parameters
    ----------
    workspace:
        Run workspace directory; receives ``samples.jsonl`` and ``samples.sqlite3``.
    fingerprint:
        Hash of the *generation-relevant* configuration. A different fingerprint
        restarts the dataset so an edited config never silently mixes output of
        two different teacher settings.
    reset:
        Delete previous data unconditionally (``--force``).
    """

    JSONL_NAME: Final[str] = "samples.jsonl"
    SQLITE_NAME: Final[str] = "samples.sqlite3"

    def __init__(
        self,
        workspace: str | Path,
        *,
        fingerprint: str,
        reset: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.workspace / self.JSONL_NAME
        self.sqlite_path = self.workspace / self.SQLITE_NAME
        self.fingerprint = fingerprint
        self.fingerprint_changed = False
        self._jsonl = _AtomicJsonl(self.jsonl_path)
        self._conn: sqlite3.Connection | None = None
        if reset:
            self._reset_files()
        else:
            self.fingerprint_changed = self._needs_reset()
            if self.fingerprint_changed:
                _log.warning(
                    "store.restarted",
                    reason="generation-relevant config changed",
                    path=str(self.jsonl_path),
                )
                self._reset_files()
        self._open()

    # ------------------------------------------------------------ lifecycle
    def _needs_reset(self) -> bool:
        if not self.sqlite_path.is_file():
            return self.jsonl_path.is_file() and self.jsonl_path.stat().st_size > 0
        try:
            conn = sqlite3.connect(self.sqlite_path)
            try:
                row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            return True
        if row is None:
            return self.jsonl_path.is_file() and self.jsonl_path.stat().st_size > 0
        return str(row[0]) != self.fingerprint

    def _reset_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.sqlite_path) + suffix).unlink()
            except FileNotFoundError:
                pass
        try:
            self.jsonl_path.unlink()
        except FileNotFoundError:
            pass

    def _open(self) -> None:
        conn = sqlite3.connect(self.sqlite_path, isolation_level=None, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(_SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('fingerprint', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.fingerprint,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('format_version', ?)",
                (str(DATASET_FORMAT_VERSION),),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            conn.close()
            raise
        self._conn = conn
        orphans = self.recover_orphans()
        if orphans:
            _log.warning("store.orphans_dropped", count=orphans)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SampleStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise CheckpointError("SampleStore is closed")
        return self._conn

    # -------------------------------------------------------------- queries
    def id_for(self, unit_id: str) -> str | None:
        row = self.conn.execute("SELECT id FROM samples WHERE unit_id=?", (unit_id,)).fetchone()
        return str(row[0]) if row else None

    def sample_id(self, unit_id: str) -> str:
        """Stable id for ``unit_id`` (created on first reservation)."""
        existing = self.id_for(unit_id)
        if existing is not None:
            return existing
        return new_sample_id()

    def has_unit(self, unit_id: str) -> bool:
        """Whether any row (any status) exists for ``unit_id``."""
        return self.id_for(unit_id) is not None

    def accepted_units(self) -> set[str]:
        rows = self.conn.execute("SELECT unit_id FROM samples WHERE status='accepted'").fetchall()
        return {str(row[0]) for row in rows}

    def processed_units(self) -> set[str]:
        """Units with any terminal outcome (resume skips these)."""
        rows = self.conn.execute(
            "SELECT unit_id FROM samples WHERE status IN ('accepted','rejected','failed')"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def sample_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT id FROM samples").fetchall()
        return {str(row[0]) for row in rows}

    def _count(self, status: Status) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM samples WHERE status=?", (status,)).fetchone()
        return int(row[0]) if row else 0

    def accepted_count(self) -> int:
        return self._count("accepted")

    def rejected_count(self) -> int:
        return self._count("rejected")

    def failed_count(self) -> int:
        return self._count("failed")

    def pending_count(self) -> int:
        return self._count("pending")

    def attempt_of(self, unit_id: str) -> int:
        row = self.conn.execute("SELECT attempt FROM samples WHERE unit_id=?", (unit_id,)).fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> StoreStats:
        rows = self.conn.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(attempt),0) FROM samples GROUP BY status"
        ).fetchall()
        stats = StoreStats()
        for status, count, attempts in rows:
            setattr(stats, str(status), int(count))
            stats.attempts += int(attempts)
        stats.total = stats.accepted + stats.rejected + stats.failed + stats.pending
        return stats

    def rejection_histogram(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT COALESCE(rule,'unknown'), COUNT(*) FROM samples "
            "WHERE status IN ('rejected','failed') GROUP BY rule ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {str(rule): int(count) for rule, count in rows}

    def pending_items(self) -> list[PendingItem]:
        """Rows left ``pending`` by an interrupted attempt, oldest first."""
        rows = self.conn.execute(
            "SELECT unit_id, id, attempt FROM samples WHERE status='pending' ORDER BY created_at"
        ).fetchall()
        return [PendingItem(unit_id=str(u), sample_id=str(i), attempt=int(a)) for u, i, a in rows]

    # -------------------------------------------------------------- writes
    def reserve(self, unit_id: str, *, sample_id: str | None = None) -> tuple[str, int]:
        """Durably reserve ``unit_id``; return ``(sample_id, attempt)``.

        The attempt counter is incremented for a row that is still ``pending``,
        which is exactly the "interrupted, now retried" case.
        """
        now = utc_now()
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id, attempt, status FROM samples WHERE unit_id=?", (unit_id,)
            ).fetchone()
            if row is None:
                sid = sample_id or new_sample_id()
                conn.execute(
                    "INSERT INTO samples(id, unit_id, attempt, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (sid, unit_id, 0, "pending", now, now),
                )
                conn.commit()
                return sid, 0
            sid, attempt, status = str(row[0]), int(row[1]), str(row[2])
            if status != "pending":
                conn.rollback()
                raise CheckpointError(
                    f"unit {unit_id!r} was already processed with status {status!r}",
                    details={"unit_id": unit_id, "status": status},
                )
            conn.execute(
                "UPDATE samples SET attempt=attempt+1, updated_at=? WHERE unit_id=?",
                (now, unit_id),
            )
            attempt += 1
            conn.commit()
            return sid, attempt
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise

    def record_accepted(
        self,
        unit_id: str,
        sample: Sample,
        *,
        sample_sha256: str,
        attempt: int,
    ) -> None:
        """Append ``sample`` to the dataset and mark the unit accepted (atomic).

        The dataset record is authoritative for the sample id: the ledger row is
        rewritten to carry ``sample.id`` so both artifacts always agree, even if
        the caller built the record with an id other than the one reserved.
        """
        line = json.dumps(sample.to_record(), ensure_ascii=False, sort_keys=True)
        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                with open(self.jsonl_path, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                conn.execute(
                    "UPDATE samples SET id=?, status='accepted', attempt=?, rule=NULL, "
                    "reason=NULL, sample_sha256=?, updated_at=? WHERE unit_id=?",
                    (sample.id, attempt, sample_sha256, utc_now(), unit_id),
                )
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                f"sample id {sample.id!r} already exists in the ledger",
                hint="sample ids must be unique; this indicates a retry race",
                details={"unit_id": unit_id, "sample_id": sample.id},
            ) from exc
        except BaseException:
            # Compensate: a failure after the append but before commit would
            # leave an orphan record; remove it so a retry cannot duplicate.
            if self._jsonl.remove_last(sample.id):
                _log.warning("store.rolled_back_partial_write", sample_id=sample.id)
            raise
        _log.debug("store.accepted", sample_id=sample.id, attempt=attempt)

    def record_outcome(
        self,
        unit_id: str,
        *,
        status: TerminalStatus,
        rule: str,
        reason: str,
        attempt: int,
    ) -> None:
        """Mark a non-accepted terminal outcome (``rejected`` or ``failed``)."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE samples SET status=?, attempt=?, rule=?, reason=?, updated_at=? "
                "WHERE unit_id=?",
                (status, attempt, rule, reason[:2000], utc_now(), unit_id),
            )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise

    # ------------------------------------------------------------- reading
    def iter_accepted(self) -> Iterator[Sample]:
        """Stream accepted samples from the dataset (skips corrupt lines)."""
        if not self.jsonl_path.is_file():
            return
        with open(self.jsonl_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Sample.from_record(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    _log.warning("store.corrupt_line_skipped", line=line_number, error=str(exc))

    def load_accepted(self, limit: int | None = None) -> list[Sample]:
        samples: list[Sample] = []
        for sample in self.iter_accepted():
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
        return samples

    # ------------------------------------------------------------ recovery
    def recover_orphans(self) -> int:
        """Drop dataset records whose id has no committed ledger row.

        An orphan can only appear if the process died while holding the commit
        pending after the fsync. Rewriting the file keeps it a faithful mirror
        of the ledger.
        """
        if not self.jsonl_path.is_file():
            return 0
        known = self.sample_ids()
        kept: list[str] = []
        dropped = 0
        for line in self._jsonl.lines():
            try:
                sample_id = str(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            if sample_id in known:
                kept.append(line)
            else:
                dropped += 1
        if dropped:
            self._jsonl.rewrite(kept)
        return dropped

    def integrity_check(self) -> tuple[bool, str]:
        """Verify the ledger and the dataset agree on accepted ids."""
        accepted_rows = {
            str(row[0])
            for row in self.conn.execute("SELECT id FROM samples WHERE status='accepted'")
        }
        file_ids: set[str] = set()
        duplicates = 0
        for line in self._jsonl.lines():
            try:
                sample_id = str(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                return False, f"{self.jsonl_path.name} contains an unparseable line"
            if sample_id in file_ids:
                duplicates += 1
            file_ids.add(sample_id)
        if duplicates:
            return False, f"{self.jsonl_path.name} contains {duplicates} duplicate ids"
        missing = accepted_rows - file_ids
        if missing:
            return False, (
                f"{len(missing)} accepted ids are missing from {self.jsonl_path.name}"
            )
        orphans = file_ids - accepted_rows
        if orphans:
            return False, f"{len(orphans)} dataset ids have no accepted ledger row"
        return True, ""

    # ------------------------------------------------------------- meta
    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default
