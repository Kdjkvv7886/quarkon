"""Hardened memory handling for adapter and prompt buffers (ADR-007).

Two physical side-channels leak secrets out of a process even when the data is
"deleted" in Python:

* **Swap**: the kernel may write a dirty page to disk, where it outlives the
  process. ``mlock(2)`` pins the page in RAM.
* **Core dumps / ptrace-style snapshots**: ``madvise(MADV_DONTDUMP)`` excludes the
  range from core files.

Zeroisation then removes the bytes. For CUDA buffers the audit requires the clear
rate to exceed 1 TB/s, which ``torch.Tensor.zero_()`` plus a stream synchronise
achieves on current hardware; :data:`VRAM_ZERO_MIN_BYTES_PER_SEC` records the
threshold so a benchmark can assert it.

The Linux-specific calls are no-ops-with-an-error on other platforms: silently
pretending to protect memory would be worse than refusing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import sys
from functools import lru_cache
from typing import Any, Final

from quark.errors import MemoryProtectionError

__all__ = [
    "MADV_DONTDUMP",
    "MCL_CURRENT",
    "MCL_FUTURE",
    "VRAM_ZERO_MIN_BYTES_PER_SEC",
    "LinuxMemoryProtector",
    "libc_available",
]

#: ``madvise`` advice value from ``linux/mman.h``.
MADV_DONTDUMP: Final[int] = 16

#: ``mlockall`` flags (kept for operators who pin the whole arena).
MCL_CURRENT: Final[int] = 1
MCL_FUTURE: Final[int] = 2

#: Audit threshold for clearing VRAM buffers (ADR-007): > 1 TB/s.
VRAM_ZERO_MIN_BYTES_PER_SEC: Final[float] = 1e12

#: Optional torch import: the engine must stay importable on CPU-only hosts.
_TORCH: Any
try:  # pragma: no cover - exercised implicitly by the import
    import torch as _torch_module
except ImportError:  # pragma: no cover - CPU-only installs
    _TORCH = None
else:  # pragma: no cover - torch is optional in CI
    _TORCH = _torch_module


def libc_available() -> bool:
    """Whether the platform exposes a C library with the hardening calls."""
    return sys.platform.startswith("linux")


@lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    """Load the platform C library once and bind the prototypes we use.

    Only Linux is accepted for the hardening calls because ``MADV_DONTDUMP`` is a
    Linux ``madvise`` advice; the generic zeroisation path also supports Windows
    through ``msvcrt``.
    """
    if sys.platform.startswith("linux"):
        name = "libc.so.6"
    elif sys.platform == "darwin":
        name = "libc.dylib"
    elif sys.platform.startswith("win"):
        name = "msvcrt"
    else:  # pragma: no cover - exotic platform
        resolved = ctypes.util.find_library("c")
        if not resolved:
            raise MemoryProtectionError(f"no C library found on platform {sys.platform!r}")
        name = resolved
    try:
        library = ctypes.CDLL(name, use_errno=True)
    except OSError as exc:  # pragma: no cover - broken installs
        raise MemoryProtectionError(f"cannot load {name}: {exc}") from exc
    for symbol, argtypes in (
        ("mlock", [ctypes.c_void_p, ctypes.c_size_t]),
        ("munlock", [ctypes.c_void_p, ctypes.c_size_t]),
        ("madvise", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        ("memset", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
    ):
        function = getattr(library, symbol, None)
        if function is not None:
            function.argtypes = argtypes
            function.restype = ctypes.c_int if symbol != "memset" else ctypes.c_void_p
    if hasattr(library, "explicit_bzero"):
        library.explicit_bzero.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.explicit_bzero.restype = None
    return library


def _page_size() -> int:
    """Operating-system page size; ``madvise`` requires page-aligned ranges."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):  # pragma: no cover - exotic hosts
        return 4096


def _aligned_range(buffer_ptr: int, size: int) -> tuple[int, int]:
    """Expand ``[buffer_ptr, buffer_ptr + size)`` to whole pages.

    ``mlock`` tolerates an unaligned start, but ``madvise(MADV_DONTDUMP)`` returns
    ``EINVAL`` unless the address is page-aligned (Linux man page). Aligning the
    range before both calls keeps them consistent and makes a small buffer or a
    tensor storage offset safe to protect.
    """
    page = _page_size()
    start = buffer_ptr - (buffer_ptr % page)
    end = buffer_ptr + size
    aligned_size = ((end + page - 1) // page) * page - start
    return start, aligned_size


def _raise_from_errno(operation: str, size: int) -> None:
    code = ctypes.get_errno()
    detail = errno.errorcode.get(code, str(code))
    raise MemoryProtectionError(
        f"{operation} failed for {size} bytes: {detail}",
        hint=(
            "raise the container memory/ulimit budget (RLIMIT_MEMLOCK), or reduce "
            "quark.serve.cache RAM tier size"
        ),
    )


class LinuxMemoryProtector:
    """Low-level ``mlock`` / ``madvise`` / zeroisation helpers.

    The class is a namespace of static methods on purpose: it carries no state, so
    it is safe to call from worker threads and from signal-safe cleanup paths.
    """

    MADV_DONTDUMP: Final[int] = MADV_DONTDUMP

    @staticmethod
    def is_linux() -> bool:
        """Whether the Linux-specific protections can be applied here."""
        return sys.platform.startswith("linux")

    @staticmethod
    def lock_and_protect(buffer_ptr: int, size: int) -> None:
        """Pin ``size`` bytes at ``buffer_ptr`` and exclude them from core dumps.

        Applies, in order:

            1. ``mlock(addr, size)`` - no swap-out (ADR-007).
            2. ``madvise(addr, size, MADV_DONTDUMP)`` - no core dump.

        Args:
            buffer_ptr: address returned by ``tensor.data_ptr()`` or
                ``ctypes.addressof``.
            size: length of the range in bytes.

        Raises:
            MemoryProtectionError: on non-Linux platforms, invalid arguments, or a
                failing syscall (``ENOMEM``/``EPERM`` are the common ones).
        """
        if not LinuxMemoryProtector.is_linux():
            raise MemoryProtectionError(
                f"memory hardening requires Linux, running on {sys.platform!r}",
                hint="run the engine in the Linux container image; macOS/Windows lack MADV_DONTDUMP",
            )
        LinuxMemoryProtector._validate(buffer_ptr, size)
        start, aligned_size = _aligned_range(buffer_ptr, size)
        library = _libc()
        # 1) Pin the pages so the kernel cannot page them out.
        if library.mlock(ctypes.c_void_p(start), aligned_size) != 0:
            _raise_from_errno("mlock", aligned_size)
        # 2) Keep the range out of core dumps and ptrace snapshots. ``madvise``
        # rejects unaligned addresses, hence the page-aligned range above.
        if library.madvise(ctypes.c_void_p(start), aligned_size, MADV_DONTDUMP) != 0:
            # Roll the pin back so a partially protected buffer cannot linger.
            library.munlock(ctypes.c_void_p(start), aligned_size)
            _raise_from_errno("madvise(MADV_DONTDUMP)", aligned_size)

    @staticmethod
    def unlock(buffer_ptr: int, size: int) -> bool:
        """Release a previous pin; returns ``False`` when nothing could be released."""
        if not LinuxMemoryProtector.is_linux():
            return False
        LinuxMemoryProtector._validate(buffer_ptr, size)
        start, aligned_size = _aligned_range(buffer_ptr, size)
        try:
            library = _libc()
            unlocked = int(library.munlock(ctypes.c_void_p(start), aligned_size))
            return unlocked == 0
        except MemoryProtectionError:  # pragma: no cover - library unavailable
            return False

    @staticmethod
    def protect_tensor(tensor: Any) -> int:
        """Pin and hide the storage backing a CPU tensor (``mlock`` + DONTDUMP).

        Only CPU tensors can be pinned; raising for CUDA tensors prevents a silent
        no-op that would leave host-invisible bytes behind.

        Returns:
            The number of bytes protected.
        """
        if _TORCH is None or not isinstance(tensor, _TORCH.Tensor):
            raise MemoryProtectionError("protect_tensor expects a torch.Tensor")
        if tensor.device.type != "cpu":
            raise MemoryProtectionError(
                "only CPU tensors can be pinned with mlock",
                hint="VRAM buffers are protected by device-memory zeroisation, not by mlock",
            )
        size = int(tensor.nelement()) * int(tensor.element_size())
        if size == 0:
            return 0
        pointer = int(tensor.data_ptr())
        LinuxMemoryProtector.lock_and_protect(pointer, size)
        return size

    @staticmethod
    def secure_zero(tensor_or_buffer: Any) -> None:
        """Overwrite the memory of ``tensor_or_buffer`` with zeros.

        Supported targets:

        * ``torch.Tensor`` - ``zero_()`` and a CUDA synchronise when applicable
          (the audit requires > 1 TB/s clear rate on VRAM).
        * ``bytearray`` / writable ``memoryview`` / ``ctypes`` array - zeroed
          through ``explicit_bzero`` when the libc provides it, ``memset``
          otherwise. Python-level loops are the last-resort fallback.
        * ``bytes`` - refused: immutable objects cannot be cleared, raising beats
          returning a false sense of security.

        Raises:
            TypeError: for unsupported targets.
        """
        if _TORCH is not None and isinstance(tensor_or_buffer, _TORCH.Tensor):
            tensor_or_buffer.zero_()
            if tensor_or_buffer.device.type == "cuda":
                import torch  # local import: no CUDA dependency on import path

                torch.cuda.synchronize()
            return
        if isinstance(tensor_or_buffer, (bytearray, memoryview)):
            buffer = tensor_or_buffer
            if isinstance(buffer, memoryview) and buffer.readonly:
                raise TypeError("memoryview is read-only; cannot clear it")
            raw = ctypes.c_char * len(buffer)
            c_array = raw.from_buffer(buffer if isinstance(buffer, bytearray) else buffer.obj)
            LinuxMemoryProtector._clear_buffer(ctypes.addressof(c_array), len(buffer))
            return
        if isinstance(tensor_or_buffer, ctypes.Array):
            size = ctypes.sizeof(tensor_or_buffer)
            if size:
                LinuxMemoryProtector._clear_buffer(ctypes.addressof(tensor_or_buffer), size)
            return
        if isinstance(tensor_or_buffer, bytes):
            raise TypeError("bytes is immutable; wrap the secret in bytearray to clear it")
        raise TypeError(
            f"unsupported target for secure_zero: {type(tensor_or_buffer).__name__}; "
            "pass a torch.Tensor, bytearray, memoryview or ctypes array"
        )

    # ------------------------------------------------------------- internals
    @staticmethod
    def _validate(buffer_ptr: int, size: int) -> None:
        if buffer_ptr <= 0:
            raise MemoryProtectionError(f"invalid buffer pointer: {buffer_ptr!r}")
        if size <= 0:
            raise MemoryProtectionError(f"size must be positive, got {size}")

    @staticmethod
    def _clear_buffer(address: int, size: int) -> None:
        """Zero a raw C buffer without relying on the Python object's lifetime."""
        try:
            library = _libc()
        except MemoryProtectionError:
            # Last resort: zero through ctypes, byte by byte. Slower, still final.
            for offset in range(size):
                ctypes.cast(address + offset, ctypes.POINTER(ctypes.c_ubyte))[0] = 0
            return
        explicit = getattr(library, "explicit_bzero", None)
        if explicit is not None:
            explicit(ctypes.c_void_p(address), size)
            return
        # ``memset`` through FFI cannot be elided by the optimiser the way a
        # compiler-visible dead store could.
        library.memset(ctypes.c_void_p(address), 0, size)
