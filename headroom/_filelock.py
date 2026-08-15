"""Best-effort interprocess file locking.

Concurrent Headroom processes serialize a handful of read-modify-write cycles
on shared JSON records — the Claude wrap marker's owner stack, an isolated run
directory's proxy/owner lists. Losing one of those writes silently drops a LIVE
owner, after which cleanup or GC acts as though it were gone.

Every lock here is advisory and best-effort by design: if the lock file cannot
be created, or a holder is wedged past the timeout, the caller still runs its
critical section unserialized. Bookkeeping must never be the reason a wrap
fails to launch.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

# `flock` is held per open file description, so a second acquisition from THIS
# process (on a fresh handle) blocks against our own outer frame forever. The
# locked sections nest — a handover calls back into the restorer, which locks
# again — so re-entrancy is tracked per lock path rather than left to the OS.
_held: dict[str, int] = {}


def acquire(handle: Any, *, timeout: float) -> bool:
    """Take an exclusive advisory lock on ``handle``, or give up after
    ``timeout`` seconds. Returns whether the lock was acquired."""

    deadline = time.monotonic() + timeout
    while True:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                cast(Any, msvcrt).locking(handle.fileno(), cast(Any, msvcrt).LK_NBLCK, 1)
            else:
                import fcntl

                cast(Any, fcntl).flock(
                    handle.fileno(), cast(Any, fcntl).LOCK_EX | cast(Any, fcntl).LOCK_NB
                )
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


def release(handle: Any) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            cast(Any, msvcrt).locking(handle.fileno(), cast(Any, msvcrt).LK_UNLCK, 1)
        else:
            import fcntl

            cast(Any, fcntl).flock(handle.fileno(), cast(Any, fcntl).LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def exclusive(path: Path, *, timeout: float = 5.0) -> Iterator[None]:
    """Serialize a critical section on ``path`` across processes.

    ``path`` is the lock file itself; it is never removed — a lock must outlive
    the section it guards, and unlinking it would let a peer lock a different
    inode and serialize nothing.
    """

    key = str(path)
    if _held.get(key, 0) > 0:
        yield  # re-entrant: an outer frame in this process already holds it
        return

    handle = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 — closed in finally
    except OSError:
        yield
        return

    acquired = acquire(handle, timeout=timeout)
    _held[key] = _held.get(key, 0) + 1
    try:
        yield
    finally:
        _held[key] -= 1
        if not _held[key]:
            del _held[key]
        if acquired:
            release(handle)
        with contextlib.suppress(OSError):
            handle.close()


def lock_path_for(target: Path, *, suffix: str = ".lock") -> Path:
    """A sibling lock path for ``target``."""

    return target.with_name(target.name + suffix)


__all__ = ["acquire", "release", "exclusive", "lock_path_for"]
