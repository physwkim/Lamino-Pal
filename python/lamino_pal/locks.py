"""Exclusive file locks between processes, as flock(2)'s on every platform:
one per open file, held until the `locked` block ends."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import IO, Iterator

if os.name == "nt":
    import msvcrt

    # Windows locks a byte range, and a locked byte refuses other handles'
    # reads and writes: lock one byte past anything written.
    _AT = 2**31 - 2

    def _lock(f: IO, op: int) -> bool:
        fd = f.fileno()
        pos = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _AT, os.SEEK_SET)
        try:
            msvcrt.locking(fd, op, 1)
            return True
        except OSError:
            if op == msvcrt.LK_UNLCK:
                raise
            return False
        finally:
            os.lseek(fd, pos, os.SEEK_SET)

    def _try(f: IO) -> bool:
        return _lock(f, msvcrt.LK_NBLCK)

    def _wait(f: IO) -> None:
        # LK_LOCK gives up after ten seconds.
        while not _try(f):
            time.sleep(0.05)

    def _unlock(f: IO) -> None:
        # Closing the handle frees it only some time later.
        _lock(f, msvcrt.LK_UNLCK)

else:
    import fcntl

    def _try(f: IO) -> bool:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _wait(f: IO) -> None:
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock(f: IO) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def locked(
    path: str | os.PathLike[str], mode: str = "a", wait_s: float | None = None, encoding: str = "utf-8"
) -> Iterator[IO | None]:
    """`path` open in `mode` and locked: waited for for good with None, else
    None once `wait_s` passes with another holding it."""
    with open(path, mode, encoding=encoding) as f:
        if wait_s is None:
            _wait(f)
        else:
            deadline = time.monotonic() + wait_s
            while not _try(f):
                if time.monotonic() >= deadline:
                    yield None
                    return
                time.sleep(0.05)
        try:
            yield f
        finally:
            f.flush()
            _unlock(f)
