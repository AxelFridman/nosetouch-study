"""Cross-process file locks and atomic replacement, for the files that parallel processes update.

Lanes run at the same time, and two files are updated by read-modify-write from any process:
``cv.update_registry`` (``results/registry*.csv``) and ``plotting.save_fig`` (``captions.json`` and
``figure_texts.json`` of a figure folder). Both now run under :func:`file_lock` and replace the file with
:func:`atomic_path`::

    with file_lock(path):                          # <path>.lock, created with os.open(O_CREAT | O_EXCL)
        data = read(path)                          # read-modify-write while holding the lock
        with atomic_path(path) as tmp:             # write a temporary file next to it ...
            write(tmp, data)                       # ... and os.replace it over the target

* **Exclusive.** The lock is a file next to the protected one (``<name>.lock``). ``O_CREAT | O_EXCL`` fails
  when it exists, atomically on every OS, so exactly one process holds it. It records the owner's pid, host,
  time and a random token; the owner removes it on release only if the token is still its own.
* **Waiting.** A busy lock is retried with exponential backoff (5 ms doubling to 0.5 s, with jitter) for up
  to ``timeout`` seconds (default 600), then :class:`LockTimeout` is raised.
* **Stale locks.** A lock file older than ``stale_after`` seconds (default 120; the protected updates take
  milliseconds, so only a process that died while holding it leaves one) is broken: renamed to a unique name
  (atomic, so one waiter breaks it), checked to be the same file that was judged stale, and deleted.
* **Atomic replacement.** A reader that takes no lock (``pipeline.merge_registries``) never sees a half-written
  file. On Windows ``os.replace`` fails while another process has the target open; it is retried for a few
  seconds, then the file is written in place (the behaviour before the lock existed).

The lock is not re-entrant: never take the same lock twice in one process.
"""
from __future__ import annotations

import os
import random
import shutil
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_SUFFIX = '.lock'
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_STALE_S = 120.0
_FIRST_DELAY_S = 0.005
_MAX_DELAY_S = 0.5


class LockTimeout(TimeoutError):
    """The lock stayed busy (and not stale) for longer than the timeout."""


def lock_path(path: str | Path) -> Path:
    """The lock file that guards ``path``: ``<path>.lock`` in the same folder."""
    p = Path(path)
    return p.with_name(p.name + LOCK_SUFFIX)


def _try_create(lock: Path, token: str) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except PermissionError:   # Windows: the previous owner's lock file is being deleted (pending delete)
        return False
    try:
        os.write(fd, token.encode('utf-8'))
    finally:
        os.close(fd)
    return True


def _read(path: Path) -> str | None:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None


def _break_stale(lock: Path, seen: os.stat_result) -> bool:
    """Remove a stale lock without ever removing a fresh one that replaced it in the meantime.
    True when the lock path is free again (broken here or by another waiter)."""
    grave = lock.with_name(f'{lock.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.stale')
    try:
        os.rename(lock, grave)                       # atomic: of several waiters, one gets this file
    except FileNotFoundError:
        return True                                  # another waiter broke it (or its owner released it)
    except OSError:
        return False
    try:
        st = os.stat(grave)
        if (st.st_mtime_ns, st.st_size) != (seen.st_mtime_ns, seen.st_size):
            try:                                     # a fresh lock replaced the stale one: give it back
                os.rename(grave, lock)
                return False
            except OSError:
                pass
        os.unlink(grave)
    except OSError:
        pass
    return True


def _release(lock: Path, token: str) -> None:
    delay = _FIRST_DELAY_S
    for _ in range(40):                              # about 10 s in all
        try:
            if _read(lock) != token:                 # gone, or broken as stale and taken by another process
                return
            os.unlink(lock)
            return
        except FileNotFoundError:
            return
        except PermissionError:                      # Windows: someone has it open right now
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


@contextmanager
def file_lock(path: str | Path, *, timeout: float = DEFAULT_TIMEOUT_S,
              stale_after: float = DEFAULT_STALE_S) -> Iterator[Path]:
    """Hold the cross-process lock of ``path`` (the file ``<path>.lock``) for the ``with`` block.

    The folder of ``path`` must exist. Raises :class:`LockTimeout` after ``timeout`` seconds of a busy,
    non-stale lock; breaks a lock file older than ``stale_after`` seconds. Yields the lock file's path."""
    lock = lock_path(path)
    token = f'pid={os.getpid()} host={socket.gethostname()} time={time.time():.3f} token={uuid.uuid4().hex}\n'
    deadline = time.monotonic() + float(timeout)
    delay = _FIRST_DELAY_S
    while not _try_create(lock, token):
        try:
            st = os.stat(lock)
        except OSError:                              # released in between (or not creatable): wait and retry
            st = None
        if st is not None and time.time() - st.st_mtime > stale_after and _break_stale(lock, st):
            continue
        if time.monotonic() >= deadline:
            raise LockTimeout(f'{lock} stayed busy for {timeout:.0f} s (holder: {(_read(lock) or "?").strip()}); '
                              f'a lock file older than {stale_after:.0f} s is broken automatically')
        time.sleep(delay * (0.5 + random.random()))
        delay = min(delay * 2, _MAX_DELAY_S)
    try:
        yield lock
    finally:
        _release(lock, token)


def replace_file(src: str | Path, dst: str | Path, *, retries: int = 30) -> None:
    """``os.replace(src, dst)``, retried while ``dst`` is open elsewhere (Windows), then an in-place copy."""
    delay = _FIRST_DELAY_S
    for _ in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
            delay = min(delay * 2, 0.25)
    shutil.copyfile(src, dst)                         # the behaviour before atomic replacement
    try:
        os.unlink(src)
    except OSError:
        pass


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; when the block succeeds it replaces ``path`` (else it is
    deleted). Write the content exactly as you would have written ``path``."""
    p = Path(path)
    tmp = p.with_name(f'.{p.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp')
    try:
        yield tmp
        replace_file(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
