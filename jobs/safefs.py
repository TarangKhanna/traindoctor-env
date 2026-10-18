"""The ONE way trusted code touches anything inside a job-writable tree (run dir, workspace, ckpt dir, stderr files).

Rule: a path string is never followed twice and never resolved. The caller names a PINNED ROOT (a directory the
trusted side created; it is opened with O_NOFOLLOW itself) and a relative path; every component is then opened with
O_NOFOLLOW relative to its parent's fd, so a symlink planted at any depth (or swapped in between two operations by a
forked process) is refused instead of followed. Only regular files with a single link are read or written
(no FIFOs, devices, or hard links to files outside the tree)."""
from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def check_rel(rel: str) -> list[str]:
    if not isinstance(rel, str) or not rel or "\0" in rel or rel.startswith("/"):
        raise ValueError("bad relative path")
    parts = rel.split("/")
    if any(x in ("", ".", "..") for x in parts):
        raise ValueError("bad relative path")
    return parts


def _root_fd(root: Path | str) -> int:
    return os.open(str(root), DIR_FLAGS)


def _walk(root: Path | str, parts: list[str], create: bool = False) -> int:
    """fd of the directory root/parts[0]/.../parts[-1] (root itself for no parts); the caller closes it."""
    fd = _root_fd(root)
    try:
        for comp in parts:
            try:
                nfd = os.open(comp, DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(comp, 0o755, dir_fd=fd)
                nfd = os.open(comp, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nfd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _regular(fd: int) -> os.stat_result:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise OSError(errno.EINVAL, "not a plain regular file")
    return st


def _open_file(root, rel: str, flags: int, create_dirs: bool = False) -> int:
    parts = check_rel(rel)
    dfd = _walk(root, parts[:-1], create=create_dirs)
    try:
        fd = os.open(parts[-1], flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o644, dir_fd=dfd)
    finally:
        os.close(dfd)
    try:
        _regular(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_bytes(root, rel: str, max_bytes: int = 64 << 20) -> bytes:
    fd = _open_file(root, rel, os.O_RDONLY)
    with os.fdopen(fd, "rb") as f:
        blob = f.read(max_bytes + 1)
    if len(blob) > max_bytes:
        raise ValueError("file too large")
    return blob


def open_write(root, rel: str, append: bool = False):
    """Binary file object for root/rel, created (with its parent directories) without following symlinks.
    Truncates unless append. The caller closes it."""
    fd = _open_file(root, rel, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else 0), create_dirs=True)
    if not append:
        os.ftruncate(fd, 0)
    return os.fdopen(fd, "ab" if append else "wb")


def write_bytes(root, rel: str, data: bytes, append: bool = False) -> None:
    with open_write(root, rel, append) as f:
        f.write(data)


def makedirs(root, rel: str, exist_ok: bool = True) -> None:
    """Create root/rel; with exist_ok=False the last component must be new."""
    parts = check_rel(rel)
    if not exist_ok:
        dfd = _walk(root, parts[:-1], create=True)
        try:
            os.mkdir(parts[-1], 0o755, dir_fd=dfd)
        finally:
            os.close(dfd)
        return
    os.close(_walk(root, parts, create=True))


def listdir(root, rel: str = "") -> list[str]:
    fd = _walk(root, check_rel(rel) if rel else [])
    try:
        return sorted(os.listdir(fd))
    finally:
        os.close(fd)


def file_size(root, rel: str) -> int:
    """Size of a plain regular file, or 0 if it is missing or not one (progress accounting only)."""
    try:
        fd = _open_file(root, rel, os.O_RDONLY)
    except (OSError, ValueError):
        return 0
    try:
        return os.fstat(fd).st_size
    finally:
        os.close(fd)
