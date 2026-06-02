#!/usr/bin/env python
"""Collapse the duplicate OpenMP runtimes that faiss and torch each bundle.

On macOS both wheels ship their own ``libomp.dylib``. Loading both in one
process aborts with::

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

The documented escape hatch, ``KMP_DUPLICATE_LIB_OK=TRUE``, is described by
OpenMP's own docs as unsafe and capable of silently producing incorrect
results -- not a trade worth making in a retrieval system.

Instead this does what the error message actually recommends: ensure a single
OpenMP runtime is linked, by pointing faiss's copy at torch's. Both are LLVM
libomp with ABI/compatibility version 5.0.0, so they are interchangeable; the
version check below refuses to proceed if that ever stops being true.

Idempotent, and safe to re-run. ``uv sync`` reinstalls the real file, so run
this again after dependency changes.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

LIB = "libomp.dylib"


def dylib_compat_version(path: Path) -> str | None:
    """Read LC_ID_DYLIB's compatibility version, or None if unreadable."""
    try:
        out = subprocess.run(
            ["otool", "-l", str(path)], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if "compatibility version" in line:
            return line.split("compatibility version")[-1].strip()
    return None


def main() -> int:
    if sys.platform != "darwin":
        print("not macOS; nothing to do")
        return 0

    site = Path(sysconfig.get_paths()["purelib"])
    faiss_omp = site / "faiss" / ".dylibs" / LIB
    torch_omp = site / "torch" / "lib" / LIB

    if not faiss_omp.exists() or not torch_omp.exists():
        print(f"skipping: expected both\n  {faiss_omp}\n  {torch_omp}")
        return 0

    if faiss_omp.is_symlink() and faiss_omp.resolve() == torch_omp.resolve():
        print("already linked; nothing to do")
        return 0

    faiss_version = dylib_compat_version(faiss_omp)
    torch_version = dylib_compat_version(torch_omp)
    if faiss_version != torch_version:
        print(
            "refusing to link: OpenMP ABI versions differ "
            f"(faiss={faiss_version}, torch={torch_version}). "
            "These are no longer interchangeable -- investigate before overriding.",
            file=sys.stderr,
        )
        return 1

    backup = faiss_omp.with_suffix(faiss_omp.suffix + ".orig")
    if not backup.exists():
        faiss_omp.rename(backup)
    else:
        faiss_omp.unlink()

    faiss_omp.symlink_to(torch_omp)
    print(f"linked {faiss_omp} -> {torch_omp}\n  (original kept at {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
