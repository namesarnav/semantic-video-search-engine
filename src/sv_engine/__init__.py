"""Semantic video search engine.

Import-order note (macOS/arm64): faiss and torch each ship their own copy of
libomp, and whichever one initializes second aborts the process with
"OMP: Error #15 ... already initialized". Importing torch first here makes the
order deterministic no matter which module an entry point touches first, so we
do not need the KMP_DUPLICATE_LIB_OK escape hatch (which openmp's own docs call
unsafe and capable of silently wrong results).

Do not remove or reorder this import, and do not let a module import faiss at
package-import time ahead of it.
"""

import torch as _torch  # noqa: F401  # must precede any faiss import

__all__ = ["__version__"]

__version__ = "0.1.0"
