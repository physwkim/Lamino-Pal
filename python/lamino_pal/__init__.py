"""BL7C laminography reconstruction on tomoxide."""

import os

# Windows looks for an extension's DLLs nowhere on PATH: a core built with CUDA
# finds the toolkit's cudart and cufft only here, in bin up to CUDA 12, bin\x64
# from 13.
if os.name == "nt" and "CUDA_PATH" in os.environ:
    for _d in (r"bin\x64", "bin"):
        if os.path.isdir(_p := os.path.join(os.environ["CUDA_PATH"], _d)):
            os.add_dll_directory(_p)

from lamino_pal._core import LaminoPalError, Refused  # noqa: E402

__all__ = ["LaminoPalError", "Refused"]
