"""Starting a process that runs interactive GPU ops (the GUI, the MCP server).

tomoxide reads TOMOXIDE_CUDA_DEVICES on every call and nothing may set it in a
running process (`dispatch.process_gpu`), so a launcher that finds it unset
puts the project's interactive GPU there by executing itself again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lamino_pal.project import Gpu, Project

ENV = "TOMOXIDE_CUDA_DEVICES"


def open_or_pin(path: str | os.PathLike[str] | None) -> Project | None:
    """Project `path`, None when it does not exist yet, with this process
    re-executed on its interactive GPU unless TOMOXIDE_CUDA_DEVICES already
    names the GPUs. A project still to be created pins the GPU it will be
    created with. Raises what reading an existing project raises."""
    if path is None:
        return None
    project = Project(path) if Path(path).exists() else None
    if ENV not in os.environ:
        gpu = (project.data.gpu if project else Gpu()).interactive
        argv, env = [sys.executable, *sys.orig_argv[1:]], {**os.environ, ENV: str(gpu)}
        if os.name == "nt":
            # Windows's exec ends this process, which whoever started it (an
            # MCP client on its pipes) takes for the server's end.
            sys.exit(subprocess.call(argv, env=env))
        os.execve(sys.executable, argv, env)
    return project
