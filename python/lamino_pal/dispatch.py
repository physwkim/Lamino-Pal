"""The one path every op call takes: validate, guard paths, lock a GPU, run, log.

The log entry is written here, in `finally`, so no op call escapes log.jsonl.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, ValidationError

from lamino_pal import LaminoPalError, Refused, _core
from lamino_pal.ops.registry import REGISTRY, Op, Params, Result, _In, _Out
from lamino_pal.project import Project, is_under

Caller = Literal["gui", "mcp", "cli"]
Progress = Callable[[float, str], None]


def log_path(project: Project) -> Path:
    return project.work_dir / "log.jsonl"


@dataclass
class Context:
    """What an op sees besides its params."""

    project: Project
    caller: Caller
    progress: Progress
    gpu: int | None = None


def run(
    name: str,
    params: dict[str, Any] | Params,
    *,
    project: Project,
    caller: Caller,
    progress: Progress | None = None,
    gpu_wait_s: float = 30.0,
) -> Result:
    t0 = time.monotonic()
    entry: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "op": name,
        "caller": caller,
        "revision": project.revision,
        "tomoxide": _core.tomoxide_rev(),
    }
    try:
        op = REGISTRY.get(name)
        if op is None:
            raise LaminoPalError(f"unknown op {name}")
        if op.cost == "job":
            raise LaminoPalError(f"{name} is a job op; submit it through jobs")
        if isinstance(params, BaseModel):
            params = params.model_dump()
        p = _resolve_paths(op, op.params.model_validate(params), project)
        entry["params"] = p.model_dump(mode="json")
        ctx = Context(project, caller, progress or (lambda frac, msg: None))
        lock = gpu_lock(project.data.gpu.interactive, gpu_wait_s) if op.gpu else nullcontext()
        with lock as gpu:
            ctx.gpu = gpu
            r = op.fn(ctx, p)
        if not isinstance(r, op.result):
            raise TypeError(f"{name} returned {type(r).__name__}, not {op.result.__name__}")
        entry.update(status="ok", result=r.model_dump(mode="json"))
        return r
    except Refused as e:
        entry.update(status="refused", error=str(e))
        raise
    except ValidationError as e:
        entry.update(status="invalid", error=str(e))
        raise
    except BaseException as e:
        entry.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        entry.setdefault("params", params)
        entry["elapsed_s"] = round(time.monotonic() - t0, 4)
        _append_log(log_path(project), entry)


def _resolve_paths(op: Op, p: Params, project: Project) -> Params:
    wd, raw = project.work_dir, project.raw_root
    upd: dict[str, str] = {}
    for k in op.path_fields(_In):
        path = (wd / getattr(p, k)).resolve()
        if not path.exists():
            raise Refused(f"{k}: {path} does not exist")
        upd[k] = str(path)
    for k in op.path_fields(_Out):
        path = (wd / getattr(p, k)).resolve()
        if not is_under(path, wd):
            raise Refused(f"{k}: {path} is outside the work directory {wd}")
        if is_under(path, raw):
            raise Refused(f"{k}: {path} is under the raw root {raw}")
        if path.exists() and not getattr(p, "overwrite"):
            raise Refused(f"{k}: {path} exists; pass overwrite to replace it")
        upd[k] = str(path)
    return p.model_copy(update=upd)


@contextmanager
def gpu_lock(n: int, wait_s: float) -> Iterator[int]:
    """Hold GPU `n` exclusively across processes; refuse after `wait_s`."""
    d = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "lamino-pal"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"gpu{n}.lock", "a+") as f:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    f.seek(0)
                    raise Refused(f"GPU {n} is busy ({f.read().strip() or 'holder unknown'})")
                time.sleep(0.05)
        f.seek(0)
        f.truncate()
        f.write(f"pid {os.getpid()}")
        f.flush()
        yield n


def _append_log(path: Path, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
