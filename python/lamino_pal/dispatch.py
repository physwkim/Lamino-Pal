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
from functools import cache
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

# The entries of this process's op calls in progress, for a job that exits
# from under one.
_open: dict[int, tuple[Path, dict[str, Any], float]] = {}


class Cancelled(Exception):
    """An op call its job's cancel cut short, whatever the op raised."""


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
    """Run instant or interactive op `name` here, a GPU op on the project's
    interactive GPU."""
    return _run(name, params, project, caller, progress, gpu_wait_s, job=False)


def run_job(
    name: str,
    params: dict[str, Any] | Params,
    *,
    project: Project,
    caller: Caller,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> Result:
    """Run job op `name` in this job process, a GPU op on one of the project's
    batch GPUs, waiting for it as long as other jobs hold it. Once
    `cancelled()`, whatever the op raises is raised as `Cancelled`."""
    return _run(name, params, project, caller, progress, None, job=True, cancelled=cancelled)


def _run(
    name: str,
    params: dict[str, Any] | Params,
    project: Project,
    caller: Caller,
    progress: Progress | None,
    gpu_wait_s: float | None,
    job: bool,
    cancelled: Callable[[], bool] = lambda: False,
) -> Result:
    t0 = time.monotonic()
    entry: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "op": name,
        "caller": caller,
        "revision": project.revision,
        "tomoxide": _core.tomoxide_rev(),
    }
    _open[id(entry)] = (log_path(project), entry, t0)
    try:
        op = REGISTRY.get(name)
        if op is None:
            raise LaminoPalError(f"unknown op {name}")
        if (op.cost == "job") != job:
            raise LaminoPalError(
                f"{name} is a job op; submit it through jobs" if not job else f"{name} is not a job op"
            )
        p = validate(op, params, project)
        entry["params"] = p.model_dump(mode="json")
        ctx = Context(project, caller, progress or (lambda frac, msg: None))
        lock = nullcontext()
        if op.gpu:
            gpu = process_gpu()
            gpus = project.data.gpu.batch if job else [project.data.gpu.interactive]
            if gpu not in gpus:
                on = "no single GPU" if gpu is None else f"GPU {gpu}"
                which = "a batch GPU" if job else "the interactive GPU"
                raise Refused(
                    f"{name} runs on {which} {gpus if job else gpus[0]}; this process's "
                    f"CUDA runs on {on} (start it with TOMOXIDE_CUDA_DEVICES set to one)"
                )
            if gpu_wait_s is None:
                ctx.progress(0.0, f"waiting for GPU {gpu}")
            lock = gpu_lock(gpu, gpu_wait_s)
        with lock as gpu:
            ctx.gpu = gpu
            r = op.fn(ctx, p)
        if not isinstance(r, op.result):
            raise TypeError(f"{name} returned {type(r).__name__}, not {op.result.__name__}")
        entry.update(status="ok", result=r.model_dump(mode="json"))
        return r
    except BaseException as e:
        # SIGTERM reaches the op's children too, so whatever the op raises
        # once the cancel arrived is the cancel's doing.
        if cancelled():
            entry.update(status="cancelled", error=f"{type(e).__name__}: {e}")
            if isinstance(e, Cancelled):
                raise
            raise Cancelled(entry["error"]) from e
        if isinstance(e, Refused):
            entry.update(status="refused", error=str(e))
        elif isinstance(e, ValidationError):
            entry.update(status="invalid", error=str(e))
        else:
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        _open.pop(id(entry), None)
        _close(log_path(project), entry, params, t0)


def abandon(status: str, error: str) -> None:
    """Log the op calls still running here as ended `status`: this process
    exits from under them."""
    for path, entry, t0 in list(_open.values()):
        entry.update(status=status, error=error)
        _close(path, entry, entry.get("params"), t0)
    _open.clear()


def _close(path: Path, entry: dict[str, Any], params: Any, t0: float) -> None:
    entry.setdefault("params", params)
    entry["elapsed_s"] = round(time.monotonic() - t0, 4)
    _append_log(path, entry)


def validate(op: Op, params: dict[str, Any] | Params, project: Project) -> Params:
    """`params` checked against `op`, with its paths resolved and checked."""
    if isinstance(params, BaseModel):
        params = params.model_dump()
    p = op.params.model_validate(params)
    if op.bind is not None:
        p = op.bind(p, project)
    return _resolve_paths(op, p, project)


def _resolve_paths(op: Op, p: Params, project: Project) -> Params:
    wd, raw = project.work_dir, project.raw_root
    unset = [k for k in op.path_fields(_Out) if getattr(p, k) is None]
    if unset:
        names = op.default_out(p, project)
        p = p.model_copy(update={k: names[k] for k in unset})
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


@cache
def process_gpu() -> int | None:
    """The GPU tomoxide runs this process's CUDA work on, or None for none or
    several. The launcher fixes it with TOMOXIDE_CUDA_DEVICES, which tomoxide
    reads on every call; nothing sets it in a running process, where setenv
    racing another thread's getenv is undefined."""
    d = _core.cuda_devices()
    return d[0] if len(d) == 1 else None


def gpu_lock_path(n: int) -> Path:
    d = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "lamino-pal"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"gpu{n}.lock"


def gpu_busy(n: int) -> bool:
    """Whether some process holds GPU `n` now."""
    with open(gpu_lock_path(n), "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


@contextmanager
def gpu_lock(n: int, wait_s: float | None) -> Iterator[int]:
    """Hold GPU `n` exclusively across processes; refuse after `wait_s`, or
    wait for good with None."""
    with open(gpu_lock_path(n), "a+") as f:
        if wait_s is None:
            fcntl.flock(f, fcntl.LOCK_EX)
        deadline = time.monotonic() + (wait_s or 0.0)
        while wait_s is not None:
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
