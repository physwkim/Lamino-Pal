"""Job ops as detached processes that outlive the GUI or MCP server that
started them. Each job owns `work_dir/jobs/<id>/`: `state.json`, written
atomically under `state.lock`, and the process's output in `out.log`.

The job process runs its op through `dispatch.run_job`, pinned to the batch
GPU picked here: CUDA's device is fixed by TOMOXIDE_CUDA_DEVICES in the
environment it starts with.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel

from lamino_pal import LaminoPalError, Refused, dispatch
from lamino_pal.dispatch import Caller
from lamino_pal.ops.registry import REGISTRY, _Out
from lamino_pal.project import Project

Status = Literal["queued", "running", "done", "failed", "cancelled"]
ACTIVE = ("queued", "running")

# argv before the job directory; tests swap it to load their own ops.
COMMAND = [sys.executable, "-m", "lamino_pal.cli", "job"]
# How long a cancelled job may take to reach its next progress call before it
# exits from under the op. An op inside one long tomoxide call hears SIGTERM
# only when the call returns.
CANCEL_GRACE_S = 10.0


class JobState(BaseModel):
    id: str
    project: str  # project.toml
    op: str
    params: dict[str, Any]
    caller: Caller
    gpu: int | None
    status: Status = "queued"
    progress: float = 0.0
    message: str = ""
    eta_s: float | None = None
    created: str
    finished: str | None = None
    pid: int | None = None  # also the process group: jobs start a new session
    proc_start: int | None = None  # its /proc starttime, against pid reuse
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


class Cancelled(Exception):
    """Raised in the op at its next progress call after SIGTERM."""


def jobs_dir(project: Project) -> Path:
    return project.work_dir / "jobs"


def submit(project: Project, name: str, params: dict[str, Any], caller: Caller) -> JobState:
    """Validate job op `name`'s params here, so a bad call is refused now and
    not minutes later, and start its process."""
    op = REGISTRY.get(name)
    if op is None:
        raise LaminoPalError(f"unknown op {name}")
    if op.cost != "job":
        raise LaminoPalError(f"{name} is not a job op; run it directly")
    p = dispatch.validate(op, params, project)
    # GUI and MCP submit concurrently: the checks and the queued state go
    # down under one lock, so the next submitter counts this job.
    jobs_dir(project).mkdir(parents=True, exist_ok=True)
    with open(jobs_dir(project) / "submit.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Each job checks its outputs when it starts; two queued at once would
        # both pass and the second overwrite the first.
        outs = {getattr(p, k) for k in op.path_fields(_Out)}
        for other in list_(project):
            o = REGISTRY.get(other.op)
            clash = outs & {other.params.get(k) for k in o.path_fields(_Out)} if other.active and o else set()
            if clash:
                raise Refused(f"job {other.id} is writing {min(clash)}")
        gpu = _pick_gpu(project) if op.gpu else None
        now = datetime.now().astimezone()
        jid = f"{now:%Y%m%d-%H%M%S}-{name}-{secrets.token_hex(2)}"
        d = jobs_dir(project) / jid
        d.mkdir(parents=True)
        st = JobState(
            id=jid,
            project=str(project.path),
            op=name,
            params=p.model_dump(mode="json"),
            caller=caller,
            gpu=gpu,
            created=now.isoformat(timespec="seconds"),
        )
        _write(d, st)
    env = dict(os.environ)
    if gpu is not None:
        env["TOMOXIDE_CUDA_DEVICES"] = str(gpu)
    try:
        with open(d / "out.log", "wb") as log:
            proc = subprocess.Popen(
                [*COMMAND, str(d)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=project.work_dir,
                start_new_session=True,
            )
    except OSError as e:
        _update(d, status="failed", error=f"could not start: {e}", finished=_now())
        raise
    # Read before anything reaps it: until then the pid cannot be reused.
    start = _proc_start(proc.pid)
    threading.Thread(target=proc.wait, daemon=True).start()
    return _update(d, pid=proc.pid, proc_start=start)


def _dir(project: Project, jid: str) -> Path:
    d = jobs_dir(project) / jid
    if Path(jid).name != jid or not (d / "state.json").is_file():
        raise Refused(f"no job {jid}")
    return d


def get(project: Project, jid: str) -> JobState:
    """Job `jid`'s state; one whose process is gone without a result is
    marked failed."""
    d = _dir(project, jid)
    with _lock(d):
        st = _read(d)
        if st.active and st.pid is not None and not _alive(st):
            st = st.model_copy(
                update={
                    "status": "failed",
                    "error": f"the job process ended without a result; see {d / 'out.log'}",
                    "finished": _now(),
                }
            )
            _write(d, st)
    return st


def list_(project: Project) -> list[JobState]:
    d = jobs_dir(project)
    if not d.is_dir():
        return []
    return sorted(
        (get(project, j.name) for j in d.iterdir() if (j / "state.json").is_file()),
        key=lambda s: s.created,
    )


def wait(project: Project, jid: str, timeout_s: float) -> JobState:
    """Job `jid`'s state once it ends or `timeout_s` passes."""
    deadline = time.monotonic() + timeout_s
    while True:
        st = get(project, jid)
        if not st.active or time.monotonic() >= deadline:
            return st
        time.sleep(0.25)


def cancel(project: Project, jid: str) -> JobState:
    """SIGTERM job `jid`'s process group, once its pid is checked to still be
    that job's process."""
    d = _dir(project, jid)
    with _lock(d):
        st = _read(d)
        if not st.active:
            raise Refused(f"job {jid} is {st.status}")
        try:
            if not _alive(st) or os.getpgid(st.pid) != st.pid:
                raise ProcessLookupError
            os.killpg(st.pid, signal.SIGTERM)
        except ProcessLookupError:
            raise Refused(f"job {jid}'s process {st.pid} is not running") from None
    return st


def run(d: Path) -> int:
    """The job process: run the op `d`'s state names and record how it ended."""
    st = _read(d)
    project = Project(st.project)
    cancel_ = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel_.set())
    t_start = time.monotonic()

    def progress(frac: float, msg: str) -> None:
        nonlocal t_start
        if cancel_.is_set():
            raise Cancelled("cancelled")
        now = time.monotonic()
        # Each restart from 0 (after a GPU wait, a stack load) restarts the ETA.
        if frac <= 0:
            t_start = now
        eta = (now - t_start) * (1 - frac) / frac if frac > 0 else None
        _update(d, progress=frac, message=msg, eta_s=eta)

    end: dict[str, Any] = {}

    def work() -> None:
        try:
            r = dispatch.run_job(st.op, st.params, project=project, caller=st.caller, progress=progress)
            end.update(status="done", progress=1.0, eta_s=0.0, result=r.model_dump(mode="json"))
        except Cancelled:
            end.update(status="cancelled")
        except Refused as e:
            end.update(status="failed", error=str(e))
        except BaseException as e:
            traceback.print_exc()
            end.update(status="failed", error=f"{type(e).__name__}: {e}")

    _update(d, status="running")
    w = threading.Thread(target=work, name=f"job {st.id}")
    w.start()
    deadline = None
    while w.is_alive():
        w.join(0.2)
        if cancel_.is_set() and deadline is None:
            deadline = time.monotonic() + CANCEL_GRACE_S
        if deadline is not None and w.is_alive() and time.monotonic() >= deadline:
            _update(d, status="cancelled", message="stopped inside a call", finished=_now())
            sys.stdout.flush()
            os._exit(128 + signal.SIGTERM)
    _update(d, **end, finished=_now())
    return {"done": 0, "cancelled": 128 + signal.SIGTERM}.get(end["status"], 1)


def _pick_gpu(project: Project) -> int:
    """A batch GPU free now and with no job of this project waiting for it,
    else the one with the fewest such jobs."""
    batch = project.data.gpu.batch
    if not batch:
        raise Refused("the project has no batch GPU (gpu.batch)")
    queued = Counter(s.gpu for s in list_(project) if s.active)
    return min(batch, key=lambda n: (queued[n] > 0 or dispatch.gpu_busy(n), queued[n], n))


def _proc_start(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat, after the parenthesised comm."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return None if fields[0] == "Z" else int(fields[19])


def _alive(st: JobState) -> bool:
    return st.pid is not None and st.proc_start is not None and _proc_start(st.pid) == st.proc_start


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def _lock(d: Path) -> Iterator[None]:
    with open(d / "state.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _read(d: Path) -> JobState:
    return JobState.model_validate_json((d / "state.json").read_text())


def _write(d: Path, st: JobState) -> None:
    tmp = d / f".state.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(st.model_dump(mode="json"), indent=1))
    os.replace(tmp, d / "state.json")


def _update(d: Path, **fields: Any) -> JobState:
    with _lock(d):
        st = _read(d).model_copy(update=fields)
        _write(d, st)
        return st
