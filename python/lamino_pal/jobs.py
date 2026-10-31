"""Job ops as detached processes that outlive the GUI or MCP server that
started them. Each job owns `work_dir/jobs/<id>/`: `state.json`, written
atomically under `state.lock`, and the process's output in `out.log`.

The job process runs its op through `dispatch.run_job`, pinned to the batch
GPU picked here: CUDA's device is fixed by TOMOXIDE_CUDA_DEVICES in the
environment it starts with.
"""

from __future__ import annotations

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
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

import psutil
from pydantic import BaseModel, ValidationError

from lamino_pal import LaminoPalError, Refused, _core, artifacts, dispatch, locks
from lamino_pal.dispatch import Caller
from lamino_pal.ops.registry import REGISTRY, _In, _Out
from lamino_pal.project import Project

Status = Literal["queued", "running", "done", "failed", "cancelled"]
ACTIVE = ("queued", "running")

# argv before the job directory; tests swap it to load their own ops.
COMMAND = [sys.executable, "-m", "lamino_pal.cli", "job"]
# How long a cancelled job may take to reach its next progress call before it
# exits from under the op. An op inside one long tomoxide call hears the
# cancel only when the call returns.
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
    message: str = ""  # the last progress line; empty once done, where it would read as the result
    eta_s: float | None = None
    created: str
    finished: str | None = None
    pid: int | None = None  # the job's while its last argument is the job's directory
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_sent: str | None = None  # when `cancel` asked the job to stop

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


def jobs_dir(project: Project) -> Path:
    return project.work_dir / "jobs"


def submit(project: Project, name: str, params: dict[str, Any], caller: Caller) -> JobState:
    """Validate job op `name`'s params here, so a bad call is refused now and
    not minutes later, and start its process. The submit is logged as a
    call of `name` that ended queued."""
    with ExitStack() as held:
        entry = held.enter_context(dispatch.logged(name, params, project, caller))
        op = dispatch.lookup(name, job=True)
        p = dispatch.validate(op, params, project)
        entry["params"] = p.model_dump(mode="json")
        jobs_dir(project).mkdir(parents=True, exist_ok=True)
        # GUI and MCP submit concurrently: the checks and the queued state go
        # down under one lock, so the next submitter counts this job.
        with locks.locked(jobs_dir(project) / "submit.lock"):
            # Each job checks its outputs when it starts; two queued at once
            # would both pass and the second overwrite the first.
            outs = {getattr(p, k) for k in op.path_fields(_Out)}
            writing = writers(list_(project))
            clash = outs & writing.keys()
            if clash:
                raise Refused(f"job {writing[min(clash)].id} is writing {min(clash)}")
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
                # To the microsecond, taken under the lock: jobs sort by it.
                created=now.isoformat(timespec="microseconds"),
            )
            # Held from the queued state to the pid: whoever reads an active
            # job under it finds its pid.
            held.enter_context(_lock(d))
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
                    # Out of the reach of the terminal or console the GUI runs in.
                    start_new_session=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as e:
            _write(d, st.model_copy(update={"status": "failed", "error": f"could not start: {e}", "finished": _now()}))
            raise
        threading.Thread(target=proc.wait, daemon=True).start()
        st = st.model_copy(update={"pid": proc.pid})
        _write(d, st)
        entry.update(status="queued", job=st.id)
    return st


def answering(project: Project, name: str, params: dict[str, Any]) -> JobState | None:
    """The done job a submit of job op `name` on `params` would only redo: the
    newest job writing its outputs, which still exist, ran these params but
    overwrite, and nothing an op reads as it runs, the project file and the
    stacks and input files, changed after that job was submitted. None when a
    submit is to start a job or be refused."""
    try:
        op = dispatch.lookup(name, job=True)
        p = dispatch.resolve(op, params, project)
    except (LaminoPalError, ValidationError):
        return None  # the submit refuses them, logged
    outs = {getattr(p, k) for k in op.path_fields(_Out)}
    if not outs or getattr(p, "overwrite") or not all(Path(o).exists() for o in outs):
        return None
    last = next((st for st in reversed(list_(project)) if outs & _outs(st)), None)
    if last is None or last.status != "done" or last.op != name:
        return None
    if {**last.params, "overwrite": False} != p.model_dump(mode="json"):
        return None
    reads = {str(project.path), *(getattr(p, k) for k in op.path_fields(_In))}
    reads |= {str(project.work_dir / s.h5) for s in project.reload().scan}
    # File times lag the clock by up to a kernel tick: a write in the second
    # the job was submitted in counts as after it.
    since = int(datetime.fromisoformat(last.created).timestamp())
    if any(Path(f).stat().st_mtime >= since for f in reads - outs if Path(f).exists()):
        return None
    return last


def _outs(st: JobState) -> set[str]:
    """The paths job `st` writes."""
    o = REGISTRY.get(st.op)
    return {st.params.get(k) for k in o.path_fields(_Out)} if o else set()


def writers(states: list[JobState]) -> dict[str, JobState]:
    """Each path the active jobs of `states` write, to its job: a submit
    writing one is refused."""
    return {o: st for st in states if st.active for o in _outs(st)}


def _dir(project: Project, jid: str) -> Path:
    d = jobs_dir(project) / jid
    if Path(jid).name != jid or not (d / "state.json").is_file():
        raise Refused(f"no job {jid}")
    return d


def get(project: Project, jid: str) -> JobState:
    """Job `jid`'s state; one whose process is gone without a result is
    marked failed, or cancelled when the cancel killed it before it ran."""
    d = _dir(project, jid)
    with _lock(d):
        return _settle(d, _read(d))


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
    """Ask job `jid` to stop: a running job stops its op and its children
    itself; one whose process has not started its op yet is terminated, once
    its pid is checked to still be that job's process."""
    d = _dir(project, jid)
    with _lock(d):
        st = _settle(d, _read(d))
        if not st.active:
            raise Refused(f"job {jid} is {st.status}")
        if st.status == "queued":
            # `run` marks it running under this lock: it has run nothing yet.
            try:
                psutil.Process(st.pid).terminate()
            except psutil.NoSuchProcess:
                pass  # settled as stopped before it ran
        st = st.model_copy(update={"cancel_sent": _now()})
        _write(d, st)
    return st


def run(d: Path) -> int:
    """The job process: run the op `d`'s state names and record how it ended."""
    st = _state(d)
    project = Project(st.project)
    cancel_ = _core.CancelToken()
    signal.signal(signal.SIGTERM, lambda *_: cancel_.cancel())
    # (time, fraction) the rate is measured from: a phase's first report, as
    # the share it opens with (a stack load, a reserved step) took no time
    # in proportion.
    since: tuple[float, float] | None = None

    def progress(frac: float, msg: str) -> None:
        nonlocal since
        now = time.monotonic()
        # Each restart from 0 (after a GPU wait, a stack load) restarts the ETA.
        if frac <= 0:
            since = None
        elif since is None:
            since = (now, frac)
        eta = None
        if since is not None and frac > since[1]:
            eta = (now - since[0]) * (1 - frac) / (frac - since[1])
        _update(d, progress=frac, message=msg, eta_s=eta)

    end: dict[str, Any] = {}

    def work() -> None:
        try:
            r = dispatch.run_job(
                st.op, st.params, project=project, caller=st.caller, progress=progress, cancel=cancel_
            )
            end.update(status="done", progress=1.0, message="", eta_s=0.0, result=r.model_dump(mode="json"))
        except dispatch.Cancelled:
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
        if deadline is None and (cancel_.is_cancelled() or _state(d).cancel_sent is not None):
            # The token first: an op whose child dies under it is cancelled.
            cancel_.cancel()
            _stop_children()
            deadline = time.monotonic() + CANCEL_GRACE_S
        if deadline is not None and w.is_alive() and time.monotonic() >= deadline:
            dispatch.abandon("cancelled", "stopped inside a call")
            artifacts.discard()
            _update(d, status="cancelled", message="stopped inside a call", finished=_now())
            sys.stdout.flush()
            os._exit(128 + signal.SIGTERM)
    _update(d, **end, finished=_now())
    return {"done": 0, "cancelled": 128 + signal.SIGTERM}.get(end["status"], 1)


def _settle(d: Path, st: JobState) -> JobState:
    """`st`, read under `d`'s lock, ended if its process is gone."""
    if not st.active or _alive(st):
        return st
    if st.pid is None:
        # `submit` holds the lock until the pid is written.
        end = {"status": "failed", "error": "the submitting process ended before starting the job"}
    elif st.status == "queued" and st.cancel_sent is not None:
        # The process records every end it lives through, but `cancel`
        # terminates it until `run` starts the op.
        end = {"status": "cancelled", "message": "stopped before it ran"}
    else:
        end = {"status": "failed", "error": f"the job process ended without a result; see {d / 'out.log'}"}
    st = st.model_copy(update={**end, "finished": _now()})
    _write(d, st)
    return st


def _pick_gpu(project: Project) -> int:
    """A batch GPU free now and with no job of this project waiting for it,
    else the one with the fewest such jobs."""
    batch = project.data.gpu.batch
    if not batch:
        raise Refused("the project has no batch GPU (gpu.batch)")
    queued = Counter(s.gpu for s in list_(project) if s.active)
    return min(batch, key=lambda n: (queued[n] > 0 or dispatch.gpu_busy(n), queued[n], n))


def _stop_children() -> None:
    """Terminate the processes this job process started, theirs too."""
    for c in psutil.Process().children(recursive=True):
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            pass


def _alive(st: JobState) -> bool:
    # Its argv, not its start time: that moves with the system clock.
    if st.pid is None:
        return False
    try:
        p = psutil.Process(st.pid)
        argv = p.cmdline()
        return p.status() != psutil.STATUS_ZOMBIE and bool(argv) and Path(argv[-1]).name == st.id
    except psutil.Error:
        return False


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def _lock(d: Path) -> Iterator[None]:
    with locks.locked(d / "state.lock"):
        yield


def _state(d: Path) -> JobState:
    with _lock(d):
        return _read(d)


def _read(d: Path) -> JobState:
    return JobState.model_validate_json((d / "state.json").read_text(encoding="utf-8"))


def _write(d: Path, st: JobState) -> None:
    tmp = d / f".state.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(st.model_dump(mode="json"), indent=1), encoding="utf-8")
    os.replace(tmp, d / "state.json")


def _update(d: Path, **fields: Any) -> JobState:
    with _lock(d):
        st = _read(d).model_copy(update=fields)
        _write(d, st)
        return st
