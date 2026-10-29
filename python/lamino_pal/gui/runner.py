"""Ops run off the Qt thread, their results handed back on it.

GPU ops share one thread, as they share the interactive GPU. Calls on one
channel are latest-wins: a call superseded before it starts is skipped and a
superseded call's result dropped; a call already running is not interrupted.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from pydantic import ValidationError
from silx.gui import qt
from silx.gui.utils.concurrent import submitToQtMainThread

from lamino_pal import LaminoPalError, dispatch, jobs
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import REGISTRY, Result
from lamino_pal.project import Project


def message(e: BaseException) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
    if isinstance(e, LaminoPalError):
        return str(e)
    return f"{type(e).__name__}: {e}"


class Runner(qt.QObject):
    failed = qt.Signal(str, str)  # op, message
    progressed = qt.Signal(str, float, str)  # channel, fraction, message
    submitted = qt.Signal(object)  # JobState

    def __init__(self, project: Project, parent: qt.QObject | None = None):
        super().__init__(parent)
        self.project = project
        self._gpu = ThreadPoolExecutor(1, "gui-gpu-op")
        self._cpu = ThreadPoolExecutor(2, "gui-op")
        self._lock = threading.Lock()
        self._gen: dict[str, int] = {}

    def run(
        self,
        name: str,
        params: dict[str, Any],
        done: Callable[[Result], None],
        channel: str | None = None,
        failed: Callable[[str], None] | None = None,
    ) -> None:
        """Run instant or interactive op `name` and call `done` with its result
        on the Qt thread, unless a later call on `channel` (default: the op)
        superseded it; `failed` also hears a refusal or error."""
        ch = channel or name
        with self._lock:
            gen = self._gen[ch] = self._gen.get(ch, 0) + 1

        def current() -> bool:
            with self._lock:
                return self._gen[ch] == gen

        def work() -> None:
            if not current():
                return
            try:
                r = dispatch.run(
                    name,
                    params,
                    project=self.project,
                    caller="gui",
                    progress=lambda frac, msg: self.progressed.emit(ch, frac, msg),
                )
            except Exception as e:
                if current():
                    self.failed.emit(name, message(e))
                    if failed is not None:
                        submitToQtMainThread(failed, message(e))
                return
            if current():
                submitToQtMainThread(done, r)

        (self._gpu if REGISTRY[name].gpu else self._cpu).submit(work)

    def submit(self, name: str, params: dict[str, Any]) -> JobState | None:
        """Start job op `name`; None, with `failed` emitted, when refused."""
        try:
            st = jobs.submit(self.project, name, params, "gui")
        except (LaminoPalError, ValidationError, OSError) as e:
            self.failed.emit(name, message(e))
            return None
        self.submitted.emit(st)
        return st

    def shutdown(self) -> None:
        for pool in (self._gpu, self._cpu):
            pool.shutdown(wait=False, cancel_futures=True)
