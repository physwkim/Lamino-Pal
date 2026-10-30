"""Ops run off the Qt thread, their results handed back on it.

GPU ops share one thread, as they share the interactive GPU. Calls on one
channel are latest-wins: a call superseded before it starts is skipped and a
superseded call's result dropped. A call a click starts asks whether to
cancel the one running on its channel; one a slider, an edit or a context
change starts leaves it running.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Callable

from pydantic import ValidationError
from silx.gui import qt
from silx.gui.utils.concurrent import submitToQtMainThread

from lamino_pal import LaminoPalError, _core, dispatch, jobs
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import REGISTRY, Result
from lamino_pal.project import Project


_by_hand: ContextVar[bool] = ContextVar("by_hand", default=False)


def by_hand(slot: Callable[[], None]) -> None:
    """Call `slot` as a click's: a call it starts asks whether to cancel the
    one running on its channel."""
    t = _by_hand.set(True)
    try:
        slot()
    finally:
        _by_hand.reset(t)


def message(e: BaseException) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
    if isinstance(e, LaminoPalError):
        return str(e)
    return f"{type(e).__name__}: {e}"


class Runner(qt.QObject):
    failed = qt.Signal(str, str)  # op, message
    progressed = qt.Signal(str, float, str)  # channel, fraction, message
    submitted = qt.Signal(object)  # JobState: started, or done on the params

    def __init__(self, project: Project, ask: Callable[[str], bool], parent: qt.QObject | None = None):
        """`ask(op)` is whether to cancel op, running on the channel of a
        call a click started."""
        super().__init__(parent)
        self.project = project
        self._ask = ask
        self._gpu = ThreadPoolExecutor(1, "gui-gpu-op")
        self._cpu = ThreadPoolExecutor(2, "gui-op")
        self._lock = threading.Lock()
        self._gen: dict[str, int] = {}
        self._running: dict[str, tuple[str, _core.CancelToken]] = {}  # channel: op, its cancel

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
        cancel = _core.CancelToken()
        with self._lock:
            gen = self._gen[ch] = self._gen.get(ch, 0) + 1
            running = self._running.get(ch)
        if running is not None and _by_hand.get():
            # The question's event loop runs other slots, which no click started.
            t = _by_hand.set(False)
            try:
                if self._ask(running[0]):
                    running[1].cancel()
            finally:
                _by_hand.reset(t)

        def current() -> bool:
            with self._lock:
                return self._gen[ch] == gen

        def work() -> None:
            with self._lock:
                if self._gen[ch] != gen:
                    return
                self._running[ch] = (name, cancel)
            try:
                r = dispatch.run(
                    name,
                    params,
                    project=self.project,
                    caller="gui",
                    progress=lambda frac, msg: self.progressed.emit(ch, frac, msg),
                    cancel=cancel,
                )
            except Exception as e:
                if current():
                    self.failed.emit(name, message(e))
                    if failed is not None:
                        submitToQtMainThread(failed, message(e))
                return
            finally:
                with self._lock:
                    if self._running.get(ch) == (name, cancel):
                        del self._running[ch]
            if current():
                submitToQtMainThread(done, r)

        (self._gpu if REGISTRY[name].gpu else self._cpu).submit(work)

    def submit(self, name: str, params: dict[str, Any]) -> JobState | None:
        """Start job op `name`, or answer with the done job that ran these
        params (`jobs.answering`); None, with `failed` emitted, when refused."""
        try:
            st = jobs.answering(self.project, name, params) or jobs.submit(self.project, name, params, "gui")
        except (LaminoPalError, ValidationError, OSError) as e:
            self.failed.emit(name, message(e))
            return None
        self.submitted.emit(st)
        return st

    def shutdown(self) -> None:
        for pool in (self._gpu, self._cpu):
            pool.shutdown(wait=False, cancel_futures=True)
