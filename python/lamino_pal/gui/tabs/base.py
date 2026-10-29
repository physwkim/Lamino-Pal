"""What every tab has: the window's context, a panel in the parameters dock,
its napari viewers, and the ops it runs through the window's runner."""

from __future__ import annotations

from typing import Any, Callable

from silx.gui import qt

from lamino_pal.gui.actions import figure_name, job_figure, save_figure
from lamino_pal.gui.runner import by_hand
from lamino_pal.gui.viewer import Viewer
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import Result
from lamino_pal.project import Method

GOOD, WARN, BAD = "#070", "#b60", "#b00"


def badge(label: qt.QLabel, text: str, color: str | None = None) -> None:
    label.setText(text if color is None else f"<span style='color:{color}'><b>{text}</b></span>")


def button(text: str, slot: Callable[[], None], tip: str = "") -> qt.QPushButton:
    b = qt.QPushButton(text)
    b.clicked.connect(lambda _=False: by_hand(slot))
    b.setToolTip(tip)
    return b


def spin(lo: int, hi: int, value: int = 0) -> qt.QSpinBox:
    s = qt.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def dspin(lo: float, hi: float, value: float, decimals: int = 2, step: float = 1.0) -> qt.QDoubleSpinBox:
    s = qt.QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    return s


def floats(text: str) -> list[float]:
    """Numbers separated by commas or spaces."""
    return [float(x) for x in text.replace(",", " ").split()]


class Tab(qt.QWidget):
    NAME = ""
    default_op: str | None = None

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.project = win.project
        self.runner = win.runner
        self.controls = qt.QWidget()
        self.panel = qt.QVBoxLayout(self.controls)
        self.docks: list[qt.QDockWidget] = []
        self.viewers: list[Viewer] = []

    def section(self, title: str) -> qt.QFormLayout:
        """A titled form in the parameters panel."""
        box = qt.QGroupBox(title)
        form = qt.QFormLayout(box)
        self.panel.addWidget(box)
        return form

    def viewer(self, title: str, docked: bool = True) -> Viewer:
        """A viewer of this tab; a docked one's layer list and controls show
        in the window while this tab does, from when its canvas is made."""
        v = Viewer(title)
        if docked:
            v.made.connect(lambda qv: self.win.attach_docks(self, qv))
        self.viewers.append(v)
        return v

    def close_viewers(self) -> None:
        for v in self.viewers:
            v.close_canvas()

    def say(self, text: str, secs: int = 20) -> None:
        self.win.statusBar().showMessage(text, secs * 1000)

    def need(self, what: str) -> Any:
        """The window's current `what` (tag, group or volume), or None with
        the user told to pick one."""
        v = getattr(self.win, what)
        if v is None:
            noun = {"tag": "a scan", "group": "a group", "volume": "a volume"}[what]
            self.say(f"{self.NAME}: pick {noun} in the project tree")
        return v

    def run(
        self,
        op: str,
        params: dict[str, Any],
        done: Callable[[Result], None],
        channel: str | None = None,
        failed: Callable[[str], None] | None = None,
    ) -> None:
        self.runner.run(op, params, done, channel=channel, failed=failed)

    def submit(self, op: str, params: dict[str, Any]) -> JobState | None:
        st = self.runner.submit(op, params)
        if st is not None:
            self.say(f"{op}: job {st.id} queued" + ("" if st.gpu is None else f" for GPU {st.gpu}"), 10)
        return st

    def figure(self, op: str, r: Result, name: str | None = None) -> str | None:
        """Op `op`'s figure of result `r`, saved to cite: its path under the
        work directory, None for an op without one."""
        fig = save_figure(self.project, op, r.model_dump(mode="json"), r.arrays, name or figure_name(op))
        return None if fig is None else str(fig[0])

    def job_figure(self, st: JobState) -> str | None:
        fig = job_figure(self.project, st)
        return None if fig is None else str(fig[0])

    def decide(self, field: str, value: Any, evidence: list[str | None], method: Method) -> None:
        """Decide `field` of the current group as the user."""
        group = self.need("group")
        evidence = [e for e in evidence if e]
        if group is None:
            return
        if not evidence:
            self.say(f"{field}: no figure to cite; run the op that shows it first")
            return
        params = {"group": group, "field": field, "value": value, "evidence": evidence, "method": method}
        self.run("group.decide", params, lambda r: self.say(f"{group}: {field} decided ({method})", 10))

    def on_context(self) -> None:
        """The group, scan or volume in view changed."""

    def on_project(self) -> None:
        """project.toml or the work directory changed."""

    def on_job(self, st: JobState) -> None:
        """A job ended, or the artifact of one of this tab's jobs was picked:
        every tab hears every job, and acts on the ops it shows."""
