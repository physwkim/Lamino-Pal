"""What every tab has: the window's context, a panel in the parameters dock,
its napari viewers and silx graphs, and the ops it runs through the window's
runner."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

from matplotlib import rcParams
from matplotlib.backend_bases import DrawEvent
from matplotlib.text import Text
from silx import config
from silx.gui import qt
from silx.gui.plot import Plot1D, PlotWidget

from lamino_pal.gui.actions import figure_name, job_figure, save_figure
from lamino_pal.gui.runner import by_hand
from lamino_pal.gui.viewer import Viewer
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import Result
from lamino_pal.project import Method

GOOD, WARN, BAD = "#070", "#b60", "#b00"

# silx's margins are fixed fractions of a graph, which in a short one clip its
# title and tick labels; its tight layout, read as a graph is made, fits them
# to the text.
config._MPL_TIGHT_LAYOUT = True
# A graph's least size: its toolbar, position bar and colorbar, its ticks and
# labels, and a plot PLOT_LEAST px or more each way; until it is drawn, with no
# title.
GRAPH_SIZE = (300, 228)
PLOT_LEAST = 80


def _wrap_width(title: Text) -> float:
    """matplotlib's width to wrap a centered title at, twice the way to the
    figure's nearer side, without its way to the figure's top: that is
    negative while the title is above the figure, as it is before the layout
    makes room, and matplotlib then gives every word a line."""
    x = title.get_transform().transform(title.get_position())[0]
    box = title.get_figure().get_window_extent()
    return 2 * min(x - box.x0, box.x1 - x)


def _hold_plot(p: PlotWidget, e: DrawEvent) -> None:
    """A graph's least height is its bars, the text drawn around its plot with
    the layout's pads, and PLOT_LEAST: its title wraps to more lines as it
    narrows, a failure can take many, and an axis's ×10ⁿ pushes the title up.
    Measured from the text, not the plot's place, which a layout too big for
    the figure leaves where it was."""
    # A draw to a file is laid out by none, on another renderer.
    if e.canvas.is_saving():
        return
    b = p.getBackend()
    fig = b.fig
    pad = 2 * fig.get_layout_engine().get()["pad"] * rcParams["font.size"] / 72
    text = fig.get_tightbbox(e.renderer).height + pad - b.ax.bbox.height / fig.dpi
    ppi = b.height() / fig.get_figheight()
    p.setMinimumHeight(p.height() - b.height() + math.ceil(text * ppi) + PLOT_LEAST)


def _blank(p: PlotWidget, clear: Callable[[], None], empty: str) -> None:
    clear()
    p.setGraphTitle(empty)
    p.resetZoom()


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
        self.viewers: list[Viewer] = []
        self.graphs: dict[str, PlotWidget] = {}

    def heightForWidth(self, w: int) -> int:
        # Read by the _Page scrolling the tab as its least height; a layout's
        # is its preferred, and a word-wrapped label in it would have the tab
        # scroll at any window size.
        return self.layout().totalMinimumHeightForWidth(w)

    def section(self, title: str) -> qt.QFormLayout:
        """A titled form in the parameters panel."""
        box = qt.QGroupBox(title)
        form = qt.QFormLayout(box)
        self.panel.addWidget(box)
        return form

    def viewer(self, title: str, docked: bool = True) -> Viewer:
        """A viewer of this tab; a docked one's layer list and controls show
        in the window while it does, from when its canvas is made."""
        v = Viewer(title)
        if docked:
            v.made.connect(lambda qv: self.win.attach_docks(v, qv))
        self.viewers.append(v)
        return v

    def graph(self, title: str, x: str, y: str, cls: type[PlotWidget] = Plot1D, empty: str = "") -> PlotWidget:
        """A graph of this tab with axes `x` and `y`, named `title` to an agent,
        and titled `empty`, what draws it, while cleared."""
        p = cls()
        p.getXAxis().setLabel(x)
        p.getYAxis().setLabel(y)
        p.setMinimumSize(*GRAPH_SIZE)
        if cls is Plot1D:
            # matplotlib's margins: a curve's peak or end on the frame is
            # hidden under it. An image fills its frame.
            p.setDataMargins(0.05, 0.05, 0.05, 0.05)
        # Its title wrapped at draw time, so the layout and the drawing agree,
        # and its least height held to that drawing.
        b = p.getBackend()
        b.ax.title.set_wrap(True)
        b.ax.title._get_wrap_line_width = partial(_wrap_width, b.ax.title)
        b.mpl_connect("draw_event", partial(_hold_plot, p))
        # silx's clear keeps the title and limits, of a result no longer shown.
        p.clear = partial(_blank, p, p.clear, empty)
        p.clear()
        self.graphs[title] = p
        return p

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
        if st is not None and not st.active:
            self.say(f"{op}: job {st.id} ran these params; its result is shown", 10)
        elif st is not None:
            self.say(f"{op}: job {st.id} queued" + ("" if st.gpu is None else f" for GPU {st.gpu}"), 10)
        return st

    def figure(self, op: str, r: Result, name: str | None = None) -> str | None:
        """Op `op`'s figure of result `r`, saved to cite: its path under the
        work directory, None for an op without one."""
        fig = save_figure(self.project, op, r.model_dump(mode="json"), r.arrays, name or figure_name(op))
        return None if fig is None else fig[0]

    def job_figure(self, st: JobState) -> str | None:
        fig = job_figure(self.project, st)
        return None if fig is None else fig[0]

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
