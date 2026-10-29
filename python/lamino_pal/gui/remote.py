"""What an agent may do to the window through `lamino_pal.remote`: show a
tab, select a project tree item, set a viewer's camera, set a graph's limits.
Each command runs on the Qt thread and returns what it left."""

from __future__ import annotations

from typing import Any

from silx.gui.plot import PlotWidget
from silx.gui.utils.concurrent import submitToQtMainThread

from lamino_pal import LaminoPalError, remote
from lamino_pal.gui.viewer import Viewer


def _tab(win, tab: str):
    t = win.tabs.get(tab)
    if t is None:
        raise LaminoPalError(f"no tab {tab}; tabs: {', '.join(win.tabs)}")
    return t


def _viewer(win, tab: str, viewer: str) -> Viewer:
    t = _tab(win, tab)
    v = next((v for v in t.viewers if v.model.title == viewer), None)
    if v is None:
        raise LaminoPalError(f"{tab} has no viewer {viewer}; viewers: {', '.join(v.model.title for v in t.viewers)}")
    if not v.model.layers:
        raise LaminoPalError(f"{tab}'s viewer {viewer} shows nothing yet")
    return v


def _graph(win, tab: str, graph: str) -> PlotWidget:
    t = _tab(win, tab)
    p = t.graphs.get(graph)
    if p is None:
        raise LaminoPalError(f"{tab} has no graph {graph}; graphs: {', '.join(t.graphs)}")
    if not p.getItems():
        raise LaminoPalError(f"{tab}'s graph {graph} shows nothing yet")
    return p


def state(win) -> dict[str, Any]:
    """The tab shown, the group, scan and volume in view, and each tab's
    viewers and graphs."""
    return {
        "tab": win.current_tab().NAME,
        "context": {"group": win.group, "tag": win.tag, "volume": win.volume},
        "tabs": {
            name: {"viewers": [v.model.title for v in t.viewers], "graphs": list(t.graphs)}
            for name, t in win.tabs.items()
        },
    }


def show_tab(win, tab: str) -> dict[str, Any]:
    win.show_tab(_tab(win, tab).NAME)
    return state(win)


def select(win, kind: str, value: str) -> dict[str, Any]:
    """Select `kind` `value` in the project tree as the user's click does."""
    keys = win.tree.keys(kind)
    if value not in keys:
        raise LaminoPalError(f"the project tree has no {kind} {value}; {kind}s: {', '.join(keys)}")
    win.tree.pick((kind, value))
    return state(win)


def view(
    win,
    tab: str,
    viewer: str,
    zoom: float | None = None,
    center: list[float] | None = None,
    angles: list[float] | None = None,
    fit: bool = False,
) -> dict[str, Any]:
    return _viewer(win, tab, viewer).look(zoom, center, angles, fit)


def graph(
    win,
    tab: str,
    graph: str,
    x: list[float] | None = None,
    y: list[float] | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    p = _graph(win, tab, graph)
    if reset:
        p.resetZoom()
    if x is not None:
        p.getXAxis().setLimits(*x)
    if y is not None:
        p.getYAxis().setLimits(*y)
    return {"x": list(p.getXAxis().getLimits()), "y": list(p.getYAxis().getLimits())}


COMMANDS = {"state": state, "show_tab": show_tab, "select": select, "view": view, "graph": graph}


def serve(win) -> remote.Server:
    """The window's commands served on its project's socket."""

    def handle(cmd: str, args: dict[str, Any]) -> Any:
        fn = COMMANDS.get(cmd)
        if fn is None:
            raise LaminoPalError(f"unknown command {cmd}; commands: {', '.join(COMMANDS)}")
        return submitToQtMainThread(fn, win, **args).result(remote.TIMEOUT_S)

    return remote.Server(win.project.path, handle)
