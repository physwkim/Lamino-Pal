"""The GUI offscreen, without GL: the layers, the steps and the widgets
around them. What the canvases draw is test_render.py's."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Else offscreen draws through DISPLAY's GLX when it has one.
os.environ["QT_QPA_OFFSCREEN_NO_GLX"] = "1"
os.environ["QT_API"] = "pyside6"

import threading
import time

import h5py
import numpy as np
import PySide6.QtCore  # noqa: F401  # silx takes the binding napari's qtpy is pinned to
import pytest
from conftest import band_at, pump
from silx.gui import qt
from silx.gui.plot.items import Shape
from test_recon import _scan

from lamino_pal import LaminoPalError, _core, dispatch, remote
from lamino_pal.artifacts import save_h5
from lamino_pal.gui import status
from lamino_pal.gui.actions import ACTIONS, RunOpDialog
from lamino_pal.gui.form import ParamsForm
from lamino_pal.gui.runner import Runner, by_hand
from lamino_pal.gui.viewer import Viewer
from lamino_pal.jobs import JobState
from lamino_pal.ops import REGISTRY
from lamino_pal.ops.align import Rival, SweepResult, Verdict
from lamino_pal.ops.volume import FocusResult


@pytest.fixture(scope="module")
def app():
    return qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture
def scene(project):
    """Scan `s` in group `m` at center 64.5 px and tilt 20, and its volume
    v.h5 with provenance; the batch GPU is this process's."""
    _scan(project, "s", shape=(32, 64))
    dispatch.run("scan.assign_group", {"tags": ["s"], "group": "m"}, project=project, caller="cli")

    def decide(f):
        f.gpu.batch = [0]
        f.group[0].center_px, f.group[0].tilt_deg = 64.5, 20.0

    project.commit(decide)
    depth = _core.lamino_depth(32, 20.0)
    with h5py.File(project.work_dir / "v.h5", "w") as f:
        f["exchange/data"] = np.random.default_rng(0).random((depth, 64, 64), dtype="f4")
        f.require_group("lamino_pal/provenance").attrs.update(tag="s", center_px=64.5, tilt_deg=20.0, bin=2)
    return project


@pytest.fixture
def win(app, scene):
    from lamino_pal.gui.window import MainWindow

    w = MainWindow(scene)
    w.show()
    pump(app, 0.2)
    yield w
    w.close()
    pump(app, 0.2)


def names(viewer):
    return [layer.name for layer in viewer.model.layers]


def test_every_op_is_an_action():
    assert set(ACTIONS) == set(REGISTRY)


def test_a_required_param_that_takes_none_is_none_when_empty(app):
    form = ParamsForm(REGISTRY["scan.assign_group"])
    form.set(tags=["s"])
    assert form.values() == {"tags": ["s"], "group": None}
    form.set(group="m")
    assert form.values() == {"tags": ["s"], "group": "m"}


def test_a_superseded_call_is_skipped_or_its_result_dropped(app, project, monkeypatch):
    started, gate, ran, done = threading.Event(), threading.Event(), [], []

    def run(name, params, **kw):
        ran.append(params["i"])
        if params["i"] == 1:
            started.set()
            gate.wait(10)
        return params["i"]

    monkeypatch.setattr(dispatch, "run", run)
    asked = []
    runner = Runner(project, lambda op: asked.append(op) or True)
    # A GPU op: its calls queue on one thread.
    name = next(n for n, o in REGISTRY.items() if o.gpu and o.cost != "job")
    runner.run(name, {"i": 1}, done.append, channel="c")
    assert started.wait(10)
    runner.run(name, {"i": 2}, done.append, channel="c")
    runner.run(name, {"i": 3}, done.append, channel="c")
    gate.set()
    assert pump(app, 10, lambda: done)
    pump(app, 0.2)
    runner.shutdown()
    # 1 ran on and was dropped; 2 was superseded before it started. No click
    # started them, so none asked.
    assert ran == [1, 3] and done == [3] and asked == []


def _blocking_run(started, gate, ran, tokens):
    """A dispatch.run whose call 1 blocks until `gate` or its cancel."""

    def run(name, params, *, cancel, **kw):
        ran.append(params["i"])
        tokens.append(cancel)
        if params["i"] == 1:
            started.set()
            while not (gate.is_set() or cancel.is_cancelled()):
                time.sleep(0.01)
            if cancel.is_cancelled():
                raise LaminoPalError("cancelled")
        return params["i"]

    return run


@pytest.mark.parametrize("answer", [True, False])
def test_a_click_asks_whether_to_cancel_the_call_running_on_its_channel(app, project, monkeypatch, answer):
    started, gate, ran, tokens, done, failed, asked = threading.Event(), threading.Event(), [], [], [], [], []
    monkeypatch.setattr(dispatch, "run", _blocking_run(started, gate, ran, tokens))
    runner = Runner(project, lambda op: asked.append(op) or answer)
    name = next(n for n, o in REGISTRY.items() if o.gpu and o.cost != "job")
    runner.run(name, {"i": 1}, done.append, channel="c", failed=failed.append)
    assert started.wait(10)
    by_hand(lambda: runner.run(name, {"i": 2}, done.append, channel="c", failed=failed.append))
    assert asked == [name] and tokens[0].is_cancelled() == answer
    gate.set()
    assert pump(app, 10, lambda: done)
    pump(app, 0.2)
    runner.shutdown()
    # Cancelled or run on, 1 was superseded: its failure or result is dropped.
    assert ran == [1, 2] and done == [2] and failed == [] and not tokens[1].is_cancelled()


def test_a_call_the_question_starts_does_not_ask(app, project, monkeypatch):
    started, gate, ran, tokens, done, asked = threading.Event(), threading.Event(), [], [], [], []
    monkeypatch.setattr(dispatch, "run", _blocking_run(started, gate, ran, tokens))
    name = next(n for n, o in REGISTRY.items() if o.gpu and o.cost != "job")

    def ask(op):
        # A slider moved while the question is up: its slot runs in the
        # question's event loop.
        asked.append(op)
        runner.run(name, {"i": 3}, done.append, channel="c")
        return False

    runner = Runner(project, ask)
    runner.run(name, {"i": 1}, done.append, channel="c")
    assert started.wait(10)
    by_hand(lambda: runner.run(name, {"i": 2}, done.append, channel="c"))
    gate.set()
    assert pump(app, 10, lambda: done)
    pump(app, 0.2)
    runner.shutdown()
    assert asked == [name] and ran == [1, 3] and done == [3]


def _remote(app, win, cmd, **args):
    """remote.call from a thread of its own, as the MCP server makes it, while
    the Qt thread runs."""
    out = {}

    def call():
        try:
            out["r"] = remote.call(win.project.path, cmd, args)
        except Exception as e:
            out["e"] = e

    t = threading.Thread(target=call)
    t.start()
    assert pump(app, 15, lambda: not t.is_alive())
    if "e" in out:
        raise out["e"]
    return out["r"]


def test_an_agent_shows_a_tab_and_zooms_a_viewer_and_a_graph(app, win):
    st = _remote(app, win, "state")
    assert st["tab"] == win.current_tab().NAME
    assert st["context"] == {"group": win.group, "tag": win.tag, "volume": win.volume}
    assert st["tabs"]["Volume"] == {"viewers": ["z slice", "xz", "yz", "3D"], "graphs": ["focus by z"]}
    assert st["tabs"]["Export"]["graphs"] == ["histogram", "MIP z", "MIP y", "MIP x"]
    assert _remote(app, win, "show_tab", tab="Align")["tab"] == "Align" == win.current_tab().NAME
    v = win.tabs["Align"].view
    v.replace("img", np.zeros((40, 60), np.float32))
    cam = _remote(app, win, "view", tab="Align", viewer="align", zoom=3.0, center=[10.0, 20.0])
    assert cam["zoom"] == v.model.camera.zoom == 3.0 and cam["center"][1:] == [10.0, 20.0]
    cam = _remote(app, win, "view", tab="Align", viewer="align", fit=True)
    assert cam["center"][1:] == [19.5, 29.5]
    plot = win.tabs["Align"].plot
    plot.addCurve([0, 10], [0, 5], legend="c")
    lim = _remote(app, win, "graph", tab="Align", graph="focus", x=[1.0, 2.0], y=[-1.0, 1.0])
    assert lim == {"x": [1.0, 2.0], "y": [-1.0, 1.0]} and plot.getXAxis().getLimits() == (1.0, 2.0)
    lim = _remote(app, win, "graph", tab="Align", graph="focus", reset=True)
    assert lim["x"][0] <= 0 and lim["x"][1] >= 10
    with pytest.raises(LaminoPalError, match="^no tab Foo; tabs: Data, Raw"):
        _remote(app, win, "show_tab", tab="Foo")
    with pytest.raises(LaminoPalError, match="^Align has no graph FRC; graphs: focus$"):
        _remote(app, win, "graph", tab="Align", graph="FRC")
    with pytest.raises(LaminoPalError, match="^unknown command zoom"):
        _remote(app, win, "zoom")


def test_status_names_how_each_field_was_decided_and_the_next_tab(scene):
    g = scene.data.group[0]
    assert status.status(scene, None).next_tab == "Data"
    assert status.source(g, "center_px") == "file" and status.status(scene, g).next_tab == "Batch"
    (scene.work_dir / "fig.png").write_bytes(b"")
    decide = {"group": "m", "field": "tilt_deg", "value": 21.0, "evidence": ["fig.png"], "method": "eye"}
    dispatch.run("group.decide", decide, project=scene, caller="mcp")
    g = scene.data.group[0]
    assert status.source(g, "tilt_deg") == "agent" and [d.field for d in status.unconfirmed(g)] == ["tilt_deg"]
    dispatch.run("group.decide", decide, project=scene, caller="gui")
    g = scene.data.group[0]
    assert status.source(g, "tilt_deg") == "eye" and not status.unconfirmed(g)
    scene.commit(lambda f: setattr(f.group[0], "center_px", None))
    assert status.status(scene, scene.data.group[0]).next_tab == "Align"


def test_an_instant_op_shows_its_figure_and_a_job_its_end(app, win):
    win.set_context(tag="s", volume="v.h5")
    dialog = RunOpDialog(win, "volume.slice")
    assert dialog.form().values() == {"volume": "v.h5"}
    dialog._run()
    assert pump(app, 20, lambda: dialog.caption.text())
    assert "Saved as artifacts/figures/" in dialog.caption.text()
    ended = []
    win.jobs.ended.connect(ended.append)
    st = win.runner.submit("export.bin", {"volume": "v.h5"})
    assert st is not None
    assert pump(app, 60, lambda: ended)
    assert ended[0].id == st.id and ended[0].status == "done"
    rows = [win.jobs.table.item(r, 0).text() for r in range(win.jobs.table.rowCount())]
    assert st.id in rows
    dialog.close()


def test_an_ambiguous_sweep_shows_every_candidate_to_pick_by_eye(app, win):
    rival = Rival(index=0, center_px=63.0, prominence=1.0, deficit=0.01)
    r = SweepResult(
        center_from="params",
        tilt_deg=20.0,
        slice=3,
        depth=8,
        candidates_px=[63.0, 64.0, 65.0, 66.0],
        focus=[1.0, 0.5, 1.01, 0.9],
        verdict=Verdict(kind="ambiguous", index=2, center_px=65.0, rivals=[rival]),
    ).with_arrays(frames=np.random.default_rng(0).random((4, 8, 8), dtype="f4"))
    win.tabs["Align"]._show_sweep(r)
    assert names(win.tabs["Align"].view) == ["c 63.00", "c 64.00", "c 65.00", "c 66.00"]


def test_a_tilt_scan_shows_each_tilts_xz_plane_over_its_depth_under_its_slice(app, win):
    rng = np.random.default_rng(0)
    xz = np.full((2, 9, 8), np.nan, np.float32)
    xz[0, :7], xz[1] = rng.random((7, 8)), rng.random((9, 8))
    tilts = [{"tilt_deg": t, "focus": 1.0, "z_peak": 2, "band": None, "depth": d} for t, d in ((19, 7), (21, 9))]
    out = win.project.work_dir / "tilts.h5"
    save_h5(out, {}, best=rng.random((2, 8, 8), dtype="f4"), xz=xz)
    result = {"center_px": 64.5, "tilts": tilts, "artifacts": [str(out)]}
    st = JobState(
        id="j", project="", op="align.tilt_scan", params={}, caller="gui", gpu=None, status="done", created="",
        result=result,
    )
    at = win.tabs["Align"]
    at.on_job(st)
    model = at.view.model
    assert names(at.view) == ["tilt 19", "tilt 21", "xz 19", "xz 21"]
    # Slices in the top row, their xz planes under them.
    assert [model.grid.position(i, 4) for i in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [model.layers[f"xz {t}"].data.shape for t in (19, 21)] == [(7, 8), (9, 8)]
    assert [layer.metadata["tilt_deg"] for layer in model.layers] == [19, 21, 19, 21]
    at._show_mode("sweep")
    assert model.grid.shape == (-1, -1)


def test_the_volume_planes_follow_one_slider_once_and_3d_shows_the_binned_volume(app, win):
    win._select("volume", "v.h5")
    vt = win.tabs["Volume"]
    nz, ny, nx = vt.shape
    assert all(names(v) == ["v.h5", "fov", "cross"] for v in vt.views.values())
    assert pump(app, 20, lambda: vt._fov is not None)  # its arrival moves the planes too
    calls = []
    goto = vt._goto
    vt._goto = lambda: (calls.append(1), goto())
    vt.views["z"].model.dims.set_current_step(0, 5)
    pump(app, 0.2)
    # Each viewer's step moved by _goto does not call it again.
    assert len(calls) == 1 and vt.at == [5, ny // 2, nx // 2]
    assert vt.views["y"].model.dims.current_step[0] == ny // 2
    vt.pages.setCurrentIndex(1)
    pump(app, 0.2)
    assert names(vt.view3d) == []  # no binned volume yet
    ended = []
    win.jobs.ended.connect(ended.append)
    vt.bin()
    assert pump(app, 60, lambda: ended) and ended[0].status == "done"
    assert pump(app, 5, lambda: names(vt.view3d) == ["3D"])
    assert vt.view3d.model.layers["3D"].scale.tolist() == [2.0, 2.0, 2.0]
    vt.rendering.setCurrentText("iso")
    assert vt.view3d.model.layers["3D"].rendering == "iso"
    for n in (2, 3):
        vt.view3d.model.dims.ndisplay = n
        pump(app, 0.1)
        assert vt.view3d.model.dims.ndisplay == n


def test_the_export_slices_follow_the_band_and_keep_a_hand_edit(app, win):
    win._select("volume", "v.h5")
    ex = win.tabs["Export"]
    nz = win.tabs["Volume"].shape[0]
    assert (ex.z0.value(), ex.z1.value()) == (0, nz)
    (win.project.work_dir / "fig.png").write_bytes(b"")
    band = {"z": [2, 20], "bin": 2, "tilt_deg": 20.0, "depth": nz}
    decide = {"group": "m", "field": "z_band", "value": band, "evidence": ["fig.png"], "method": "focus"}
    dispatch.run("group.decide", decide, project=win.project, caller="cli")
    # The window hears project.toml change on its own.
    assert pump(app, 10, lambda: (ex.z0.value(), ex.z1.value()) == (2, 21))
    ex.z0.setValue(5)
    heard = []
    on_project = ex.on_project
    ex.on_project = lambda: (heard.append(1), on_project())
    win.project.commit(lambda f: f.recon.update(filter_name="shepp"))
    assert pump(app, 10, lambda: heard)
    assert (ex.z0.value(), ex.z1.value()) == (5, 21)


def test_the_volume_tab_shows_the_band_carried_to_the_volumes_tilt(app, win):
    (win.project.work_dir / "fig.png").write_bytes(b"")
    d20, d30 = _core.lamino_depth(32, 20.0), _core.lamino_depth(32, 30.0)
    band = {"z": [2, 20], "bin": 2, "tilt_deg": 20.0, "depth": d20}
    decide = {"group": "m", "field": "z_band", "value": band, "evidence": ["fig.png"], "method": "eye"}
    dispatch.run("group.decide", decide, project=win.project, caller="cli")
    with h5py.File(win.project.work_dir / "w.h5", "w") as f:
        f["exchange/data"] = np.random.default_rng(0).random((d30, 8, 8), dtype="f4")
        f.require_group("lamino_pal/provenance").attrs.update(tag="s", center_px=64.5, tilt_deg=30.0, bin=2)
    lo, hi = band_at((2, 20), d20, 20.0, 30.0, d30)
    assert (lo, hi) != (2, 20)
    win._select("volume", "w.h5")
    vt = win.tabs["Volume"]
    assert (vt.z0.value(), vt.z1.value()) == (lo, hi)


def test_the_focus_curve_shades_the_band_air_and_z_wrap(app, win):
    vt = win.tabs["Volume"]
    f = [0.3] * 30 + [0.01] * 40 + [1.0] * 80
    r = FocusResult(shape=(150, 8, 8), step=1, focus=f, peak_z=70, band=(70, 149), air=[(30, 69)], wrap=(0, 29))
    vt._show_focus(r)
    x = {s.getName(): s.getPoints()[:, 0] for s in vt.plot.getItems() if isinstance(s, Shape)}
    shapes = {k: (v.min(), v.max()) for k, v in x.items()}
    assert shapes == {"suggested band 0": (70, 149), "air 0": (30, 69), "z wrap 0": (0, 29)}
    assert vt.plot.getGraphTitle() == "suggested band [70, 149] (gray), air [30, 69] (blue), z wrap [0, 29] (orange)"


@pytest.mark.parametrize("ndisplay, grid", [(2, True), (3, False)])
def test_a_canvas_made_over_layers_is_the_one_that_heard_them_inserted(app, ndisplay, grid):
    data = np.random.default_rng(0).random((3, 4, 8, 8), dtype="f4")

    def made(before: bool):
        v = Viewer("t")
        v.model.dims.ndisplay = ndisplay
        v.model.grid.enabled = grid
        for i in range(3 if before else 0):
            v.model.add_image(data[i], name=f"c {i}")
        v.show()
        pump(app, 0.1)
        for i in range(0 if before else 3):
            v.model.add_image(data[i], name=f"c {i}")
        qv, layers = v.qt_viewer, v.model.layers
        visuals = [qv.canvas.layer_to_visual[layer] for layer in layers]
        controls = qv.controls
        shown = [layer.name for layer in layers if controls.widgets[layer] is controls.currentWidget()]
        state = (
            [x.order for x in visuals],
            len({id(x.node.parent) for x in visuals}),
            shown,
            qv.canvas.camera._3D_camera.depth_value,
        )
        v.close_canvas()
        v.close()
        return state

    assert made(True) == made(False)


def test_a_viewer_makes_its_canvas_when_first_shown(app, win):
    at, vt = win.tabs["Align"], win.tabs["Volume"]
    assert not any(v._qt_viewer for t in win.tabs.values() for v in t.viewers)
    win.show_tab("Align")
    pump(app, 0.2)
    assert at.view._qt_viewer and len(at.docks) == 2 and all(d.isVisible() for d in at.docks)
    win._select("volume", "v.h5")
    pump(app, 0.2)
    # The 3D viewer is on a page not yet shown.
    assert [v._qt_viewer is not None for v in vt.viewers] == [True, True, True, False]
    assert all(d.isVisible() for d in vt.docks) and not any(d.isVisible() for d in at.docks)
