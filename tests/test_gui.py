"""The GUI offscreen, without GL: the layers, the steps and the widgets
around them. What the canvases draw is test_render.py's."""

import gc
import io
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Else offscreen draws through DISPLAY's GLX when it has one.
os.environ["QT_QPA_OFFSCREEN_NO_GLX"] = "1"
os.environ["QT_API"] = "pyside6"

import re
import threading
import time
import warnings

import h5py
import matplotlib
import numpy as np
import PySide6.QtCore  # noqa: F401  # silx takes the binding napari's qtpy is pinned to
import pytest
from conftest import band_at, pump
from napari._vispy.canvas import VispyCanvas
from napari._vispy.utils.visual import create_vispy_overlay
from napari.layers import Image
from PIL import Image as PILImage
from silx.gui import qt
from silx.gui.plot.items import Shape
from test_recon import _scan
from vispy.app.canvas import MouseEvent

from lamino_pal import LaminoPalError, _core, dispatch, jobs, remote
from lamino_pal.artifacts import save_h5
from lamino_pal.gui import status
from lamino_pal.gui.actions import ACTIONS, RunOpDialog
from lamino_pal.gui.form import ParamsForm
from lamino_pal.gui.runner import Runner, by_hand
from lamino_pal.gui.table import fill
from lamino_pal.gui.tree import _key
from lamino_pal.gui.viewer import Viewer
from lamino_pal.jobs import JobState
from lamino_pal.ops import REGISTRY
from lamino_pal.ops.align import Rival, SweepResult, Verdict
from lamino_pal.ops.volume import FocusResult
from lamino_pal.render import to_u8, window


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


def test_an_agent_selects_a_tree_item_as_a_click_does(app, win):
    st = _remote(app, win, "select", kind="volume", value="v.h5")
    assert st["tab"] == "Volume" and st["context"] == {"group": "m", "tag": "s", "volume": "v.h5"}
    assert [win.tree.current()] == [_key(i) for i in win.treeProject.selectedItems()] == [("volume", "v.h5")]
    win.set_context(volume=None)
    # Selected already, it is selected again.
    assert _remote(app, win, "select", kind="volume", value="v.h5")["context"]["volume"] == "v.h5"
    st = _remote(app, win, "select", kind="group", value="m")
    assert st["context"] == {"group": "m", "tag": "s", "volume": "v.h5"}
    assert [_key(i) for i in win.treeProject.selectedItems()] == [("group", "m")]
    with pytest.raises(LaminoPalError, match="^the project tree has no volume w.h5; volumes: v.h5, s.h5$"):
        _remote(app, win, "select", kind="volume", value="w.h5")


def test_an_agent_is_refused_a_viewer_or_graph_that_shows_nothing(app, win):
    with pytest.raises(LaminoPalError, match="^Align's viewer align shows nothing yet$"):
        _remote(app, win, "view", tab="Align", viewer="align", zoom=2.0)
    with pytest.raises(LaminoPalError, match="^Align's graph focus shows nothing yet$"):
        _remote(app, win, "graph", tab="Align", graph="focus", reset=True)


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


def test_a_job_resubmitted_on_a_done_jobs_params_shows_that_job_until_its_input_changes(app, win):
    old = time.time() - 60
    for f in win.project.work_dir.rglob("*"):
        os.utime(f, (old, old))
    win.set_context(tag="s", volume="v.h5")
    ended = []
    win.jobs.ended.connect(ended.append)
    win.tabs["Volume"].bin()
    assert pump(app, 60, lambda: ended) and ended[0].status == "done"
    first = ended[0].id
    win.tabs["Volume"].bin()
    assert pump(app, 5, lambda: len(ended) == 2) and ended[1].id == first
    assert win.statusBar().currentMessage() == f"export.bin: job {first} ran these params; its result is shown"
    # The dialog holds the id before the job's end is polled.
    dialog = RunOpDialog(win, "export.bin")
    dialog._run()
    assert pump(app, 5, lambda: len(ended) == 3) and ended[2].id == dialog._job == first
    assert f'"id": "{first}"' in dialog.text.toPlainText()
    assert [st.id for st in jobs.list_(win.project)] == [first]
    os.utime(win.project.work_dir / "v.h5")
    failed = []
    win.runner.failed.connect(lambda op, m: failed.append(m))
    assert win.runner.submit("export.bin", {"volume": "v.h5"}) is None
    assert "exists; pass overwrite to replace it" in failed[0]
    dialog.close()


def test_a_batch_row_whose_final_exists_reads_so_and_a_rerun_shows_the_job_that_made_it(app, win):
    old = time.time() - 60
    for f in win.project.work_dir.rglob("*"):
        os.utime(f, (old, old))
    win.set_context(tag="s")
    tab = win.tabs["Batch"]
    tab.refresh()
    assert tab.table.item(0, 6).text() == "to make"
    ended = []
    win.jobs.ended.connect(ended.append)
    tab.table.selectRow(0)
    tab.run_selected()
    assert pump(app, 120, lambda: ended) and ended[0].status == "done"
    first = ended[0].id
    tab.refresh()
    assert [tab.table.item(0, c).text() for c in (1, 6)] == ["64.50", "exists"]
    assert status.finals(win.project, win.find_group()) == (1, 1)
    tab.table.selectRow(0)
    tab.run_selected()
    assert pump(app, 5, lambda: len(ended) == 2) and ended[1].id == first
    assert win.statusBar().currentMessage() == f"recon.run: job {first} ran these params; its result is shown"


def selected(table: qt.QTableWidget) -> list[str]:
    return [table.item(i.row(), 0).text() for i in table.selectionModel().selectedRows()]


def test_a_table_filled_again_keeps_its_selected_and_current_rows_by_their_first_cell(app):
    t = qt.QTableWidget(0, 2)
    t.setSelectionBehavior(qt.QAbstractItemView.SelectRows)

    def rows(*keys):
        return [[qt.QTableWidgetItem(k), qt.QTableWidgetItem("")] for k in keys]

    fill(t, rows("j2", "j1"))
    t.selectRow(1)
    # A job submitted, newest first.
    fill(t, rows("j3", "j2", "j1"))
    assert selected(t) == ["j1"] and t.currentRow() == 2
    fill(t, rows("j3", "j2"))
    assert selected(t) == [] and t.currentRow() == -1


def test_the_data_table_keeps_its_selected_folder_as_it_is_filled_again_sorted(app, win):
    tab = win.tabs["Data"]
    keys = ("raw", "tag", "kind", "n", "n_flat", "range_deg", "speed_dps", "exptime_s", "energy_ev")
    tab._found = [{**dict.fromkeys(keys), "raw": raw, "tag": raw, "imported": []} for raw in ("c", "a", "b")]
    tab._fill()
    tab.table.sortByColumn(0, qt.Qt.AscendingOrder)
    tab.table.selectRow(1)
    assert [r[0] for r in tab._rows()] == ["b"]
    tab._fill()
    assert [r[0] for r in tab._rows()] == ["b"]


def _ambiguous_sweep(app, win):
    """Align showing a sweep of four 8×8 candidates, 65 best and 63 its rival."""
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
    win.show_tab("Align")
    at = win.tabs["Align"]
    at._show_sweep(r)
    pump(app, 0.1)
    return at, r.arrays["frames"]


def _click(view, y, x, drag=0):
    """A left click at world (y, x) on `view`'s canvas, released `drag` px right."""
    canvas = view.qt_viewer.canvas
    at = (canvas.view.transform * canvas.view.scene.transform).map([x, y])[:2]
    press = MouseEvent("mouse_press", pos=at, button=1)
    canvas._on_mouse_press(press)
    for dx in range(1, drag + 1):
        canvas._on_mouse_move(MouseEvent("mouse_move", pos=at + [dx, 0], button=1, press_event=press))
    canvas._on_mouse_release(MouseEvent("mouse_release", pos=at + [drag, 0], button=1, press_event=press))


def framed(view):
    """The labels whose tiles are framed."""
    labels = view.model.layers["labels"]
    return [t for t, c in zip(labels.features["label"], labels.edge_color) if c[3] > 0]


def test_an_ambiguous_sweep_shows_every_candidate_to_pick_by_eye(app, win):
    at, frames = _ambiguous_sweep(app, win)
    # One layer, a candidate on the slider at a time, the best first, all on its window.
    model = at.view.model
    assert names(at.view) == ["sweep"] and model.layers["sweep"].data.shape == (4, 8, 8)
    assert tuple(model.layers["sweep"].contrast_limits) == window(frames[2])
    assert model.dims.current_step[0] == 2 and model.text_overlay.text == "center 65.00 px (best)"
    assert at._picked("center_px") == 65.0


def test_the_slider_or_a_click_on_the_plot_picks_a_sweep_candidate(app, win):
    at, frames = _ambiguous_sweep(app, win)
    model = at.view.model
    model.dims.set_current_step(0, 0)
    assert at._picked("center_px") == 63.0 and model.text_overlay.text == "center 63.00 px (rival)"
    assert at.plot._getMarker("picked").getXPosition() == 63.0
    at._clicked({"event": "mouseClicked", "x": 65.9})
    assert at._picked("center_px") == 66.0 and model.dims.current_step[0] == 3
    # The decision cites the candidate as shown.
    cited = PILImage.open(win.project.work_dir / at._candidate(66.0))
    assert np.array_equal(np.asarray(cited), to_u8(frames[3], window(frames[2])))


def test_a_tilt_scan_shows_each_tilts_xz_plane_over_its_depth_under_its_slice(app, win):
    rng = np.random.default_rng(0)
    xz = np.full((2, 9, 8), np.nan, np.float32)
    xz[0, :7], xz[1] = rng.random((7, 8)), rng.random((9, 8))
    tilts = [
        {"tilt_deg": t, "focus": 1.0, "z_peak": 2, "band": None, "depth": d, "slice": d // 2}
        for t, d in ((19, 7), (21, 9))
    ]
    out = win.project.work_dir / "tilts.h5"
    save_h5(out, {}, layer=rng.random((2, 8, 8), dtype="f4"), xz=xz)
    result = {"center_px": 64.5, "tilts": tilts, "artifacts": [str(out)]}
    st = JobState(
        id="j", project="", op="align.tilt_scan", params={}, caller="gui", gpu=None, status="done", created="",
        result=result,
    )
    at = win.tabs["Align"]
    win.show_tab("Align")
    at.on_job(st)
    pump(app, 0.1)
    assert names(at.view) == ["tilts", "labels"]
    # Slices in the top row, their xz planes under them.
    assert at._tiles == [(0, 0, 8, 8), (0, 12, 8, 8), (12, 0, 7, 8), (12, 12, 9, 8)]
    assert at._values == [19, 21, 19, 21]
    assert list(at.view.model.layers["labels"].features["label"]) == ["19°", "21°", "xz 19°", "xz 21°"]
    # None picked until a tile is clicked, not dragged from, nor clicked between.
    _click(at.view, 3, 3, drag=20)
    _click(at.view, 10, 10)
    assert at._picked("tilt_deg") is None and framed(at.view) == []
    # Either of a tilt's tiles picks it.
    _click(at.view, 20, 15)
    assert at._picked("tilt_deg") == 21 and framed(at.view) == ["21°", "xz 21°"]
    assert at.plot._getMarker("picked").getXPosition() == 21


def test_a_graph_draws_its_hangul_title_in_a_font_that_has_it(app, scene, monkeypatch):
    from lamino_pal.gui.tabs.align import TILT_FOCUS
    from lamino_pal.gui.window import MainWindow

    if not qt.QFontDatabase.families(qt.QFontDatabase.WritingSystem.Korean):
        pytest.skip("no font here has Hangul")
    monkeypatch.setitem(matplotlib.rcParams, "font.family", ["sans-serif"])
    w = MainWindow(scene)
    plot = w.tabs["Align"].plot
    plot.setGraphTitle(TILT_FOCUS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot.saveGraph(io.BytesIO(), fileFormat="png")
    w.close()
    assert [str(c.message) for c in caught if "missing from font" in str(c.message)] == []


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


def test_the_fov_disk_is_centered_on_the_z_slice(app, win):
    win._select("volume", "v.h5")
    vt = win.tabs["Volume"]
    assert pump(app, 20, lambda: vt._fov is not None)
    _, ny, nx = vt.shape
    rad = vt._fov.arrays["radius"][vt.at[0]]
    corners = vt.views["z"].model.layers["fov"].data[0]
    assert rad > 0
    center = np.array([ny / 2, nx / 2])
    np.testing.assert_allclose([corners.min(0), corners.max(0)], [center - rad, center + rad])


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
    r = FocusResult(shape=(150, 8, 8), step=1, peak_z=70, band=(70, 149), air=[(30, 69)], wrap=(0, 29))
    r.with_arrays(focus=np.array(f))
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
        # The scene is rebuilt as the event loop runs.
        pump(app, 0.1)
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


def test_a_canvas_rebuilds_its_scene_once_for_the_layers_changed_in_one_pass(app, monkeypatch):
    # napari rebuilds it, every layer's overlays, per layer, and collects
    # garbage, 0.3 s, per layer removed: 49 took minutes.
    built = []
    rebuild, collect = VispyCanvas._update_scenegraph, gc.collect

    def counted(self, event=None):
        built.append(len(self.viewer.layers))
        rebuild(self, event)

    def collected(*args):
        built.append("gc")
        return collect(*args)

    monkeypatch.setattr(VispyCanvas, "_update_scenegraph", counted)
    monkeypatch.setattr(gc, "collect", collected)
    v = Viewer("t")
    v.show()
    pump(app, 0.1)
    built.clear()
    v.model.grid.enabled = True
    for i in range(5):
        v.model.add_image(np.zeros((4, 4), "f4"), name=f"c {i}")
    pump(app, 0.1)
    v.clear()
    pump(app, 0.1)
    v.close_canvas()
    v.close()
    # The grid's at once, the layers' once each pass.
    assert built == [0, 5, "gc", 0]


def test_an_overlays_visual_is_made_once_a_pass_and_only_while_it_is_shown(app, monkeypatch):
    # napari makes every layer's four, hidden, as it maps the layer and again
    # as it rebuilds the scene: half of a sweep's 27 s.
    made = []

    def counted(overlay, **kwargs):
        made.append(type(overlay).__name__)
        return create_vispy_overlay(overlay, **kwargs)

    monkeypatch.setattr("napari._vispy.canvas.create_vispy_overlay", counted)
    v = Viewer("t")
    v.show()
    v.model.grid.enabled = True
    boxed = Image(np.zeros((4, 4), "f4"), name="boxed")
    boxed._overlays["bounding_box"].visible = True
    v.model.layers.extend([Image(np.zeros((4, 4), "f4"), name="c 0"), boxed])
    pump(app, 0.1)
    assert made == ["BoundingBoxOverlay"]
    made.clear()
    scale_bar, colorbar = v.model._overlays["scale_bar"], v.model.layers["c 0"]._overlays["colorbar"]
    scale_bar.visible = colorbar.visible = True
    pump(app, 0.1)
    canvas = v.qt_viewer.canvas
    try:
        assert sorted(made) == ["ColorBarOverlay", "ScaleBarOverlay"]
        assert canvas._overlay_to_visual[scale_bar][0].node.visible
        assert canvas._layer_overlay_to_visual[v.model.layers["c 0"]][colorbar].node.visible
    finally:
        v.close_canvas()
        v.close()


def test_a_layer_added_as_its_grid_is_turned_on_is_drawn_in_it(app):
    v = Viewer("t")
    v.show()
    # Align's rings, in one view, then its sweep in a grid.
    v.model.add_image(np.zeros((4, 4), "f4"), name="mean")
    pump(app, 0.1)
    v.clear()
    v.model.grid.enabled = True
    v.model.add_image(np.zeros((4, 4), "f4"), name="c 0")
    pump(app, 0.1)
    canvas = v.qt_viewer.canvas
    try:
        assert canvas.layer_to_visual[v.model.layers["c 0"]].node.parent is canvas.grid_views[0].scene
    finally:
        v.close_canvas()
        v.close()


def test_a_viewers_napari_buttons_and_its_docks_find_their_icons(app):
    v = Viewer("t")
    qv = v.qt_viewer
    for w in (qv, qv.dockLayerControls, qv.dockLayerList):
        icons = re.findall(r'url\("(theme_light:/[^"]+)"\)', w.styleSheet())
        assert icons and all(qt.QFile.exists(i) for i in icons)
    v.close_canvas()
    v.close()


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
