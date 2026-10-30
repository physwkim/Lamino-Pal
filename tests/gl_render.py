"""Run by test_render.py on xcb with GL. Each window has WA_DontShowOnScreen:
drawn, never mapped."""

import os

import h5py
import numpy as np
import PySide6.QtCore  # noqa: F401  # silx takes the binding napari's qtpy is pinned to
import pytest
from conftest import pump
from silx.gui import qt

from lamino_pal.gui.tabs.volume import VIEWS
from lamino_pal.gui.viewer import Viewer


@pytest.fixture(scope="module")
def app():
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"
    return qt.QApplication.instance() or qt.QApplication([])


def drawn(v: Viewer) -> np.ndarray:
    """What `v`'s canvas draws, gray."""
    return v.qt_viewer.screenshot(flash=False)[..., :3].mean(-1)


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_a_canvas_made_over_layers_draws_as_one_that_heard_them_inserted(app, ndisplay):
    # Three blocks of 1/3, 2/3 and 1 in the middle of a dark volume each.
    data = np.zeros((3, 8, 32, 32), "f4")
    for i in range(3):
        data[i, 2:6, 8:24, 8:24] = (i + 1) / 3

    def made(before: bool) -> np.ndarray:
        v = Viewer("t")
        # Not 600 wide: napari fits to a 600-wide canvas before it is made,
        # and three tiles in a row fit the same at any height that width.
        v.resize(900, 300)
        v.setAttribute(qt.Qt.WA_DontShowOnScreen)
        v.model.dims.ndisplay = ndisplay
        v.model.grid.enabled = ndisplay == 2
        v.model.grid.shape = (1, 3)
        for i in range(3 if before else 0):
            v.model.add_image(data[i], name=f"c {i}", contrast_limits=(0, 1))
        v.show()
        pump(app, 0.2)
        for i in range(0 if before else 3):
            v.model.add_image(data[i], name=f"c {i}", contrast_limits=(0, 1))
        if not before:
            # napari fits on the first layer only.
            v.model.reset_view()
        pump(app, 0.2)
        img = drawn(v)
        v.close_canvas()
        v.close()
        return img

    img = made(True)
    h, w = img.shape
    if ndisplay == 2:
        # One block a tile, in the layers' order.
        assert [round(img[h // 2, (2 * k + 1) * w // 6]) for k in range(3)] == [85, 170, 255]
    else:
        assert img[h // 2, w // 2] > 0 and img[h // 10, w // 10] == 0
    assert np.array_equal(img, made(False))


def test_a_camera_set_by_hand_is_kept_as_its_canvas_is_first_drawn(app):
    v = Viewer("t")
    v.model.add_image(np.ones((40, 60), "f4"))
    # Before the canvas has its size: a fit would be redone at it.
    v.look(zoom=3.0, center=[10.0, 20.0])
    v.resize(600, 300)
    v.setAttribute(qt.Qt.WA_DontShowOnScreen)
    v.show()
    pump(app, 0.3)
    try:
        assert v.model.camera.zoom == 3.0 and v.model.camera.center[1:] == (10.0, 20.0)
    finally:
        v.close_canvas()
        v.close()


@pytest.mark.parametrize("painted", [False, True])
def test_layers_added_to_a_grid_after_others_were_drawn_are_fit_to_it(app, painted):
    v = Viewer("t")
    v.resize(900, 300)
    v.setAttribute(qt.Qt.WA_DontShowOnScreen)
    v.show()
    # Align's rings, drawn, then its sweep: napari fits on the first layer,
    # to a grid of one.
    v.model.add_image(np.ones((30, 50), "f4"), name="mean")
    pump(app, 0.2)
    v.clear()
    v.model.grid.enabled = True
    for i in range(9):
        v.model.add_image(np.ones((32, 32), "f4"), name=f"c {i}")
    if painted:
        # A draw before the event loop rebuilds the scene.
        v.qt_viewer.canvas.native.repaint()
    pump(app, 0.2)
    cam = v.model.camera
    try:
        shown = cam.zoom, cam.center
        v.model.reset_view()
        assert (cam.zoom, cam.center) == shown
    finally:
        v.close_canvas()
        v.close()


@pytest.mark.parametrize("by_hand", [False, True])
def test_layers_added_after_a_draw_in_the_same_pass_are_fit_to_unless_zoomed_by_hand(app, by_hand):
    v = Viewer("t")
    v.resize(900, 600)
    v.setAttribute(qt.Qt.WA_DontShowOnScreen)
    v.show()
    pump(app, 0.2)
    canvas = v.qt_viewer.canvas
    cam = v.model.camera
    # Preview's planes: the first fit alone, drawn as the window lays out
    # the controls it brings, then the others.
    v.model.add_image(np.ones((64, 64), "f4"), name="slice")
    canvas.native.repaint()
    if by_hand:
        canvas._scene_canvas.events.mouse_wheel(pos=(450, 300), delta=(0, 1))
        # Into the model as drawn.
        canvas.native.repaint()
    shown = cam.zoom, cam.center
    v.model.add_image(np.ones((64, 64), "f4"), name="xz", translate=(72, 0))
    pump(app, 0.2)
    try:
        if by_hand:
            assert (cam.zoom, cam.center) == shown
        else:
            shown = cam.zoom, cam.center
            v.model.reset_view()
            assert (cam.zoom, cam.center) == shown
    finally:
        v.close_canvas()
        v.close()


def fill(img: np.ndarray) -> tuple[float, float]:
    """The height and width of what `img` shows on its black ground, as
    fractions of its own."""
    rows, cols = np.flatnonzero(img.any(1)), np.flatnonzero(img.any(0))
    h, w = img.shape
    return (rows[-1] - rows[0] + 1) / h, (cols[-1] - cols[0] + 1) / w


def test_every_volume_view_is_fit_to_the_canvas_it_is_drawn_on(app, project):
    from lamino_pal.gui.window import MainWindow

    vol = np.ones((40, 64, 64), "f4")
    vol[:, 40:48, 8:16] = 0  # the window's low end, inside the z plane
    for name, v in (("v.h5", vol), ("v_binned.h5", vol[::2, ::2, ::2])):
        with h5py.File(project.work_dir / name, "w") as f:
            f["exchange/data"] = v
            f.require_group("lamino_pal/provenance").attrs.update(tag="s", bin=2)
    w = MainWindow(project)
    w.resize(1600, 1000)
    w.setAttribute(qt.Qt.WA_DontShowOnScreen)
    w.show()
    pump(app, 0.3)
    # The planes are added to viewers whose canvases are not made yet.
    w._select("volume", "v.h5")
    w.show_tab("Volume")
    pump(app, 0.5)
    vt = w.tabs["Volume"]
    vt.pages.setCurrentIndex(1)
    pump(app, 0.5)
    try:
        for name, v in (*vt.views.items(), ("3D", vt.view3d)):
            # The lines over the planes are coloured, the planes gray.
            img = v.qt_viewer.screenshot(flash=False)[..., :3].min(-1) > 60
            # napari's fit: 95% of the canvas on the axis the data fills most.
            assert 0.93 < max(fill(img)) < 0.97, (name, fill(img), img.shape)
    finally:
        w.close()
        pump(app, 0.3)


def test_each_montage_label_is_drawn_inside_its_tile(app, project):
    from lamino_pal.artifacts import save_h5
    from lamino_pal.gui.window import MainWindow
    from lamino_pal.jobs import JobState

    rng = np.random.default_rng(0)
    tilts = [
        {"tilt_deg": t, "focus": 1.0, "z_peak": 2, "band": None, "depth": 48, "slice": 24} for t in (19, 21)
    ]
    out = project.work_dir / "tilts.h5"
    save_h5(out, {}, layer=rng.random((2, 96, 96), dtype="f4"), xz=rng.random((2, 48, 96), dtype="f4"))
    st = JobState(
        id="j", project="", op="align.tilt_scan", params={}, caller="gui", gpu=None, status="done", created="",
        result={"center_px": 64.5, "tilts": tilts, "artifacts": [str(out)]},
    )
    w = MainWindow(project)
    w.resize(1600, 1000)
    w.setAttribute(qt.Qt.WA_DontShowOnScreen)
    w.show()
    w.show_tab("Align")
    at = w.tabs["Align"]
    at.on_job(st)
    pump(app, 0.5)
    try:
        img = at.view.qt_viewer.screenshot(flash=False)[..., :3].astype(int)
        ys, xs = np.nonzero((img[..., 0] > 150) & (img[..., 1] > 150) & (img[..., 2] < 100))
        assert len(ys), "no label drawn"
        cam = at.view.model.camera
        h, w_ = img.shape[:2]
        rows = cam.center[-2] + (ys + 0.5 - h / 2) / cam.zoom
        cols = cam.center[-1] + (xs + 0.5 - w_ / 2) / cam.zoom
        # Pixel i spans i ± 0.5; antialiasing may bleed a screen px.
        e = 0.5 + 1.5 / cam.zoom
        inside = np.zeros(len(ys), bool)
        for y, x, th, tw in at._tiles:
            inside |= (rows > y - e) & (rows < y + th - 1 + e) & (cols > x - e) & (cols < x + tw - 1 + e)
        assert inside.all(), (rows[~inside].min(), rows[~inside].max(), at._tiles, cam.zoom)
    finally:
        w.close()
        pump(app, 0.3)


def test_the_volume_views_cross_is_drawn_whole_zoomed_out(app, project):
    from lamino_pal.gui.window import MainWindow

    with h5py.File(project.work_dir / "v.h5", "w") as f:
        f["exchange/data"] = np.random.default_rng(0).random((40, 128, 128), dtype="f4")
    w = MainWindow(project)
    w.resize(1600, 1000)
    w.setAttribute(qt.Qt.WA_DontShowOnScreen)
    w.show()
    w._select("volume", "v.h5")
    w.show_tab("Volume")
    pump(app, 0.5)
    vt = w.tabs["Volume"]
    try:
        for v in vt.views.values():
            v.look(zoom=0.26)
        missing = []
        for at in ([20, 64, 64], [5, 17, 101], [33, 90, 9], [12, 41, 77]):
            vt.at = at
            vt._goto()
            pump(app, 0.2)
            for name, v in vt.views.items():
                img = v.qt_viewer.screenshot(flash=False)[..., :3].astype(int)
                yellow = (img[..., 0] > 150) & (img[..., 1] > 150) & (img[..., 2] < 100)
                _, h, wd = (vt.shape[i] for i in VIEWS[name])
                # Its line across the rows and its line down the columns.
                if yellow.sum(1).max() < 0.8 * wd * 0.26 or yellow.sum(0).max() < 0.8 * h * 0.26:
                    missing.append((at, name))
        assert missing == []
    finally:
        w.close()
        pump(app, 0.3)


def test_the_drift_pair_is_drawn_alone_over_the_projections(app, project):
    from types import SimpleNamespace

    from lamino_pal.gui.window import MainWindow

    w = MainWindow(project)
    w.resize(1600, 1000)
    w.setAttribute(qt.Qt.WA_DontShowOnScreen)
    w.show()
    w.show_tab("Raw")
    rt = w.tabs["Raw"]
    rt.view.replace("projections", np.ones((3, 64, 64), "f4"), contrast_limits=(0, 1), colormap="gray")
    first = np.zeros((64, 64), "f4")
    first[:, :32] = 1
    rt._drift(SimpleNamespace(arrays={"first": first, "turn": np.zeros_like(first)}, first=0, turn=1, dy_px=0, dx_px=0))
    pump(app, 0.5)
    try:
        img = rt.view.qt_viewer.screenshot(flash=False)[..., :3].astype(int)
        cam = rt.view.model.camera
        h, w_ = img.shape[:2]
        row = round(h / 2 + (32 - cam.center[-2]) * cam.zoom)
        at = [img[row, round(w_ / 2 + (c - cam.center[-1]) * cam.zoom)] for c in (16, 48)]
        # Green where only the first is bright, black where neither is.
        assert [list(p) for p in at] == [[0, 255, 0], [0, 0, 0]]
    finally:
        w.close()
        pump(app, 0.3)
