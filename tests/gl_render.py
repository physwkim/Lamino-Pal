"""Run by test_render.py on xcb with GL. Each window has WA_DontShowOnScreen:
drawn, never mapped."""

import os

import h5py
import numpy as np
import PySide6.QtCore  # noqa: F401  # silx takes the binding napari's qtpy is pinned to
import pytest
from conftest import pump
from silx.gui import qt

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
        v.resize(600, 300)
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
