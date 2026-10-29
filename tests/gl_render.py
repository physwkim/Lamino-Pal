"""Run by test_render.py on xcb with GL. Each window has WA_DontShowOnScreen:
drawn, never mapped."""

import os

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
