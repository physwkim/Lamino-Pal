"""An embedded napari viewer: its model made at once, its canvas the first
time it is needed."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

from napari._vispy.canvas import VispyCanvas
from napari.components import ViewerModel
from napari.qt import QtViewer, get_stylesheet
from napari.resources._icons import _theme_path
from napari.utils.theme import available_themes
from silx.gui import qt


@functools.cache
def _napari_style() -> str:
    """napari's light stylesheet in the window's font size. napari's buttons
    draw their icons only through it, as `theme_<id>:` urls on search paths
    napari's own QApplication would have registered."""
    for name in available_themes():
        qt.QDir.addSearchPath(f"theme_{name}", str(_theme_path(name)))
    return get_stylesheet("light", extra_variables={"font_size": f"{qt.QApplication.font().pointSize()}pt"})


class _Canvas(VispyCanvas):
    """napari 0.6.6's canvas, made on a model that may already hold layers:
    it reorders and regrids every layer of the model as each one is mapped,
    a KeyError until the last is, so it waits for that.

    A fit (fit_to_view: napari's on the first layer and on ndisplay, or
    reset_view) is to the size the canvas last had, napari's default 800×600
    before it was made, so it is made again as the canvas is resized until
    it is next drawn shown. A camera set by hand is kept."""

    _unfit = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unfit = True
        self.viewer.events.reset_view.connect(self._fitted)
        # Painted: napari calls on_draw itself as layers change.
        self._scene_canvas.events.draw.connect(self._drawn)

    def _fitted(self, event=None) -> None:
        self._unfit = True

    def _drawn(self, event) -> None:
        if self.native.isVisible():
            self._unfit = False

    def keep(self) -> None:
        """The camera as now set is no fit to redo."""
        self._unfit = False

    def on_resize(self, event) -> None:
        super().on_resize(event)
        if self._unfit and self.viewer.layers:
            self.viewer.fit_to_view()

    def _mapped(self) -> bool:
        return all(layer in self.layer_to_visual for layer in self.viewer.layers)

    def _reorder_layers(self) -> None:
        if self._mapped():
            super()._reorder_layers()

    def _update_scenegraph(self, event=None) -> None:
        if self._mapped():
            super()._update_scenegraph(event)


class Viewer(qt.QWidget):
    # Its QtViewer was made.
    made = qt.Signal(object)

    def __init__(self, title: str, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.model = ViewerModel(title=title)
        self._qt_viewer: QtViewer | None = None
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

    @property
    def qt_viewer(self) -> QtViewer:
        """The canvas, made on first need: a QtViewer takes 0.16 s, and most
        of a session's viewers are never shown."""
        if self._qt_viewer is None:
            qv = QtViewer(self.model, canvas_class=_Canvas)
            # napari 0.6.6 makes a layer's controls, and sets the 3D camera
            # depth, only as the layer is inserted: these layers were before.
            for layer in self.model.layers:
                qv.controls._add(SimpleNamespace(value=layer))
            qv.controls._display(SimpleNamespace(value=self.model.layers.selection.active))
            qv._update_camera_depth()
            # The docks too: the window takes them out from under qv.
            for w in (qv, qv.dockLayerControls, qv.dockLayerList):
                w.setStyleSheet(_napari_style())
            self._qt_viewer = qv
            self.layout().addWidget(qv)
            self.made.emit(qv)
        return self._qt_viewer

    def showEvent(self, event: qt.QShowEvent) -> None:
        # Made to be shown.
        self.qt_viewer
        super().showEvent(event)

    def close_canvas(self) -> None:
        if self._qt_viewer is not None:
            self._qt_viewer.close()

    def look(
        self,
        zoom: float | None = None,
        center: list[float] | None = None,
        angles: list[float] | None = None,
        fit: bool = False,
    ) -> dict[str, Any]:
        """The camera, fitted first when `fit`, then set to what is given and
        kept as the canvas is resized."""
        cam = self.model.camera
        if fit:
            self.model.reset_view()
        if zoom is not None or center is not None or angles is not None:
            # Made first: a canvas made later would fit over the camera.
            canvas = self.qt_viewer.canvas
            if zoom is not None:
                cam.zoom = zoom
            if center is not None:
                cam.center = tuple(center)
            if angles is not None:
                cam.angles = tuple(angles)
            canvas.keep()
        return {"zoom": cam.zoom, "center": list(cam.center), "angles": list(cam.angles)}

    def replace(self, name: str, data, **kwargs):
        """Image layer `name` holding `data`, made or refilled."""
        layers = self.model.layers
        if name in layers:
            layer = layers[name]
            layer.data = data
            for k, v in kwargs.items():
                setattr(layer, k, v)
            return layer
        return self.model.add_image(data, name=name, **kwargs)

    def clear(self) -> None:
        self.model.layers.clear()
