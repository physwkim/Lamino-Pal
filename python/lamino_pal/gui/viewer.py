"""An embedded napari viewer: its model made at once, its canvas the first
time it is needed."""

from __future__ import annotations

import functools
import gc
from types import SimpleNamespace
from typing import Any

from napari._vispy.canvas import VispyCanvas
from napari.components import ViewerModel
from napari.qt import QtViewer, get_stylesheet
from napari.resources._icons import _theme_path
from napari.utils.events import disconnect_events
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
    """napari 0.6.6's canvas, its scene graph rebuilt once for all the layers
    added, removed or reordered in one event-loop pass, or before a draw that
    comes first. napari rebuilds it, every layer's overlays, per layer: O(n²),
    minutes for a sweep's 49 candidates. A canvas made on a model already
    holding layers is so rebuilt with all of them mapped.

    A fit (fit_to_view: napari's on the first layer and on ndisplay, or
    reset_view) is to the size the canvas last had, napari's default 800×600
    before it was made, so it is made again as the canvas is resized until
    it is next drawn shown. A camera set by hand is kept."""

    _unfit = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Active while a rebuild is owed; deleted with the widget it is for.
        self._owed = qt.QTimer(self.native, singleShot=True, interval=0)
        self._owed.timeout.connect(self._rebuild)
        # Not owed: napari makes a layer's overlays in the grid the scene has,
        # which must be the model's.
        grid = self.viewer.grid.events
        for e in (grid.stride, grid.shape, grid.enabled, grid.spacing):
            e.disconnect(self._update_scenegraph)
            e.connect(self._rebuild)
        # A layer was removed since the last rebuild.
        self._closed = False
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

    def _update_scenegraph(self, event=None) -> None:
        self._owed.start()

    def _remove_layer(self, event) -> None:
        # napari's, but for its gc.collect and context.finish (0.3 s), which
        # free the layer's GL objects before the rebuild: left to it, once.
        layer = event.value
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self._overlay_callbacks[layer])
        disconnect_events(layer._overlays.events, self._overlay_callbacks[layer])
        del self._overlay_callbacks[layer]
        vispy_layer = self.layer_to_visual.pop(layer)
        disconnect_events(self.viewer.camera.events, vispy_layer)
        vispy_layer.close()
        self._remove_layer_overlays(layer)
        del self._layer_overlay_to_visual[layer]
        self._closed = True
        self._update_scenegraph()

    def _rebuild(self) -> None:
        self._owed.stop()
        if self._closed:
            self._closed = False
            gc.collect()
            self._scene_canvas.context.finish()
        super()._update_scenegraph()

    def on_draw(self, event) -> None:
        if self._owed.isActive():
            self._rebuild()
        super().on_draw(event)


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
