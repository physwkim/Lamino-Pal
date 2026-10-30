"""An embedded napari viewer: its model made at once, its canvas the first
time it is needed."""

from __future__ import annotations

import functools
import gc
from types import SimpleNamespace
from typing import Any

from napari._vispy.canvas import VispyCanvas
from napari.components import ViewerModel
from napari.layers import Shapes
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
    reset_view) is to the layers, their grid and the size the canvas then
    had, napari's default 800×600 before it was made, so it is made again as
    the canvas is resized or its scene rebuilt until it is next drawn shown.
    A camera set by hand is kept."""

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

    def _fitted(self, event=None) -> None:
        self._unfit = True

    def keep(self) -> None:
        """The camera as now set is no fit to redo."""
        self._unfit = False

    def on_resize(self, event) -> None:
        super().on_resize(event)
        if self._unfit and self.viewer.layers:
            self.viewer.fit_to_view()

    def _update_scenegraph(self, event=None) -> None:
        self._owed.start()

    def add_layer_visual_mapping(self, napari_layer, vispy_layer) -> None:
        # Owed before napari makes the layer's overlays, then left to the rebuild.
        self._owed.start()
        super().add_layer_visual_mapping(napari_layer, vispy_layer)

    def _update_layer_overlays(self, layer) -> None:
        # An owed rebuild remakes every layer's overlays.
        if not self._owed.isActive():
            super()._update_layer_overlays(layer)

    # An overlay's visual is made only as it is shown. napari makes one for
    # each, all hidden by default: a layer's four, 0.12 s, and the axes in
    # every grid view.

    def _add_layer_overlay(self, layer, overlay, parent) -> None:
        if overlay.visible:
            super()._add_layer_overlay(layer, overlay, parent)
        else:
            overlay.events.visible.connect(self._shown)

    def _add_viewer_overlay(self, overlay, parent) -> None:
        if overlay.visible:
            super()._add_viewer_overlay(overlay, parent)
        else:
            overlay.events.visible.connect(self._shown)

    def _shown(self, event) -> None:
        overlay = event.source
        overlay.events.visible.disconnect(self._shown)
        if any(o is overlay for o in self.viewer._overlays.values()):
            self._update_viewer_overlays()
        for layer in self.layer_to_visual:
            if any(o is overlay for o in layer._overlays.values()):
                self._update_layer_overlays(layer)

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
        if self._unfit and self.viewer.layers:
            self.viewer.fit_to_view()

    def on_draw(self, event) -> None:
        if self._owed.isActive():
            self._rebuild()
        super().on_draw(event)
        # Painted, after the rebuild's fit: napari calls on_draw(None) itself
        # as layers change.
        if event is not None and self.native.isVisible():
            self._unfit = False


class Viewer(qt.QWidget):
    # Its QtViewer was made.
    made = qt.Signal(object)
    # It was shown or hidden, not by the window's minimizing.
    shown = qt.Signal(bool)

    def __init__(self, title: str, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.model = ViewerModel(title=title)
        self._qt_viewer: QtViewer | None = None
        # Each layer of add_outlines, to its edges' width in screen px.
        self._px: dict[Shapes, float] = {}
        self.model.camera.events.zoom.connect(self._widen)
        self.model.layers.events.removed.connect(lambda e: self._px.pop(e.value, None))
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
        if not event.spontaneous():
            self.shown.emit(True)

    def hideEvent(self, event: qt.QHideEvent) -> None:
        super().hideEvent(event)
        if not event.spontaneous():
            self.shown.emit(False)

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

    def add_outlines(self, data=None, *, px: float = 1.0, **kwargs) -> Shapes:
        """A shapes layer whose edges are `px` screen px wide at any zoom.
        napari's are as wide in data px, and GL draws one thinner than a
        screen px only where it covers a pixel's center: zoomed out, a line
        across the pixel rows is missing."""
        layer = self.model.add_shapes(data, **kwargs)
        self._px[layer] = px
        # napari gives a shape set as data a width of 1.
        layer.events.data.connect(self._widen)
        self._widen()
        return layer

    def _widen(self, event=None) -> None:
        for layer, px in self._px.items():
            w = px / self.model.camera.zoom
            # For shapes drawn by hand too.
            layer.current_edge_width = w
            layer.edge_width = w

    def clear(self) -> None:
        self.model.layers.clear()
