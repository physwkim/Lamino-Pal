"""An embedded napari viewer: its model made at once, its canvas the first
time it is needed."""

from __future__ import annotations

from types import SimpleNamespace

from napari._vispy.canvas import VispyCanvas
from napari.components import ViewerModel
from napari.qt import QtViewer
from silx.gui import qt


class _Canvas(VispyCanvas):
    """napari 0.6.6's canvas, made on a model that may already hold layers:
    it reorders and regrids every layer of the model as each one is mapped,
    a KeyError until the last is, so it waits for that."""

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
