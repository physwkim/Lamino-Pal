"""An embedded napari viewer, its model and canvas made together."""

from __future__ import annotations

from napari.components import ViewerModel
from napari.qt import QtViewer
from silx.gui import qt


class Viewer(qt.QWidget):
    def __init__(self, title: str, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.model = ViewerModel(title=title)
        # napari 0.6.6's QtViewer cannot start on a model holding two layers
        # (_reorder_layers looks up visuals not yet made): never defer it.
        self.qt_viewer = QtViewer(self.model)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.qt_viewer)

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
