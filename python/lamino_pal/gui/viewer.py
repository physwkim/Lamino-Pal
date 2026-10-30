"""An embedded napari viewer: its model made at once, its canvas the first
time it is needed."""

from __future__ import annotations

import functools
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
from napari import viewer as napari_viewer
from napari._app_model import get_app_model
from napari._vispy.canvas import VispyCanvas
from napari.components import LayerList, ViewerModel
from napari.layers import Image, Layer, Shapes
from napari.qt import QtViewer, get_stylesheet
from napari.resources._icons import _theme_path
from napari.utils.events import disconnect_events
from napari.utils.theme import available_themes
from silx.gui import icons, qt

# silx's zoom mode: a press is a drag once 5 px from where it was, and a drag
# zooms to its box if that covers 5 px².
_DRAG = 5
_AREA = 5


@functools.cache
def _napari_style() -> str:
    """napari's light stylesheet in the window's font size. napari's buttons
    draw their icons only through it, as `theme_<id>:` urls on search paths
    napari's own QApplication would have registered."""
    for name in available_themes():
        qt.QDir.addSearchPath(f"theme_{name}", str(_theme_path(name)))
    return get_stylesheet("light", extra_variables={"font_size": f"{qt.QApplication.font().pointSize()}pt"})


def _shown() -> ViewerModel | None:
    """The viewer whose layer list shows, as its buttons and its layer
    controls' do: those of the others are hidden."""
    shown = [v.model for v in _viewers if v._qt_viewer is not None and v._qt_viewer.dockLayerList.isVisible()]
    return shown[0] if len(shown) == 1 else None


@functools.cache
def _act_on_shown() -> None:
    """napari's buttons run its actions on its Window's viewer, or that
    viewer's active layer, which it gives them only from a Window: these
    viewers have none, and act on the one shown."""
    get_app_model().injection_store.register(
        providers={
            napari_viewer.Viewer: _shown,
            ViewerModel: _shown,
            Layer: lambda: None if (m := _shown()) is None else m.layers.selection.active,
            LayerList: lambda: None if (m := _shown()) is None else m.layers,
        }
    )


# Every Viewer made, for _shown.
_viewers: weakref.WeakSet[Viewer] = weakref.WeakSet()


class _Canvas(VispyCanvas):
    """napari 0.6.6's canvas, its scene graph rebuilt once for all the layers
    added, removed or reordered in one event-loop pass, or before a draw that
    comes first. napari rebuilds it, every layer's overlays, per layer: O(n²),
    minutes for a sweep's 49 candidates. A canvas made on a model already
    holding layers is so rebuilt with all of them mapped.

    A fit (fit_to_view: napari's on the first layer and on ndisplay, or
    reset_view) is to the layers, their grid and the size the canvas then
    had, napari's default 800×600 before it was made, so it is made again as
    the canvas is resized or its scene rebuilt until the camera is set by
    hand: moved other than by a fit, or by keep. Not until it is drawn: Qt
    may draw it between the layers added in one pass."""

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
        # A fit moves the camera, then emits reset_view. A press that leaves
        # the camera, a click on a plane or a shape drawn, is no setting.
        cam = self.viewer.camera.events
        for e in (cam.zoom, cam.center, cam.angles):
            e.connect(self.keep)
        self.viewer.events.reset_view.connect(self._fitted)

    def _fitted(self, event=None) -> None:
        self._unfit = True

    def keep(self, event=None) -> None:
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


class _Outlines(Shapes):
    """Shapes only shown: napari draws the active layer's shape under the
    cursor in its highlight colour, over the colour it is to be read by."""

    _highlight_visible = property(lambda self: False, lambda self, visible: None)

    @property
    def _type_string(self) -> str:
        # Its icon in the layer list.
        return "shapes"


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
        # napari's: drag_to_zoom, Alt+drag, now _drag's.
        self.model.mouse_drag_callbacks[:] = [self._drag]
        self.model.mouse_wheel_callbacks.append(self._wheel)
        # The camera before each box zoom, for Zoom Back: on the images then
        # shown, forgotten as one is added, removed or refilled.
        self._back: list[tuple[float, tuple]] = []
        self.model.layers.events.inserted.connect(self._new_image)
        self.model.layers.events.removed.connect(self._new_image)
        self._menu = qt.QMenu(self)
        self._menu.addAction(icons.getQIcon("zoom-back"), "Zoom Back", self.zoom_back)
        # silx's modes, put by napari's buttons under the layer list: a left
        # drag zooms to a box, in 3D turns the view, or pans.
        modes = qt.QButtonGroup(self)
        self.zoom_mode, self.pan_mode = (qt.QPushButton(checkable=True, iconSize=qt.QSize(20, 20)) for _ in range(2))
        for b in (self.zoom_mode, self.pan_mode):
            # napari's size; its stylesheet draws the rest.
            b.setFixedSize(28, 28)
            modes.addButton(b)
        self.pan_mode.setIcon(icons.getQIcon("pan"))
        self.pan_mode.setToolTip("Pan mode")
        self.zoom_mode.setChecked(True)
        self.model.dims.events.ndisplay.connect(self._name_modes)
        self._name_modes()
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        _viewers.add(self)

    @property
    def qt_viewer(self) -> QtViewer:
        """The canvas, made on first need: a QtViewer takes 0.16 s, and most
        of a session's viewers are never shown."""
        if self._qt_viewer is None:
            qv = QtViewer(self.model, canvas_class=_Canvas)
            _act_on_shown()
            # Its console is a Window's.
            qv.viewerButtons.consoleButton.hide()
            # napari 0.6.6 makes a layer's controls, and sets the 3D camera
            # depth, only as the layer is inserted: these layers were before.
            for layer in self.model.layers:
                qv.controls._add(SimpleNamespace(value=layer))
            qv.controls._display(SimpleNamespace(value=self.model.layers.selection.active))
            qv._update_camera_depth()
            for i, b in enumerate((self.zoom_mode, self.pan_mode)):
                qv.viewerButtons.layout().insertWidget(i, b)
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

    def zoom_back(self) -> None:
        """silx's: the camera before the last box zoom, else the fit."""
        if self._back:
            zoom, center = self._back.pop()
            self.look(zoom=zoom, center=list(center))
        else:
            self.model.reset_view()

    def _new_image(self, event) -> None:
        layer = event.value
        if isinstance(layer, Image):
            self._forget(event)
            if event.type == "inserted":
                layer.events.data.connect(self._forget)
            else:
                layer.events.data.disconnect(self._forget)

    def _forget(self, event) -> None:
        self._back.clear()

    def _name_modes(self, event=None) -> None:
        if self.model.dims.ndisplay == 2:
            self.zoom_mode.setIcon(icons.getQIcon("zoom"))
            self.zoom_mode.setToolTip("Zoom mode")
        else:
            self.zoom_mode.setIcon(icons.getQIcon("rotate-3d"))
            self.zoom_mode.setToolTip("Rotate mode")

    def _drag(self, model: ViewerModel, event):
        """silx's mouse wherever napari's would move the camera: in 2D a
        left drag zooms to a box and a right click offers Zoom Back; a
        middle drag pans, and in 3D, where a left drag turns the view, a
        Ctrl+left one too; in Pan mode a left drag pans. A press taken here
        never reaches the vispy camera, which keeps only the wheel in 2D."""
        if not model.camera.mouse_pan:
            return
        flat, button = model.dims.ndisplay == 2, event.button
        pans = self.pan_mode.isChecked() or (not flat and "Control" in event.modifiers)
        if button == 3 or (button == 1 and pans):
            gen = self._pan(model, event)
        elif button == 1 and (flat or "Alt" in event.modifiers):
            gen = self._box(model, event)
        elif button == 2 and flat:
            gen = self._click(event)
        else:
            return
        event.handled = True
        yield from gen

    def _wheel(self, model: ViewerModel, event) -> None:
        """In 3D, vispy's zoom, 1.1× a step, toward the point under the
        cursor, as its 2D camera's is: its 3D camera's is to its center. With
        a modifier the wheel is napari's, and vispy's camera never hears it."""
        if model.dims.ndisplay != 3 or event.modifiers or not model.camera.mouse_zoom:
            return
        cam = model.camera
        f = 1.1 ** event.delta[1]
        center = np.asarray(cam.center, float)
        at = np.asarray(event.position)[list(event.dims_displayed)]
        cam.zoom *= f
        cam.center = tuple(center + (1 - 1 / f) * (at - center))
        event.handled = True

    def _pan(self, model: ViewerModel, event):
        """The point grabbed held under the cursor."""
        dims = list(event.dims_displayed)
        grabbed = np.asarray(event.position)[dims]
        while True:
            yield
            center = np.array(model.camera.center, float)
            center[-len(dims) :] += grabbed - np.asarray(event.position)[dims]
            model.camera.center = tuple(center)

    def _box(self, model: ViewerModel, event):
        """napari's zoom box drawn from the press; on release, silx's zoom to it."""
        box = model._zoom_box
        a, at = np.asarray(event.pos), event.position
        dragged = False
        yield
        while True:
            b = np.asarray(event.pos)
            dragged = dragged or ((b - a) ** 2).sum() >= _DRAG**2
            if event.type == "mouse_release":
                break
            if dragged:
                # Canvas rows, columns.
                box.position = (tuple(a[::-1]), tuple(b[::-1]))
                box.visible = True
            yield
        box.visible = False
        if dragged and np.prod(np.abs(b - a)) >= _AREA:
            cam = model.camera
            self._back.append((cam.zoom, cam.center))
            # The box fit whole, at its middle. Not by napari's _on_boxzoom,
            # which divides the canvas's width by the box's rows.
            width_height = np.asarray(self.qt_viewer.canvas._current_viewbox_size)
            middle = np.mean([at, event.position], 0)[list(event.dims_displayed)]
            self.look(zoom=cam.zoom * np.min(width_height / np.abs(b - a)), center=list(middle))

    def _click(self, event):
        """Zoom Back offered by a click, not a drag."""
        a = np.asarray(event.pos)
        yield
        while event.type == "mouse_move":
            yield
        if ((np.asarray(event.pos) - a) ** 2).sum() < _DRAG**2:
            self._menu.popup(qt.QCursor.pos())

    def replace(self, name: str, data, **kwargs):
        """Image layer `name` holding `data`, made or refilled; made, under
        the outlines, which it would hide."""
        layers = self.model.layers
        if name in layers:
            layer = layers[name]
            layer.data = data
            for k, v in kwargs.items():
                setattr(layer, k, v)
            return layer
        layer = self.model.add_image(data, name=name, **kwargs)
        over = [i for i, lay in enumerate(layers) if lay in self._px]
        if over:
            layers.move(len(layers) - 1, over[0])
        return layer

    def add_overlay(self, layer: Layer) -> Layer:
        """`layer` drawn over the image, which stays the active layer: napari
        activates each layer it inserts, and the layer controls then show the
        overlay's tools, not the image's contrast. A layer to draw on is
        activated by its caller."""
        layers = self.model.layers
        # napari 0.6.6's flag for its insert.
        layers._activate_on_insert = False
        try:
            return self.model.add_layer(layer)
        finally:
            layers._activate_on_insert = True

    def add_outlines(self, data=None, *, px: float = 1.0, by_hand: bool = False, **kwargs) -> Shapes:
        """A shapes layer whose edges are `px` screen px wide at any zoom.
        napari's are as wide in data px, and GL draws one thinner than a
        screen px only where it covers a pixel's center: zoomed out, a line
        across the pixel rows is missing. Highlighted as napari's only when
        drawn `by_hand`."""
        layer = self.add_overlay((Shapes if by_hand else _Outlines)(data, **kwargs))
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
