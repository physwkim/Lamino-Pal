"""Volume: a final volume sliced three ways, the disk every projection saw,
the focus of every slice to read the z band from, and the 3D view."""

from __future__ import annotations

import contextlib
import io
from datetime import datetime
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
from PIL import Image
from silx.gui import qt

from lamino_pal import LaminoPalError, Refused, units
from lamino_pal.gui.tabs.base import BAD, GOOD, WARN, Tab, badge, button, spin
from lamino_pal.jobs import JobState
from lamino_pal.mcp import figures
from lamino_pal.ops.registry import Result
from lamino_pal.ops.volume import data, provenance
from lamino_pal.render import window

RENDERING = ("mip", "attenuated_mip", "translucent", "iso", "average", "minip")
# The axes each viewer's slider and plane are: the volume's [z, y, x] transposed.
VIEWS = {"z": (0, 1, 2), "y": (1, 0, 2), "x": (2, 0, 1)}


def lazy(d: np.ndarray | h5py.Dataset):
    """A mapped volume as it is; a chunked h5 one read a slice at a time."""
    return d if isinstance(d, np.ndarray) else da.from_array(d, chunks=(1, *d.shape[1:]))


def binned(path: Path) -> Path | None:
    """The volume the 3D view loads for `path`: itself when binned already,
    else its `export.bin` sibling when made."""
    if "binned_from" in provenance(path):
        return path
    sib = path.with_name(f"{path.stem}_binned.h5")
    return sib if sib.exists() else None


def _fill(v, name: str, shapes: list[np.ndarray], kind: str) -> None:
    """Shapes layer `name` of viewer `v` holding `shapes`, each a `kind`."""
    v.model.layers[name].data = [(a, kind) for a in shapes]


class VolumeTab(Tab):
    NAME = "Volume"
    default_op = "volume.focus_by_z"

    def __init__(self, win):
        super().__init__(win)
        self.views = {
            "z": self.viewer("z slice"),
            "y": self.viewer("xz", docked=False),
            "x": self.viewer("yz", docked=False),
        }
        self.view3d = self.viewer("3D", docked=False)
        self.view3d.model.dims.ndisplay = 3
        self.info = qt.QLabel(wordWrap=True)
        self.fov = qt.QLabel(wordWrap=True)
        self.plot = self.graph("focus by z")
        self.plot.setMinimumHeight(170)
        self.plot.sigPlotSignal.connect(self._marker_moved)
        aux = qt.QSplitter(qt.Qt.Vertical)
        aux.addWidget(self.views["y"])
        aux.addWidget(self.views["x"])
        slices = qt.QSplitter(qt.Qt.Horizontal)
        slices.addWidget(self.views["z"])
        slices.addWidget(aux)
        slices.setSizes([700, 500])
        self.pages = qt.QTabWidget()
        self.pages.addTab(slices, "Slices")
        self.pages.addTab(self.view3d, "3D")
        self.pages.currentChanged.connect(self._page)
        split = qt.QSplitter(qt.Qt.Vertical)
        split.addWidget(self.pages)
        split.addWidget(self.plot)
        split.setSizes([750, 200])
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.fov)
        layout.addWidget(split, 1)
        self.path: Path | None = None
        self.shape: tuple[int, int, int] | None = None
        self.at = [0, 0, 0]  # (z, y, x) the three planes cross at
        self._focus: Result | None = None
        self._fov: Result | None = None
        self._syncing = False
        for axis, v in self.views.items():
            v.model.dims.events.current_step.connect(lambda e, axis=axis: self._stepped(axis))
            v.model.mouse_double_click_callbacks.append(lambda m, ev, axis=axis: self._clicked(axis, ev))

        band = self.section("z band")
        self.step = spin(1, 64, 4)
        band.addRow("every n-th slice", self.step)
        band.addRow(button("Focus by z", self.focus))
        self.z0, self.z1 = spin(0, 0), spin(0, 0)
        for s in (self.z0, self.z1):
            s.valueChanged.connect(self._band_edited)
        band.addRow("first slice", self.z0)
        band.addRow("last slice", self.z1)
        band.addRow(button("Decide z band", self.decide_band, "the slices above, inclusive, for the group"))
        view3d = self.section("3D")
        self.rendering = qt.QComboBox()
        self.rendering.addItems(RENDERING)
        self.rendering.currentTextChanged.connect(lambda t: self._set3d(rendering=t))
        self.depiction = qt.QComboBox()
        self.depiction.addItems(("volume", "plane"))
        self.depiction.currentTextChanged.connect(lambda t: self._set3d(depiction=t))
        view3d.addRow("rendering", self.rendering)
        view3d.addRow("depiction", self.depiction)
        view3d.addRow(button("Bin 2×2×2 for 3D", self.bin, "export.bin of this volume (job)"))
        snap = self.section("Snapshot")
        snap.addRow(button("Save the view", self.snapshot, "the shown canvas as a PNG under artifacts/figures"))
        self.panel.addStretch(1)

    # Volume ----------------------------------------------------------------

    def on_context(self) -> None:
        vol = self.win.volume
        path = None if vol is None else self.project.work_dir / vol
        if path == self.path:
            return
        self.path = path
        self.shape = None  # no plane follows the sliders until the layers exist
        self._focus = self._fov = None
        self.plot.clear()
        self.fov.clear()
        for v in (*self.views.values(), self.view3d):
            v.clear()
        if path is None:
            self.info.setText("Pick a volume in the project tree.")
            return
        try:
            d = data(path)
            prov = provenance(path)
        except (LaminoPalError, OSError) as e:
            self.info.setText(f"{vol}: {e}")
            self.path = None
            return
        nz, ny, nx = d.shape
        mid = np.asarray(d[nz // 2], dtype=np.float32)
        w = window(mid)
        src = lazy(d)
        # The overlays are refilled, never re-added: adding or removing a
        # layer moves the slider.
        for axis, v in self.views.items():
            v.model.add_image(src.transpose(VIEWS[axis]), name=vol, contrast_limits=w, colormap="gray")
            v.add_outlines(name="fov", edge_color="cyan", face_color="transparent")
            v.add_outlines(name="cross", edge_color="yellow")
        self.shape = d.shape
        self.at = [nz // 2, ny // 2, nx // 2]
        what = ", ".join(f"{k} {prov[k]}" for k in ("tag", "bin", "center_px", "tilt_deg") if k in prov)
        self.info.setText(f"<b>{vol}</b> {nz}×{ny}×{nx} {d.dtype}" + (f"; {what}" if what else "; no provenance"))
        for s in (self.z0, self.z1):
            s.setRange(0, nz - 1)
        g = self.win.find_group()
        zb = None if g is None else g.z_band
        if zb is not None and {"bin", "tilt_deg"} <= prov.keys():
            # A band read at another binning is not shown.
            with contextlib.suppress(Refused):
                z0, z1 = units.band_for(zb, prov["bin"], prov["tilt_deg"], nz)
                self.z0.setValue(z0)
                self.z1.setValue(z1)
        self._goto()
        self.run("volume.valid_fov", {"volume": vol}, self._show_fov, channel="volume-fov", failed=self._no_fov)
        if self.pages.currentIndex() == 1:
            self._load3d()

    def on_project(self) -> None:
        if self.win.volume is not None and not (self.project.work_dir / self.win.volume).exists():
            self.win.set_context(volume=None)

    # The three planes ------------------------------------------------------

    def _stepped(self, axis: str) -> None:
        if self._syncing or self.shape is None:
            return
        i = "zyx".index(axis)
        step = int(self.views[axis].model.dims.current_step[0])
        # napari repeats the event for one step, and sends it when a displayed axis moves.
        if step != self.at[i]:
            self.at[i] = step
            self._goto()

    def _clicked(self, axis: str, event) -> None:
        """A double click sets the plane's two other coordinates."""
        if self.shape is None:
            return
        r, c = event.position[-2:]
        order = VIEWS[axis]
        for k, val in zip(order[1:], (r, c)):
            self.at[k] = int(np.clip(round(val), 0, self.shape[k] - 1))
        self._goto()

    def _goto(self) -> None:
        """Every viewer at `at`, its crosshair through the other two planes."""
        self._syncing = True
        try:
            for axis, v in self.views.items():
                order = VIEWS[axis]
                v.model.dims.set_current_step(0, self.at[order[0]])
                r, c = self.at[order[1]], self.at[order[2]]
                h, w = self.shape[order[1]], self.shape[order[2]]
                _fill(v, "cross", [np.array([[r, 0], [r, w - 1]]), np.array([[0, c], [h - 1, c]])], "line")
            self._draw_fov()
        finally:
            self._syncing = False

    # Field of view ---------------------------------------------------------

    def _show_fov(self, r: Result) -> None:
        self._fov = r
        full = "none" if r.full_z is None else f"slices {r.full_z[0]}–{r.full_z[1]}"
        badge(
            self.fov,
            f"every projection saw a disk of up to {r.r_lateral:.0f} px; its whole disk on {full} (the cyan lines)",
            GOOD if r.full_z is not None else WARN,
        )
        if self.shape is not None:
            self._goto()

    def _no_fov(self, msg: str) -> None:
        badge(self.fov, f"field of view: {msg}", BAD)

    def _draw_fov(self) -> None:
        """The disk every projection saw on the z slice; the slices it is
        whole on across the xz and yz planes."""
        r = self._fov
        if r is None or self.shape is None:
            return
        nz, ny, nx = self.shape
        z, _, _ = self.at
        rad = r.arrays["radius"][z]
        # napari reads an ellipse of two rows as its center and radii.
        disk = [np.array([[ny / 2, nx / 2], [rad, rad]])] if rad > 0 else []
        _fill(self.views["z"], "fov", disk, "ellipse")
        rows = [] if r.full_z is None else list(r.full_z)
        for axis in ("y", "x"):
            n = self.shape[VIEWS[axis][2]]
            _fill(self.views[axis], "fov", [np.array([[q, 0], [q, n - 1]]) for q in rows], "line")

    # Focus and the z band --------------------------------------------------

    def focus(self) -> None:
        vol = self.need("volume")
        if vol:
            self.plot.setGraphTitle("focus by z …")
            params = {"volume": vol, "step": self.step.value()}
            fail = self.plot.setGraphTitle
            self.run("volume.focus_by_z", params, self._show_focus, channel="volume-focus", failed=fail)

    def _show_focus(self, r: Result) -> None:
        self._focus = r
        self.plot.clear()
        spans = figures.shaded(r.band, r.air, r.wrap)
        self.plot.setGraphTitle(figures.shades_key(spans))
        f = r.arrays["focus"]
        top = max(f, default=0.0)
        for k, v in spans.items():
            for i, (a, b) in enumerate(v):
                color = figures.SHADES[k][0]
                self.plot.addShape((a, b), (0, top), legend=f"{k} {i}", shape="rectangle", color=color, z=0)
        z = np.arange(len(f)) * r.step
        self.plot.addCurve(z, f, legend="focus", color="black")
        self.plot.addXMarker(r.peak_z, legend="peak", text="peak", color="gray")
        self.plot.getXAxis().setLabel("slice z")
        self.plot.getYAxis().setLabel("focus (mean |∇|²)")
        if r.band is not None:
            self.z0.setValue(r.band[0])
            self.z1.setValue(r.band[1])
        self._band_edited()

    def _band_edited(self) -> None:
        if self._focus is None:
            return
        for name, s in (("z0", self.z0), ("z1", self.z1)):
            self.plot.addXMarker(s.value(), legend=name, text=name, color="blue", draggable=True)

    def _marker_moved(self, event: dict) -> None:
        if event.get("event") == "markerMoved" and event.get("label") in ("z0", "z1"):
            s = self.z0 if event["label"] == "z0" else self.z1
            s.blockSignals(True)
            s.setValue(int(round(event["x"])))
            s.blockSignals(False)

    def decide_band(self) -> None:
        r = self._focus
        if r is None or self.path is None:
            self.say("Volume: run Focus by z first")
            return
        prov = provenance(self.path)
        if "bin" not in prov or "tilt_deg" not in prov:
            self.say("Volume: this volume has no provenance: its bin and tilt are unknown")
            return
        z0, z1 = sorted((self.z0.value(), self.z1.value()))
        band = {"z": [z0, z1], "bin": int(prov["bin"]), "tilt_deg": float(prov["tilt_deg"]), "depth": self.shape[0]}
        suggested = r.band is not None and tuple(r.band) == (z0, z1)
        self.decide("z_band", band, [self.figure("volume.focus_by_z", r)], "focus" if suggested else "eye")

    # 3D --------------------------------------------------------------------

    def _page(self, i: int) -> None:
        if i == 1:
            self._load3d()

    def _load3d(self) -> None:
        if self.path is None or "3D" in self.view3d.model.layers:
            return
        b = binned(self.path)
        if b is None:
            self.say("Volume: no binned volume yet: Bin 2×2×2 for 3D", 30)
            return
        d = np.asarray(data(b), dtype=np.float32)
        k = 1 if b == self.path else 2
        mid = d[d.shape[0] // 2]
        self.view3d.model.add_image(
            d,
            name="3D",
            scale=(k, k, k),
            contrast_limits=window(mid),
            colormap="gray",
            rendering=self.rendering.currentText(),
            depiction=self.depiction.currentText(),
        )
        self.view3d.model.reset_view()

    def _set3d(self, **kw) -> None:
        if "3D" in self.view3d.model.layers:
            for k, v in kw.items():
                setattr(self.view3d.model.layers["3D"], k, v)

    def bin(self) -> None:
        vol = self.need("volume")
        if vol:
            self.submit("export.bin", {"volume": vol})

    def on_job(self, st: JobState) -> None:
        if st.op == "export.bin" and st.status == "done" and self.pages.currentIndex() == 1:
            self._load3d()

    # Snapshot --------------------------------------------------------------

    def snapshot(self) -> None:
        what = "3d" if self.pages.currentIndex() == 1 else f"z{self.at[0]}"
        v = self.view3d if what == "3d" else self.views["z"]
        if self.path is None:
            return
        img = v.qt_viewer.screenshot(flash=False)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        name = f"{self.path.stem}-{datetime.now():%Y%m%d-%H%M%S}-{what}.png"
        rel = figures.save(self.project.work_dir, name, buf.getvalue())
        self.say(f"Volume: saved {rel}", 20)
