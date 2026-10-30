"""Preview: one slice and the xz and yz planes through it as `recon.run`
would make them, recipes side by side with their CNR, the even/odd split."""

from __future__ import annotations

import json

import numpy as np
from silx.gui import qt

from lamino_pal import _core
from lamino_pal.artifacts import load_h5
from lamino_pal.gui.tabs.base import Tab, button, spin
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import Result

# tomoxide's names; its CUDA laminography takes these three algorithms only.
CHOICES = {
    "algorithm": ("fourierrec", "fbp", "linerec"),
    "filter_name": ("parzen", "shepp", "ramp", "cosine", "cosine2", "hamming", "hann", "none"),
    "remove_stripe_method": (
        "none", "vo-all", "fw", "ti", "sf", "vo-sort", "vo-filter", "vo-large", "vo-dead", "vo-fit",
    ),
    "retrieve_phase_method": ("none", "paganin", "Gpaganin", "farago"),
}
GAP = 8  # px between the planes
ROI_HELP = "ROI layer: draw the signal rectangle first, the background second"


def planes(view, sl: np.ndarray, xz: np.ndarray, yz: np.ndarray, w, prefix: str = "") -> None:
    """The slice top left, xz below it, yz turned to stand right of it, all on
    window `w`."""
    ny, nx = sl.shape
    view.replace(f"{prefix}slice", sl, contrast_limits=w, colormap="gray", translate=(0, 0))
    view.replace(f"{prefix}xz", xz, contrast_limits=w, colormap="gray", translate=(ny + GAP, 0))
    view.replace(f"{prefix}yz", yz.T, contrast_limits=w, colormap="gray", translate=(0, nx + GAP))


def _rect(data: np.ndarray) -> tuple[int, int, int, int]:
    """A Shapes rectangle as (y0, y1, x0, x1) of the slice."""
    y, x = data[:, 0], data[:, 1]
    return int(round(y.min())), int(round(y.max())), int(round(x.min())), int(round(x.max()))


class PreviewTab(Tab):
    NAME = "Preview"
    default_op = "recon.preview"

    def __init__(self, win):
        super().__init__(win)
        self.view = self.viewer("preview")
        self.info = qt.QLabel(wordWrap=True)
        self.cnr = qt.QTableWidget(0, 6)
        self.cnr.setHorizontalHeaderLabels(("variant", "signal μ", "background μ", "background σ", "CNR", "window"))
        self.cnr.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.cnr.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.cnr.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.cnr.itemSelectionChanged.connect(self._variant)
        self.cnr.hide()
        self.frc = self.graph("FRC")
        self.frc.hide()
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.cnr)
        layout.addWidget(self.frc)

        rec = self.section("Recipe (blank: the project's)")
        self.combos: dict[str, qt.QComboBox] = {}
        for k, items in CHOICES.items():
            c = qt.QComboBox()
            c.addItems(["", *items])
            c.currentIndexChanged.connect(lambda _: self.preview())
            rec.addRow(k, c)
            self.combos[k] = c
        self.extra = qt.QLineEdit(placeholderText='{"num_iter": 10}')
        self.extra.editingFinished.connect(self.preview)
        rec.addRow("more fields", self.extra)
        geo = self.section("Geometry (blank: the group's)")
        self.center = qt.QLineEdit()
        self.tilt = qt.QLineEdit()
        for e in (self.center, self.tilt):
            e.editingFinished.connect(self.preview)
        geo.addRow("center px", self.center)
        geo.addRow("tilt °", self.tilt)
        at = self.section("Planes")
        self.z = qt.QSlider(qt.Qt.Horizontal)
        self.z.setRange(0, 0)
        self.zl = qt.QLabel("z: z band middle")
        self.y = spin(-1, 100000, -1)
        self.x = spin(-1, 100000, -1)
        self.y.setSpecialValueText("middle")
        self.x.setSpecialValueText("middle")
        self._z_set = False
        self.z.valueChanged.connect(self._z_moved)
        self.y.valueChanged.connect(lambda _: self.preview())
        self.x.valueChanged.connect(lambda _: self.preview())
        at.addRow(self.zl, self.z)
        at.addRow("y", self.y)
        at.addRow("x", self.x)
        at.addRow(button("Preview", lambda: self.preview(again=True)))
        cmp_ = self.section("Compare recipes (job)")
        self.variants = qt.QPlainTextEdit()
        self.variants.setPlaceholderText("one JSON object per line laid over the recipe; blank: the defaults")
        self.variants.setMaximumHeight(100)
        cmp_.addRow(self.variants)
        cmp_.addRow(button("Add ROI layer", self._roi_layer, ROI_HELP))
        cmp_.addRow(button("Compare", self.compare))
        cmp_.addRow(button("Even/odd split", self.split, "the slice from each half of the projections (job)"))
        self.panel.addStretch(1)
        # The inputs, as `_inputs` keys them, when the view was last filled.
        self._filled: str | None = None

    # Params ----------------------------------------------------------------

    def setup(self) -> dict | None:
        """tag, center, tilt and recipe as set here; None, told, when unusable."""
        tag = self.need("tag")
        if tag is None:
            return None
        recipe = dict(self.project.data.recon)
        recipe.update({k: c.currentText() for k, c in self.combos.items() if c.currentText()})
        try:
            if self.extra.text().strip():
                recipe.update(json.loads(self.extra.text()))
            c, t = self.center.text().strip(), self.tilt.text().strip()
            return {
                "tag": tag,
                "center_px": float(c) if c else None,
                "tilt_deg": float(t) if t else None,
                "recipe": recipe,
            }
        except ValueError as e:
            self.say(f"Preview: {e}")
            return None

    def _where(self) -> dict:
        return {
            "z": self.z.value() if self._z_set else None,
            "y": None if self.y.value() < 0 else self.y.value(),
        }

    def _params(self) -> dict | None:
        p = self.setup()
        if p is not None:
            p.update(self._where(), x=None if self.x.value() < 0 else self.x.value())
        return p

    def _inputs(self, p: dict) -> str:
        """Params `p` and what the op binds a blank one to: the group's center,
        tilt and z band, and the scan's file."""
        scan = self.project.data.find_scan(p["tag"])
        g = next((g for g in self.project.data.group if g.name == scan.group), None)
        bound = None if g is None else (g.center_px, g.tilt_deg, g.z_band and g.z_band.model_dump())
        return json.dumps([p, scan.model_dump(), bound], sort_keys=True)

    def _z_moved(self, z: int) -> None:
        self._z_set = True
        self.zl.setText(f"z: {z}")
        self.preview()

    # Preview ---------------------------------------------------------------

    def preview(self, again: bool = False) -> None:
        """The view filled by a preview, unless `again` is unset and the inputs
        are those it was last filled on: a project change that leaves them is
        not one, and a compare or split shown stays."""
        if not self.isVisible():
            return
        p = self._params()
        if p is None:
            return
        key = self._inputs(p)
        if key == self._filled and not again:
            return
        self._filled = key
        self.info.setText("reconstructing …")
        self.run("recon.preview", p, self._show, channel="preview", failed=self._failed)

    def _failed(self, msg: str) -> None:
        self._filled = None
        self.info.setText(msg)

    def _show(self, r: Result) -> None:
        a = r.arrays
        self.cnr.hide()
        self.frc.hide()
        for name in [lay.name for lay in self.view.model.layers if lay.name.startswith(("v", "half"))]:
            self.view.model.layers.remove(name)
        planes(self.view, a["slice"], a["xz"], a["yz"], r.window)
        z, y, x = r.at
        self.z.blockSignals(True)
        self.z.setRange(0, r.depth - 1)
        self.z.setValue(z)
        self.z.blockSignals(False)
        self.zl.setText(f"z: {z}")
        c = json.loads(_core.recipe_config(json.dumps(r.recipe)))
        fields = ", ".join(f"{k}={c[k]}" for k in CHOICES)
        self.info.setText(
            f"center {r.center_px:.2f} px, tilt {r.tilt_deg:g}°, z {z} y {y} x {x} of depth {r.depth}; {fields}"
        )

    def on_context(self) -> None:
        self._z_set = False
        self.on_project()

    def on_project(self) -> None:
        g = self.win.find_group()
        self.center.setPlaceholderText("" if g is None or g.center_px is None else f"{g.center_px:.2f}")
        self.tilt.setPlaceholderText("20" if g is None or g.tilt_deg is None else f"{g.tilt_deg:g}")
        self.preview()

    def showEvent(self, event: qt.QShowEvent) -> None:
        super().showEvent(event)
        qt.QTimer.singleShot(0, self.preview)

    # Compare and split -----------------------------------------------------

    def _roi_layer(self) -> None:
        if "ROI" not in self.view.model.layers:
            self.view.add_outlines(
                name="ROI", shape_type="rectangle", edge_color="yellow", face_color="transparent", by_hand=True
            )
        roi = self.view.model.layers["ROI"]
        self.view.model.layers.selection.active = roi
        roi.mode = "add_rectangle"
        self.say(ROI_HELP)

    def compare(self) -> None:
        p = self.setup()
        if p is None:
            return
        p.update(self._where())
        try:
            lines = [ln for ln in self.variants.toPlainText().splitlines() if ln.strip()]
            if lines:
                p["variants"] = [json.loads(ln) for ln in lines]
        except ValueError as e:
            self.say(f"Preview: variants: {e}")
            return
        if "ROI" in self.view.model.layers:
            roi = self.view.model.layers["ROI"]
            rects = [d for d, t in zip(roi.data, roi.shape_type) if t == "rectangle"]
            if len(rects) >= 2:
                p["signal"], p["background"] = _rect(rects[0]), _rect(rects[1])
        self.submit("recon.compare", p)

    def split(self) -> None:
        p = self.setup()
        if p is not None:
            p.update(self._where())
            self.submit("recon.split", p)

    def on_job(self, st: JobState) -> None:
        if st.status != "done" or st.op not in ("recon.compare", "recon.split"):
            return
        _, a = load_h5(st.result["artifacts"][0])
        r = st.result
        self._job = st
        self._arrays = a
        # Shown on the inputs as now set: a preview replaces it as they change.
        p = self._params() if self.win.tag is not None else None
        self._filled = None if p is None else self._inputs(p)
        if st.op == "recon.compare":
            self.frc.hide()
            vs = r["variants"]
            self.cnr.setRowCount(len(vs))
            for row, v in enumerate(vs):
                stats = ("signal_mean", "background_mean", "background_std", "cnr")
                cells = [v["label"]] + ["" if v[k] is None else f"{v[k]:.4g}" for k in stats]
                cells.append(f"[{v['window'][0]:.3g}, {v['window'][1]:.3g}]")
                for col, text in enumerate(cells):
                    self.cnr.setItem(row, col, qt.QTableWidgetItem(text))
            self.cnr.resizeColumnsToContents()
            self.cnr.show()
            self.info.setText(f"job {st.id}: {len(vs)} variants at z {r['at'][0]}, xz at y {r['at'][1]}; pick a row")
            self.cnr.selectRow(0)
            self._variant()
        else:
            self.cnr.hide()
            self.frc.clear()
            f = a["frc"]
            n = len(f)
            self.frc.addCurve(np.arange(n) / (2 * n), f, legend="FRC", color="black")
            self.frc.addYMarker(1 / 7, legend="1/7", text="1/7", color="gray")
            self.frc.getXAxis().setLabel("cycles per pixel")
            self.frc.show()
            for i, half in enumerate(("even", "odd")):
                w = (float(np.percentile(a["slice"][i], 0.5)), float(np.percentile(a["slice"][i], 99.5)))
                self.view.replace(f"half {half}", a["slice"][i], contrast_limits=w, colormap="gray", visible=i == 0)
            res = "none below 1/7" if r["resolution_px"] is None else f"{r['resolution_px']:.1f} px"
            self.info.setText(
                f"job {st.id}: halves {r['nproj'][0]}/{r['nproj'][1]} at z {r['at'][0]}, "
                f"correlation {r['correlation']:.3f}, FRC resolution {res}; toggle the half layers"
            )

    def _variant(self) -> None:
        """The selected variant's slice and xz plane, on its own window."""
        rows = self.cnr.selectionModel().selectedRows()
        if not rows or getattr(self, "_job", None) is None or self._job.op != "recon.compare":
            return
        i = rows[0].row()
        v = self._job.result["variants"][i]
        sl, xz = self._arrays["slice"][i], self._arrays["xz"][i]
        w = tuple(v["window"])
        ny, _ = sl.shape
        self.view.replace("variant slice", sl, contrast_limits=w, colormap="gray", translate=(0, 0))
        self.view.replace("variant xz", xz, contrast_limits=w, colormap="gray", translate=(ny + GAP, 0))
