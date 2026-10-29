"""Align, in the guide's order: the rings estimate, the center sweep, the
tilt by eye. Each decision cites the figure it was read from."""

from __future__ import annotations

import numpy as np
from silx.gui import qt
from silx.gui.plot import Plot1D

from lamino_pal import scans, units
from lamino_pal.artifacts import load_h5
from lamino_pal.gui.actions import figure_name
from lamino_pal.gui.tabs.base import BAD, GOOD, WARN, Tab, badge, button, dspin, floats, spin
from lamino_pal.jobs import JobState
from lamino_pal.mcp import figures
from lamino_pal.ops.registry import Result
from lamino_pal.render import png, to_u8, window

TRUSTWORTHY = 8.0  # ring prominence, `align.rings`
OPEN_ARC = "open arc — ring 무효 (README: 462–467, 40 px 오차)"
TILT_FOCUS = "참고용: tilt에 따라 단조 증가하므로 판단 근거 아님"


def _opt(edit: qt.QLineEdit) -> float | None:
    t = edit.text().strip()
    return float(t) if t else None


class AlignTab(Tab):
    NAME = "Align"
    default_op = "align.center_sweep"

    def __init__(self, win):
        super().__init__(win)
        self.view = self.viewer("align")
        self.plot = Plot1D()
        self.plot.setMinimumHeight(180)
        self.plot.sigPlotSignal.connect(self._clicked)
        self.verdict = qt.QLabel(wordWrap=True)
        split = qt.QSplitter(qt.Qt.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.plot)
        split.setSizes([700, 220])
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.verdict)
        layout.addWidget(split, 1)
        self.mode: str | None = None  # what the viewer shows
        self._rings: Result | None = None
        self._sweep: Result | None = None
        self._job: JobState | None = None  # the tilt scan or center compare shown
        self._tag: str | None = None

        rings = self.section("1. Rings")
        self.ring_step = spin(1, 1000, 10)
        self.ring_bands = spin(1, 16, 4)
        rings.addRow("every n-th", self.ring_step)
        rings.addRow("row bands", self.ring_bands)
        self.rings_button = button("Rings", self.rings)
        rings.addRow(self.rings_button)
        self.rings_out = qt.QLabel(wordWrap=True)
        rings.addRow(self.rings_out)
        rings.addRow(button("Decide center (rings)", self.decide_rings))

        sweep = self.section("2. Center sweep")
        self.center = qt.QLineEdit(placeholderText="group's, else rings")
        self.half = dspin(0, 2000, 24, 1)
        self.cstep = dspin(0.05, 100, 1, 2, 0.5)
        self.tilt = qt.QLineEdit(placeholderText="group's, else 20")
        self.slice = qt.QLineEdit(placeholderText="z band middle")
        sweep.addRow("center px", self.center)
        sweep.addRow("± px", self.half)
        sweep.addRow("step px", self.cstep)
        sweep.addRow("tilt °", self.tilt)
        sweep.addRow("slice", self.slice)
        sweep.addRow(button("Sweep", self.sweep))
        self.widen = button("범위 넓히기", self._widen, "the peak is at an edge: sweep again around it, twice as wide")
        self.widen.hide()
        sweep.addRow(self.widen)
        sweep.addRow(button("Compare by eye", self.compare, "the best and its rivals reconstructed whole (job)"))
        sweep.addRow(button("Decide selected center", self.decide_center))

        tilt = self.section("3. Tilt")
        self.tilts = qt.QLineEdit("16 18 20 22 24")
        tilt.addRow("tilts °", self.tilts)
        tilt.addRow(button("Tilt scan", self.tilt_scan, "each tilt's best slice, by the group's center (job)"))
        tilt.addRow(button("Decide selected tilt", self.decide_tilt))
        self.panel.addStretch(1)

    # Context ---------------------------------------------------------------

    def on_context(self) -> None:
        if self.win.tag == self._tag:
            return
        self._tag = self.win.tag
        self._show_mode(None)
        self.verdict.clear()
        self.rings_out.clear()
        self.rings_button.setEnabled(True)
        if self._tag is not None:
            self.run("raw.range_check", {"tag": self._tag}, self._range, channel="align-range")

    def _range(self, r: Result) -> None:
        if r.kind == "full":
            return
        self.rings_button.setEnabled(False)
        g = self.win.find_group()
        hint = (
            f"use group {g.name}'s center {g.center_px:.2f} px"
            if g is not None and g.center_px is not None
            else "use the center of a full-turn scan of this mount"
        )
        badge(self.rings_out, f"{r.kind} turn: {OPEN_ARC}; {hint}", BAD)

    def _show_mode(self, mode: str | None) -> None:
        self.mode = mode
        self.view.clear()
        self.view.model.grid.enabled = mode in ("sweep", "tilts", "centers")
        self.plot.clear()
        self.plot.setGraphTitle("")
        self.widen.hide()

    def _layer(self, name: str, a: np.ndarray, **meta) -> None:
        self.view.model.add_image(a, name=name, contrast_limits=window(a), colormap="gray", metadata=meta)

    def _picked(self, key: str) -> float | None:
        """The `key` of the selected layer, None with the user told to pick one."""
        layer = self.view.model.layers.selection.active
        if layer is None or key not in layer.metadata:
            self.say(f"{self.NAME}: select one {key} layer in the layer list")
            return None
        return layer.metadata[key]

    # Rings -----------------------------------------------------------------

    def rings(self) -> None:
        tag = self.need("tag")
        if tag:
            params = {"tag": tag, "step": self.ring_step.value(), "bands": self.ring_bands.value()}
            self.run("align.rings", params, lambda r: self._show_rings(tag, r), failed=self.rings_out.setText)

    def _show_rings(self, tag: str, r: Result) -> None:
        self._rings = r
        self._show_mode("rings")
        mean = r.arrays["mean"]
        scan, _ = scans.find(self.project, tag)
        col = units.center_for_scan(scan, r.center_px)
        h = mean.shape[0]
        self.view.model.add_image(mean, name="mean", contrast_limits=window(mean), colormap="gray")
        lines = [np.array([[0, col], [h - 1, col]])]
        for b in r.bands:
            c = units.center_for_scan(scan, b.center_px)
            lines.append(np.array([[b.rows[0], c], [b.rows[1] - 1, c]]))
        self.view.model.add_shapes(
            lines,
            shape_type="line",
            name="centers",
            edge_color=["yellow"] + ["cyan"] * len(r.bands),
            edge_width=2,
        )
        color = GOOD if r.prominence >= TRUSTWORTHY else WARN
        bands = ", ".join(f"{b.center_px:.1f}" for b in r.bands)
        badge(
            self.rings_out,
            f"center {r.center_px:.2f} px, prominence {r.prominence:.1f}; bands {bands} "
            f"(spread {r.band_spread_px:.1f} px)",
            color,
        )

    def decide_rings(self) -> None:
        r = self._rings
        if r is None:
            self.say("Align: run Rings first")
            return
        self.decide("center_px", r.center_px, [self.figure("align.rings", r)], "rings")

    # Center sweep ----------------------------------------------------------

    def sweep(self, center: float | None = None, half: float | None = None) -> None:
        tag = self.need("tag")
        if not tag:
            return
        try:
            params = {
                "tag": tag,
                "center_px": center if center is not None else _opt(self.center),
                "half_width_px": half if half is not None else self.half.value(),
                "step_px": self.cstep.value(),
                "tilt_deg": _opt(self.tilt),
                "slice": None if not self.slice.text().strip() else int(self.slice.text()),
            }
        except ValueError as e:
            self.say(f"Align: {e}")
            return
        self.verdict.setText(f"sweeping {tag} …")
        self.run("align.center_sweep", params, self._show_sweep, channel="align-sweep", failed=self.verdict.setText)

    def _show_sweep(self, r: Result) -> None:
        self._sweep = r
        self._show_mode("sweep")
        v = r.verdict
        frames = r.arrays["frames"]
        keep = [v.index] + [x.index for x in v.rivals] if v.kind == "ambiguous" else range(len(frames))
        for i in keep:
            c = r.candidates_px[i]
            self._layer(f"c {c:.2f}", frames[i], center_px=c)
        self.view.model.layers.selection.active = self.view.model.layers[f"c {r.candidates_px[v.index]:.2f}"]
        self.plot.addCurve(r.candidates_px, r.focus, legend="focus", symbol="o", color="black")
        self.plot.addXMarker(v.center_px, legend="best", text="best", color="green")
        for x in v.rivals:
            self.plot.addXMarker(x.center_px, legend=f"rival {x.index}", text="rival", color="orange")
        self.plot.getXAxis().setLabel("center px (full resolution)")
        self.plot.getYAxis().setLabel("focus")
        head = f"{v.kind}: center {v.center_px:.2f} px on slice {r.slice} at tilt {r.tilt_deg:g}°"
        head += f" (center from {r.center_from})"
        if v.kind == "resolved":
            badge(self.verdict, head, GOOD)
        elif v.kind == "railed":
            badge(self.verdict, f"{head}; the peak is at the sweep's edge", WARN)
            self.widen.show()
        elif v.kind == "ambiguous":
            rivals = ", ".join(f"{x.center_px:.2f} (−{x.deficit:.1%})" for x in v.rivals)
            badge(self.verdict, f"{head}; rivals {rivals}: compare them by eye", WARN)
        else:
            badge(self.verdict, f"{head}; no sample on this slice: check the z band", BAD)

    def _clicked(self, event: dict) -> None:
        """A click on the sweep curve selects the nearest candidate's layer."""
        if event.get("event") != "mouseClicked" or self.mode != "sweep":
            return
        names = [lay.name for lay in self.view.model.layers if "center_px" in lay.metadata]
        if names:
            layers = self.view.model.layers
            best = min(names, key=lambda n: abs(layers[n].metadata["center_px"] - event["x"]))
            layers.selection.active = layers[best]

    def _widen(self) -> None:
        r = self._sweep
        if r is not None:
            self.sweep(center=r.verdict.center_px, half=r.candidates_px[-1] - r.candidates_px[0])

    def compare(self) -> None:
        r = self._sweep
        tag = self.need("tag")
        if r is None or not tag:
            self.say("Align: sweep first")
            return
        v = r.verdict
        centers = [v.center_px] + [x.center_px for x in v.rivals]
        layer = self.view.model.layers.selection.active
        if layer is not None and layer.metadata.get("center_px") not in (None, *centers):
            centers.append(layer.metadata["center_px"])
        if len(centers) < 2:
            self.say("Align: no rivals; select a candidate layer to compare with the best")
            return
        self.submit("align.center_compare", {"tag": tag, "centers_px": centers, "tilt_deg": r.tilt_deg})

    def decide_center(self) -> None:
        if self.mode == "sweep" and self._sweep is not None:
            c = self._picked("center_px")
            if c is None:
                return
            r = self._sweep
            method = "sweep" if r.verdict.kind == "resolved" and c == r.verdict.center_px else "eye"
            evidence = [self.figure("align.center_sweep", r), self._candidate(c)]
            self.decide("center_px", c, evidence, method)
        elif self.mode == "centers" and self._job is not None:
            c = self._picked("center_px")
            if c is not None:
                self.decide("center_px", c, [self.job_figure(self._job)], "eye")
        else:
            self.say("Align: sweep or compare first, then select a center layer")

    def _candidate(self, c: float) -> str:
        """The selected candidate's slice alone, saved to cite."""
        layer = self.view.model.layers.selection.active
        img = png(to_u8(layer.data, tuple(layer.contrast_limits)))
        name = figure_name("align.center_sweep").replace(".png", f"-c{c:.2f}.png")
        return str(figures.save(self.project.work_dir, name, img))

    # Tilt ------------------------------------------------------------------

    def tilt_scan(self) -> None:
        tag = self.need("tag")
        if not tag:
            return
        try:
            tilts = floats(self.tilts.text())
        except ValueError as e:
            self.say(f"Align: tilts: {e}")
            return
        self.submit("align.tilt_scan", {"tag": tag, "tilts_deg": tilts})

    def on_job(self, st: JobState) -> None:
        if st.status != "done" or st.op not in ("align.tilt_scan", "align.center_compare"):
            return
        self._job = st
        _, arrays = load_h5(st.result["artifacts"][0])
        best = arrays["best"]
        if st.op == "align.tilt_scan":
            self._show_mode("tilts")
            rows = st.result["tilts"]
            for t, a in zip(rows, best):
                self._layer(f"tilt {t['tilt_deg']:g}", a, tilt_deg=t["tilt_deg"])
            x = [t["tilt_deg"] for t in rows]
            self.plot.addCurve(x, [t["focus"] for t in rows], legend="focus", symbol="o", color="gray")
            self.plot.setGraphTitle(TILT_FOCUS)
            self.plot.getXAxis().setLabel("tilt °")
            self.verdict.setText(f"job {st.id}: each tilt's best slice at center {st.result['center_px']:.2f} px")
        else:
            self._show_mode("centers")
            rows = st.result["centers"]
            for c, a in zip(rows, best):
                self._layer(f"c {c['center_px']:.2f}", a, center_px=c["center_px"])
            x = [c["center_px"] for c in rows]
            self.plot.addCurve(x, [c["focus"] for c in rows], legend="focus", symbol="o", color="black")
            self.plot.getXAxis().setLabel("center px (full resolution)")
            self.verdict.setText(f"job {st.id}: each center's best slice at tilt {st.result['tilt_deg']:g}°")

    def decide_tilt(self) -> None:
        if self.mode != "tilts" or self._job is None:
            self.say("Align: run a tilt scan, then select a tilt layer")
            return
        t = self._picked("tilt_deg")
        if t is not None:
            self.decide("tilt_deg", t, [self.job_figure(self._job)], "eye")
