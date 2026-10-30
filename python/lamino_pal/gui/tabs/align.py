"""Align, in the guide's order: the rings estimate, the center sweep, the
tilt by eye. Each decision cites the figure it was read from."""

from __future__ import annotations

import math

import numpy as np
from silx.gui import qt

from lamino_pal import scans, units
from lamino_pal.artifacts import load_h5
from lamino_pal.gui.actions import figure_name
from lamino_pal.gui.tabs.base import BAD, GOOD, WARN, Tab, badge, button, dspin, floats, spin
from lamino_pal.jobs import JobState
from lamino_pal.mcp import figures
from lamino_pal.ops.registry import Result
from lamino_pal.render import GAP, grid, origins, png, to_u8, window

TRUSTWORTHY = 8.0  # ring prominence, `align.rings`
OPEN_ARC = "open arc — ring 무효 (README: 462–467, 40 px 오차)"
TILT_FOCUS = "참고용: tilt에 따른 경향이 run마다 달라 판단 근거 아님"
# The graph's markers are named in its title: silx writes a marker's text at
# the plot's top, where the texts of close markers overlap.
PICKED = "picked (blue)"


def _opt(edit: qt.QLineEdit) -> float | None:
    t = edit.text().strip()
    return float(t) if t else None


class AlignTab(Tab):
    NAME = "Align"
    default_op = "align.center_sweep"

    def __init__(self, win):
        super().__init__(win)
        self.view = self.viewer("align")
        self.plot = self.graph("focus")
        self.plot.setMinimumHeight(180)
        self.plot.sigPlotSignal.connect(self._clicked)
        self.view.model.mouse_drag_callbacks.append(self._press)
        self.view.model.dims.events.current_step.connect(self._stepped)
        text = self.view.model.text_overlay
        text.position, text.color, text.font_size = "top_left", "yellow", 14
        self.verdict = qt.QLabel(wordWrap=True)
        split = qt.QSplitter(qt.Qt.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.plot)
        split.setSizes([700, 220])
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.verdict)
        layout.addWidget(split, 1)
        self.mode: str | None = None  # what the viewer shows
        # The value of `_key` each sweep candidate or montage tile shows, its
        # label, the one picked, and the tiles as (y, x, height, width).
        self._values: list[float] = []
        self._labels: list[str] = []
        self._key: str | None = None
        self._pick: float | None = None
        self._tiles: list[tuple[int, int, int, int]] = []
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
        tilt.addRow(button("Tilt scan", self.tilt_scan, "each tilt's band-middle slice, by the group's center (job)"))
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
        self._values, self._labels, self._key, self._pick, self._tiles = [], [], None, None, []
        self.view.clear()
        self.view.model.text_overlay.visible = False
        self.plot.clear()
        self.plot.setGraphTitle("")
        self.widen.hide()

    def _montage(self, rows: list[list[tuple[np.ndarray, float, str]]], key: str) -> None:
        """Each (image, value of `key`, label) in `rows`, on its own window, a
        labelled tile of one montage layer: napari takes 0.1 s a layer. A tile
        is picked by a click on it or the plot."""
        tiles = [[to_u8(a, window(a)) for a, _, _ in r] for r in rows]
        # A frame drawn in the gap stays visible zoomed out to the whole montage.
        gap = max(GAP, max(max(t.shape) for r in tiles for t in r) // 32)
        at, _ = origins(tiles, gap)
        self._tiles = [(y, x, *t.shape) for r, ats in zip(tiles, at) for t, (y, x) in zip(r, ats)]
        self._values = [v for r in rows for _, v, _ in r]
        self._labels = [label for r in rows for _, _, label in r]
        self._key = key
        model = self.view.model
        img = model.add_image(grid(tiles, gap), name=self.mode, contrast_limits=(0, 255), colormap="gray")
        # Pixel i spans i ± 0.5: the frame's line runs mid-gap.
        e = (gap + 1) / 2
        model.add_shapes(
            [np.array([[y - e, x - e], [y + h - 1 + e, x + w - 1 + e]]) for y, x, h, w in self._tiles],
            shape_type="rectangle",
            name="frames",
            edge_width=gap,
            edge_color="transparent",
            face_color="transparent",
        )
        # A label in screen px hangs from its tile's corner into the tile
        # ("lower_left" of a point): the gaps are too narrow for it zoomed out.
        model.add_points(
            [(y - 0.5, x - 0.5) for y, x, _, _ in self._tiles],
            name="labels",
            size=0,
            border_width=0,
            face_color="transparent",
            features={"label": self._labels},
            text={"string": "{label}", "anchor": "lower_left", "color": "yellow", "size": 12},
        )
        model.layers.selection.active = img

    def _choose(self, v: float) -> None:
        """`v` picked: the sweep's candidate showing it on the slider, or the
        tiles showing it framed; marked on the plot."""
        self._pick = v
        model = self.view.model
        if self.mode == "sweep":
            i = self._values.index(v)
            model.dims.set_current_step(0, i)
            model.text_overlay.text = self._labels[i]
        else:
            model.layers["frames"].edge_color = ["blue" if x == v else "transparent" for x in self._values]
        self.plot.removeMarker("picked")
        self.plot.addXMarker(v, legend="picked", color="blue")

    def _stepped(self, event=None) -> None:
        # The slider moved: its candidate picked.
        if self.mode == "sweep" and self._values:
            self._choose(self._values[self.view.model.dims.current_step[0]])

    def _press(self, viewer, event):
        """A click on a tile, not a drag of the view, picks it."""
        if event.button != 1 or not self._tiles:
            return
        at, pos = event.position[-2:], np.asarray(event.pos)
        yield
        while event.type == "mouse_move":
            yield
        if np.hypot(*(np.asarray(event.pos) - pos)) > 4:
            return
        y, x = (math.floor(p + 0.5) for p in at)
        for (ty, tx, h, w), v in zip(self._tiles, self._values):
            if ty <= y < ty + h and tx <= x < tx + w:
                self._choose(v)

    def _picked(self, key: str) -> float | None:
        """The picked tile's `key`, None with the user told to pick one."""
        if self._key != key or self._pick is None:
            self.say(f"{self.NAME}: click a {key} tile")
            return None
        return self._pick

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
        self.view.add_outlines(
            lines,
            shape_type="line",
            name="centers",
            edge_color=["yellow"] + ["cyan"] * len(r.bands),
            px=2,
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
        model = self.view.model
        # One candidate at a time, full size, flipped through on the slider:
        # side by side, 49 are too small to judge. On the best's window, so a
        # flip changes the focus alone.
        frames = r.arrays["frames"]
        model.add_image(frames, name="sweep", contrast_limits=window(frames[v.index]), colormap="gray")
        model.dims.axis_labels = ("candidate", "y", "x")
        role = {x.index: " (rival)" for x in v.rivals} | {v.index: " (best)"}
        self._values = list(r.candidates_px)
        self._labels = [f"center {c:.2f} px{role.get(i, '')}" for i, c in enumerate(self._values)]
        self._key = "center_px"
        model.text_overlay.visible = True
        self._choose(self._values[v.index])
        self.plot.addCurve(r.candidates_px, r.focus, legend="focus", symbol="o", color="black")
        self.plot.addXMarker(v.center_px, legend="best", color="green")
        for x in v.rivals:
            self.plot.addXMarker(x.center_px, legend=f"rival {x.index}", color="orange")
        self.plot.setGraphTitle(", ".join(["best (green)", *(["rivals (orange)"] if v.rivals else []), PICKED]))
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
        """A click on the plot picks the candidate or tiles nearest it."""
        if event.get("event") == "mouseClicked" and self._values:
            self._choose(min(self._values, key=lambda v: abs(v - event["x"])))

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
        if self._key == "center_px" and self._pick not in (None, *centers):
            centers.append(self._pick)
        if len(centers) < 2:
            self.say("Align: no rivals; flip to another candidate to compare with the best")
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
            self.say("Align: sweep or compare first, then pick a center")

    def _candidate(self, c: float) -> str:
        """The picked candidate alone, as shown, saved to cite."""
        layer = self.view.model.layers["sweep"]
        img = png(to_u8(layer.data[self._values.index(c)], tuple(layer.contrast_limits)))
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
        layer = arrays["layer"]
        if st.op == "align.tilt_scan":
            self._show_mode("tilts")
            rows = st.result["tilts"]
            # The tilts' slices in one row, their xz planes in the row below.
            self._montage(
                [
                    [(a, t["tilt_deg"], f"{t['tilt_deg']:g}°") for t, a in zip(rows, layer)],
                    [(a[: t["depth"]], t["tilt_deg"], f"xz {t['tilt_deg']:g}°") for t, a in zip(rows, arrays["xz"])],
                ],
                "tilt_deg",
            )
            x = [t["tilt_deg"] for t in rows]
            self.plot.addCurve(x, [t["focus"] for t in rows], legend="focus", symbol="o", color="gray")
            self.plot.setGraphTitle(f"{TILT_FOCUS}; {PICKED}")
            self.plot.getXAxis().setLabel("tilt °")
            self.verdict.setText(
                f"job {st.id}: each tilt's band-middle slice at center {st.result['center_px']:.2f} px,"
                " its xz plane below"
            )
        else:
            self._show_mode("centers")
            rows = st.result["centers"]
            self._montage([[(a, c["center_px"], f"{c['center_px']:.2f}") for c, a in zip(rows, layer)]], "center_px")
            x = [c["center_px"] for c in rows]
            self.plot.addCurve(x, [c["focus"] for c in rows], legend="focus", symbol="o", color="black")
            self.plot.setGraphTitle(PICKED)
            self.plot.getXAxis().setLabel("center px (full resolution)")
            self.verdict.setText(f"job {st.id}: each center's band-middle slice at tilt {st.result['tilt_deg']:g}°")

    def decide_tilt(self) -> None:
        if self.mode != "tilts" or self._job is None:
            self.say("Align: run a tilt scan, then click a tilt tile")
            return
        t = self._picked("tilt_deg")
        if t is not None:
            self.decide("tilt_deg", t, [self.job_figure(self._job)], "eye")
