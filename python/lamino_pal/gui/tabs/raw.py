"""Raw: a scan's corrected projections, read lazily, and what its angles,
drift and optical flow say before any reconstruction."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
from silx.gui import qt

from lamino_pal import _core, scans
from lamino_pal.gui.tabs.base import BAD, GOOD, WARN, Tab, badge, button, spin
from lamino_pal.ops.registry import Result


def projections(path, n: int, shape: tuple[int, int]) -> da.Array:
    """Scan `path`'s n flat/dark-corrected −log projections, each read when
    napari shows it."""
    one = dask.delayed(lambda i: _core.prep_frames(path, [i])[0], pure=True)
    return da.stack([da.from_delayed(one(i), shape, np.float32) for i in range(n)])


class RawTab(Tab):
    NAME = "Raw"
    default_op = "raw.frame"

    def __init__(self, win):
        super().__init__(win)
        self.view = self.viewer("raw")
        self.theta = np.zeros(0)
        self.angle = qt.QLabel()
        self.range = qt.QLabel(wordWrap=True)
        top = qt.QHBoxLayout()
        top.addWidget(self.angle)
        top.addWidget(self.range, 1)
        layout = qt.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.view, 1)
        self.view.model.dims.events.current_step.connect(self._angle)

        mean = self.section("Mean projection")
        self.step = spin(1, 1000, 10)
        mean.addRow("every n-th", self.step)
        mean.addRow(button("Mean", self.mean))
        drift = self.section("Drift over one turn")
        self.drift_out = qt.QLabel(wordWrap=True)
        drift.addRow(button("Drift", self.drift, "the projection at first + 360° against the first"))
        drift.addRow(self.drift_out)
        flow = self.section("Tilt from optical flow")
        self.flow_out = qt.QLabel(wordWrap=True)
        flow.addRow(button("Tilt flow", self.tilt_flow, "a cross-check of the tilt scan, not a decision"))
        flow.addRow(self.flow_out)
        self.panel.addStretch(1)
        self._tag: str | None = None

    def on_context(self) -> None:
        if self.win.tag == self._tag:
            return
        self._tag = self.win.tag
        self.view.clear()
        for label in (self.angle, self.range, self.drift_out, self.flow_out):
            label.clear()
        if self._tag is None:
            return
        tag = self._tag
        self.run("raw.frame", {"tag": tag}, lambda r: self._stack(tag, r), channel="raw-stack")
        self.run("raw.range_check", {"tag": tag}, self._range, channel="raw-range")

    def _stack(self, tag: str, r: Result) -> None:
        if tag != self._tag:
            return
        _, path = scans.find(self.project, tag)
        self.theta = scans.theta_deg(path)
        stack = projections(str(path), len(self.theta), tuple(r.shape))
        self.view.replace("projections", stack, contrast_limits=r.window, colormap="gray")
        self._angle()

    def _angle(self, *_) -> None:
        if not self.theta.size or self.view.model.dims.ndim < 3:
            return
        i = self.view.model.dims.current_step[0]
        if i < self.theta.size:
            self.angle.setText(f"projection {i}  θ = {self.theta[i]:.3f}°")

    def _range(self, r: Result) -> None:
        text = (
            f"{r.kind} turn: {r.nproj} projections, {r.first_deg:.2f}° to {r.last_deg:.2f}°, "
            f"step {r.step_deg:.4f}° (±{r.step_spread_deg:.4f})"
        )
        if r.turn_index is not None:
            text += f", first + 360° at {r.turn_index}"
        color = GOOD if r.kind == "full" and not r.warnings else WARN if r.kind == "full" else BAD
        badge(self.range, text + "".join(f"<br>⚠ {w}" for w in r.warnings), color)

    def mean(self) -> None:
        tag = self.need("tag")
        if tag:
            self.run(
                "raw.mean_projection",
                {"tag": tag, "step": self.step.value()},
                lambda r: self.view.replace("mean", r.arrays["mean"], contrast_limits=r.window, colormap="gray"),
            )

    def drift(self) -> None:
        tag = self.need("tag")
        if tag:
            self.run("raw.drift", {"tag": tag}, self._drift, failed=self.drift_out.setText)

    def _drift(self, r: Result) -> None:
        a = r.arrays
        w = (float(min(a["first"].min(), a["turn"].min())), float(max(a["first"].max(), a["turn"].max())))
        self.view.replace("first", a["first"], contrast_limits=w, colormap="green", blending="additive")
        self.view.replace("turn", a["turn"], contrast_limits=w, colormap="magenta", blending="additive")
        self.drift_out.setText(
            f"projection {r.turn} against {r.first}: dy {r.dy_px:+.1f} px, dx {r.dx_px:+.1f} px (full resolution)"
        )

    def tilt_flow(self) -> None:
        tag = self.need("tag")
        if tag:
            self.run("raw.tilt_flow", {"tag": tag}, self._flow, failed=self.flow_out.setText)

    def _flow(self, r: Result) -> None:
        text = (
            f"tilt {r.tilt_deg:.2f}° ± {r.tilt_sigma_deg:.2f}, roll {r.roll_deg:.2f}° ± {r.roll_sigma_deg:.2f}, "
            f"axis {r.axis_px:.1f} ± {r.axis_sigma_px:.1f} px over {len(r.pairs)} pairs"
        )
        if r.group_tilt_deg is None:
            self.flow_out.setText(text + "; the group's tilt is undecided")
            return
        self.flow_out.setText(f"{text}; the group's decided tilt {r.group_tilt_deg:g}°")
