"""Export: the window a 16-bit TIFF maps, the TIFF itself, the binned volume
and the maximum intensity projections of the z band."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.plot import Plot2D

from lamino_pal import LaminoPalError, dispatch
from lamino_pal.artifacts import load_h5
from lamino_pal.gui.runner import message
from lamino_pal.gui.tabs.base import Tab, button, spin
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import REGISTRY, Result
from lamino_pal.ops.volume import data
from lamino_pal.render import window

MIPS = (("z", "down z [y, x]"), ("y", "along y [z, x]"), ("x", "along x [z, y]"))


class ExportTab(Tab):
    NAME = "Export"
    default_op = "export.tiff16"

    def __init__(self, win):
        super().__init__(win)
        self.info = qt.QLabel(wordWrap=True)
        self.hist = self.graph("histogram")
        self.hist.getXAxis().setLabel("value")
        self.hist.getYAxis().setLabel("voxels")
        self.mips = {}
        row = qt.QSplitter(qt.Qt.Horizontal)
        for k, title in MIPS:
            p = self.graph(f"MIP {k}", Plot2D)
            p.setGraphTitle(f"MIP {title}")
            p.setKeepDataAspectRatio(True)
            p.setYAxisInverted(True)
            row.addWidget(p)
            self.mips[k] = p
        split = qt.QSplitter(qt.Qt.Vertical)
        split.addWidget(self.hist)
        split.addWidget(row)
        split.setSizes([250, 650])
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(split, 1)
        self._vol: str | None = None
        self._z: tuple | None = None  # the default z range last set, with its reason

        rng = self.section("Slices [z0, z1)")
        self.z0, self.z1 = spin(0, 0), spin(0, 0)
        rng.addRow("z0", self.z0)
        rng.addRow("z1", self.z1)
        rng.addRow(button("Window", self.window, "the window the TIFF maps to 0–65535, and the histogram"))
        out = self.section("Export (jobs)")
        out.addRow(button("16-bit TIFF", lambda: self.export("export.tiff16")))
        out.addRow(button("Maximum intensity projections", lambda: self.export("export.mip")))
        out.addRow(button("Bin 2×2×2", self.bin, "the whole volume, for the 3D view"))
        self.panel.addStretch(1)

    def on_context(self) -> None:
        if self.win.volume != self._vol:
            self._vol = self.win.volume
            self._z = None
            self.hist.clear()
            for p in self.mips.values():
                p.clear()
        self._default_z()

    def on_project(self) -> None:
        self.on_context()

    def _default_z(self) -> None:
        """The group's z band of the volume, else every slice; set again only
        when that changes, so slices set by hand outlive other project edits."""
        vol = self._vol
        if vol is None:
            self.info.setText("Pick a volume in the project tree.")
            return
        try:
            nz = data(self.project.work_dir / vol).shape[0]
        except (LaminoPalError, OSError) as e:
            self.info.setText(f"{vol}: {e}")
            return
        try:
            z0, z1 = dispatch.validate(REGISTRY["export.window"], {"volume": vol}, self.project).z
            how = "the group's z band"
        except (LaminoPalError, ValidationError) as e:
            z0, z1 = 0, nz
            how = f"every slice ({message(e)})"
        if (z0, z1, nz, how) == self._z:
            return
        self._z = (z0, z1, nz, how)
        for s in (self.z0, self.z1):
            s.setRange(0, nz)
        self.z0.setValue(z0)
        self.z1.setValue(z1)
        self.info.setText(f"<b>{vol}</b>: slices [{z0}, {z1}) of {nz}, {how}")

    def _params(self) -> dict | None:
        vol = self.need("volume")
        if vol is None:
            return None
        return {"volume": vol, "z": [self.z0.value(), self.z1.value()]}

    def window(self) -> None:
        p = self._params()
        if p is not None:
            self.hist.setGraphTitle("window …")
            self.run("export.window", p, self._show_window, channel="export-window", failed=self.hist.setGraphTitle)

    def _show_window(self, r: Result) -> None:
        self.hist.clear()
        n = len(r.hist)
        edges = np.linspace(r.span[0], r.span[1], n + 1)
        self.hist.addHistogram(r.hist, edges, legend="voxels", color="black")
        for name, x in zip(("p0.1", "p99.9"), r.window):
            self.hist.addXMarker(x, legend=name, text=name, color="blue")
        self.hist.setGraphTitle(
            f"slices [{r.z[0]}, {r.z[1]}), every 8th: window [{r.window[0]:.4g}, {r.window[1]:.4g}] → 0–65535"
        )

    def export(self, op: str) -> None:
        p = self._params()
        if p is not None:
            self.submit(op, p)

    def bin(self) -> None:
        vol = self.need("volume")
        if vol:
            self.submit("export.bin", {"volume": vol})

    def on_job(self, st: JobState) -> None:
        if st.status != "done" or st.op not in ("export.mip", "export.tiff16", "export.bin"):
            return
        r = st.result
        out = r["artifacts"][0]
        if st.op == "export.mip":
            _, a = load_h5(out)
            for k, p in self.mips.items():
                lo, hi = window(a[k])
                p.addImage(a[k], legend="mip", colormap=Colormap("gray", vmin=lo, vmax=hi), resetzoom=True)
            self.info.setText(f"job {st.id}: MIPs of slices [{r['z'][0]}, {r['z'][1]}) in {out}")
        elif st.op == "export.tiff16":
            w = r["window"]
            self.info.setText(f"job {st.id}: {out}, {r['shape'][0]} slices, window [{w[0]:.4g}, {w[1]:.4g}]")
        elif st.op == "export.bin":
            self.info.setText(f"job {st.id}: {out}, {'×'.join(map(str, r['shape']))}")
