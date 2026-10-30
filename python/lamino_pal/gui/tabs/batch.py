"""Batch: the group's final volumes, each with the center, tilt and recipe it
will be made at, queued as jobs or copied as shell commands for another GPU."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from pydantic import ValidationError
from silx.gui import qt

from lamino_pal import LaminoPalError, _core, dispatch
from lamino_pal.gui.runner import message
from lamino_pal.gui.table import fill
from lamino_pal.gui.tabs.base import BAD, GOOD, Tab, badge, button
from lamino_pal.jobs import JobState
from lamino_pal.launch import ENV
from lamino_pal.ops.registry import REGISTRY

COLUMNS = ("scan", "center px", "tilt °", "algorithm", "filter", "output", "state")


class BatchTab(Tab):
    NAME = "Batch"
    default_op = "recon.run"

    def __init__(self, win):
        super().__init__(win)
        self.table = qt.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.head = qt.QLabel(wordWrap=True)
        self.gpus = qt.QLabel()
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.head)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.gpus)
        self._params: dict[str, dict] = {}  # tag -> the params that bind

        run = self.section("Run")
        self.overwrite = qt.QCheckBox("overwrite an existing volume")
        self.overwrite.toggled.connect(lambda _: self.refresh())
        run.addRow(self.overwrite)
        run.addRow(button("Run selected", self.run_selected, "one job per scan, each on a free batch GPU"))
        self.gpu = qt.QComboBox()
        run.addRow("copy for GPU", self.gpu)
        run.addRow(button("Copy commands", self.copy, "shell commands for the selected scans, to run elsewhere"))
        self.panel.addStretch(1)
        self.timer = qt.QTimer(self, interval=2000)
        self.timer.timeout.connect(self._locks)

    def showEvent(self, event: qt.QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
        self.timer.start()

    def hideEvent(self, event: qt.QHideEvent) -> None:
        self.timer.stop()
        super().hideEvent(event)

    def on_context(self) -> None:
        if self.isVisible():
            self.refresh()

    def on_project(self) -> None:
        self.on_context()

    def on_job(self, st: JobState) -> None:
        if st.op == "recon.run":
            self.on_context()

    def refresh(self) -> None:
        g = self.win.find_group()
        self.gpu.blockSignals(True)
        keep = self.gpu.currentText()
        self.gpu.clear()
        self.gpu.addItems([str(n) for n in self.project.data.gpu.batch])
        self.gpu.setCurrentText(keep)
        self.gpu.blockSignals(False)
        self._locks()
        tags = self.win.group_tags()
        self._params.clear()
        if g is None:
            fill(self.table, [])
            self.head.setText("Pick a group in the project tree.")
            return
        recipe = self.project.data.recon
        fields = ", ".join(f"{k} = {v!r}" for k, v in recipe.items()) or "tomoxide's defaults"
        self.head.setText(f"group <b>{g.name}</b>: recipe [recon] {fields}")
        op = REGISTRY["recon.run"]
        rows = []
        for tag in tags:
            params = {"tag": tag, "overwrite": self.overwrite.isChecked()}
            try:
                p = dispatch.resolve(op, params, self.project)
            except (LaminoPalError, ValidationError) as e:
                cells = [tag, "", "", "", "", "", message(e)]
                color = BAD
            else:
                self._params[tag] = params
                out = Path(p.out)
                c = json.loads(_core.recipe_config(json.dumps(p.recipe)))
                cells = [
                    tag,
                    f"{p.center_px:.2f}",
                    f"{p.tilt_deg:g}",
                    c["algorithm"],
                    c["filter_name"],
                    out.name,
                    "exists" if out.exists() else "to make",
                ]
                color = GOOD if out.exists() else None
            items = []
            for col, text in enumerate(cells):
                item = qt.QTableWidgetItem(text)
                item.setToolTip(text)
                if col == len(cells) - 1 and color is not None:
                    item.setForeground(qt.QColor(color))
                items.append(item)
            rows.append(items)
        fill(self.table, rows)
        self.table.resizeColumnsToContents()

    def _locks(self) -> None:
        held = [f"GPU {n} {'busy' if dispatch.gpu_busy(n) else 'free'}" for n in self.project.data.gpu.batch]
        badge(self.gpus, "batch GPUs: " + (", ".join(held) or "none (gpu.batch is empty)"))

    def _selected(self) -> list[str]:
        """The selected scans that bind; the others named on the status bar."""
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        tags = [self.table.item(r, 0).text() for r in rows]
        bad = [t for t in tags if t not in self._params]
        if bad:
            self.say(f"Batch: skipped {', '.join(bad)}: see the state column")
        return [t for t in tags if t in self._params]

    def run_selected(self) -> None:
        for tag in self._selected():
            self.submit("recon.run", self._params[tag])

    def commands(self, tags: list[str], gpu: str) -> str:
        project = shlex.quote(str(self.project.path))
        lines = []
        for tag in tags:
            js = shlex.quote(json.dumps(self._params[tag]))
            lines.append(f"echo {js} | {ENV}={gpu} lamino-pal run --project {project} recon.run -")
        return "\n".join(lines)

    def copy(self) -> None:
        tags = self._selected()
        if tags:
            qt.QApplication.clipboard().setText(self.commands(tags, self.gpu.currentText()))
            self.say(f"Batch: {len(tags)} commands for GPU {self.gpu.currentText()} copied", 10)
