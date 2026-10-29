"""Data: the raw root's scan folders, imported at a bin and grouped by mount."""

from __future__ import annotations

from silx.gui import qt

from lamino_pal.gui.actions import RunOpDialog
from lamino_pal.gui.tabs.base import Tab, button, spin
from lamino_pal.jobs import JobState
from lamino_pal.ops.registry import Result

COLUMNS = (
    "folder", "tag", "kind", "N", "flats", "range °", "speed °/s", "exposure s", "energy eV", "imported", "group",
)


def _num(x: float | None, fmt: str = "g") -> str:
    return "" if x is None else format(x, fmt)


class DataTab(Tab):
    NAME = "Data"
    default_op = "scan.discover"

    def __init__(self, win):
        super().__init__(win)
        self._found: list[dict] = []
        self._shown = False
        self.root = qt.QLabel()
        self.root.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.table = qt.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(qt.QAbstractItemView.DoubleClicked)
        self.table.setSortingEnabled(True)
        self.skipped = qt.QLabel(wordWrap=True)
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.root)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.skipped)

        imp = self.section("Import")
        self.bin = spin(1, 8, 2)
        imp.addRow("bin", self.bin)
        imp.addRow(button("Discover", self.discover, "list the scan folders under the raw root"))
        imp.addRow(button("Import selected", self.import_, "one job per folder; the tag column is editable"))
        imp.addRow(button("Adopt an h5…", lambda: RunOpDialog(self.win, "scan.adopt").show()))
        grp = self.section("Mount group")
        self.group = qt.QLineEdit(placeholderText="group name")
        grp.addRow("group", self.group)
        grp.addRow(button("Assign selected", lambda: self.assign(self.group.text().strip() or None)))
        grp.addRow(button("Ungroup selected", lambda: self.assign(None)))
        self.panel.addStretch(1)

    def showEvent(self, event: qt.QShowEvent) -> None:
        if not self._shown:
            self._shown = True
            self.discover()
        super().showEvent(event)

    def discover(self) -> None:
        self.root.setText(f"raw root (read only): {self.project.raw_root}")
        self.run("scan.discover", {}, self._show)

    def _show(self, r: Result) -> None:
        d = r.model_dump(mode="json")
        self._found = d["scans"]
        self._fill()
        self.skipped.setText(
            "" if not d["skipped"] else "skipped: " + "; ".join(f"{raw} ({why})" for raw, why in d["skipped"])
        )

    def _fill(self) -> None:
        groups = {s.tag: s.group for s in self.project.data.scan}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._found))
        for row, f in enumerate(self._found):
            cells = (
                f["raw"],
                f["imported"][0] if f["imported"] else f["tag"],
                f["kind"],
                str(f["n"]),
                str(f["n_flat"]),
                _num(f["range_deg"]),
                _num(f["speed_dps"]),
                _num(f["exptime_s"]),
                _num(f["energy_ev"]),
                ", ".join(f["imported"]),
                ", ".join(sorted({groups.get(t) or "" for t in f["imported"]} - {""})),
            )
            for col, text in enumerate(cells):
                item = qt.QTableWidgetItem(text)
                if col != 1 or f["imported"]:
                    item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def _rows(self) -> list[tuple[str, str, bool]]:
        """The selected rows' (folder, tag, imported)."""
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [
            (self.table.item(r, 0).text(), self.table.item(r, 1).text(), bool(self.table.item(r, 9).text()))
            for r in rows
        ]

    def import_(self) -> None:
        rows = [r for r in self._rows() if not r[2]]
        if not rows:
            self.say("Data: select folders not imported yet")
        for raw, tag, _ in rows:
            self.submit("scan.import", {"raw": raw, "tag": tag, "bin": self.bin.value()})

    def assign(self, group: str | None) -> None:
        tags = [t for _, t, imported in self._rows() if imported]
        if not tags:
            self.say("Data: select imported scans")
            return
        self.run("scan.assign_group", {"tags": tags, "group": group}, lambda r: self._assigned(group))

    def _assigned(self, group: str | None) -> None:
        if group is not None:
            self.win.set_context(group=group)

    def on_project(self) -> None:
        self._fill()

    def on_job(self, st: JobState) -> None:
        if st.op == "scan.import":
            self.discover()
