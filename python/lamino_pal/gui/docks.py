"""The bottom docks: the project's jobs, and log.jsonl as it grows."""

from __future__ import annotations

import json

from silx.gui import qt

from lamino_pal import LaminoPalError, dispatch, jobs
from lamino_pal.jobs import JobState
from lamino_pal.project import Project
from lamino_pal.gui.runner import message
from lamino_pal.gui.table import fill

POLL_MS = 1000
COLUMNS = ("job", "status", "progress", "ETA", "GPU", "message")


def _eta(s: float | None) -> str:
    if s is None:
        return ""
    m, s = divmod(int(s), 60)
    return f"{m}:{s:02d}"


class JobsWidget(qt.QWidget):
    """Every job of the project, newest first, polled; `ended` carries each
    job this window submitted once it ends."""

    ended = qt.Signal(object)

    def __init__(self, project: Project, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self._mine: set[str] = set()
        self.table = qt.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.cancel = qt.QPushButton("Cancel selected")
        self.cancel.clicked.connect(self._cancel)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addWidget(self.cancel, alignment=qt.Qt.AlignRight)
        self.timer = qt.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)

    def track(self, st: JobState) -> None:
        self._mine.add(st.id)
        self.poll()

    def poll(self) -> None:
        states = jobs.list_(self.project)[::-1]
        rows = []
        for st in states:
            cells = (
                f"{st.id}",
                st.status,
                f"{st.progress:.0%}",
                _eta(st.eta_s) if st.active else "",
                "" if st.gpu is None else str(st.gpu),
                st.error or st.message,
            )
            rows.append([qt.QTableWidgetItem(text) for text in cells])
            if st.id in self._mine and not st.active:
                self._mine.discard(st.id)
                self.ended.emit(st)
        fill(self.table, rows)

    def _cancel(self) -> None:
        for idx in self.table.selectionModel().selectedRows():
            jid = self.table.item(idx.row(), 0).text()
            try:
                jobs.cancel(self.project, jid)
            except LaminoPalError as e:  # it ended meanwhile
                self.window().statusBar().showMessage(message(e), 10000)
        self.poll()


def _line(e: dict) -> str:
    head = f"{e.get('ts', '')[11:23]}  {e.get('caller', ''):3}  {e.get('op', '')}  {e.get('status', '')}"
    tail = e.get("error") or ""
    return f"{head}  {e.get('elapsed_s', 0):.2f} s  {tail}".rstrip()


class LogWidget(qt.QPlainTextEdit):
    """log.jsonl's entries, one line each, followed from where it was read."""

    def __init__(self, project: Project, keep: int = 2000, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(keep)
        self.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
        self.path = dispatch.log_path(project)
        self._offset = 0
        self.timer = qt.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)
        self.poll()

    def poll(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1  # a line still being written waits
        self._offset += end
        for raw in chunk[:end].splitlines():
            try:
                self.appendPlainText(_line(json.loads(raw)))
            except ValueError:
                self.appendPlainText(raw.decode(errors="replace"))
