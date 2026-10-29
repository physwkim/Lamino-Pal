"""The main window: the project tree, one tab per step, the jobs and log."""

from __future__ import annotations

import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from silx.gui import qt

from lamino_pal import jobs
from lamino_pal.jobs import JobState
from lamino_pal.launch import ENV
from lamino_pal.ops.registry import REGISTRY
from lamino_pal.ops.volume import provenance
from lamino_pal.project import Group, Project
from lamino_pal.gui import status
from lamino_pal.gui.actions import ACTIONS, RunOpDialog
from lamino_pal.gui.docks import JobsWidget, LogWidget
from lamino_pal.gui.runner import Runner, by_hand, message
from lamino_pal.gui.tabs import TABS
from lamino_pal.gui.tree import ProjectTree

DEBOUNCE_MS = 200


def _relaunch(path: Path) -> None:
    """This GUI again on project `path`, pinned to that project's GPU."""
    env = {k: v for k, v in os.environ.items() if k != ENV}
    os.execve(sys.executable, [sys.executable, "-m", "lamino_pal.gui", "--project", str(path)], env)


class MainWindow(qt.QMainWindow):
    # The group, scan or volume in view changed.
    context_changed = qt.Signal()

    def __init__(self, project: Project, parent: qt.QWidget | None = None):
        super().__init__(parent)
        qt.loadUi(str(files("lamino_pal.gui") / "ui" / "main.ui"), self)
        self.project = project
        self.setWindowTitle(f"Lamino-Pal — {project.path}")
        self.group: str | None = None
        self.tag: str | None = None
        self.volume: str | None = None

        self.runner = Runner(project, self._cancel_running, self)
        self.runner.failed.connect(lambda op, msg: self.statusBar().showMessage(f"{op}: {msg}", 20000))
        self.runner.progressed.connect(self._progress)
        self.jobs = JobsWidget(project)
        self.layoutJobs.addWidget(self.jobs)
        self.runner.submitted.connect(self.jobs.track)
        self.jobs.ended.connect(self._job_ended)
        self.layoutLog.addWidget(LogWidget(project))
        self.tabifyDockWidget(self.dockJobs, self.dockLog)
        self.dockJobs.raise_()

        self.tree = ProjectTree(self.treeProject, project)
        self.tree.selected.connect(self._select)
        self.tabs = {}
        for cls in TABS:
            tab = cls(self)
            self.tabs[cls.NAME] = tab
            getattr(self, f"layout{cls.NAME}").addWidget(tab)
            self.stackParams.addWidget(tab.controls)
        self.tabWidget.currentChanged.connect(self._tab_changed)
        self.splitterMain.setSizes([260, 1340])

        self.pushButtonNext.clicked.connect(self._next)
        self.pushButtonConfirm.clicked.connect(lambda _=False: by_hand(self._confirm))
        self.actionRunOp.triggered.connect(lambda: RunOpDialog(self, self.current_tab().default_op).show())
        self.actionOpen.triggered.connect(self._open)
        self.actionQuit.triggered.connect(self.close)

        self.watcher = qt.QFileSystemWatcher(self)
        self._debounce = qt.QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._debounce.timeout.connect(self.on_project)
        self.watcher.fileChanged.connect(lambda _: self._debounce.start())
        self.watcher.directoryChanged.connect(lambda _: self._debounce.start())
        self._watch()
        self.on_project()
        self._tab_changed(self.tabWidget.currentIndex())

    # Context ---------------------------------------------------------------

    def find_group(self) -> Group | None:
        return next((g for g in self.project.data.group if g.name == self.group), None)

    def group_tags(self) -> list[str]:
        return [s.tag for s in self.project.data.scan if self.group and s.group == self.group]

    def set_context(self, **kw: Any) -> None:
        """Set any of group, tag and volume; a scan brings its group along."""
        if "tag" in kw and "group" not in kw and kw["tag"] is not None:
            s = self.project.data.find_scan(kw["tag"])
            kw["group"] = s.group if s else None
        changed = False
        for k, v in kw.items():
            if getattr(self, k) != v:
                setattr(self, k, v)
                changed = True
        if changed:
            self._status()
            self.context_changed.emit()
            for tab in self.tabs.values():
                tab.on_context()

    def context_params(self, op: str) -> dict[str, Any]:
        """The params of op `op` the context fills."""
        fields = REGISTRY[op].params.model_fields
        have = {"group": self.group, "tag": self.tag, "volume": self.volume}
        return {k: v for k, v in have.items() if k in fields and v is not None}

    def _select(self, kind: str, value: str) -> None:
        if kind == "group":
            tags = [s.tag for s in self.project.data.scan if s.group == value]
            self.set_context(group=value, tag=self.tag if self.tag in tags else (tags[0] if tags else None))
        elif kind == "scan":
            self.set_context(tag=value)
        elif kind == "volume":
            try:
                tag = provenance(self.project.work_dir / value).get("tag")
            except OSError as e:
                self.statusBar().showMessage(f"{value}: {e}", 20000)
                return
            if tag is not None and self.project.data.find_scan(tag) is not None:
                self.set_context(tag=tag, volume=value)
            else:
                self.set_context(volume=value)
            self.show_tab("Volume")
        elif kind == "artifact":
            self._artifact(value)

    def _artifact(self, rel: str) -> None:
        """The tab of the job that wrote artifact `rel` shows it."""
        path = str((self.project.work_dir / rel).resolve())
        for st in reversed(jobs.list_(self.project)):
            if st.status == "done" and path in [str(Path(a).resolve()) for a in st.result.get("artifacts", [])]:
                tab = self.tabs[ACTIONS[st.op]]
                self.set_context(tag=st.params.get("tag", self.tag))
                self.show_tab(tab.NAME)
                tab.on_job(st)
                return
        self.statusBar().showMessage(f"{rel}: no job of this project wrote it", 20000)

    # Tabs ------------------------------------------------------------------

    def current_tab(self):
        return list(self.tabs.values())[self.tabWidget.currentIndex()]

    def show_tab(self, name: str) -> None:
        self.tabWidget.setCurrentIndex(list(self.tabs).index(name))

    def _tab_changed(self, i: int) -> None:
        self.stackParams.setCurrentIndex(i)
        self._show_docks()

    def _show_docks(self) -> None:
        """Each tab's docks shown while it is the current tab."""
        cur = self.current_tab()
        for tab in self.tabs.values():
            for dock in tab.docks:
                dock.setVisible(tab is cur)

    def attach_docks(self, tab, qv) -> None:
        """QtViewer `qv`'s layer controls and list beside the parameters,
        shown while `tab` is."""
        for dock in (qv.dockLayerControls, qv.dockLayerList):
            self.addDockWidget(qt.Qt.RightDockWidgetArea, dock)
            tab.docks.append(dock)
        self._show_docks()

    def _job_ended(self, st: JobState) -> None:
        if st.status == "failed":
            self.statusBar().showMessage(f"{st.op} job {st.id} failed: {st.error}", 30000)
        # Not only the op's own tab: Volume bins for its 3D view.
        for tab in self.tabs.values():
            tab.on_job(st)

    def _progress(self, channel: str, frac: float, msg: str) -> None:
        self.statusBar().showMessage(f"{channel}: {frac:.0%} {msg}", 5000)

    # Project ---------------------------------------------------------------

    def _watch(self) -> None:
        """Watch project.toml, the work directory and artifacts/, again after
        an atomic rename replaced a watched file."""
        want = [self.project.path, self.project.work_dir, self.project.work_dir / "artifacts"]
        have = set(self.watcher.files()) | set(self.watcher.directories())
        add = [str(p) for p in want if p.exists() and str(p) not in have]
        if add:
            self.watcher.addPaths(add)

    def on_project(self) -> None:
        self._watch()
        try:
            self.project.reload()
        except Exception as e:  # a hand edit that does not parse
            self.statusBar().showMessage(f"project.toml: {message(e)}", 30000)
            return
        if self.group is not None and self.find_group() is None:
            self.group = None
        if self.tag is not None and self.project.data.find_scan(self.tag) is None:
            self.tag = None
        self.tree.refresh()
        self._status()
        for tab in self.tabs.values():
            tab.on_project()

    def _status(self) -> None:
        g = self.find_group()
        st = status.status(self.project, g)
        self._status_now = st
        self.labelGroup.setText(f"group <b>{g.name}</b>" if g else "no group")

        def show(label: qt.QLabel, what: str, value: str | None, how: str | None) -> None:
            if value is None:
                label.setText(f"{what}: <span style='color:#b00'>undecided</span>")
            else:
                color = "#b60" if how == "agent" else "#070"
                label.setText(f"{what}: {value} <span style='color:{color}'>({how})</span>")

        if g is None:
            for label in (self.labelCenter, self.labelTilt, self.labelZBand, self.labelRecon):
                label.clear()
        else:
            show(self.labelCenter, "center", None if g.center_px is None else f"{g.center_px:.2f} px", st.center)
            show(self.labelTilt, "tilt", None if g.tilt_deg is None else f"{g.tilt_deg:g}°", st.tilt)
            zb = g.z_band
            show(self.labelZBand, "z band", None if zb is None else f"{zb.z[0]}–{zb.z[1]} (bin {zb.bin})", st.z_band)
            self.labelRecon.setText(f"finals {st.finals[0]}/{st.finals[1]}")
        self.pushButtonConfirm.setVisible(bool(g and status.unconfirmed(g)))
        self.pushButtonNext.setText(f"Next: {st.next_tab}")

    def _next(self) -> None:
        self.show_tab(self._status_now.next_tab)

    def _cancel_running(self, op: str) -> bool:
        q = f"{op} is still running. Cancel it?"
        return qt.QMessageBox.question(self, "Lamino-Pal", q) == qt.QMessageBox.Yes

    def _confirm(self) -> None:
        """The agent's standing decisions decided again by the user."""
        g = self.find_group()
        if g is None:
            return
        for d in status.unconfirmed(g):
            params = {
                "group": g.name,
                "field": d.field,
                "value": d.value if isinstance(d.value, float) else d.value.model_dump(),
                "evidence": d.evidence,
                "method": d.method,
                "note": d.note,
            }
            self.runner.run("group.decide", params, lambda r: None, channel=f"confirm-{d.field}")

    def _open(self) -> None:
        path, _ = qt.QFileDialog.getOpenFileName(self, "Open project", str(self.project.work_dir), "project.toml")
        if path:
            self.close()
            _relaunch(Path(path))

    def closeEvent(self, event: qt.QCloseEvent) -> None:
        self.runner.shutdown()
        for tab in self.tabs.values():
            tab.close_viewers()
        super().closeEvent(event)
