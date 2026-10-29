"""Where each op lives in the GUI, and a dialog that runs any op."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from silx.gui import qt

from lamino_pal.artifacts import load_h5
from lamino_pal.jobs import JobState
from lamino_pal.mcp import figures
from lamino_pal.ops.registry import REGISTRY, Result
from lamino_pal.project import Project
from lamino_pal.gui.form import ParamsForm

# The tab each op belongs to; ops are reached nowhere else but the dialog.
ACTIONS = {
    "scan.discover": "Data",
    "scan.import": "Data",
    "scan.adopt": "Data",
    "scan.assign_group": "Data",
    "raw.frame": "Raw",
    "raw.mean_projection": "Raw",
    "raw.range_check": "Raw",
    "raw.drift": "Raw",
    "raw.tilt_flow": "Raw",
    "align.rings": "Align",
    "align.center_sweep": "Align",
    "align.tilt_scan": "Align",
    "align.center_compare": "Align",
    "group.decide": "Align",
    "recon.preview": "Preview",
    "recon.compare": "Preview",
    "recon.split": "Preview",
    "recon.run": "Batch",
    "volume.slice": "Volume",
    "volume.ortho": "Volume",
    "volume.focus_by_z": "Volume",
    "volume.valid_fov": "Volume",
    "export.window": "Export",
    "export.tiff16": "Export",
    "export.bin": "Export",
    "export.mip": "Export",
    "log.tail": "Tools",
}


def figure_name(op: str, st: JobState | None = None) -> str:
    """The name a figure is saved under: the job's id, else the time."""
    return f"{st.id}.png" if st is not None else f"{datetime.now():%Y%m%d-%H%M%S-%f}-{op}.png"


def save_figure(project: Project, op: str, result: dict[str, Any], arrays: figures.Arrays, name: str):
    """Op `op`'s figure saved under the work directory as `name`: (path
    relative to it, PNG, caption), or None for an op without one."""
    fig = figures.render(op, result, arrays)
    if fig is None:
        return None
    png, caption = fig
    return figures.save(project.work_dir, name, png), png, caption


def job_figure(project: Project, st: JobState):
    """`save_figure` for an ended job, its arrays read from its artifact."""
    if st.status != "done" or not figures.has(st.op):
        return None
    _, arrays = load_h5(st.result["artifacts"][0])
    return save_figure(project, st.op, st.result, arrays, figure_name(st.op, st))


def pixmap(png: bytes, width: int | None = None) -> qt.QPixmap:
    pm = qt.QPixmap()
    pm.loadFromData(png, "PNG")
    return pm if width is None or pm.width() <= width else pm.scaledToWidth(width, qt.Qt.SmoothTransformation)


class RunOpDialog(qt.QDialog):
    """Any op with any params: a job is submitted, the rest run here; the
    result shows as JSON and, for an op with one, its figure."""

    def __init__(self, win, op: str | None = None):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Run op")
        self.resize(900, 700)
        self.combo = qt.QComboBox()
        self.forms = qt.QStackedWidget()
        self._names: list[str] = []
        tabs = list(dict.fromkeys(ACTIONS.values()))
        for tab in tabs:
            for name in [n for n, t in ACTIONS.items() if t == tab]:
                o = REGISTRY[name]
                self.combo.addItem(f"{tab} · {name}  [{o.cost}{', GPU' if o.gpu else ''}]", name)
                self._names.append(name)
                form = ParamsForm(o)
                scroll = qt.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(form)
                self.forms.addWidget(scroll)
        self.doc = qt.QLabel()
        self.doc.setWordWrap(True)
        self.doc.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.run = qt.QPushButton("Run")
        self.run.clicked.connect(self._run)
        self.text = qt.QPlainTextEdit(readOnly=True)
        self.image = qt.QLabel()
        self.image.setAlignment(qt.Qt.AlignCenter)
        self.caption = qt.QLabel(wordWrap=True)
        out = qt.QSplitter(qt.Qt.Vertical)
        out.addWidget(self.text)
        pic = qt.QScrollArea()
        pic.setWidgetResizable(True)
        pic.setWidget(self.image)
        out.addWidget(pic)
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.combo)
        layout.addWidget(self.doc)
        layout.addWidget(self.forms, 1)
        layout.addWidget(self.run, alignment=qt.Qt.AlignRight)
        layout.addWidget(out, 2)
        layout.addWidget(self.caption)
        self.combo.currentIndexChanged.connect(self._pick)
        self._job: str | None = None
        win.jobs.ended.connect(self._ended)
        if op is not None:
            self.combo.setCurrentIndex(self._names.index(op))
        self._pick(self.combo.currentIndex())

    def form(self) -> ParamsForm:
        return self.forms.currentWidget().widget()

    def _pick(self, i: int) -> None:
        self.forms.setCurrentIndex(i)
        self.doc.setText(REGISTRY[self._names[i]].doc.strip())
        self.form().set(**self.win.context_params(self._names[i]))

    def _run(self) -> None:
        name = self._names[self.combo.currentIndex()]
        params = self.form().values()
        self.text.setPlainText(f"running {name} …")
        self.image.clear()
        self.caption.clear()
        if REGISTRY[name].cost == "job":
            st = self.win.runner.submit(name, params)
            if st is None:
                self.text.setPlainText("refused: see the status bar")
                return
            self._job = st.id
            self.text.setPlainText(f"job {st.id} submitted on GPU {st.gpu}" if st.gpu is not None else f"job {st.id}")
            return
        fail = self.text.setPlainText
        self.win.runner.run(name, params, lambda r: self._show(name, r), channel="dialog", failed=fail)

    def _show(self, name: str, r: Result) -> None:
        result = r.model_dump(mode="json")
        self.text.setPlainText(json.dumps(result, indent=1))
        self._figure(save_figure(self.win.project, name, result, r.arrays, figure_name(name)))

    def _ended(self, st: JobState) -> None:
        if st.id != self._job:
            return
        self.text.setPlainText(json.dumps(st.model_dump(mode="json"), indent=1))
        self._figure(job_figure(self.win.project, st))

    def _figure(self, fig) -> None:
        if fig is None:
            return
        rel, png, caption = fig
        self.image.setPixmap(pixmap(png, 1600))
        self.caption.setText(f"{caption}\nSaved as {rel}.")
