"""`lamino-pal-gui [--project project.toml]`: the GUI on one project, its CUDA
on the project's interactive GPU."""

from __future__ import annotations

import argparse
import os
import sys

from lamino_pal import Refused, launch


def _choose():
    """A project opened or created by hand; None when cancelled."""
    from silx.gui import qt

    from lamino_pal.project import Project

    while True:
        box = qt.QMessageBox(qt.QMessageBox.Question, "Lamino-Pal", "Open a project or create one?")
        opened = box.addButton("Open…", qt.QMessageBox.AcceptRole)
        created = box.addButton("Create…", qt.QMessageBox.AcceptRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is opened:
            path, _ = qt.QFileDialog.getOpenFileName(None, "Open project", "", "project.toml")
            if path:
                return path
        elif box.clickedButton() is created:
            work = qt.QFileDialog.getExistingDirectory(None, "Work directory (project.toml goes here)")
            raw = work and qt.QFileDialog.getExistingDirectory(None, "Raw data root (read only)")
            if work and raw:
                try:
                    return Project.create(os.path.join(work, "project.toml"), raw).path
                except (Refused, OSError) as e:
                    qt.QMessageBox.warning(None, "Lamino-Pal", str(e))
        else:
            return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lamino-pal-gui")
    ap.add_argument("--project", help="project.toml, asked for when missing; created when it does not exist")
    a = ap.parse_args(argv)
    try:
        project = launch.open_or_pin(a.project)
    except (OSError, ValueError) as e:
        ap.error(f"cannot read project {a.project}: {e}")
    # napari's qtpy and silx take the binding the environment pins.
    os.environ.setdefault("QT_API", "pyside6")
    import PySide6.QtCore  # noqa: F401
    from silx.gui import qt

    from lamino_pal.gui.window import MainWindow, _relaunch
    from lamino_pal.ops import load_all
    from lamino_pal.project import Project

    app = qt.QApplication.instance() or qt.QApplication(sys.argv[:1])
    app.setApplicationName("Lamino-Pal")
    if a.project is None:
        path = _choose()
        if path is not None:
            # Again with --project, so the chosen project's GPU is pinned.
            _relaunch(path)
        return 1
    if project is None:
        raw = qt.QFileDialog.getExistingDirectory(None, f"Raw data root for the new {a.project}")
        if not raw:
            return 1
        try:
            project = Project.create(a.project, raw)
        except (Refused, OSError) as e:
            qt.QMessageBox.warning(None, "Lamino-Pal", str(e))
            return 1
    load_all()
    win = MainWindow(project)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
