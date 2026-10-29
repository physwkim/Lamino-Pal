"""What the canvases draw. Qt has one platform a process and test_gui's
offscreen has no GL, so gl_render.py runs in a process of its own on xcb,
where DISPLAY has GLX."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
PROBE = """
from PySide6.QtGui import QGuiApplication, QOpenGLContext
app = QGuiApplication([])
raise SystemExit(0 if QOpenGLContext().create() else 1)
"""


def test_the_canvases_draw_with_gl():
    env = {**os.environ, "QT_QPA_PLATFORM": "xcb", "QT_API": "pyside6"}
    if "DISPLAY" not in env or subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True).returncode:
        pytest.skip("DISPLAY has no GLX")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(HERE / "gl_render.py")],
        env=env,
        cwd=HERE.parent,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
