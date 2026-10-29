"""A project scan as the analysis ops read it: its file, angles and stack."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from lamino_pal import Refused, _core
from lamino_pal.project import Group, Project, Scan


def find(project: Project, tag: str) -> tuple[Scan, Path]:
    """Scan `tag` as the project file has it now, and its h5."""
    s = project.reload().find_scan(tag)
    if s is None:
        raise Refused(f"no scan {tag}")
    path = project.work_dir / s.h5
    if not path.exists():
        raise Refused(f"scan {tag}: {path} does not exist")
    return s, path


def group_of(project: Project, scan: Scan) -> Group | None:
    return next((g for g in project.data.group if g.name == scan.group), None)


def theta_deg(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["exchange/theta"][()].astype("f8")


@dataclass(frozen=True)
class Turn:
    """What a scan's angles cover."""

    nproj: int
    first_deg: float
    last_deg: float
    step_deg: float  # median
    step_spread_deg: float  # largest |step - median|
    kind: Literal["full", "half", "partial"]
    turn_index: int | None  # the projection at first + 360, if any
    warnings: list[str]


def turn(path: Path) -> Turn:
    t = theta_deg(path)
    if t.size < 2:
        raise Refused(f"{path}: {t.size} angles")
    d = np.diff(t)
    step = float(np.median(d))
    span = float(t[-1] - t[0])
    warnings = []
    if (d <= 0).any():
        warnings.append(f"angles do not increase at projection {int(np.argmax(d <= 0)) + 1}")
    spread = float(np.abs(d - step).max())
    if spread > step / 2:
        warnings.append(f"uneven steps: one is {spread:.3g} deg off the median {step:.3g}")

    # A turn is covered with or without its closing frame: span is 360 - step
    # (fly, endpoint excluded) or 360 (step scans repeat 0 deg at 360).
    def covers(r: float) -> bool:
        return min(abs(span - r), abs(span + step - r)) <= step / 2

    kind = "full" if covers(360) else "half" if covers(180) else "partial"
    hit = np.flatnonzero(np.abs(t - (t[0] + 360)) <= step / 2)
    return Turn(
        nproj=int(t.size),
        first_deg=float(t[0]),
        last_deg=float(t[-1]),
        step_deg=step,
        step_spread_deg=spread,
        kind=kind,
        turn_index=int(hit[0]) if hit.size else None,
        warnings=warnings,
    )


_lock = threading.Lock()
_cached: tuple[tuple[str, int, int], _core.Stack] | None = None


def stack(path: Path) -> _core.Stack:
    """The prepped stack of `path`, kept for the next op on the same file; one
    at a time, as a bin-2 stack is 4 GB."""
    global _cached
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = None  # drop the old one before loading the next
            _cached = (key, _core.Stack(path))
        return _cached[1]
