"""A group's progress, computed from the project each time it is shown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from lamino_pal import LaminoPalError, dispatch
from lamino_pal.ops.registry import REGISTRY
from lamino_pal.project import Decision, Group, Project

FIELDS = ("center_px", "tilt_deg", "z_band")


def latest(group: Group, field: str) -> Decision | None:
    return next((d for d in reversed(group.decisions) if d.field == field), None)


def source(group: Group, field: str) -> str | None:
    """How `field` was decided: None while it is not, "file" when no decision
    records its value, "agent" for an agent's still to be confirmed, else its
    method or "user"."""
    value = getattr(group, field)
    if value is None:
        return None
    d = latest(group, field)
    if d is None or d.value != value:
        return "file"
    if d.by == "agent":
        return "agent"
    return d.method or "user"


def unconfirmed(group: Group) -> list[Decision]:
    """The agent's decisions that still stand."""
    return [d for f in FIELDS if source(group, f) == "agent" for d in [latest(group, f)]]


def finals(project: Project, group: Group) -> tuple[int, int]:
    """How many of the group's scans have their final volume, of how many: a
    scan `recon.run` refuses to name one for has none."""
    tags = [s.tag for s in project.data.scan if s.group == group.name]
    done = 0
    for t in tags:
        try:
            p = dispatch.resolve(REGISTRY["recon.run"], {"tag": t}, project)
        except (LaminoPalError, ValidationError):
            continue
        done += Path(p.out).exists()
    return done, len(tags)


@dataclass(frozen=True)
class GroupStatus:
    center: str | None
    tilt: str | None
    z_band: str | None
    finals: tuple[int, int]
    next_tab: str


def status(project: Project, group: Group | None) -> GroupStatus:
    if group is None:
        return GroupStatus(None, None, None, (0, 0), "Data")
    s = {f: source(group, f) for f in FIELDS}
    n, m = finals(project, group)
    if m == 0:
        nxt = "Data"
    elif s["center_px"] is None or s["tilt_deg"] is None:
        nxt = "Align"
    elif s["z_band"] is None:
        # The band is read off a volume at the decided geometry.
        nxt = "Volume" if n else "Batch"
    elif n < m:
        nxt = "Batch"
    else:
        nxt = "Export"
    return GroupStatus(s["center_px"], s["tilt_deg"], s["z_band"], (n, m), nxt)
