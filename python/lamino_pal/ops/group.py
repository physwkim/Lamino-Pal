"""Mount group decisions: center, tilt and sample band, each with its evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from lamino_pal import Refused
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import Params, Result, op
from lamino_pal.project import Decision, ProjectFile, ZBand, is_under


class DecideParams(Params):
    group: str
    field: Literal["center_px", "tilt_deg", "z_band"]
    value: float | ZBand  # center_px full resolution; z_band rows of a bin-`bin` volume
    evidence: list[str] = Field(min_length=1)  # files under the work directory
    note: str = ""


class DecideResult(Result):
    previous: float | ZBand | None
    by: Literal["user", "agent"]


@op("group.decide", cost="instant", params=DecideParams, result=DecideResult)
def decide(ctx: Context, p: DecideParams) -> DecideResult:
    """Set mount group `group`'s center_px, tilt_deg or z_band and record the
    decision with its evidence: the images or artifacts it was read from, under
    the work directory. A decision made over MCP is recorded as the agent's,
    for a person to confirm."""
    if (p.field == "z_band") != isinstance(p.value, ZBand):
        raise Refused(f"{p.field} takes {'a z band' if p.field == 'z_band' else 'a number'}")
    wd = ctx.project.work_dir
    evidence = []
    for e in p.evidence:
        path = (wd / e).resolve()
        if not is_under(path, wd) or not path.exists():
            raise Refused(f"evidence {e}: no such file under the work directory {wd}")
        evidence.append(str(path.relative_to(wd.resolve())))
    by = "agent" if ctx.caller == "mcp" else "user"
    previous = None

    def mutate(f: ProjectFile) -> None:
        nonlocal previous
        g = next((g for g in f.group if g.name == p.group), None)
        if g is None:
            raise Refused(f"no group {p.group}")
        previous = getattr(g, p.field)
        setattr(g, p.field, p.value)
        g.decisions.append(Decision(field=p.field, value=p.value, by=by, evidence=evidence, note=p.note))

    ctx.project.commit(mutate)
    return DecideResult(previous=previous, by=by)
