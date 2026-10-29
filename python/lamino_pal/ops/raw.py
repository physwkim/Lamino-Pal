"""Raw projections: look at them, and check the scan before aligning it.

Detector positions and distances in results are full-resolution pixels, as a
group's center is; arrays stay at the scan's binning.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from lamino_pal import Refused, _core, scans, units
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import Params, Result, op
from lamino_pal.render import window


class TagParams(Params):
    tag: str


class FrameParams(TagParams):
    index: int = Field(0, ge=0)


class FrameResult(Result):
    index: int
    theta_deg: float
    shape: tuple[int, int]
    window: tuple[float, float]  # p0.5, p99.5


@op("raw.frame", cost="instant", params=FrameParams, result=FrameResult)
def frame(ctx: Context, p: FrameParams) -> FrameResult:
    """Projection `index` of scan `tag`, flat/dark corrected and −log, with its
    angle and a p0.5–p99.5 display window. Array: `frame`."""
    _, path = scans.find(ctx.project, p.tag)
    a = _core.prep_frames(path, [p.index])[0]
    return FrameResult(
        index=p.index,
        theta_deg=float(scans.theta_deg(path)[p.index]),
        shape=a.shape,
        window=window(a),
    ).with_arrays(frame=a)


class MeanParams(TagParams):
    step: int = Field(10, ge=1)  # every `step`th projection


class MeanResult(Result):
    n: int  # projections averaged
    shape: tuple[int, int]
    window: tuple[float, float]


@op("raw.mean_projection", cost="interactive", params=MeanParams, result=MeanResult)
def mean_projection(ctx: Context, p: MeanParams) -> MeanResult:
    """The mean of every `step`th −log projection of scan `tag`; over a full
    turn its rings are centred on the rotation axis. Array: `mean`."""
    _, path = scans.find(ctx.project, p.tag)
    a = _core.mean_projection(path, p.step)
    n = len(range(0, scans.turn(path).nproj, p.step))
    return MeanResult(n=n, shape=a.shape, window=window(a)).with_arrays(mean=a)


class RangeResult(Result):
    nproj: int
    first_deg: float
    last_deg: float
    step_deg: float
    step_spread_deg: float
    kind: Literal["full", "half", "partial"]
    turn_index: int | None
    warnings: list[str]


@op("raw.range_check", cost="instant", params=TagParams, result=RangeResult)
def range_check(ctx: Context, p: TagParams) -> RangeResult:
    """What scan `tag`'s angles cover: a full turn (with or without the frame
    at first + 360), a half turn or neither, with the median step and how far
    any step strays from it. A half turn's rings are open arcs and cannot give
    the center. The angles only: the reconstruction range sweep of the
    alignment guide §5 needs a known center and is not run here."""
    _, path = scans.find(ctx.project, p.tag)
    return RangeResult(**scans.turn(path).__dict__)


class DriftResult(Result):
    first: int
    turn: int  # the projection one turn after `first`
    dy_px: float  # where `turn` sits relative to `first`, full resolution
    dx_px: float
    bin: int


@op("raw.drift", cost="interactive", params=TagParams, result=DriftResult)
def drift(ctx: Context, p: TagParams) -> DriftResult:
    """How far the sample moved over one turn of scan `tag`: the shift of the
    projection at first + 360 deg against the first, to 0.1 scan pixel.
    Refused for a scan without that projection. Arrays: `first`, `turn`."""
    scan, path = scans.find(ctx.project, p.tag)
    t = scans.turn(path)
    if t.turn_index is None:
        raise Refused(
            f"scan {p.tag} has no projection at {t.first_deg + 360:.2f} deg "
            f"(last {t.last_deg:.2f})"
        )
    a = _core.prep_frames(path, [0, t.turn_index])
    # register_shift moves `moving` onto `reference`: the drift is its negative.
    dy, dx = _core.register_shift(a[0], a[1], 10.0)
    return DriftResult(
        first=0,
        turn=t.turn_index,
        dy_px=-dy * scan.bin,
        dx_px=-dx * scan.bin,
        bin=scan.bin,
    ).with_arrays(first=a[0], turn=a[1])


class FlowPair(BaseModel):
    first: int
    gap: int
    tilt_deg: float
    roll_deg: float
    axis_px: float  # full resolution
    resid_px: float  # full resolution


class TiltFlowResult(Result):
    tilt_deg: float
    tilt_sigma_deg: float
    roll_deg: float
    roll_sigma_deg: float
    axis_px: float  # full-resolution column
    axis_sigma_px: float
    pairs: list[FlowPair]
    group_tilt_deg: float | None  # the scan's group's decided tilt


@op("raw.tilt_flow", cost="interactive", params=TagParams, result=TiltFlowResult)
def tilt_flow(ctx: Context, p: TagParams) -> TiltFlowResult:
    """The axis tilt read off the optical flow between neighbouring
    projections of scan `tag`, with its spread over the projection pairs, and
    the group's decided tilt to compare. A cross-check: the center sweep and
    the tilt scan decide."""
    scan, path = scans.find(ctx.project, p.tag)
    f = json.loads(scans.stack(path).tilt_flow())
    b = scan.bin
    group = scans.group_of(ctx.project, scan)
    return TiltFlowResult(
        tilt_deg=f["tilt_deg"],
        tilt_sigma_deg=f["tilt_sigma_deg"],
        roll_deg=f["roll_deg"],
        roll_sigma_deg=f["roll_sigma_deg"],
        axis_px=units.center_full(scan, f["axis_col"]),
        axis_sigma_px=f["axis_col_sigma"] * b,
        pairs=[
            FlowPair(
                first=q["first"],
                gap=q["gap"],
                tilt_deg=q["tilt_deg"],
                roll_deg=q["roll_deg"],
                axis_px=units.center_full(scan, q["axis_col"]),
                resid_px=q["resid_px"] * b,
            )
            for q in f["pairs"]
        ],
        group_tilt_deg=group.tilt_deg if group else None,
    )
