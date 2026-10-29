"""Alignment, in the order of tomoxide's `docs/LAMINOGRAPHY_ALIGNMENT.md`:
rings estimate, center sweep, tilt scan by eye.

Centers in params and results are full-resolution detector columns, as a
group stores them; slices and z rows are the scan's own reconstruction rows.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from lamino_pal import Refused, _core, scans, units
from lamino_pal.artifacts import save_h5
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import OptOutPath, OutputParams, Params, Result, op
from lamino_pal.project import Group, Project, Scan

DEFAULT_TILT_DEG = 20.0


def _tilt(group: Group | None, tilt_deg: float | None) -> float:
    if tilt_deg is not None:
        return tilt_deg
    if group is not None and group.tilt_deg is not None:
        return group.tilt_deg
    return DEFAULT_TILT_DEG


def _band(
    stack: _core.Stack, scan: Scan, group: Group | None, tilt: float | None = None
) -> tuple[tuple[int, int], float] | None:
    """The group's sample band as rows of this scan's volume at `tilt`, by
    default the tilt it was read at, and that tilt."""
    zb = group.z_band if group else None
    if zb is None:
        return None
    t = zb.tilt_deg if tilt is None else tilt
    return units.band_for(zb, scan.bin, t, stack.depth(t)), t


def _slice(stack: _core.Stack, scan: Scan, group: Group | None, tilt: float, slice: int | None) -> int:
    if slice is not None:
        return slice
    band = _band(stack, scan, group, tilt)
    return stack.default_slice(tilt, None if band is None else band[0])


class RingsParams(Params):
    tag: str
    step: int = Field(10, ge=1)  # every `step`th projection
    bands: int = Field(4, ge=1)  # row bands checked on their own


class RingBand(BaseModel):
    rows: tuple[int, int]  # [r0, r1) of the scan's detector rows
    center_px: float
    prominence: float


class RingsResult(Result):
    center_px: float
    prominence: float
    trustworthy: bool
    bands: list[RingBand]
    band_spread_px: float  # largest minus smallest band center


@op("align.rings", cost="interactive", params=RingsParams, result=RingsResult)
def rings(ctx: Context, p: RingsParams) -> RingsResult:
    """The rotation axis read off the rings of the mean projection over one
    full turn, with its prominence (trustworthy at 8 or more) and the center
    each row band gives on its own. A first estimate for the center sweep,
    not a decision. Refused for a scan short of a full turn: its rings are
    open arcs (the 261003 half turns read 462–467 against 504 at bin 2).
    Array: `mean`."""
    scan, path = scans.find(ctx.project, p.tag)
    t = scans.turn(path)
    if t.kind != "full":
        raise Refused(
            f"scan {p.tag} is a {t.kind} turn: its rings are open arcs and miss the "
            f"axis; use the center of a full-turn scan of its mount"
        )
    mean = _core.mean_projection(path, p.step)
    whole = json.loads(_core.rings_from_mean(mean))
    bands = []
    for rows in np.array_split(np.arange(mean.shape[0]), p.bands):
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        b = json.loads(_core.rings_from_mean(mean[r0:r1]))
        bands.append(
            RingBand(
                rows=(r0, r1),
                center_px=units.center_full(scan, b["center"]),
                prominence=b["prominence"],
            )
        )
    c = [b.center_px for b in bands]
    return RingsResult(
        center_px=units.center_full(scan, whole["center"]),
        prominence=whole["prominence"],
        trustworthy=whole["trustworthy"],
        bands=bands,
        band_spread_px=max(c) - min(c),
    ).with_arrays(mean=mean)


class SweepParams(Params):
    tag: str
    center_px: float | None = None  # default: the group's, else the rings estimate
    half_width_px: float = Field(24.0, ge=0)
    step_px: float = Field(1.0, gt=0)
    tilt_deg: float | None = None  # default: the group's, else 20
    slice: int | None = Field(None, ge=0)  # default: the group's z_band middle


class Rival(BaseModel):
    index: int
    center_px: float
    prominence: float
    deficit: float  # its focus below the winner's, relative


class Verdict(BaseModel):
    kind: Literal["resolved", "railed", "ambiguous", "flat"]
    index: int
    center_px: float
    rivals: list[Rival]


class SweepResult(Result):
    center_from: Literal["params", "group", "rings"]
    tilt_deg: float
    slice: int
    depth: int
    candidates_px: list[float]
    focus: list[float]
    verdict: Verdict


def _verdict(scan: Scan, v: dict) -> Verdict:
    full = lambda c: units.center_full(scan, c)  # noqa: E731
    return Verdict(
        kind=v["kind"],
        index=v["index"],
        center_px=full(v["value"]),
        rivals=[
            Rival(index=r["index"], center_px=full(r["value"]), prominence=r["prominence"], deficit=r["deficit"])
            for r in v["rivals"]
        ],
    )


@op("align.center_sweep", cost="interactive", params=SweepParams, result=SweepResult, gpu=True)
def center_sweep(ctx: Context, p: SweepParams) -> SweepResult:
    """Reconstruct one slice of scan `tag` at every center of center_px ±
    half_width_px and judge the focus curve: resolved, railed (the peak is at
    the edge: widen), ambiguous (rivals close behind: compare them by eye) or
    flat (the slice holds no sample: check the z band). Centers are
    full-resolution columns; the grid is cut in the scan's own pixels.
    Array: `frames`, one slice per candidate."""
    scan, path = scans.find(ctx.project, p.tag)
    group = scans.group_of(ctx.project, scan)
    tilt = _tilt(group, p.tilt_deg)
    if p.center_px is not None:
        c, src = p.center_px, "params"
    elif group is not None and group.center_px is not None:
        c, src = group.center_px, "group"
    else:
        ctx.progress(0.0, "rings estimate")
        c, src = rings(ctx, RingsParams(tag=p.tag)).center_px, "rings"
    b = scan.bin
    cands = _core.grid(units.center_for_scan(scan, c), p.half_width_px / b, p.step_px / b)
    ctx.progress(0.1, "loading the stack")
    stack = scans.stack(path)
    s = _slice(stack, scan, group, tilt, p.slice)
    ctx.progress(0.3, f"{len(cands)} candidates on slice {s}")
    frames, focus = stack.center_sweep(tilt, s, cands)
    return SweepResult(
        center_from=src,
        tilt_deg=tilt,
        slice=s,
        depth=stack.depth(tilt),
        candidates_px=[units.center_full(scan, x) for x in cands],
        focus=focus,
        verdict=_verdict(scan, json.loads(_core.judge_sweep(cands, focus))),
    ).with_arrays(frames=frames)


def _decided_center(scan: Scan, group: Group | None, center_px: float | None) -> float:
    if center_px is not None:
        return center_px
    if group is None or group.center_px is None:
        raise Refused(f"scan {scan.tag} has no decided center: pass center_px")
    return group.center_px


def _stem(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


class TiltScanParams(OutputParams):
    tag: str
    center_px: float | None = None  # default: the group's decided center
    tilts_deg: list[float] = Field(default_factory=lambda: [16.0, 18.0, 20.0, 22.0, 24.0], min_length=1)
    out: OptOutPath = None  # default artifacts/<tag>_tilts_c<center>_t<lo>-<hi>.h5


class TiltFocus(BaseModel):
    tilt_deg: float
    focus: float
    z_peak: int
    band: tuple[int, int] | None
    depth: int
    focus_by_z: list[float]


class TiltScanResult(Result):
    center_px: float
    tilts: list[TiltFocus]


def _tilt_scan_out(p: TiltScanParams, _: Project) -> dict[str, str]:
    c = "group" if p.center_px is None else _stem(p.center_px)
    t = p.tilts_deg
    return {"out": f"artifacts/{p.tag}_tilts_c{c}_t{_stem(min(t))}-{_stem(max(t))}.h5"}


@op(
    "align.tilt_scan",
    cost="job",
    params=TiltScanParams,
    result=TiltScanResult,
    gpu=True,
    default_out=_tilt_scan_out,
)
def tilt_scan(ctx: Context, p: TiltScanParams) -> TiltScanResult:
    """Reconstruct scan `tag` fully at each tilt and keep each one's best
    slice, inside the group's z band carried to that tilt, and its xz plane
    through the middle row. Decide the tilt by eye on the slices and xz planes
    in `out`: the focus value grows with tilt on the 261003 sample and does not
    pick it. Arrays in `out`: `best` [tilt, y, x], `xz` [tilt, z, x], each
    tilt's first `depth` rows and NaN below them."""
    scan, path = scans.find(ctx.project, p.tag)
    group = scans.group_of(ctx.project, scan)
    c = _decided_center(scan, group, p.center_px)
    ctx.progress(0.0, "loading the stack")
    stack = scans.stack(path)
    band = _band(stack, scan, group)
    best: list[np.ndarray] = []
    xz: list[np.ndarray] = []

    def on_tilt(f: str, slice: np.ndarray, plane: np.ndarray) -> None:
        best.append(slice)
        xz.append(plane)
        f = json.loads(f)
        ctx.progress(len(best) / len(p.tilts_deg), f"tilt {f['tilt_deg']:g}: z {f['z_peak']}")

    foci = json.loads(stack.tilt_scan(units.center_for_scan(scan, c), p.tilts_deg, band, on_tilt))
    r = TiltScanResult(center_px=c, tilts=[TiltFocus(**f) for f in foci], artifacts=[p.out])
    stacked = np.stack(best)
    planes = np.full((len(xz), max(len(x) for x in xz), stacked.shape[2]), np.nan, np.float32)
    for x, plane in zip(planes, xz):
        x[: len(plane)] = plane
    save_h5(p.out, r.model_dump(mode="json", exclude={"artifacts"}), best=stacked, xz=planes)
    return r.with_arrays(best=stacked, xz=planes)


class CompareParams(OutputParams):
    tag: str
    centers_px: list[float] = Field(min_length=2)
    tilt_deg: float | None = None  # default: the group's, else 20
    out: OptOutPath = None  # default artifacts/<tag>_centers_<c0>-<c1>_t<tilt>.h5


class CenterFocus(BaseModel):
    center_px: float
    focus: float
    z_peak: int
    band: tuple[int, int] | None
    focus_by_z: list[float]


class CompareResult(Result):
    tilt_deg: float
    centers: list[CenterFocus]


def _compare_out(p: CompareParams, _: Project) -> dict[str, str]:
    c = p.centers_px
    t = "group" if p.tilt_deg is None else _stem(p.tilt_deg)
    return {"out": f"artifacts/{p.tag}_centers_{_stem(min(c))}-{_stem(max(c))}_t{t}.h5"}


@op(
    "align.center_compare",
    cost="job",
    params=CompareParams,
    result=CompareResult,
    gpu=True,
    default_out=_compare_out,
)
def center_compare(ctx: Context, p: CompareParams) -> CompareResult:
    """Reconstruct scan `tag` fully at each center of `centers_px` and keep
    each one's best slice inside the group's z band, to compare by eye the
    candidates a sweep left (an ambiguous verdict's rivals), each image on its
    own p0.5–p99.5 window. Arrays in `out`: `best` [center, y, x]."""
    scan, path = scans.find(ctx.project, p.tag)
    group = scans.group_of(ctx.project, scan)
    tilt = _tilt(group, p.tilt_deg)
    ctx.progress(0.0, "loading the stack")
    stack = scans.stack(path)
    band = _band(stack, scan, group)
    best: list[np.ndarray] = []
    centers = []
    for c in p.centers_px:
        f = json.loads(
            stack.tilt_scan(units.center_for_scan(scan, c), [tilt], band, lambda f, s, xz: best.append(s))
        )[0]
        centers.append(
            CenterFocus(center_px=c, focus=f["focus"], z_peak=f["z_peak"], band=f["band"], focus_by_z=f["focus_by_z"])
        )
        ctx.progress(len(centers) / len(p.centers_px), f"center {c:g}: z {f['z_peak']}")
    r = CompareResult(tilt_deg=tilt, centers=centers, artifacts=[p.out])
    stacked = np.stack(best)
    save_h5(p.out, r.model_dump(mode="json", exclude={"artifacts"}), best=stacked)
    return r.with_arrays(best=stacked)
