"""A reconstructed volume: a slice, the planes through a voxel, the focus of
every slice and the field of view every projection saw.

A volume is an h5 with `exchange/data` [z, y, x], as `tomoxide recon` and
`recon.run` write it; `recon.run`'s also carries `/lamino_pal/provenance`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import Field

from lamino_pal import Refused, _core, scans, units
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import InPath, Params, Result, op
from lamino_pal.render import window

_lock = threading.Lock()
_cached: tuple[tuple[str, int, int, int], np.ndarray | h5py.Dataset] | None = None


def data(path: str | Path) -> np.ndarray | h5py.Dataset:
    """`exchange/data` of volume `path`, kept for the next call on the same
    file. Mapped when contiguous, as tomoxide writes it: a yz plane then costs
    its page faults once per process instead of an h5py read every time (bin
    1: 1.5 s, then 0.05 s, against 5 s each)."""
    global _cached
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size, st.st_ino)
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = None
            f = h5py.File(path, "r")
            d = f.get("exchange/data")
            if not isinstance(d, h5py.Dataset) or d.ndim != 3:
                f.close()
                raise Refused(f"{path} has no 3-D exchange/data")
            offset = d.id.get_offset()
            if offset is not None:
                d = np.memmap(path, dtype=d.dtype, mode="r", offset=offset, shape=d.shape)
                f.close()
            _cached = (key, d)
        return _cached[1]


def provenance(path: str | Path) -> dict[str, Any]:
    """`/lamino_pal/provenance` of volume `path`, empty when it has none."""
    with h5py.File(path, "r") as f:
        g = f.get("lamino_pal/provenance")
        return {} if g is None else {k: v.item() if hasattr(v, "item") else v for k, v in g.attrs.items()}


def _within(what: str, i: int | None, n: int, of: str) -> int:
    if i is None:
        return n // 2
    if i >= n:
        raise Refused(f"{what} {i} is outside the {n}-{of} volume")
    return i


class VolumeParams(Params):
    volume: InPath  # h5 with exchange/data [z, y, x]


class SliceParams(VolumeParams):
    z: int | None = Field(None, ge=0)  # default: the middle slice


class SliceResult(Result):
    shape: tuple[int, int, int]
    z: int
    window: tuple[float, float]  # p0.5–p99.5


@op("volume.slice", cost="instant", params=SliceParams, result=SliceResult)
def slice_(ctx: Context, p: SliceParams) -> SliceResult:
    """Slice z of volume `volume` and its p0.5–p99.5 display window.
    Array: `slice` [y, x]."""
    d = data(p.volume)
    z = _within("slice", p.z, d.shape[0], "slice")
    sl = np.asarray(d[z], dtype=np.float32)
    return SliceResult(shape=d.shape, z=z, window=window(sl)).with_arrays(slice=sl)


class OrthoParams(SliceParams):
    y: int | None = Field(None, ge=0)  # default: the middle row
    x: int | None = Field(None, ge=0)  # default: the middle column


class OrthoResult(Result):
    shape: tuple[int, int, int]
    at: tuple[int, int, int]  # (z, y, x) the planes cross at
    window: tuple[float, float]  # the slice's p0.5–p99.5


# Interactive, not instant: a yz plane reads a page from every row of the volume.
@op("volume.ortho", cost="interactive", params=OrthoParams, result=OrthoResult)
def ortho(ctx: Context, p: OrthoParams) -> OrthoResult:
    """The slice at z and the xz and yz planes through (z, y, x) of volume
    `volume`, the planes `recon.preview` shows of a volume not yet made.
    Arrays: `slice` [y, x], `xz` [z, x], `yz` [z, y]."""
    d = data(p.volume)
    nz, ny, nx = d.shape
    at = (_within("slice", p.z, nz, "slice"), _within("row", p.y, ny, "row"), _within("column", p.x, nx, "column"))
    z, y, x = at
    sl = np.asarray(d[z], dtype=np.float32)
    xz = np.asarray(d[:, y, :], dtype=np.float32)
    yz = np.asarray(d[:, :, x], dtype=np.float32)
    return OrthoResult(shape=d.shape, at=at, window=window(sl)).with_arrays(slice=sl, xz=xz, yz=yz)


BAND_WINDOW = 31  # slices of the running median the band is read from
CHUNK = 64  # slices read per focus call


def _runs(on: np.ndarray) -> np.ndarray:
    """[first, last] of each run of True in `on`."""
    e = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    return np.stack((np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1), axis=1)


def z_regions(
    focus: np.ndarray, step: int = 1
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], tuple[int, int] | None]:
    """The sample band, the air and the z wrap: slices (inclusive) of the
    focus median-filtered over `BAND_WINDOW` slices; `focus` holds every
    `step`-th slice's. The band is the longest run at least half the peak;
    air, what lies outside it within twice the least focus, none when that
    reaches half the peak. Fourier reconstruction's z is periodic, so a sample
    the volume cuts at one end shows at the other: the wrap is the run sharper
    than air at the far end when the band's run reaches the near one."""
    w = max(1, BAND_WINDOW // step) | 1
    s = np.median(sliding_window_view(np.pad(focus, w // 2, mode="edge"), w), axis=1)
    if not s.size or s.max() <= 0:
        return None, [], None

    def at(run) -> tuple[int, int]:
        return int(run[0]) * step, int(run[1]) * step

    on = _runs(s >= s.max() / 2)
    b0, b1 = on[np.argmax(on[:, 1] - on[:, 0])]
    floor = s.min()
    if 4 * floor >= s.max():
        return at((b0, b1)), [], None
    out = np.ones(s.size, bool)
    out[b0 : b1 + 1] = False
    air = [at(r) for r in _runs((s <= 2 * floor) & out)]
    sharp = _runs(s > 2 * floor)
    lo, hi = sharp[(sharp[:, 0] <= b0) & (sharp[:, 1] >= b1)][0]
    wrap = None
    if hi == s.size - 1 and lo > 0 and sharp[0, 0] == 0:
        wrap = at(sharp[0])
    elif lo == 0 and hi < s.size - 1 and sharp[-1, 1] == s.size - 1:
        wrap = at(sharp[-1])
    return at((b0, b1)), air, wrap


class FocusParams(VolumeParams):
    step: int = Field(1, ge=1)  # score every step-th slice


class FocusResult(Result):
    shape: tuple[int, int, int]
    step: int
    peak_z: int
    band: tuple[int, int] | None  # suggested sample slices, inclusive
    air: list[tuple[int, int]]  # slices outside the band within twice the least focus, inclusive
    wrap: tuple[int, int] | None  # slices the sample cut by the other end wraps into, inclusive


@op("volume.focus_by_z", cost="interactive", params=FocusParams, result=FocusResult)
def focus_by_z(ctx: Context, p: FocusParams) -> FocusResult:
    """The focus of every `step`-th slice of volume `volume` (mean |∇|² inside
    a 0.92-FOV disk, the align ops' score) drawn as a curve, a suggested
    sample band, the longest run of slices at least half as sharp as the
    sharpest stretch, and the air and the z wrap marked outside it
    (`z_regions`). Confirm the band by eye before `group.decide`."""
    d = data(p.volume)
    nz = d.shape[0]
    focus: list[float] = []
    for z0 in range(0, nz, CHUNK * p.step):
        z1 = min(nz, z0 + CHUNK * p.step)
        focus += _core.focus_by_z(np.ascontiguousarray(d[z0 : z1 : p.step], dtype=np.float32))
        ctx.progress(z1 / nz, f"slice {z1}/{nz}")
    f = np.array(focus)
    band, air, wrap = z_regions(f, p.step)
    return FocusResult(
        shape=d.shape,
        step=p.step,
        peak_z=int(np.argmax(f)) * p.step,
        band=band,
        air=air,
        wrap=wrap,
    ).with_arrays(focus=f)


class FovParams(VolumeParams):
    tag: str | None = None  # the scan it was made from; default: its provenance
    center_px: float | None = None  # full-resolution column; default: its provenance
    tilt_deg: float | None = None  # default: its provenance


class FovResult(Result):
    tag: str
    center_px: float
    tilt_deg: float
    r_lateral: float  # the radius the detector width allows
    full_z: tuple[int, int] | None  # the slices whose whole r_lateral disk is valid


@op("volume.valid_fov", cost="instant", params=FovParams, result=FovResult)
def valid_fov(ctx: Context, p: FovParams) -> FovResult:
    """The disk of volume `volume` every projection saw: what lies outside it
    is reconstructed from part of the turn. Its radius is at most `r_lateral`,
    reached on the slices `full_z`. The scan, center and tilt come from the
    volume's provenance unless given."""
    prov = provenance(p.volume)
    tag = p.tag if p.tag is not None else prov.get("tag")
    c = p.center_px if p.center_px is not None else prov.get("center_px")
    t = p.tilt_deg if p.tilt_deg is not None else prov.get("tilt_deg")
    if tag is None or c is None or t is None:
        raise Refused(f"{p.volume} has no provenance: pass tag, center_px and tilt_deg")
    scan, path = scans.find(ctx.project, tag)
    with h5py.File(path, "r") as f:
        _, h, w = f["exchange/data"].shape
    fov = json.loads(_core.valid_fov(w, h, units.center_for_scan(scan, c), t))
    nz, ny, nx = data(p.volume).shape
    if (nz, ny, nx) != (fov["depth"], w, w):
        raise Refused(
            f"{p.volume} is {nz}x{ny}x{nx}; scan {tag} at tilt {t} reconstructs {fov['depth']}x{w}x{w}"
        )
    return FovResult(
        tag=tag,
        center_px=c,
        tilt_deg=t,
        r_lateral=fov["r_lateral"],
        full_z=fov["full_z"],
    ).with_arrays(radius=np.array(fov["radius"]))
