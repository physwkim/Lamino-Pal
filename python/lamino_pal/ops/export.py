"""A volume's exports, as the 261003 scripts made them: a 16-bit TIFF of a z
range (export_tiff.py), a 2×2×2 mean-binned volume for the 3D view
(export_bin.py's rule) and the maximum intensity projection along each axis.

numpy and tifffile made the scripts' files, so these use them too, reading the
volume in bands of about `BAND_BYTES`: a bin-1 volume is 36.5 GB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, Iterator

import h5py
import numpy as np
import tifffile
from pydantic import AfterValidator

from lamino_pal import Refused, scans, units
from lamino_pal.artifacts import save_h5, written
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import InPath, OptOutPath, OutputParams, Params, Result, op
from lamino_pal.ops.volume import data, provenance
from lamino_pal.project import Project

BAND_BYTES = 256 << 20  # float32 read per band
SAMPLE = 8  # every SAMPLE-th slice sets the window, as export_tiff.py
HIST_BINS = 256


def _range(z: tuple[int, int]) -> tuple[int, int]:
    if not 0 <= z[0] < z[1]:
        raise ValueError(f"z {z} is not [z0, z1) with 0 <= z0 < z1")
    return z


ZRange = Annotated[tuple[int, int], AfterValidator(_range)]


class RangeParams(Params):
    volume: InPath  # h5 with exchange/data [z, y, x]
    z: ZRange | None = None  # slices [z0, z1); default: the group's z_band


def _bind_z(p: RangeParams, project: Project) -> RangeParams:
    """`z` from the z_band of the group of the scan `volume` was made from."""
    path = project.work_dir / p.volume
    if p.z is not None or not path.exists():
        # The dispatcher refuses a missing volume.
        return p
    prov = provenance(path)
    if "tag" not in prov:
        raise Refused(f"{p.volume} has no provenance naming its scan: pass z")
    scan, _ = scans.find(project, prov["tag"])
    group = scans.group_of(project, scan)
    zb = group.z_band if group else None
    if zb is None:
        raise Refused(f"scan {scan.tag} has no group z_band: pass z")
    if "bin" not in prov or "tilt_deg" not in prov:
        raise Refused(f"{p.volume} has no provenance naming its bin and tilt: pass z")
    z0, z1 = units.band_for(zb, prov["bin"], prov["tilt_deg"], data(path).shape[0])
    # The band is inclusive.
    return p.model_copy(update={"z": (z0, z1 + 1)})


def _within(z: tuple[int, int], nz: int) -> tuple[int, int]:
    if z[1] > nz:
        raise Refused(f"slices [{z[0]}, {z[1]}) are outside the {nz}-slice volume")
    return z


def _band(slice_bytes: int) -> int:
    return max(1, BAND_BYTES // slice_bytes)


def window_of(path: str | Path, z: tuple[int, int]) -> tuple[float, float, np.ndarray]:
    """export_tiff.py's window of slices [z0, z1) of volume `path`: p0.1 and
    p99.9 of every `SAMPLE`-th slice; with that sample."""
    d = data(path)
    z0, z1 = _within(z, d.shape[0])
    sample = np.asarray(d[z0:z1:SAMPLE], dtype=np.float32)
    lo, hi = np.percentile(sample, [0.1, 99.9])
    if not lo < hi:
        raise Refused(f"slices [{z0}, {z1}) of {path} are flat: p0.1 = p99.9 = {lo}")
    return float(lo), float(hi), sample


class WindowResult(Result):
    z: tuple[int, int]
    window: tuple[float, float]  # p0.1–p99.9 of every 8th slice
    span: tuple[float, float]  # the histogram's: the window widened by half its width each side
    hist: list[int]  # of the sample's voxels, HIST_BINS bins over span


@op("export.window", cost="interactive", params=RangeParams, result=WindowResult, bind=_bind_z)
def window(ctx: Context, p: RangeParams) -> WindowResult:
    """The window `export.tiff16` maps slices [z0, z1) of volume `volume` to
    0–65535 with (p0.1–p99.9 of every 8th slice), and the histogram of those
    slices around it."""
    ctx.progress(0.0, f"window: every {SAMPLE}th slice")
    lo, hi, sample = window_of(p.volume, p.z)
    span = (lo - (hi - lo) / 2, hi + (hi - lo) / 2)
    hist, _ = np.histogram(sample, HIST_BINS, span)
    return WindowResult(z=p.z, window=(lo, hi), span=span, hist=hist.tolist())


class TiffParams(RangeParams, OutputParams):
    out: OptOutPath = None  # default: <volume stem>_z<z0>-<z1>_16bit.tif


class TiffResult(Result):
    z: tuple[int, int]
    window: tuple[float, float]  # the floats 0 and 65535 stand for
    shape: tuple[int, int, int]  # of the TIFF stack


def _tiff_out(p: TiffParams, _: Project) -> dict[str, str]:
    return {"out": f"{Path(p.volume).stem}_z{p.z[0]}-{p.z[1]}_16bit.tif"}


@op("export.tiff16", cost="job", params=TiffParams, result=TiffResult, default_out=_tiff_out, bind=_bind_z)
def tiff16(ctx: Context, p: TiffParams) -> TiffResult:
    """Slices [z0, z1) of volume `volume` as one 16-bit BigTIFF stack `out`,
    the window (`export.window`) mapped to 0–65535 and clipped, as
    export_tiff.py wrote it: page 0's description holds z_range and window,
    so the floats can be recovered."""
    ctx.progress(0.0, f"window: every {SAMPLE}th slice")
    lo, hi, _ = window_of(p.volume, p.z)
    d = data(p.volume)
    (z0, z1), (_, ny, nx) = p.z, d.shape
    # export_tiff.py's steps, all float32 under numpy 1.26's value-based casting.
    lo32, scale = np.float32(lo), np.float32(hi - lo)
    k = _band(4 * ny * nx)

    def bands():
        for a in range(z0, z1, k):
            b = min(z1, a + k)
            v = np.asarray(d[a:b], dtype=np.float32)
            yield np.clip((v - lo32) / scale * np.float32(65535), 0, 65535).astype(np.uint16)
            ctx.progress((b - z0) / (z1 - z0), f"slice {b - z0}/{z1 - z0}")

    shape = (z1 - z0, ny, nx)
    with written(p.out) as part, tifffile.TiffWriter(part, bigtiff=True) as tw:
        meta = {"z_range": [z0, z1], "window": [lo, hi]}
        tw.write(bands(), shape=shape, dtype=np.uint16, metadata=meta)
    return TiffResult(z=p.z, window=(lo, hi), shape=shape, artifacts=[p.out])


class BinParams(OutputParams):
    volume: InPath  # h5 with exchange/data [z, y, x], or a 16-bit TIFF stack
    out: OptOutPath = None  # default: <volume stem>_binned.h5, or .tif of a TIFF


class BinResult(Result):
    shape: tuple[int, int, int]


def _tiff(path: str | Path) -> bool:
    return Path(path).suffix.lower() in (".tif", ".tiff")


def _bin_out(p: BinParams, _: Project) -> dict[str, str]:
    return {"out": f"{Path(p.volume).stem}_binned{'.tif' if _tiff(p.volume) else '.h5'}"}


def _half(p: BinParams, shape: tuple[int, ...]) -> tuple[int, int, int]:
    nz, ny, nx = (n // 2 for n in shape)
    if not (nz and ny and nx):
        raise Refused(f"{p.volume} is {'x'.join(map(str, shape))}: too thin to bin 2x2x2")
    return nz, ny, nx


def _binned(
    ctx: Context, half: tuple[int, int, int], read: Callable[[int, int], Any]
) -> Iterator[tuple[int, int, np.ndarray]]:
    """export_bin.py's 2×2×2 means in float32 of the [z, y, x] stack whose
    slices [z0, z1) `read(z0, z1)` reads, a band [a, b) at a time."""
    nz, ny, nx = half
    k = _band(4 * 8 * ny * nx)
    for a in range(0, nz, k):
        b = min(nz, a + k)
        v = np.asarray(read(2 * a, 2 * b)[:, : 2 * ny, : 2 * nx], dtype=np.float32)
        yield a, b, v.reshape(b - a, 2, ny, 2, nx, 2).mean((1, 3, 5))
        ctx.progress(b / nz, f"slice {b}/{nz}")


@op("export.bin", cost="job", params=BinParams, result=BinResult, default_out=_bin_out)
def bin_(ctx: Context, p: BinParams) -> BinResult:
    """Volume `volume` 2×2×2 mean binned, an odd extent's last slice, row or
    column dropped, into `out`. Of an h5 volume, a float32 h5 volume: the
    bin-4 volume the 3D view loads of a bin-2 final, its provenance the
    source's with bin doubled and binned_from naming the source. Of a 16-bit
    TIFF stack (`export.tiff16`'s), a 16-bit ImageJ TIFF of the means
    truncated, as export_bin.py wrote the 261003 bin-4 stacks."""
    if _tiff(p.volume):
        with tifffile.TiffFile(p.volume) as t:
            s = t.series[0]
            if s.dtype != np.uint16 or s.ndim != 3:
                raise Refused(f"{p.volume} is a {s.dtype} {'x'.join(map(str, s.shape))} TIFF, not a 16-bit stack")
            half = _half(p, s.shape)
            bands = (v.astype(np.uint16) for _, _, v in _binned(ctx, half, lambda a, b: t.asarray(key=range(a, b))))
            with written(p.out) as part:
                tifffile.imwrite(
                    part, bands, shape=half, dtype=np.uint16, imagej=True, metadata={"axes": "ZYX"}, compression=None
                )
        return BinResult(shape=half, artifacts=[p.out])
    d = data(p.volume)
    half = _half(p, d.shape)
    prov = provenance(p.volume)
    if "bin" in prov:
        prov["bin"] *= 2
    with written(p.out) as part, h5py.File(part, "w") as f:
        out = f.create_dataset("exchange/data", half, np.float32)
        for a, b, v in _binned(ctx, half, lambda a, b: d[a:b]):
            out[a:b] = v
        f.create_group("lamino_pal/provenance").attrs.update({**prov, "binned_from": str(p.volume)})
    return BinResult(shape=half, artifacts=[p.out])


class MipParams(RangeParams, OutputParams):
    out: OptOutPath = None  # default: artifacts/<volume stem>_z<z0>-<z1>_mip.h5


class MipResult(Result):
    z: tuple[int, int]
    shape: tuple[int, int, int]  # of the volume


def _mip_out(p: MipParams, _: Project) -> dict[str, str]:
    return {"out": f"artifacts/{Path(p.volume).stem}_z{p.z[0]}-{p.z[1]}_mip.h5"}


@op("export.mip", cost="job", params=MipParams, result=MipResult, default_out=_mip_out, bind=_bind_z)
def mip(ctx: Context, p: MipParams) -> MipResult:
    """The maximum intensity projection of slices [z0, z1) of volume `volume`
    along each axis, saved to h5 `out`. Arrays: `z` [y, x] down the stack,
    `y` [z, x] and `x` [z, y] through it."""
    d = data(p.volume)
    z0, z1 = _within(p.z, d.shape[0])
    _, ny, nx = d.shape
    mz = np.full((ny, nx), -np.inf, np.float32)
    my, mx = np.empty((z1 - z0, nx), np.float32), np.empty((z1 - z0, ny), np.float32)
    k = _band(4 * ny * nx)
    for a in range(z0, z1, k):
        b = min(z1, a + k)
        v = np.asarray(d[a:b], dtype=np.float32)
        np.maximum(mz, v.max(0), out=mz)
        my[a - z0 : b - z0], mx[a - z0 : b - z0] = v.max(1), v.max(2)
        ctx.progress((b - z0) / (z1 - z0), f"slice {b - z0}/{z1 - z0}")
    r = MipResult(z=p.z, shape=d.shape, artifacts=[p.out])
    save_h5(p.out, r.model_dump(mode="json", exclude={"artifacts"}), z=mz, y=my, x=mx)
    return r.with_arrays(z=mz, y=my, x=mx)
