"""Reconstruction: a preview, recipes side by side, the even/odd split and the
final volume.

A recipe is a table of tomoxide `Config` fields, read as `tomoxide recon
--config` reads its file; the project's `[recon]` is the default. Lamino-Pal
sets the scan, its geometry and the output itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import tomli_w
from pydantic import AfterValidator, BaseModel, Field, model_validator

import lamino_pal
from lamino_pal import LaminoPalError, Refused, _core, scans, units
from lamino_pal.artifacts import save_h5
from lamino_pal.dispatch import Context
from lamino_pal.ops.align import _decided_center, _slice, _stem, _tilt
from lamino_pal.ops.registry import OptOutPath, OutputParams, Params, Result, op
from lamino_pal.project import Group, Project, Scan
from lamino_pal.render import window

OWNED = ("file_name", "rotation_axis", "lamino_angle", "output", "save_format", "backend")


def _checked(recipe: dict[str, Any]) -> dict[str, Any]:
    owned = sorted(set(recipe) & set(OWNED))
    if owned:
        raise Refused(f"recipe: Lamino-Pal sets {owned[0]} itself")
    _core.recipe_config(json.dumps(recipe))
    return recipe


Recipe = Annotated[dict[str, Any], AfterValidator(_checked)]


def _rect(r: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = r
    if not (0 <= y0 < y1 and 0 <= x0 < x1):
        raise ValueError(f"rectangle {r} is not (y0, y1, x0, x1) with y0 < y1, x0 < x1")
    return r


Rect = Annotated[tuple[int, int, int, int], AfterValidator(_rect)]


class Setup(Params):
    tag: str
    center_px: float | None = None  # default: the group's decided center
    tilt_deg: float | None = None  # default: the group's, else 20
    recipe: Recipe | None = None  # default: the project's [recon]


class Where(Params):
    z: int | None = Field(None, ge=0)  # default: the group's z_band middle
    y: int | None = Field(None, ge=0)  # default: the middle row


def _bind(p: Setup, project: Project, tilt_decided: bool = False) -> Setup:
    scan, _ = scans.find(project, p.tag)
    group = scans.group_of(project, scan)
    upd: dict[str, Any] = {}
    if p.center_px is None:
        upd["center_px"] = _decided_center(scan, group, None)
    if p.tilt_deg is None:
        if tilt_decided and (group is None or group.tilt_deg is None):
            raise Refused(f"scan {scan.tag} has no decided tilt: pass tilt_deg")
        upd["tilt_deg"] = _tilt(group, None)
    if p.recipe is None:
        upd["recipe"] = _checked(dict(project.data.recon))
    return p.model_copy(update=upd)


def _stack(ctx: Context, tag: str) -> tuple[Scan, Group | None, _core.Stack]:
    scan, path = scans.find(ctx.project, tag)
    ctx.progress(0.0, "loading the stack")
    return scan, scans.group_of(ctx.project, scan), scans.stack(path)


def _at(
    stack: _core.Stack, scan: Scan, group: Group | None, p: Setup, z: int | None, y: int | None, x: int | None = None
) -> tuple[int, int, int]:
    n = stack.shape[2]
    z = _slice(stack, scan, group, p.tilt_deg, z)
    return z, n // 2 if y is None else y, n // 2 if x is None else x


def _planes(
    ctx: Context,
    stack: _core.Stack,
    scan: Scan,
    p: Setup,
    recipe: dict[str, Any],
    at: tuple[int, int, int],
    span: tuple[float, float],
    what: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The slice, xz and yz planes through `at`, reporting progress over
    `span` of the op's."""
    lo, hi = span

    def on_rows(done: int, rh: int) -> None:
        ctx.progress(lo + (hi - lo) * done / rh, f"{what}: row {done}/{rh}")

    c = units.center_for_scan(scan, p.center_px)
    return stack.planes(json.dumps(recipe), c, p.tilt_deg, at, on_rows)


class PreviewParams(Setup, Where):
    x: int | None = Field(None, ge=0)  # default: the middle column


class PreviewResult(Result):
    center_px: float
    tilt_deg: float
    recipe: dict[str, Any]
    depth: int
    at: tuple[int, int, int]  # (z, y, x) the planes cross at
    window: tuple[float, float]  # the slice's p0.5–p99.5


@op("recon.preview", cost="interactive", params=PreviewParams, result=PreviewResult, gpu=True, bind=_bind)
def preview(ctx: Context, p: PreviewParams) -> PreviewResult:
    """Reconstruct scan `tag` with the recipe, as `recon.run` will, and keep
    the slice at z and the xz and yz planes through (z, y, x): the volume a
    final run writes, three planes of it, in seconds on a loaded stack.
    Arrays: `slice` [y, x], `xz` [z, x], `yz` [z, y]."""
    scan, group, stack = _stack(ctx, p.tag)
    at = _at(stack, scan, group, p, p.z, p.y, p.x)
    sl, xz, yz = _planes(ctx, stack, scan, p, p.recipe, at, (0.05, 1.0), "preview")
    return PreviewResult(
        center_px=p.center_px,
        tilt_deg=p.tilt_deg,
        recipe=p.recipe,
        depth=xz.shape[0],
        at=at,
        window=window(sl),
    ).with_arrays(slice=sl, xz=xz, yz=yz)


DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {},
    {"filter_name": "shepp"},
    {"filter_name": "ramp"},
    {"remove_stripe_method": "vo-all"},
]


class RecipesParams(Setup, Where, OutputParams):
    # Each variant's fields laid over the recipe. Paganin is not a default: its
    # pixel size, distance and energy belong to the data.
    variants: list[Recipe] = Field(default_factory=lambda: [dict(v) for v in DEFAULT_VARIANTS], min_length=1)
    signal: Rect | None = None  # (y0, y1, x0, x1) of the slice
    background: Rect | None = None
    out: OptOutPath = None  # default artifacts/<tag>_recipes_c<center>_t<tilt>.h5

    @model_validator(mode="after")
    def _both_rois(self) -> RecipesParams:
        if (self.signal is None) != (self.background is None):
            raise ValueError("signal and background go together")
        return self


def _bind_recipes(p: RecipesParams, project: Project) -> RecipesParams:
    p = _bind(p, project)
    for v in p.variants:
        _checked({**p.recipe, **v})
    return p


class Variant(BaseModel):
    label: str
    fields: dict[str, Any]  # laid over the recipe
    window: tuple[float, float]
    signal_mean: float | None
    background_mean: float | None
    background_std: float | None
    cnr: float | None  # |signal − background| / background std


class RecipesResult(Result):
    center_px: float
    tilt_deg: float
    recipe: dict[str, Any]
    at: tuple[int, int]  # (z, y) of the slice and the xz plane
    variants: list[Variant]


def _recipes_out(p: RecipesParams, _: Project) -> dict[str, str]:
    return {"out": f"artifacts/{p.tag}_recipes_c{_stem(p.center_px)}_t{_stem(p.tilt_deg)}.h5"}


def _label(v: dict[str, Any]) -> str:
    return ",".join(f"{k}={x}" for k, x in v.items()) or "recipe"


def _roi(a: np.ndarray, r: Rect) -> np.ndarray:
    y0, y1, x0, x1 = r
    if y1 > a.shape[0] or x1 > a.shape[1]:
        raise Refused(f"rectangle {r} leaves the {a.shape[0]} x {a.shape[1]} slice")
    return a[y0:y1, x0:x1]


@op(
    "recon.compare",
    cost="job",
    params=RecipesParams,
    result=RecipesResult,
    gpu=True,
    default_out=_recipes_out,
    bind=_bind_recipes,
)
def compare(ctx: Context, p: RecipesParams) -> RecipesResult:
    """Reconstruct scan `tag` once per variant (fields laid over the recipe:
    by default the recipe itself, the shepp and ramp filters and vo-all
    stripe removal) and keep each
    one's slice at z and xz plane at y, to compare by eye. With a signal and
    a background rectangle of the slice, each variant's CNR = |μs − μb| / σb.
    Arrays in `out`: `slice` [variant, y, x], `xz` [variant, z, x]."""
    scan, group, stack = _stack(ctx, p.tag)
    z, y, x = _at(stack, scan, group, p, p.z, p.y)
    slices, xzs, variants = [], [], []
    k = len(p.variants)
    for i, v in enumerate(p.variants):
        span = (0.05 + 0.95 * i / k, 0.05 + 0.95 * (i + 1) / k)
        sl, xz, _ = _planes(ctx, stack, scan, p, {**p.recipe, **v}, (z, y, x), span, _label(v))
        stats: dict[str, float | None] = dict.fromkeys(["signal_mean", "background_mean", "background_std", "cnr"])
        if p.signal is not None:
            s, b = _roi(sl, p.signal), _roi(sl, p.background)
            stats.update(
                signal_mean=float(s.mean()), background_mean=float(b.mean()), background_std=float(b.std())
            )
            stats["cnr"] = abs(stats["signal_mean"] - stats["background_mean"]) / stats["background_std"]
        variants.append(Variant(label=_label(v), fields=v, window=window(sl), **stats))
        slices.append(sl)
        xzs.append(xz)
    r = RecipesResult(
        center_px=p.center_px,
        tilt_deg=p.tilt_deg,
        recipe=p.recipe,
        at=(z, y),
        variants=variants,
        artifacts=[p.out],
    )
    arrays = {"slice": np.stack(slices), "xz": np.stack(xzs)}
    save_h5(p.out, r.model_dump(mode="json", exclude={"artifacts"}), **arrays)
    return r.with_arrays(**arrays)


class SplitParams(Setup, Where, OutputParams):
    out: OptOutPath = None  # default artifacts/<tag>_split_c<center>_t<tilt>.h5


class SplitResult(Result):
    center_px: float
    tilt_deg: float
    recipe: dict[str, Any]
    at: tuple[int, int]  # (z, y) of the slices and the xz planes
    nproj: tuple[int, int]  # (even, odd)
    radius_px: float  # of the disk every projection saw at z
    correlation: float  # Pearson, inside the disk
    frc: list[float]  # per ring of 1/n cycles per pixel
    resolution_px: float | None  # 1 / the first frequency below 1/7


def _split_out(p: SplitParams, _: Project) -> dict[str, str]:
    return {"out": f"artifacts/{p.tag}_split_c{_stem(p.center_px)}_t{_stem(p.tilt_deg)}.h5"}


FRC_THRESHOLD = 1 / 7
TAPER_PX = 16.0


def frc(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """The Fourier ring correlation of square slices `a` and `b` inside the
    disk of `radius` about the middle (raised-cosine edge, so the shared edge
    does not correlate), and their Pearson correlation inside it."""
    n = a.shape[0]
    yy, xx = np.indices(a.shape)
    d = np.hypot(yy - n / 2, xx - n / 2)
    inside = d < radius
    taper = np.clip((radius - d) / TAPER_PX, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * taper)
    a = (a - a[inside].mean()) * w
    b = (b - b[inside].mean()) * w
    pearson = float(np.corrcoef(a[inside], b[inside])[0, 1])
    fa, fb = np.fft.fftshift(np.fft.fft2(a)), np.fft.fftshift(np.fft.fft2(b))
    ring = np.rint(np.hypot(yy - n // 2, xx - n // 2)).astype(int).ravel()
    m = n // 2
    num = np.bincount(ring, (fa * fb.conj()).real.ravel(), m + 1)[:m]
    pa = np.bincount(ring, (abs(fa) ** 2).ravel(), m + 1)[:m]
    pb = np.bincount(ring, (abs(fb) ** 2).ravel(), m + 1)[:m]
    den = np.sqrt(pa * pb)
    return num / np.where(den > 0, den, 1), pearson


@op("recon.split", cost="job", params=SplitParams, result=SplitResult, gpu=True, default_out=_split_out, bind=_bind)
def split(ctx: Context, p: SplitParams) -> SplitResult:
    """Reconstruct scan `tag` from its even and from its odd projections and
    compare the two slices at z inside the disk every projection saw: what
    both halves show is the sample, what one shows is noise or artifact.
    Reports their Pearson correlation and Fourier ring correlation, with the
    resolution where it falls below 1/7. Arrays in `out`: `slice` [half, y,
    x], `xz` [half, z, x]."""
    scan, group, stack = _stack(ctx, p.tag)
    nproj, nz, n = stack.shape
    z, y, x = _at(stack, scan, group, p, p.z, p.y)
    fov = json.loads(_core.valid_fov(n, nz, units.center_for_scan(scan, p.center_px), p.tilt_deg))
    if z >= fov["depth"]:
        raise Refused(f"slice {z} is outside the {fov['depth']}-slice volume")
    radius = fov["radius"][z]
    if radius < 2 * TAPER_PX:
        raise Refused(f"slice {z} is outside the field of view every projection saw (radius {radius:.0f} px)")
    slices, xzs, counts = [], [], []
    for i, first in enumerate((0, 1)):
        half = stack.select(list(range(first, nproj, 2)))
        counts.append(half.shape[0])
        span = (0.05 + 0.475 * i, 0.05 + 0.475 * (i + 1))
        sl, xz, _ = _planes(ctx, half, scan, p, p.recipe, (z, y, x), span, ("even", "odd")[i])
        del half
        slices.append(sl)
        xzs.append(xz)
    curve, pearson = frc(slices[0], slices[1], radius)
    below = np.flatnonzero(curve[1:] < FRC_THRESHOLD)
    r = SplitResult(
        center_px=p.center_px,
        tilt_deg=p.tilt_deg,
        recipe=p.recipe,
        at=(z, y),
        nproj=tuple(counts),
        radius_px=radius,
        correlation=pearson,
        frc=curve.tolist(),
        resolution_px=n / (below[0] + 1) if below.size else None,
        artifacts=[p.out],
    )
    arrays = {"slice": np.stack(slices), "xz": np.stack(xzs)}
    save_h5(p.out, r.model_dump(mode="json", exclude={"artifacts"}), **arrays)
    return r.with_arrays(**arrays)


class RunParams(Setup, OutputParams):
    # tilt_deg defaults to the group's decided tilt only: a final volume is
    # not made at a guessed one.
    out: OptOutPath = None  # default final_<tag>_bin<b>_c<center>_t<tilt>_<algorithm>_<filter>.h5


class RunResult(Result):
    center_px: float
    tilt_deg: float
    recipe: dict[str, Any]
    shape: tuple[int, int, int]
    secs: float  # the reconstruction's own wall clock


def _bind_run(p: RunParams, project: Project) -> RunParams:
    return _bind(p, project, tilt_decided=True)


def _final_out(p: RunParams, project: Project) -> dict[str, str]:
    scan, _ = scans.find(project, p.tag)
    c = json.loads(_core.recipe_config(json.dumps(p.recipe)))
    name = f"final_{p.tag}_bin{scan.bin}_c{_stem(p.center_px)}_t{_stem(p.tilt_deg)}"
    return {"out": f"{name}_{c['algorithm']}_{c['filter_name']}.h5"}


# python/lamino_pal/__init__.py -> the source tree with the tomoxide submodule.
SOURCE = Path(lamino_pal.__file__).resolve().parents[2]


def tomoxide_cli() -> Path:
    """The tomoxide CLI, built by cargo from the submodule the core was built
    from: `tomoxide --version` names no commit, so the build is the check."""
    tox = SOURCE / "lib" / "tomoxide"
    if not (tox / "Cargo.toml").is_file():
        raise Refused(f"recon.run builds the tomoxide CLI from {tox}, which is missing")
    rev = subprocess.run(
        ["git", "-C", str(tox), "describe", "--always", "--dirty", "--abbrev=12"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if rev.endswith("-dirty"):
        raise Refused(f"{tox} has uncommitted changes ({rev}): the CLI and the core cannot be shown to match")
    if rev != _core.tomoxide_rev():
        raise Refused(f"the core was built from tomoxide {_core.tomoxide_rev()}, {tox} is at {rev}: rebuild the core")
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    argv = [cargo, "build", "--release", "--features", "cuda", "-p", "tomoxide-cli"]
    argv += ["--manifest-path", str(tox / "Cargo.toml"), "--message-format=json-render-diagnostics"]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise Refused(f"recon.run builds the tomoxide CLI with cargo, which is not at {cargo}") from None
    if r.returncode != 0:
        raise LaminoPalError(f"building the tomoxide CLI failed:\n{r.stderr[-4000:]}")
    for line in r.stdout.splitlines():
        m = json.loads(line)
        if m.get("reason") == "compiler-artifact" and m["target"]["name"] == "tomoxide" and m.get("executable"):
            return Path(m["executable"])
    raise LaminoPalError("cargo built no tomoxide executable")


@op(
    "recon.run",
    cost="job",
    params=RunParams,
    result=RunResult,
    gpu=True,
    default_out=_final_out,
    bind=_bind_run,
)
def run(ctx: Context, p: RunParams) -> RunResult:
    """Reconstruct scan `tag` whole with `tomoxide recon` (the CLI built from
    the core's own tomoxide commit) at the decided center and tilt, into an
    h5 volume `out` (`exchange/data` [z, y, x]) with its provenance in
    `/lamino_pal/provenance`: center, tilt, recipe, source, tomoxide commit
    and command line."""
    scan, path = scans.find(ctx.project, p.tag)
    ctx.progress(0.0, "building the tomoxide CLI")
    exe = tomoxide_cli()
    out = Path(p.out)
    # The CLI appends .h5 to its output base.
    part = out.with_name(f".{out.stem}.part")
    part_h5 = part.with_name(part.name + ".h5")
    config = out.with_name(f".{out.stem}.toml")
    argv = [str(exe), "--backend", "cuda", "recon", str(path), "--config", str(config)]
    argv += ["--center", repr(units.center_for_scan(scan, p.center_px)), "--lamino_angle", repr(p.tilt_deg)]
    argv += ["--save_format", "h5", "--output", str(part), "--progress_json"]
    tail: deque[str] = deque(maxlen=40)
    done = (0, 1, 0.0)
    try:
        config.write_text(tomli_w.dumps(p.recipe))
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            for line in proc.stdout:
                # The job's out.log keeps the CLI's own account.
                sys.stdout.write(line)
                if line.startswith("{"):
                    j = json.loads(line)
                    done = (j["end"], j["total"], j["secs"])
                    ctx.progress(0.05 + 0.95 * j["end"] / j["total"], f"row {j['end']}/{j['total']}")
                else:
                    tail.append(line.rstrip())
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if rc != 0:
            raise LaminoPalError(f"tomoxide recon exited {rc}:\n" + "\n".join(tail))
        with h5py.File(part_h5, "r+") as f:
            shape = f["exchange/data"].shape
            prov = f.require_group("lamino_pal/provenance")
            prov.attrs.update(
                tag=p.tag,
                bin=scan.bin,
                center_px=p.center_px,
                tilt_deg=p.tilt_deg,
                recipe=json.dumps(p.recipe),
                source=str(path),
                tomoxide=_core.tomoxide_rev(),
                argv=json.dumps(argv),
            )
        os.replace(part_h5, out)
    finally:
        part_h5.unlink(missing_ok=True)
        config.unlink(missing_ok=True)
    return RunResult(
        center_px=p.center_px,
        tilt_deg=p.tilt_deg,
        recipe=p.recipe,
        shape=tuple(shape),
        secs=done[2],
        artifacts=[p.out],
    )
