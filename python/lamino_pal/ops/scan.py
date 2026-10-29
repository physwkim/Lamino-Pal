"""Scans: find them under the raw root, bring them in as h5, group them."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import h5py
from pydantic import BaseModel, Field

from lamino_pal import Refused, _core
from lamino_pal.artifacts import written
from lamino_pal.dispatch import Context
from lamino_pal.ops.registry import InPath, OptOutPath, OutputParams, Params, Result, op
from lamino_pal.project import Group, ProjectFile, Scan, Tag, is_under


class RawScan(BaseModel):
    """A scan folder as `bl7c` reads it."""

    kind: Literal["step", "fly"]
    raw: str
    tag: str  # proposed
    n: int
    n_flat: int
    range_deg: float
    speed_dps: float | None
    exptime_s: float | None
    energy_ev: float | None


class Found(RawScan):
    imported: list[str]  # project tags made from this folder


class DiscoverParams(Params):
    pass


class DiscoverResult(Result):
    scans: list[Found]
    skipped: list[tuple[str, str]]  # (raw, why)


@op("scan.discover", cost="instant", params=DiscoverParams, result=DiscoverResult)
def discover(ctx: Context, p: DiscoverParams) -> DiscoverResult:
    """List every BL7C step and fly scan under the raw root, with the project
    tags already made from each, and the scan folders that cannot be imported."""
    d = json.loads(_core.bl7c_discover(str(ctx.project.raw_root)))
    scans = ctx.project.reload().scan
    return DiscoverResult(
        scans=[
            Found(**s, imported=[x.tag for x in scans if x.raw == s["raw"]])
            for s in d["scans"]
        ],
        skipped=d["skipped"],
    )


def _put(f: ProjectFile, scan: Scan, overwrite: bool) -> None:
    """Add `scan`, or replace the scan with its tag, keeping that one's group."""
    old = f.find_scan(scan.tag)
    if old is None:
        f.scan.append(scan)
        return
    if not overwrite:
        raise Refused(f"scan {scan.tag} exists; pass overwrite to replace it")
    scan.group = old.group
    f.scan[f.scan.index(old)] = scan


def _relative(ctx: Context, path: str) -> str:
    wd = ctx.project.work_dir
    return str(Path(path).relative_to(wd)) if is_under(Path(path), wd) else path


@contextmanager
def _import_lock(ctx: Context, raw: str) -> Iterator[None]:
    """One import per project at a time, from its checks through its commit:
    an import's output path and tag cannot change hands while it runs."""
    with open(ctx.project.work_dir / ".import.lock", "w") as f:
        with ctx.waiting("another import", f"importing {raw}"):
            fcntl.flock(f, fcntl.LOCK_EX)
        yield


class ImportParams(OutputParams):
    raw: str  # a `scan.discover` raw
    tag: Tag
    bin: int = Field(2, ge=1)
    out: OptOutPath = None  # default h5/<tag>_bin<bin>.h5


class ImportResult(Result):
    scan: RawScan
    tag: str
    h5: str
    bin: int
    shape: tuple[int, int, int]
    saturated: int  # projection pixels at 65535, before binning
    saturated_frac: float


@op(
    "scan.import",
    cost="job",
    params=ImportParams,
    result=ImportResult,
    default_out=lambda p, _: {"out": f"h5/{p.tag}_bin{p.bin}.h5"},
)
def import_(ctx: Context, p: ImportParams) -> ImportResult:
    """Import raw scan folder `raw` as DXchange h5 `out`, bin×bin mean binned,
    and add it to the project as `tag`. An existing tag is replaced only with
    `overwrite`, and keeps its group."""
    out = Path(p.out)
    with _import_lock(ctx, p.raw):
        # Rechecked under the lock; the dispatcher checked `out` before it.
        if ctx.project.reload().find_scan(p.tag) and not p.overwrite:
            raise Refused(f"scan {p.tag} exists; pass overwrite to replace it")
        existed = out.exists()
        if existed and not p.overwrite:
            raise Refused(f"out: {out} exists; pass overwrite to replace it")
        with written(out) as part:
            r = json.loads(
                _core.bl7c_import(
                    str(ctx.project.raw_root),
                    p.raw,
                    str(part),
                    p.bin,
                    lambda done, total: ctx.progress(done / total, f"{done}/{total} frames"),
                )
            )
        scan = RawScan(**r["scan"])
        h5 = _relative(ctx, p.out)
        entry = Scan(tag=p.tag, kind=scan.kind, raw=scan.raw, h5=h5, bin=p.bin)
        try:
            ctx.project.commit(lambda f: _put(f, entry, p.overwrite))
        except BaseException:
            # No entry (an adopt may have taken the tag): a file this import
            # created must not outlive it.
            if not existed:
                out.unlink(missing_ok=True)
            raise
    return ImportResult(
        scan=scan,
        tag=p.tag,
        h5=h5,
        bin=p.bin,
        shape=r["shape"],
        saturated=r["saturated"],
        saturated_frac=r["saturated_frac"],
        artifacts=[p.out],
    )


class AdoptParams(Params):
    tag: Tag
    h5: InPath
    bin: int = Field(ge=1)
    kind: Literal["step", "fly"]
    raw: str | None = None  # default: the file's /lamino_pal source
    overwrite: bool = False  # replace the scan entry with this tag


class AdoptResult(Result):
    tag: str
    h5: str
    shape: tuple[int, int, int]


@op("scan.adopt", cost="instant", params=AdoptParams, result=AdoptResult)
def adopt(ctx: Context, p: AdoptParams) -> AdoptResult:
    """Add an existing DXchange h5 to the project as `tag`, with its binning
    stated: `bin` and `kind` must agree with the file's /lamino_pal attributes
    when it has them. An existing tag is replaced only with `overwrite`."""
    with h5py.File(p.h5, "r") as f:
        data, theta = f.get("exchange/data"), f.get("exchange/theta")
        if data is None or theta is None:
            raise Refused(f"{p.h5} has no exchange/data and exchange/theta")
        if data.ndim != 3 or theta.shape != data.shape[:1]:
            raise Refused(f"{p.h5}: data {data.shape} with theta {theta.shape}")
        shape = data.shape
        attrs = dict(f["lamino_pal"].attrs) if "lamino_pal" in f else {}
    raw = p.raw
    for k, want in [("binning", p.bin), ("kind", p.kind), ("source", raw)]:
        have = attrs.get(k)
        if have is not None and want is not None and have != want:
            raise Refused(f"{p.h5}: /lamino_pal {k} is {have!r}, not {want!r}")
    raw = raw or attrs.get("source")
    h5 = _relative(ctx, p.h5)
    entry = Scan(tag=p.tag, kind=p.kind, raw=raw, h5=h5, bin=p.bin)
    ctx.project.commit(lambda f: _put(f, entry, p.overwrite))
    return AdoptResult(tag=p.tag, h5=h5, shape=shape)


class AssignParams(Params):
    tags: list[str] = Field(min_length=1)
    group: Tag | None  # None takes the scans out of their group


class AssignResult(Result):
    created: bool  # `group` was new


@op("scan.assign_group", cost="instant", params=AssignParams, result=AssignResult)
def assign_group(ctx: Context, p: AssignParams) -> AssignResult:
    """Put the scans `tags` in mount group `group`, creating it if new; scans
    in one group share a center, tilt and sample band. `group` None takes them
    out of any group."""
    created = False

    def mutate(f: ProjectFile) -> None:
        nonlocal created
        missing = [t for t in p.tags if f.find_scan(t) is None]
        if missing:
            raise Refused(f"no scan {missing}")
        if p.group is not None and all(g.name != p.group for g in f.group):
            f.group.append(Group(name=p.group))
            created = True
        for t in p.tags:
            f.find_scan(t).group = p.group

    ctx.project.commit(mutate)
    return AssignResult(created=created)
