"""project.toml: the one store of scans, mount groups and their decisions.

The work directory is the directory holding project.toml; it is not stored.
Every write goes through `Project.commit`.
"""

from __future__ import annotations

import fcntl
import os
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from lamino_pal import Refused


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# A scan tag or group name: it becomes part of file names.
Tag = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class Meta(_Model):
    raw_root: str
    revision: int = 0


class Gpu(_Model):
    interactive: int = 0
    batch: list[int] = [1, 2, 3]


class ZBand(_Model):
    """Sample rows `z` of a bin-`bin` volume reconstructed at `tilt_deg`."""

    z: tuple[int, int]
    bin: int = Field(ge=1)
    tilt_deg: float


# How a value was read off its evidence.
Method = Literal["rings", "sweep", "eye", "focus"]


class Decision(_Model):
    field: Literal["center_px", "tilt_deg", "z_band"]
    value: float | ZBand
    by: Literal["user", "agent"]
    evidence: list[str] = Field(min_length=1)
    method: Method | None = None
    note: str = ""


class Group(_Model):
    """Scans sharing one mount, hence one center, tilt and sample band."""

    name: Tag
    center_px: float | None = None  # full-resolution detector column
    tilt_deg: float | None = None
    z_band: ZBand | None = None
    decisions: list[Decision] = []


class Scan(_Model):
    tag: Tag
    kind: Literal["step", "fly"]
    raw: str | None = None  # under raw_root, the on-disk (NFD) name
    h5: str  # resolved against the work directory
    bin: int = Field(ge=1)
    group: str | None = None


class ProjectFile(_Model):
    project: Meta
    gpu: Gpu = Gpu()
    group: list[Group] = []
    scan: list[Scan] = []
    recon: dict[str, Any] = {}  # tomoxide Config fields

    @model_validator(mode="after")
    def _references(self) -> ProjectFile:
        for what, names in [
            ("scan tag", [s.tag for s in self.scan]),
            ("group", [g.name for g in self.group]),
        ]:
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise ValueError(f"{what} {sorted(dup)} repeated")
        groups = {g.name for g in self.group}
        for s in self.scan:
            if s.group is not None and s.group not in groups:
                raise ValueError(f"scan {s.tag}: no group {s.group}")
        return self

    def find_scan(self, tag: str) -> Scan | None:
        return next((s for s in self.scan if s.tag == tag), None)


def is_under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


class Project:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).resolve()
        self.data = self._read()

    @classmethod
    def create(cls, path: str | os.PathLike[str], raw_root: str) -> Project:
        path = Path(path).resolve()
        if path.exists():
            raise Refused(f"{path} exists")
        if is_under(path.parent, Path(raw_root)):
            raise Refused(f"work directory {path.parent} is under the raw root")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, ProjectFile(project=Meta(raw_root=raw_root)))
        return cls(path)

    @property
    def work_dir(self) -> Path:
        return self.path.parent

    @property
    def raw_root(self) -> Path:
        return Path(self.data.project.raw_root)

    @property
    def revision(self) -> int:
        return self.data.project.revision

    def reload(self) -> ProjectFile:
        self.data = self._read()
        return self.data

    def commit(self, mutation: Callable[[ProjectFile], None]) -> ProjectFile:
        """Apply `mutation` to the file as it is now on disk.

        Concurrent committers serialize on the lock, and each one rereads, so
        none overwrites another's change and revisions never repeat.
        """
        with self._lock():
            cur = self._read()
            mutation(cur)
            cur.project.revision += 1
            new = ProjectFile.model_validate(cur.model_dump())
            _write(self.path, new)
            self.data = new
            return new

    def _read(self) -> ProjectFile:
        with open(self.path, "rb") as f:
            return ProjectFile.model_validate(tomllib.load(f))

    @contextmanager
    def _lock(self) -> Iterator[None]:
        # A side file: the atomic rename replaces project.toml's inode.
        with open(self.path.with_name(self.path.name + ".lock"), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield


def _write(path: Path, data: ProjectFile) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        tomli_w.dump(data.model_dump(mode="json", exclude_none=True), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
