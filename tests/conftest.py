import math
import os

# Before anything can start a thread: GPU ops run on this process's one GPU.
os.environ.setdefault("TOMOXIDE_CUDA_DEVICES", "0")

from pathlib import Path

import pytest

from lamino_pal import dispatch
from lamino_pal.ops import load_all
from lamino_pal.project import Project

load_all()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return Project.create(tmp_path / "work" / "project.toml", str(raw))


@pytest.fixture
def run_job():
    """Run a job op in-process as its job process will, with this process's
    GPU 0 as the project's batch GPU."""

    def run(project, name, params, progress=None):
        if project.data.gpu.batch != [0]:
            project.commit(lambda f: setattr(f.gpu, "batch", [0]))
        return dispatch.run_job(name, params, project=project, caller="cli", progress=progress)

    return run


@pytest.fixture(autouse=True)
def gpu_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))


DATA = Path(os.environ.get("LAMINO_PAL_DATA", "~/work/261003_Lamino_test")).expanduser()


def band_at(z, depth_from, tilt_from, tilt_to, depth_to):
    """tomoxide's SampleBand::at restated, so a test does not check the
    binding against itself."""

    def row(v):
        r = (v - depth_from / 2) * math.cos(math.radians(tilt_from))
        return min(max(round(r / math.cos(math.radians(tilt_to)) + depth_to / 2), 0), depth_to - 1)

    return row(z[0]), row(z[1])


@pytest.fixture(scope="session")
def golden_project(tmp_path_factory):
    """A project of two 261003 bin-2 scans: step1000 (a full turn closed at
    360 deg) and the fly half turn f180_N500."""
    assert DATA.is_dir(), f"LAMINO_PAL_DATA {DATA} is missing"
    p = Project.create(tmp_path_factory.mktemp("golden") / "project.toml", str(DATA))
    for tag, h5, kind in [
        ("step1000", "tomoxide/step1000_bin2.h5", "step"),
        ("f180_N500", "tomoxide/fly/f180_N500_2dps_0p18s_1p5ET_bin2.h5", "fly"),
    ]:
        params = {"tag": tag, "h5": str(DATA / h5), "bin": 2, "kind": kind}
        dispatch.run("scan.adopt", params, project=p, caller="cli")
    return p
