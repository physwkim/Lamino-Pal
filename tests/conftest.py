import os

# Before anything can start a thread: GPU ops run on this process's one GPU.
os.environ.setdefault("TOMOXIDE_CUDA_DEVICES", "0")

import pytest

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
    """Run a job op in-process as its job will: validated, paths resolved."""
    from lamino_pal import dispatch
    from lamino_pal.ops import REGISTRY

    def run(project, name, params, progress=None):
        o = REGISTRY[name]
        p = dispatch._resolve_paths(o, o.params.model_validate(params), project)
        ctx = dispatch.Context(project, "cli", progress or (lambda frac, msg: None))
        return o.fn(ctx, p)

    return run


@pytest.fixture(autouse=True)
def gpu_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
