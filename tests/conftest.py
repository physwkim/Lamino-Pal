import pytest

from lamino_pal.ops import load_all
from lamino_pal.project import Project

load_all()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return Project.create(tmp_path / "work" / "project.toml", str(raw))


@pytest.fixture(autouse=True)
def gpu_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
