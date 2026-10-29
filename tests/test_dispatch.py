import json
import threading

import pytest
from pydantic import ValidationError

from lamino_pal import LaminoPalError, Refused, dispatch
from lamino_pal.ops.registry import (
    REGISTRY,
    OptOutPath,
    OutPath,
    OutputParams,
    Params,
    Result,
    build,
)


def _log(project):
    return [json.loads(x) for x in dispatch.log_path(project).read_text().splitlines()]


class WriteParams(OutputParams):
    out: OutPath


class WriteResult(Result):
    path: str


def _write(ctx, p):
    """Write a marker file."""
    with open(p.out, "w") as f:
        f.write("x")
    return WriteResult(path=p.out, artifacts=[p.out])


@pytest.fixture
def write_op(monkeypatch):
    o = build("test.write", cost="instant", params=WriteParams, result=WriteResult, fn=_write)
    monkeypatch.setitem(REGISTRY, o.name, o)
    return o


def test_every_outcome_is_logged(project, write_op):
    dispatch.run("log.tail", {}, project=project, caller="cli")
    with pytest.raises(ValidationError):
        dispatch.run("log.tail", {"n": 0}, project=project, caller="mcp")
    with pytest.raises(LaminoPalError):
        dispatch.run("no.such", {}, project=project, caller="gui")
    with pytest.raises(Refused):
        dispatch.run("test.write", {"out": "/etc/x"}, project=project, caller="cli")
    log = _log(project)
    assert [(e["op"], e["status"], e["caller"]) for e in log] == [
        ("log.tail", "ok", "cli"),
        ("log.tail", "invalid", "mcp"),
        ("no.such", "error", "gui"),
        ("test.write", "refused", "cli"),
    ]
    assert all(e["tomoxide"] and "elapsed_s" in e for e in log)


def test_output_paths_resolve_under_the_work_dir(project, write_op):
    r = dispatch.run("test.write", {"out": "a/../m.txt"}, project=project, caller="cli")
    assert r.path == str(project.work_dir / "m.txt")
    assert _log(project)[-1]["params"]["out"] == r.path
    with pytest.raises(Refused, match="exists"):
        dispatch.run("test.write", {"out": "m.txt"}, project=project, caller="cli")
    dispatch.run("test.write", {"out": "m.txt", "overwrite": True}, project=project, caller="cli")


def test_raw_root_inside_work_dir_is_still_read_only(tmp_path, write_op):
    from lamino_pal.project import Project

    work = tmp_path / "w"
    p = Project.create(work / "project.toml", str(work / "raw"))
    with pytest.raises(Refused, match="raw root"):
        dispatch.run("test.write", {"out": "raw/x"}, project=p, caller="cli")


def test_registration_requires_output_params():
    class P(Params):
        out: OutPath

    with pytest.raises(TypeError, match="OutputParams"):
        build("test.bad", cost="instant", params=P, result=Result, fn=_write)


class NamedParams(OutputParams):
    tag: str
    out: OptOutPath = None


def test_omitted_output_path_takes_the_default_name(project, monkeypatch):
    o = build(
        "test.named",
        cost="instant",
        params=NamedParams,
        result=WriteResult,
        fn=_write,
        default_out=lambda p: {"out": f"{p.tag}.h5"},
    )
    monkeypatch.setitem(REGISTRY, o.name, o)
    r = dispatch.run("test.named", {"tag": "a"}, project=project, caller="mcp")
    assert r.path == str(project.work_dir / "a.h5")
    assert _log(project)[-1]["params"]["out"] == r.path
    with pytest.raises(Refused, match="exists"):
        dispatch.run("test.named", {"tag": "a"}, project=project, caller="mcp")
    with pytest.raises(Refused, match="outside"):
        dispatch.run("test.named", {"tag": "../x"}, project=project, caller="mcp")
    r = dispatch.run("test.named", {"tag": "a", "out": "b.h5"}, project=project, caller="mcp")
    assert r.path == str(project.work_dir / "b.h5")


def test_registration_requires_default_out_for_an_optional_output():
    with pytest.raises(TypeError, match="default_out"):
        build("test.bad", cost="instant", params=NamedParams, result=Result, fn=_write)


def test_registration_refuses_a_path_marker_inside_a_union():
    class P(OutputParams):
        out: OutPath | None = None

    with pytest.raises(TypeError, match="hides its path marker"):
        build("test.bad", cost="instant", params=P, result=Result, fn=_write)


def test_gpu_lock_is_exclusive():
    held = threading.Event()
    release = threading.Event()

    def holder():
        with dispatch.gpu_lock(3, 1.0):
            held.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    held.wait(5)
    try:
        with pytest.raises(Refused, match="GPU 3 is busy"):
            with dispatch.gpu_lock(3, 0.1):
                pass
    finally:
        release.set()
        t.join()
    with dispatch.gpu_lock(3, 0.1) as n:
        assert n == 3
