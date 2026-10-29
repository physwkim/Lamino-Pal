import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import jobops
import numpy as np
import pytest
from mcp import Client, StdioServerParameters
from PIL import Image
from pydantic import Field
from test_align import _group
from test_recon import _scan

from lamino_pal import dispatch, jobs, launch
from lamino_pal.mcp.server import FIGURES, build
from lamino_pal.ops import REGISTRY
from lamino_pal.ops.registry import Params, Result
from lamino_pal.ops.registry import build as build_op

pytestmark = pytest.mark.anyio

FIXED = {"project_open", "project_info", "job_list", "job_status", "job_wait", "job_cancel", "job_result"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


class CountParams(Params):
    fracs: list[float]
    dt: list[float] = Field(default_factory=list)  # seconds to sleep before each report


def _count(ctx, p):
    """Report each of `fracs` as progress, sleeping `dt` before each."""
    for frac, dt in zip(p.fracs, p.dt or [0.0] * len(p.fracs)):
        time.sleep(dt)
        ctx.progress(frac, f"at {frac}")
    return Result()


@pytest.fixture
def test_ops(monkeypatch):
    """jobops' job ops run by job_main.py, and an interactive op reporting
    progress, registered before the server is built."""
    for o in [*jobops.OPS, build_op("test.count", cost="interactive", params=CountParams, result=Result, fn=_count)]:
        monkeypatch.setitem(REGISTRY, o.name, o)
    monkeypatch.setattr(jobs, "COMMAND", [sys.executable, str(Path(__file__).with_name("job_main.py")), "job"])


def _volume(project, vol, name="v.h5"):
    path = project.work_dir / name
    with h5py.File(path, "w") as f:
        f["exchange/data"] = vol
    return str(path)


def _last_log(project):
    return json.loads(dispatch.log_path(project).read_text().splitlines()[-1])


def _image(r):
    (img,) = [c for c in r.content if c.type == "image"]
    return Image.open(io.BytesIO(base64.b64decode(img.data)))


async def test_every_registry_op_is_a_tool_named_with_underscores(project):
    async with Client(build(project)) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert names - FIXED == {name.replace(".", "_") for name in REGISTRY}
    assert FIXED <= names


async def test_an_op_returns_its_result_and_is_logged_as_the_agents(project):
    async with Client(build(project)) as c:
        await c.call_tool("log_tail", {})
        r = await c.call_tool("log_tail", {"n": 1})
    assert not r.is_error
    (entry,) = r.structured_content["entries"]
    assert (entry["op"], entry["caller"], entry["status"]) == ("log.tail", "mcp", "ok")
    assert json.loads(r.content[0].text) == r.structured_content


async def test_an_op_judged_by_eye_returns_its_image_saved_to_cite(project):
    vol = np.random.default_rng(0).normal(size=(6, 20, 30)).astype(np.float32)
    path = _volume(project, vol)
    _scan(project, "s")
    _group(project, ["s"])
    async with Client(build(project)) as c:
        r = await c.call_tool("volume_slice", {"volume": path, "z": 2})
        assert not r.is_error and r.structured_content["z"] == 2
        caption = r.content[1].text
        assert caption.startswith("Slice z=2 [y, x], window [")
        (saved,) = (project.work_dir / FIGURES).iterdir()
        assert f"Saved as {FIGURES / saved.name}:" in caption and saved.name.endswith("-volume.slice.png")
        im = _image(r)
        assert (im.size, im.mode) == ((30, 20), "L")
        assert saved.read_bytes() == base64.b64decode(r.content[2].data)
        evidence = [str(FIGURES / saved.name)]
        r = await c.call_tool("group_decide", {"group": "m", "field": "center_px", "value": 30.0, "evidence": evidence})
    assert not r.is_error
    (d,) = project.reload().group[0].decisions
    assert (d.by, d.evidence) == ("agent", evidence)


async def test_a_refusal_and_invalid_params_reach_the_agent_and_the_log(project):
    path = project.work_dir / "flat.h5"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = np.zeros((4, 4), np.float32)
    async with Client(build(project)) as c:
        r = await c.call_tool("volume_slice", {"volume": str(path)})
        assert r.is_error and "has no 3-D exchange/data" in r.content[0].text
        assert _last_log(project)["status"] == "refused"
        r = await c.call_tool("volume_slice", {"volume": str(path), "z": -1, "bogus": 1})
        text = r.content[0].text
        assert r.is_error and "greater than or equal to 0" in text and "Extra inputs are not permitted" in text
        e = _last_log(project)
        assert (e["op"], e["caller"], e["status"], e["params"]["bogus"]) == ("volume.slice", "mcp", "invalid", 1)


async def test_progress_rises_and_is_throttled_but_the_end_is_sent(project, test_ops):
    seen = []

    async def progress(frac, total, msg):
        seen.append((frac, total, msg))

    # 0.2 arrives within PROGRESS_EVERY_S of 0.1, 0.05 does not rise.
    args = {"fracs": [0.1, 0.2, 0.05, 0.5, 1.0], "dt": [0, 0, 0.3, 0, 0]}
    async with Client(build(project)) as c:
        r = await c.call_tool("test_count", args, progress_callback=progress)
    assert not r.is_error
    assert seen == [(0.1, 1.0, "at 0.1"), (0.5, 1.0, "at 0.5"), (1.0, 1.0, "at 1.0")]


async def test_a_job_op_returns_its_job_to_wait_on_cancel_and_read(project, test_ops):
    async with Client(build(project)) as c:
        r = await c.call_tool("test_steps", {"n": 4})
        sub = r.structured_content
        assert (sub["status"], sub["gpu"], sub["last_run_s"]) == ("queued", None, None)
        seen = []

        async def progress(frac, total, msg):
            seen.append(frac)

        r = await c.call_tool("job_wait", {"job_id": sub["job_id"], "timeout_s": 20}, progress_callback=progress)
        assert r.structured_content["status"] == "done"
        assert seen == sorted(set(seen)) and seen[-1] == 1.0
        r = await c.call_tool("job_result", {"job_id": sub["job_id"]})
        assert r.structured_content["n"] == 4 and len(r.content) == 1
        r = await c.call_tool("job_cancel", {"job_id": sub["job_id"]})
        assert r.is_error and "done" in r.content[0].text

        r = await c.call_tool("test_steps", {"n": 400, "dt": 0.05, "out": "artifacts/long.txt"})
        long = r.structured_content
        assert isinstance(long["last_run_s"], float)
        r = await c.call_tool("job_list", {"active_only": True})
        assert [s["id"] for s in r.structured_content["result"]] == [long["job_id"]]
        r = await c.call_tool("job_result", {"job_id": long["job_id"]})
        assert r.is_error and f"job {long['job_id']} is " in r.content[0].text
        await c.call_tool("job_cancel", {"job_id": long["job_id"]})
        r = await c.call_tool("job_wait", {"job_id": long["job_id"], "timeout_s": 20})
        assert r.structured_content["status"] == "cancelled", r.structured_content["error"]
        r = await c.call_tool("job_wait", {"job_id": long["job_id"], "timeout_s": 46})
        assert r.is_error and "less than or equal to 45" in r.content[0].text


async def test_a_job_judged_by_eye_returns_its_image_from_its_artifact(project, test_ops):
    vol = np.random.default_rng(0).normal(size=(12, 16, 20)).astype(np.float32)
    path = _volume(project, vol)
    async with Client(build(project)) as c:
        r = await c.call_tool("export_mip", {"volume": path, "z": [2, 10]})
        jid = r.structured_content["job_id"]
        r = await c.call_tool("job_wait", {"job_id": jid, "timeout_s": 30})
        assert r.structured_content["status"] == "done", r.structured_content
        r = await c.call_tool("job_result", {"job_id": jid})
    assert r.structured_content["z"] == [2, 10]
    assert r.content[1].text.startswith("Maximum intensity projections of slices [2, 10)")
    # z [y, x] beside x turned to [y, z], above y [z, x].
    gap = 4
    assert _image(r).size == (20 + gap + 8, 16 + gap + 8)
    assert (project.work_dir / FIGURES / f"{jid}.png").exists()


async def test_project_open_opens_or_creates_and_resources_follow_it(project, tmp_path):
    other = tmp_path / "other" / "project.toml"
    raw = tmp_path / "raw2"
    raw.mkdir()
    async with Client(build(None)) as c:
        r = await c.call_tool("project_info", {})
        assert r.is_error and "no project is open: call project_open" in r.content[0].text
        r = await c.call_tool("project_open", {"path": str(other)})
        assert r.is_error and "pass raw_root to create it" in r.content[0].text
        r = await c.call_tool("project_open", {"path": str(other), "raw_root": str(raw)})
        assert r.structured_content["path"] == str(other) and other.exists()
        r = await c.call_tool("project_open", {"path": str(project.path), "raw_root": str(raw)})
        assert r.is_error and f"exists with raw root {project.raw_root}" in r.content[0].text
        r = await c.call_tool("project_open", {"path": str(project.path)})
        info = r.structured_content
        assert (info["work_dir"], info["process_gpu"]) == (str(project.work_dir), dispatch.process_gpu())
        await c.call_tool("log_tail", {"n": 1})
        toml = (await c.read_resource("lamino://project")).contents[0].text
        log = (await c.read_resource("lamino://log")).contents[0].text
        guide = (await c.read_resource("lamino://guide/alignment")).contents[0].text
        prompt = await c.get_prompt("align_mount", {"group": "g1"})
    assert toml == project.path.read_text()
    assert log == dispatch.log_path(project).read_text()
    assert guide.startswith("# Aligning a mount group with Lamino-Pal")
    assert "Align mount group g1 of the open project" in prompt.messages[0].content.text


def _env():
    return {k: v for k, v in os.environ.items() if k != launch.ENV}


async def test_the_stdio_server_runs_on_the_projects_interactive_gpu(project):
    project.commit(lambda f: setattr(f.gpu, "interactive", 1))
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "lamino_pal.mcp", "--project", str(project.path)], env=_env()
    )
    async with Client(server) as c:
        r = await c.call_tool("project_info", {})
    assert (r.structured_content["path"], r.structured_content["process_gpu"]) == (str(project.path), 1)


def test_an_unreadable_project_stops_the_server(tmp_path):
    bad = tmp_path / "project.toml"
    bad.write_text("not toml [")
    p = subprocess.run(
        [sys.executable, "-m", "lamino_pal.mcp", "--project", str(bad)],
        env=_env(), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
    )
    assert p.returncode == 2 and f"cannot read project {bad}" in p.stderr
