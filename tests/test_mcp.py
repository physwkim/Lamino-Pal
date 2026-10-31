import base64
import io
import json
import os
import re
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
from conftest import DATA
from test_align import _group
from test_recon import _scan

from lamino_pal import dispatch, jobs, launch, remote
from lamino_pal.mcp import figures
from lamino_pal.mcp.figures import FIGURES
from lamino_pal.render import to_u8, window
from lamino_pal.mcp.server import build
from lamino_pal.ops import REGISTRY, recon
from lamino_pal.ops.registry import Params, Result
from lamino_pal.ops.registry import build as build_op
from lamino_pal.project import Project

pytestmark = pytest.mark.anyio

FIXED = {
    "project_open",
    "project_info",
    "job_list",
    "job_status",
    "job_wait",
    "job_cancel",
    "job_result",
    "gui_state",
    "gui_show_tab",
    "gui_select",
    "gui_view",
    "gui_graph",
}


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


async def test_a_tuple_params_order_is_in_its_tool_schema(project):
    async with Client(build(project)) as c:
        tools = {t.name: t.input_schema["properties"] for t in (await c.list_tools()).tools}

    def described(prop):
        return [arm["description"] for arm in prop["anyOf"] if "description" in arm]

    for rect in ("signal", "background"):
        assert described(tools["recon_compare"][rect]) == [
            "(y0, y1, x0, x1): rows [y0, y1) and columns [x0, x1) of the slice"
        ]
    assert described(tools["export_tiff16"]["z"]) == ["slices [z0, z1)"]


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
        # In the structured content, which a client may show alone.
        fig = r.structured_content["figure"]
        assert json.loads(r.content[0].text) == r.structured_content
        assert fig["caption"].startswith("Slice z=2 [y, x], window [")
        (saved,) = (project.work_dir / FIGURES).iterdir()
        assert fig["path"] == (FIGURES / saved.name).as_posix() and saved.name.endswith("-volume.slice.png")
        im = _image(r)
        assert (im.size, im.mode) == ((30, 20), "L")
        assert saved.read_bytes() == base64.b64decode(r.content[1].data)
        evidence = [fig["path"]]
        r = await c.call_tool("group_decide", {"group": "m", "field": "center_px", "value": 30.0, "evidence": evidence})
    assert not r.is_error
    (d,) = project.reload().group[0].decisions
    assert (d.by, d.evidence) == ("agent", evidence)


async def test_the_gui_tools_reach_the_gui_open_on_the_project(project):
    async with Client(build(project)) as c:
        r = await c.call_tool("gui_state", {})
        assert r.is_error and r.content[0].text.endswith(f": no Lamino-Pal GUI is open on {project.path}")
        seen = []
        gui = remote.Server(project.path, lambda cmd, args: seen.append((cmd, args)) or {"x": args["x"]})
        try:
            r = await c.call_tool("gui_graph", {"tab": "Align", "graph": "focus", "x": [1, 2]})
        finally:
            gui.close()
    assert not r.is_error and r.structured_content == {"x": [1.0, 2.0]}
    assert seen == [("graph", {"tab": "Align", "graph": "focus", "x": [1.0, 2.0], "y": None, "reset": False})]


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
    fig = r.structured_content["figure"]
    assert fig["caption"].startswith("Maximum intensity projections of slices [2, 10)")
    # z [y, x] beside x turned to [y, z], above y [z, x].
    gap = 4
    assert _image(r).size == (20 + gap + 8, 16 + gap + 8)
    assert fig["path"] == (FIGURES / f"{jid}.png").as_posix() and (project.work_dir / fig["path"]).exists()


def test_slices_compared_by_eye_show_their_detail_zoomed_at_full_resolution():
    n, yc, xc = 1200, 800, 350
    yy, xx = np.mgrid[0:n, 0:n]
    # Detail fading out from (yc, xc) inside the 0.92-FOV disk, and noise
    # stronger than it in the corners outside the disk.
    detail = ((yy + xx) % 2) * np.exp(-((yy - yc) ** 2 + (xx - xc) ** 2) / (2 * 60.0**2))
    noise = np.random.default_rng(0).normal(size=(n, n)) * (np.hypot(yy - n / 2, xx - n / 2) > 0.5 * n)
    layer = np.stack([detail + noise, 2 * detail + noise, 3 * detail]).astype(np.float32)
    tilts = [{"tilt_deg": t, "slice": 6, "z_peak": 5, "focus": 1.0, "depth": 300} for t in (18, 20, 22)]
    xz = np.zeros((3, 300, n), np.float32)
    image, caption = figures.render("align.tilt_scan", {"center_px": 504.0, "tilts": tilts}, {"layer": layer, "xz": xz})
    gap, s = 4, (1568 - 2 * 4) // 3
    y0, y1, x0, x1 = map(int, re.search(r"zoom \[(\d+):(\d+), (\d+):(\d+)\] at full", caption).groups())
    assert (y1 - y0, x1 - x0) == (s, s)
    assert abs((y0 + y1) / 2 - yc) <= 2 and abs((x0 + x1) / 2 - xc) <= 2
    im = np.asarray(Image.open(io.BytesIO(image)))
    # Zooms in a row above the whole slices shrunk to s and the xz planes
    # shrunk alike, nothing scaled again.
    assert im.shape == (2 * s + round(300 * s / n) + 2 * gap, 3 * s + 2 * gap)
    for i, a in enumerate(layer):
        zoom = a[y0:y1, x0:x1]
        assert np.array_equal(im[:s, i * (s + gap) : i * (s + gap) + s], to_u8(zoom, window(zoom)))
    # The slice shown is the band middle, the same layer at every tilt; the
    # focus peak is named beside it.
    assert caption.startswith("Each tilt's slice through the band middle, one layer at every tilt, at center 504.00")
    assert caption.splitlines()[1].startswith("- tilt 18 deg, slice 6 (focus 1, peak z 5), zoom window [")


def test_a_tilt_scan_shows_each_tilts_xz_plane_over_its_depth_below_its_slice():
    rng = np.random.default_rng(0)
    layer = rng.random((3, 64, 64), dtype="f4")
    depths = [30, 35, 40]
    xz = np.full((3, 40, 64), np.nan, np.float32)
    for x, d in zip(xz, depths):
        x[:d] = rng.random((d, 64), dtype="f4")
    tilts = [{"tilt_deg": t, "slice": 6, "z_peak": 5, "focus": 1.0, "depth": d} for t, d in zip((18, 20, 22), depths)]
    image, caption = figures.render("align.tilt_scan", {"center_px": 504.0, "tilts": tilts}, {"layer": layer, "xz": xz})
    assert "and the xz plane through y=32 [z, x] over the tilt's whole depth below it" in caption.splitlines()[0]
    im = np.asarray(Image.open(io.BytesIO(image)))
    gap, top = 4, 2 * (64 + 4)
    assert im.shape == (top + 40, 3 * 64 + 2 * gap)
    for i, (x, d, line) in enumerate(zip(xz, depths, caption.splitlines()[1:])):
        col = slice(i * (64 + gap), i * (64 + gap) + 64)
        w = window(x[:d])
        assert line.endswith(f", xz window [{w[0]:.4g}, {w[1]:.4g}]")
        assert np.array_equal(im[top : top + d, col], to_u8(x[:d], w))
        assert (im[top + d :, col] == 255).all()


def test_a_focus_curve_shades_its_band_air_and_z_wrap():
    f = [0.3] * 30 + [0.01] * 40 + [1.0] * 80
    r = {"step": 1, "shape": [150, 8, 8], "peak_z": 70}
    r |= {"band": [70, 149], "air": [[30, 69]], "wrap": [0, 29]}
    image, caption = figures.render("volume.focus_by_z", r, {"focus": np.array(f)})
    assert "suggested band [70, 149] (gray), air [30, 69] (blue), z wrap [0, 29] (orange)," in caption
    im = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    for color, _ in figures.SHADES.values():
        assert (im == Image.new("RGB", (1, 1), color).getpixel((0, 0))).all(-1).any(), color


def test_a_sweep_shows_its_best_rivals_and_candidates_spread_over_its_range():
    c = [976.0 + i for i in range(49)]
    frames = np.random.default_rng(0).random((49, 32, 32), dtype="f4")
    verdict = {"kind": "ambiguous", "index": 10, "rivals": [{"index": 30}]}
    r = {"slice": 7, "tilt_deg": 20.0, "candidates_px": c, "focus": [1.0] * 49, "verdict": verdict}
    _, caption = figures.render("align.center_sweep", r, {"frames": frames})
    head, *lines = caption.splitlines()
    assert "for 7 of the 49 candidates, the best, its rivals and 5 evenly spaced over the sweep" in head
    shown = [float(re.match(r"- center (\S+) px", line).group(1)) for line in lines]
    assert shown == [976.0, 986.0, 988.0, 1000.0, 1006.0, 1012.0, 1024.0]
    assert lines[1].startswith("- center 986.00 px (best, focus 1)") and "(rival, focus 1)" in lines[4]
    assert lines[0].startswith("- center 976.00 px (focus 1)")


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


async def test_the_stdio_server_answers_once_it_has_pinned_its_gpu(project):
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "lamino_pal.mcp", "--project", str(project.path)], env=_env()
    )
    async with Client(server) as c:
        r = await c.call_tool("project_info", {})
    assert r.structured_content["path"] == str(project.path)


@pytest.mark.gpu
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


@pytest.mark.gpu
@pytest.mark.golden
async def test_the_stdio_server_aligns_previews_and_cancels_a_run_on_step1000(tmp_path):
    p = Project.create(tmp_path / "e2e" / "project.toml", str(DATA))
    dispatch.run("scan.adopt", {"tag": "step1000", "h5": str(DATA / "tomoxide/step1000_bin2.h5"), "bin": 2,
                                "kind": "step"}, project=p, caller="cli")
    # Built here, so the run's cancel below stops the reconstruction, not cargo.
    recon.tomoxide_cli()
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "lamino_pal.mcp", "--project", str(p.path)], env=_env()
    )
    async with Client(server) as c:
        r = await c.call_tool("align_rings", {"tag": "step1000"})
        assert r.structured_content["trustworthy"] is False and _image(r).mode == "L"
        r = await c.call_tool("align_center_sweep", {"tag": "step1000"})
        assert r.structured_content["verdict"]["kind"] == "ambiguous"
        caption = r.structured_content["figure"]["caption"]
        assert "(best, focus" in caption and "(rival, focus" in caption
        args = {"tag": "step1000", "center_px": 1008.5, "tilt_deg": 20.0, "recipe": {"algorithm": "fourierrec"}}
        r = await c.call_tool("recon_preview", args)
        assert r.structured_content["at"] == [545, 512, 512] and _image(r).mode == "L"
        r = await c.call_tool("recon_run", args)
        jid = r.structured_content["job_id"]
        st = r.structured_content
        while st["status"] == "queued" or st["status"] == "running" and st["message"] != "reconstructing":
            st = (await c.call_tool("job_wait", {"job_id": jid, "timeout_s": 0.5})).structured_content
        assert (st["status"], st["message"]) == ("running", "reconstructing"), st
        await c.call_tool("job_cancel", {"job_id": jid})
        st = (await c.call_tool("job_wait", {"job_id": jid, "timeout_s": 45})).structured_content
    # Ended by the CLI's death, not by the grace kill ("stopped inside a call").
    assert (st["status"], st["message"]) == ("cancelled", "reconstructing"), st
