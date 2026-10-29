"""`lamino-pal-mcp`: every registry op as an MCP tool over stdio, with job
control, the project and op log as resources and the alignment guide.

Op tools pass their arguments through untouched: the dispatcher validates
them, so an invalid call is refused and logged like any other caller's.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.server.mcpserver.tools.base import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp_types import CallToolResult, ContentBlock, ImageContent, TextContent
from pydantic import ConfigDict, Field, ValidationError

from lamino_pal import LaminoPalError, dispatch, jobs, launch, remote
from lamino_pal.artifacts import load_h5
from lamino_pal.mcp import figures
from lamino_pal.ops import REGISTRY, load_all
from lamino_pal.ops.registry import Op
from lamino_pal.project import Project

WAIT_MAX_S = 45.0  # one job_wait call, under Codex's default 60 s tool timeout
PROGRESS_EVERY_S = 0.2
LOG_LINES = 200  # of lamino://log

INSTRUCTIONS = """\
Laminography reconstruction of PLS-II BL7C scans on tomoxide. Every analysis
op is a tool; a job op returns a job_id to follow with job_wait and
job_result. Read lamino://guide/alignment before aligning a mount group.
With a GUI open on the project, the gui_ tools show its user what you mean:
a tab, a viewer's zoom, a graph's range."""


class Session:
    """The project the tools act on: the one the server started with, or the
    one `project_open` opened since."""

    def __init__(self, project: Project | None):
        self.project = project

    def need(self, error: type[Exception] = ToolError) -> Project:
        if self.project is None:
            raise error("no project is open: call project_open")
        return self.project


@contextmanager
def _errors() -> Iterator[None]:
    """A refusal or invalid params reach the agent with their text, and so
    does a crash, as the op log records it: the server is local."""
    try:
        yield
    except ToolError:
        raise
    except (LaminoPalError, ValidationError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


class _Raw(ArgModelBase):
    model_config = ConfigDict(extra="allow")

    def model_dump_one_level(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class _Progress:
    """An op's progress calls, from its worker thread, as MCP progress
    notifications: throttled, and rising only, as MCP requires."""

    def __init__(self, ctx: Context):
        self.ctx, self.last, self.at = ctx, -1.0, 0.0

    def __call__(self, frac: float, msg: str) -> None:
        now = time.monotonic()
        if frac <= self.last or (frac < 1.0 and now - self.at < PROGRESS_EVERY_S):
            return
        self.last, self.at = frac, now
        anyio.from_thread.run(self.ctx.report_progress, frac, 1.0, msg)


Figure = tuple[dict[str, str], bytes]  # ({path, caption}, PNG)


def _figure(project: Project, name: str, op: str, result: dict[str, Any], arrays: figures.Arrays) -> Figure | None:
    """Op `op`'s figure, saved by `figures.save` as `name` so a decision can
    cite what was seen; None for an op without one."""
    fig = figures.render(op, result, arrays)
    if fig is None:
        return None
    png, caption = fig
    return {"path": str(figures.save(project.work_dir, name, png)), "caption": caption}, png


def _reply(result: dict[str, Any], figure: Figure | None) -> CallToolResult:
    """`result`, with its figure's path and caption as `figure`, as the
    structured content and as its text alike: a client may show either
    one."""
    image: list[ContentBlock] = []
    if figure is not None:
        meta, png = figure
        result = {**result, "figure": meta}
        image.append(ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png"))
    text = TextContent(type="text", text=json.dumps(result))
    return CallToolResult(content=[text, *image], structured_content=result)


def _last_run_s(project: Project, op: str) -> float | None:
    """How long op `op`'s last successful run took, by the op log."""
    path = dispatch.log_path(project)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for line in reversed(lines):
        e = json.loads(line)
        if e.get("op") == op and e.get("status") == "ok":
            return e.get("elapsed_s")
    return None


def _submit(project: Project, op: Op, args: dict[str, Any]) -> dict[str, Any]:
    st = jobs.submit(project, op.name, args, "mcp")
    return {"job_id": st.id, "status": st.status, "gpu": st.gpu, "last_run_s": _last_run_s(project, op.name)}


def _run(project: Project, op: Op, args: dict[str, Any], progress: _Progress) -> list[Any]:
    r = dispatch.run(op.name, args, project=project, caller="mcp", progress=progress)
    result = r.model_dump(mode="json")
    name = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{op.name}.png"
    return [result, _figure(project, name, op.name, result, r.arrays)]


def _job_figure(project: Project, st: jobs.JobState) -> Figure | None:
    _, arrays = load_h5(st.result["artifacts"][0])
    return _figure(project, f"{st.id}.png", st.op, st.result, arrays)


def op_tool(session: Session, op: Op) -> Tool:
    """Registry op `op` as a tool named with `_` for `.`."""

    async def call(ctx: Context, **args: Any) -> CallToolResult:
        project = session.need()
        with _errors():
            if op.cost == "job":
                return _reply(await anyio.to_thread.run_sync(_submit, project, op, args), None)
            result, fig = await anyio.to_thread.run_sync(_run, project, op, args, _Progress(ctx))
            return _reply(result, fig)

    doc = inspect.cleandoc(op.doc or op.name)
    if op.cost == "job":
        doc += (
            "\n\nRuns as a job: returns {job_id, status, gpu, last_run_s} at once (last_run_s: how long its "
            "last successful run took). Follow it with job_wait, then job_result."
        )
    if figures.has(op.name):
        doc += (
            "\n\nReturns an image of the result as well; `figure` gives its caption and its path under the "
            "work directory, to cite as group_decide evidence."
        )
    return Tool(
        fn=call,
        name=op.name.replace(".", "_"),
        description=doc,
        parameters=op.params.model_json_schema(),
        fn_metadata=FuncMetadata(arg_model=_Raw),
        is_async=True,
        context_kwarg="ctx",
    )


def _info(project: Project) -> dict[str, Any]:
    data = project.reload()
    return {
        "path": str(project.path),
        "work_dir": str(project.work_dir),
        "process_gpu": dispatch.process_gpu(),
        **data.model_dump(mode="json"),
    }


def build(project: Project | None) -> MCPServer:
    """The server, its tools acting on `project` until `project_open`."""
    session = Session(project)
    server = MCPServer(
        "lamino-pal",
        instructions=INSTRUCTIONS,
        tools=[op_tool(session, op) for op in REGISTRY.values()],
    )

    @server.tool()
    def project_open(path: str, raw_root: str | None = None) -> dict[str, Any]:
        """Open project.toml `path` for every tool from now on, or create it
        with raw data root `raw_root` when it does not exist. Returns
        project_info. GPU ops run only when this server's GPU (process_gpu) is
        the project's interactive GPU, fixed when the server started."""
        with _errors():
            p = Path(path)
            if p.exists():
                opened = Project(p)
                if raw_root is not None and Path(raw_root).resolve() != opened.raw_root.resolve():
                    raise ToolError(f"{opened.path} exists with raw root {opened.raw_root}, not {raw_root}")
            elif raw_root is None:
                raise ToolError(f"{p} does not exist: pass raw_root to create it")
            else:
                opened = Project.create(p, raw_root)
            session.project = opened
            return _info(opened)

    @server.tool()
    def project_info() -> dict[str, Any]:
        """The open project as it is on disk now: its path, work directory,
        GPUs, groups with their decisions, scans and reconstruction recipe,
        and the GPU this server's CUDA runs on (process_gpu)."""
        with _errors():
            return _info(session.need())

    @server.tool()
    def job_list(active_only: bool = False) -> list[dict[str, Any]]:
        """The project's jobs, oldest first, without their results."""
        with _errors():
            states = jobs.list_(session.need())
            return [s.model_dump(mode="json", exclude={"result"}) for s in states if s.active or not active_only]

    @server.tool()
    def job_status(job_id: str) -> dict[str, Any]:
        """Job `job_id`'s state: status, progress, message, error, result."""
        with _errors():
            return jobs.get(session.need(), job_id).model_dump(mode="json")

    @server.tool()
    async def job_wait(
        job_id: str, ctx: Context, timeout_s: Annotated[float, Field(ge=0, le=WAIT_MAX_S)] = 30.0
    ) -> dict[str, Any]:
        """Job `job_id`'s state once it ends or `timeout_s` passes, sending its
        progress meanwhile. Call again while it is queued or running."""
        project = session.need()
        deadline = time.monotonic() + timeout_s
        last = -1.0
        with _errors():
            while True:
                st = await anyio.to_thread.run_sync(jobs.get, project, job_id)
                if st.progress > last:
                    last = st.progress
                    await ctx.report_progress(st.progress, 1.0, st.message)
                if not st.active or time.monotonic() >= deadline:
                    return st.model_dump(mode="json")
                await anyio.sleep(0.25)

    @server.tool()
    def job_cancel(job_id: str) -> dict[str, Any]:
        """Cancel queued or running job `job_id`: its op stops at its next
        progress call, or the job exits 10 s later."""
        with _errors():
            return jobs.cancel(session.need(), job_id).model_dump(mode="json")

    @server.tool()
    async def job_result(job_id: str) -> CallToolResult:
        """Finished job `job_id`'s result, with the image of an op judged by
        eye: `figure` gives its caption and its path to cite."""
        project = session.need()
        with _errors():
            st = await anyio.to_thread.run_sync(jobs.get, project, job_id)
            if st.status != "done":
                raise ToolError(f"job {job_id} is {st.status}" + (f": {st.error}" if st.error else ""))
            fig = await anyio.to_thread.run_sync(_job_figure, project, st) if figures.has(st.op) else None
            return _reply(st.result, fig)

    def gui(cmd: str, args: dict[str, Any]) -> Any:
        with _errors():
            return remote.call(session.need().path, cmd, args)

    @server.tool()
    async def gui_state() -> dict[str, Any]:
        """The GUI open on the project: the tab shown, the group, scan and
        volume in view, and each tab's viewers and graphs. Every gui_ tool
        fails when no GUI is open on the project."""
        return await anyio.to_thread.run_sync(gui, "state", {})

    @server.tool()
    async def gui_show_tab(tab: str) -> dict[str, Any]:
        """Show GUI tab `tab`. Returns gui_state."""
        return await anyio.to_thread.run_sync(gui, "show_tab", {"tab": tab})

    @server.tool()
    async def gui_view(
        tab: str,
        viewer: str,
        zoom: float | None = None,
        center: list[float] | None = None,
        angles: tuple[float, float, float] | None = None,
        fit: bool = False,
    ) -> dict[str, Any]:
        """Set the camera of viewer `viewer` of GUI tab `tab`: fit the data to
        it first when `fit`, then set zoom (screen pixels per data pixel),
        center ([y, x], or [z, y, x] in 3D, in data pixels) and the 3D view's
        angles (degrees). Returns the camera."""
        args = {"tab": tab, "viewer": viewer, "zoom": zoom, "center": center, "angles": angles, "fit": fit}
        return await anyio.to_thread.run_sync(gui, "view", args)

    @server.tool()
    async def gui_graph(
        tab: str,
        graph: str,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Zoom graph `graph` of GUI tab `tab`: autoscale it first when
        `reset`, then set the x and y axis ranges [lo, hi]. Returns the
        ranges."""
        args = {"tab": tab, "graph": graph, "x": x, "y": y, "reset": reset}
        return await anyio.to_thread.run_sync(gui, "graph", args)

    @server.resource("lamino://project", name="project", description="project.toml", mime_type="application/toml")
    def project_toml() -> str:
        return session.need(ResourceError).path.read_text(encoding="utf-8")

    @server.resource(
        "lamino://log", name="log", description=f"The last {LOG_LINES} op calls", mime_type="application/jsonl"
    )
    def op_log() -> str:
        path = dispatch.log_path(session.need(ResourceError))
        if not path.exists():
            return ""
        with open(path, encoding="utf-8") as f:
            return "".join(deque(f, LOG_LINES))

    @server.resource(
        "lamino://guide/alignment",
        name="alignment guide",
        description="Finding a mount's center, tilt and sample band, and the methods that failed",
        mime_type="text/markdown",
    )
    def guide() -> str:
        return files("lamino_pal.mcp").joinpath("alignment.md").read_text(encoding="utf-8")

    @server.prompt()
    def align_mount(group: str) -> str:
        """Find mount group `group`'s center, tilt and sample band."""
        return f"""\
Align mount group {group} of the open project. Read lamino://guide/alignment
first and do not retry the methods its section 6 lists.

1. project_info: the group's scans and what is decided. Take a full-turn scan
   (raw_range_check).
2. align_rings on it. Look at the image: closed rings or open arcs? Arcs mean
   the scan was misaligned at acquisition: report that and stop.
3. align_center_sweep from the rings center. railed: move or widen the grid;
   flat: the slice holds no sample, fix the band; ambiguous: run
   align_center_compare on the winner and rivals and judge by eye (round
   particles, not dashes in a swirl).
4. group_decide center_px, citing the figure it was read from.
5. align_tilt_scan at that center; decide tilt_deg by eye on the slices, not
   by the focus values.
6. Read the band (recon_preview, or volume_focus_by_z on a volume) and
   group_decide z_band.

Report each decision with the figure it cites, and what a person should
confirm."""

    @server.prompt()
    def reconstruct_group(group: str) -> str:
        """Reconstruct and export every scan of mount group `group`."""
        return f"""\
Reconstruct every scan of mount group {group} of the open project.

1. project_info: the group needs center_px, tilt_deg and z_band decided; if
   one is missing, align it first (prompt align_mount).
2. recon_preview of one scan: check the slice and planes before the runs.
3. recon_run for each scan; follow each job with job_wait and job_result.
4. For each final volume: export_window, export_tiff16 of the band,
   export_mip, and export_bin for the 3D view.

Report the files written and anything that looked wrong."""

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lamino-pal-mcp")
    ap.add_argument(
        "--project",
        help="project.toml the tools act on, once project_open creates it if it does not exist; "
        "this process's CUDA runs on its interactive GPU",
    )
    a = ap.parse_args(argv)
    try:
        project = launch.open_or_pin(a.project)
    except (OSError, ValueError) as e:
        ap.error(f"cannot read project {a.project}: {e}")
    load_all()
    build(project).run("stdio")
