"""Job ops for test_jobs, registered in the test process by its fixture and
in the job process by job_main.py."""

import subprocess
import time
from pathlib import Path

from pydantic import Field

from lamino_pal.ops.registry import OptOutPath, OutputParams, Result, build


class StepsParams(OutputParams):
    n: int = 3
    dt: float = Field(0.05, ge=0)
    fail_at: int | None = None
    child: bool = False  # start `sleep 60` in the job's process group; its pid in out.child
    out: OptOutPath = None


class StepsResult(Result):
    n: int
    gpu: int | None
    child: int | None


def _steps(ctx, p):
    """Sleep `dt` `n` times, reporting progress after each, and write `out`."""
    Path(p.out).parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(["sleep", "60"]).pid if p.child else None
    if child:
        Path(p.out + ".child").write_text(str(child))
    for i in range(p.n):
        if i == p.fail_at:
            raise RuntimeError(f"failed at step {i}")
        time.sleep(p.dt)
        ctx.progress((i + 1) / p.n, f"step {i + 1}")
    Path(p.out).write_text(str(p.n))
    return StepsResult(n=p.n, gpu=ctx.gpu, child=child, artifacts=[p.out])


def _out(p, _):
    return {"out": f"artifacts/steps{p.n}.txt"}


OPS = [
    build("test.steps", cost="job", params=StepsParams, result=StepsResult, fn=_steps, default_out=_out),
    build("test.gpu_steps", cost="job", params=StepsParams, result=StepsResult, fn=_steps, gpu=True,
          default_out=_out),
]
