"""Job ops for test_jobs, registered in the test process by its fixture and
in the job process by job_main.py."""

import subprocess
import time
from pathlib import Path

from pydantic import Field

from lamino_pal.artifacts import written
from lamino_pal.ops.registry import OptOutPath, OutputParams, Result, build


class StepsParams(OutputParams):
    n: int = 3
    dt: float = Field(0.05, ge=0)
    fail_at: int | None = None
    child: bool = False  # start `sleep 60` in the job's process group; its pid in out.child
    child_step: bool = False  # first wait on `sleep 60`, failing if it fails; its pid in out.child
    out: OptOutPath = None


class StepsResult(Result):
    n: int
    gpu: int | None
    child: int | None


def _steps(ctx, p):
    """Sleep `dt` `n` times, reporting progress after each, while writing `out`."""
    with written(p.out) as part:
        part.write_text("")
        child = subprocess.Popen(["sleep", "60"]).pid if p.child else None
        if child:
            Path(p.out + ".child").write_text(str(child))
        if p.child_step:
            proc = subprocess.Popen(["sleep", "60"])
            Path(p.out + ".child").write_text(str(proc.pid))
            if proc.wait() != 0:
                raise RuntimeError(f"sleep exited {proc.returncode}")
        for i in range(p.n):
            if i == p.fail_at:
                raise RuntimeError(f"failed at step {i}")
            time.sleep(p.dt)
            ctx.progress((i + 1) / p.n, f"step {i + 1}")
        part.write_text(str(p.n))
    return StepsResult(n=p.n, gpu=ctx.gpu, child=child, artifacts=[p.out])


def _out(p, _):
    return {"out": f"artifacts/steps{p.n}.txt"}


OPS = [
    build("test.steps", cost="job", params=StepsParams, result=StepsResult, fn=_steps, default_out=_out),
    build("test.gpu_steps", cost="job", params=StepsParams, result=StepsResult, fn=_steps, gpu=True,
          default_out=_out),
]
