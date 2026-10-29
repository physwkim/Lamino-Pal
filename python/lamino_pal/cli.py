"""`lamino-pal run`: one op from the shell, its result as JSON on stdout.

A job op run this way runs here, in the foreground, on the GPU the shell gave
TOMOXIDE_CUDA_DEVICES. `lamino-pal job <dir>` is the body of a job that
`jobs.submit` started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from lamino_pal import LaminoPalError, dispatch, jobs
from lamino_pal.ops import REGISTRY, load_all
from lamino_pal.project import Project


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lamino-pal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one op and print its result")
    r.add_argument("--project", required=True, help="project.toml")
    r.add_argument("op")
    r.add_argument("params", nargs="?", help="params JSON file, - for stdin; default {}")
    j = sub.add_parser("job")
    j.add_argument("dir")
    a = ap.parse_args(argv)
    load_all()
    if a.cmd == "job":
        return jobs.run(Path(a.dir))
    if a.params is None:
        params = {}
    else:
        params = json.load(sys.stdin if a.params == "-" else open(a.params))
    project = Project(a.project)
    op = REGISTRY.get(a.op)
    run = dispatch.run_job if op is not None and op.cost == "job" else dispatch.run

    def progress(frac: float, msg: str) -> None:
        print(f"{frac:6.1%} {msg}", file=sys.stderr)

    try:
        res = run(a.op, params, project=project, caller="cli", progress=progress)
    except (LaminoPalError, ValidationError) as e:
        print(f"lamino-pal: {e}", file=sys.stderr)
        return 2
    print(res.model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
