from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from lamino_pal.dispatch import Context, log_path
from lamino_pal.ops.registry import Params, Result, op


class TailParams(Params):
    n: int = Field(20, ge=1, le=1000)
    op: str | None = None


class TailResult(Result):
    entries: list[dict[str, Any]]


@op("log.tail", cost="instant", params=TailParams, result=TailResult)
def tail(ctx: Context, p: TailParams) -> TailResult:
    """Return the last `n` op-log entries, oldest first, optionally of one op."""
    path = log_path(ctx.project)
    if not path.exists():
        return TailResult(entries=[])
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if p.op is not None:
        entries = [e for e in entries if e.get("op") == p.op]
    return TailResult(entries=entries[-p.n :])
