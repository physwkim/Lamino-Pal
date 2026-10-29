"""The op registry: every analysis exists only as a registered op.

GUI actions, MCP tools and the CLI are generated from `REGISTRY` and run ops
through `lamino_pal.dispatch`.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, PrivateAttr

Cost = Literal["instant", "interactive", "job"]


class _Out:
    """Marks a params field as an output path (see `OutPath`)."""


class _In:
    """Marks a params field as an input path (see `InPath`)."""


# Resolved against the work directory; the dispatcher refuses one outside it,
# under the raw root, or existing without `overwrite`.
OutPath = Annotated[str, _Out]
# An `OutPath` the caller may omit (`= None`); the op's `default_out` names it.
# Not `OutPath | None`: pydantic drops a marker inside a Union arm, which
# `build` refuses.
OptOutPath = Annotated[str | None, _Out]
# Resolved against the work directory and must exist.
InPath = Annotated[str, _In]


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputParams(Params):
    overwrite: bool = False


class Result(BaseModel):
    """JSON result plus in-process arrays, which never reach the log or MCP."""

    artifacts: list[str] = []
    _arrays: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def arrays(self) -> dict[str, Any]:
        return self._arrays

    def with_arrays(self, **arrays: Any) -> Result:
        self._arrays.update(arrays)
        return self


@dataclass(frozen=True)
class Op:
    name: str
    cost: Cost
    params: type[Params]
    result: type[Result]
    fn: Callable[..., Result]
    gpu: bool
    # Output-path field -> name, from the validated params; covers every
    # optional `OutPath` field (checked in `build`).
    default_out: Callable[[Any], dict[str, str]] | None = None
    # (params, project) -> params with the defaults the project decides filled
    # in. It runs at validation, so a job runs with the values it was
    # submitted with and `default_out` names the file from them. It must
    # leave filled params as they are: the job validates them again.
    bind: Callable[[Any, Any], Any] | None = None

    @property
    def doc(self) -> str:
        return self.fn.__doc__ or ""

    def path_fields(self, marker: type) -> list[str]:
        return [
            k for k, f in self.params.model_fields.items() if marker in f.metadata
        ]


REGISTRY: dict[str, Op] = {}


def _buried(ann: Any, marker: type) -> bool:
    if get_origin(ann) is Annotated:
        return marker in ann.__metadata__ or _buried(get_args(ann)[0], marker)
    return any(_buried(a, marker) for a in get_args(ann))

_NAME = re.compile(r"^[a-z]+\.[a-z0-9_]+$")


def build(
    name: str,
    *,
    cost: Cost,
    params: type[Params],
    result: type[Result],
    fn: Callable[..., Result],
    gpu: bool = False,
    default_out: Callable[[Any], dict[str, str]] | None = None,
    bind: Callable[[Any, Any], Any] | None = None,
) -> Op:
    if not _NAME.match(name):
        raise ValueError(f"op name {name!r} is not `area.verb`")
    if not (fn.__doc__ or "").strip():
        raise ValueError(f"op {name} has no docstring (its MCP description)")
    if not (issubclass(params, Params) and issubclass(result, Result)):
        raise TypeError(f"op {name}: params/result must derive Params/Result")
    for k, f in params.model_fields.items():
        for m in (_Out, _In):
            if m not in f.metadata and _buried(f.annotation, m):
                raise TypeError(f"op {name}: {k} hides its path marker from the dispatcher")
    o = Op(name, cost, params, result, fn, gpu, default_out, bind)
    if o.path_fields(_Out) and not issubclass(params, OutputParams):
        raise TypeError(f"op {name} writes files; its params must derive OutputParams")
    optional = [k for k in o.path_fields(_Out) if not params.model_fields[k].is_required()]
    if optional and default_out is None:
        raise TypeError(f"op {name}: optional output paths {optional} need default_out")
    return o


def op(
    name: str,
    *,
    cost: Cost,
    params: type[Params],
    result: type[Result],
    gpu: bool = False,
    default_out: Callable[[Any], dict[str, str]] | None = None,
    bind: Callable[[Any, Any], Any] | None = None,
) -> Callable[[Callable[..., Result]], Callable[..., Result]]:
    def register(fn: Callable[..., Result]) -> Callable[..., Result]:
        if name in REGISTRY:
            raise ValueError(f"op {name} registered twice")
        REGISTRY[name] = build(
            name,
            cost=cost,
            params=params,
            result=result,
            fn=fn,
            gpu=gpu,
            default_out=default_out,
            bind=bind,
        )
        return fn

    return register


def load_all() -> dict[str, Op]:
    """Import every module of `lamino_pal.ops`, registering its ops."""
    pkg = importlib.import_module("lamino_pal.ops")
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name != "registry":
            importlib.import_module(f"lamino_pal.ops.{m.name}")
    return REGISTRY
