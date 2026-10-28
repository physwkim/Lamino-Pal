"""The op registry: every analysis exists only as a registered op.

GUI actions, MCP tools and the CLI are generated from `REGISTRY` and run ops
through `lamino_pal.dispatch`.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

Cost = Literal["instant", "interactive", "job"]


class _Out:
    """Marks a params field as an output path (see `OutPath`)."""


class _In:
    """Marks a params field as an input path (see `InPath`)."""


# Resolved against the work directory; the dispatcher refuses one outside it,
# under the raw root, or existing without `overwrite`.
OutPath = Annotated[str, _Out]
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

    @property
    def doc(self) -> str:
        return self.fn.__doc__ or ""

    def path_fields(self, marker: type) -> list[str]:
        return [
            k for k, f in self.params.model_fields.items() if marker in f.metadata
        ]


REGISTRY: dict[str, Op] = {}

_NAME = re.compile(r"^[a-z]+\.[a-z0-9_]+$")


def build(
    name: str,
    *,
    cost: Cost,
    params: type[Params],
    result: type[Result],
    fn: Callable[..., Result],
    gpu: bool = False,
) -> Op:
    if not _NAME.match(name):
        raise ValueError(f"op name {name!r} is not `area.verb`")
    if not (fn.__doc__ or "").strip():
        raise ValueError(f"op {name} has no docstring (its MCP description)")
    if not (issubclass(params, Params) and issubclass(result, Result)):
        raise TypeError(f"op {name}: params/result must derive Params/Result")
    o = Op(name, cost, params, result, fn, gpu)
    if o.path_fields(_Out) and not issubclass(params, OutputParams):
        raise TypeError(f"op {name} writes files; its params must derive OutputParams")
    return o


def op(
    name: str,
    *,
    cost: Cost,
    params: type[Params],
    result: type[Result],
    gpu: bool = False,
) -> Callable[[Callable[..., Result]], Callable[..., Result]]:
    def register(fn: Callable[..., Result]) -> Callable[..., Result]:
        if name in REGISTRY:
            raise ValueError(f"op {name} registered twice")
        REGISTRY[name] = build(
            name, cost=cost, params=params, result=result, fn=fn, gpu=gpu
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
