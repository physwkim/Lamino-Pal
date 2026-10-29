"""Files job ops leave for the eye: the arrays their results describe."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# This process's scratch files in open `scratch` blocks.
_scratch: set[Path] = set()


@contextmanager
def scratch(path: Path) -> Iterator[Path]:
    """`path`, removed when the block ends, or by `discard` when the process
    exits from under the block."""
    _scratch.add(path)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        _scratch.discard(path)


def discard() -> None:
    """Remove the scratch files of the open `scratch` blocks: the process is
    exiting from under them."""
    for path in list(_scratch):
        path.unlink(missing_ok=True)


@contextmanager
def written(path: str | os.PathLike[str]) -> Iterator[Path]:
    """A scratch name beside `path` to write, renamed to `path` when the block
    ends, so a stopped job leaves no half file. It keeps `path`'s suffix: a
    writer that appends a missing one writes the name given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with scratch(path.with_name(f".{path.stem}.part{path.suffix}")) as part:
        yield part
        os.replace(part, path)


def save_h5(path: str | os.PathLike[str], meta: dict[str, Any], **arrays: np.ndarray) -> None:
    """`arrays` as datasets and `meta` as JSON in attribute `meta`."""
    with written(path) as part, h5py.File(part, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        for k, a in arrays.items():
            f.create_dataset(k, data=a)


def load_h5(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["meta"]), {k: f[k][()] for k in f}
