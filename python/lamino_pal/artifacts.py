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


@contextmanager
def written(path: str | os.PathLike[str]) -> Iterator[Path]:
    """A `.part` name beside `path` to write, renamed to `path` when the block
    ends and removed when it raises, so a killed job leaves no half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    try:
        yield part
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def save_h5(path: str | os.PathLike[str], meta: dict[str, Any], **arrays: np.ndarray) -> None:
    """`arrays` as datasets and `meta` as JSON in attribute `meta`."""
    with written(path) as part, h5py.File(part, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        for k, a in arrays.items():
            f.create_dataset(k, data=a)


def load_h5(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["meta"]), {k: f[k][()] for k in f}
