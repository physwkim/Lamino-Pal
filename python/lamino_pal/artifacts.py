"""Files job ops leave for the eye: the arrays their results describe."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def save_h5(path: str | os.PathLike[str], meta: dict[str, Any], **arrays: np.ndarray) -> None:
    """`arrays` as datasets and `meta` as JSON in attribute `meta`, written
    under a `.part` name and renamed, so a killed job leaves no half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    try:
        with h5py.File(part, "w") as f:
            f.attrs["meta"] = json.dumps(meta)
            for k, a in arrays.items():
                f.create_dataset(k, data=a)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def load_h5(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["meta"]), {k: f[k][()] for k in f}
