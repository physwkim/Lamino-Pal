"""Display windows for images ops return, and the 8-bit montages and PNGs
the MCP server shows them as."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

LONG_SIDE = 1568  # px: a model sees no more of an image than this
GAP = 4  # px of white between tiles


def window(a: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    """The `lo`-`hi` percentile window of the finite values of `a`."""
    f = a[np.isfinite(a)]
    if f.size == 0:
        return 0.0, 0.0
    w = np.percentile(f, [lo, hi])
    return float(w[0]), float(w[1])


def to_u8(a: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    """`a` over window `w` as 0–255, clipped; NaN is 0."""
    lo, hi = w
    scale = 255 / (hi - lo) if hi > lo else 0.0
    return np.clip(np.nan_to_num((np.asarray(a, np.float32) - lo) * scale), 0, 255).astype(np.uint8)


def grid(rows: list[list[np.ndarray | None]]) -> np.ndarray:
    """8-bit tiles in rows and columns, each top-left in a cell as tall as its
    row's tallest tile and as wide as its column's widest; None leaves a cell
    empty."""
    ncol = max(len(r) for r in rows)
    cells = [r + [None] * (ncol - len(r)) for r in rows]
    h = [max((t.shape[0] for t in r if t is not None), default=0) for r in cells]
    w = [max((r[j].shape[1] for r in cells if r[j] is not None), default=0) for j in range(ncol)]
    out = np.full((sum(h) + GAP * (len(h) - 1), sum(w) + GAP * (ncol - 1)), 255, np.uint8)
    y = 0
    for i, r in enumerate(cells):
        x = 0
        for j, t in enumerate(r):
            if t is not None:
                out[y : y + t.shape[0], x : x + t.shape[1]] = t
            x += w[j] + GAP
        y += h[i] + GAP
    return out


def montage(tiles: list[np.ndarray]) -> np.ndarray:
    """8-bit tiles left to right, then top to bottom, in about a square."""
    ncol = int(np.ceil(np.sqrt(len(tiles))))
    return grid([tiles[i : i + ncol] for i in range(0, len(tiles), ncol)])


def png(a: np.ndarray) -> bytes:
    """8-bit image `a` as PNG, shrunk to at most `LONG_SIDE` on its longer side."""
    im = Image.fromarray(a)
    k = LONG_SIDE / max(im.size)
    if k < 1:
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
