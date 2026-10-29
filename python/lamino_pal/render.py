"""Display windows for images ops return."""

from __future__ import annotations

import numpy as np


def window(a: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    """The `lo`-`hi` percentile window of the finite values of `a`."""
    f = a[np.isfinite(a)]
    if f.size == 0:
        return 0.0, 0.0
    w = np.percentile(f, [lo, hi])
    return float(w[0]), float(w[1])
