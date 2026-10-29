"""The one place a detector column converts between binnings.

A group stores its center as a full-resolution (bin 1) column; a bin-`b` file
is the `b`-pixel block mean, so full column `c` sits at `(c - (b-1)/2) / b`.
"""

from __future__ import annotations

from lamino_pal.project import Scan


def center_for_scan(scan: Scan, c_full: float) -> float:
    """The column of `scan`'s h5 that is full-resolution column `c_full`."""
    b = scan.bin
    return (c_full - (b - 1) / 2) / b


def center_full(scan: Scan, c_b: float) -> float:
    """The full-resolution column that is column `c_b` of `scan`'s h5."""
    b = scan.bin
    return c_b * b + (b - 1) / 2
