"""The one place a decided value converts into a scan's or volume's own
units: a detector column between binnings, a z band between volumes.

A group stores its center as a full-resolution (bin 1) column; a bin-`b` file
is the `b`-pixel block mean, so full column `c` sits at `(c - (b-1)/2) / b`.
"""

from __future__ import annotations

from lamino_pal import Refused, _core
from lamino_pal.project import Scan, ZBand


def center_for_scan(scan: Scan, c_full: float) -> float:
    """The column of `scan`'s h5 that is full-resolution column `c_full`."""
    b = scan.bin
    return (c_full - (b - 1) / 2) / b


def center_full(scan: Scan, c_b: float) -> float:
    """The full-resolution column that is column `c_b` of `scan`'s h5."""
    b = scan.bin
    return c_b * b + (b - 1) / 2


def band_for(zb: ZBand, bin: int, tilt_deg: float, depth: int) -> tuple[int, int]:
    """The rows of z band `zb` in a `depth`-deep bin-`bin` volume at
    `tilt_deg`, carried from the volume it was read from through the detector
    rows. A band is not carried across binnings."""
    if zb.bin != bin:
        raise Refused(f"the z band was read at bin {zb.bin}; this volume is bin {bin}: pass the slices")
    return _core.band_at(tuple(zb.z), zb.depth, zb.tilt_deg, tilt_deg, depth)
