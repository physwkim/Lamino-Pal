"""What an op judged by eye shows the agent: its arrays as one 8-bit image,
each panel named and windowed in the caption rather than drawn on it. A job
op's arrays come from its artifact, the others' with the result. The GUI
saves the evidence its decisions cite here too."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from lamino_pal.artifacts import written
from lamino_pal.render import GAP, LONG_SIDE, grid, png, to_u8, window

Arrays = dict[str, np.ndarray]
FIGURES = Path("artifacts/figures")  # under the work directory
_FIGURES: dict[str, Callable[[dict[str, Any], Arrays], tuple[np.ndarray, str]]] = {}


def _figure(*ops: str):
    def register(fn):
        _FIGURES.update(dict.fromkeys(ops, fn))
        return fn

    return register


def has(op: str) -> bool:
    return op in _FIGURES


def render(op: str, result: dict[str, Any], arrays: Arrays) -> tuple[bytes, str] | None:
    """Op `op`'s figure as PNG, with its caption; None for an op without one."""
    f = _FIGURES.get(op)
    if f is None:
        return None
    img, caption = f(result, arrays)
    return png(img), caption


def save(work_dir: Path, name: str, image: bytes) -> Path:
    """PNG `image` as `FIGURES`/`name` under `work_dir`, kept when it exists,
    so a decision can cite what was seen; the path relative to `work_dir`."""
    rel = FIGURES / name
    path = work_dir / rel
    if not path.exists():
        with written(path) as part:
            part.write_bytes(image)
    return rel


def _w(w: tuple[float, float]) -> str:
    return f"[{w[0]:.4g}, {w[1]:.4g}]"


def _own(a: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    w = window(a)
    return to_u8(a, w), w


def _zoom_at(stack: np.ndarray, s: int) -> tuple[int, int]:
    """The top left of the s×s zoom, centred where the slices' gradient
    energy, each slice on its own window and the sum smoothed over s/4,
    peaks inside the 0.92-FOV disk the focus score is measured in: the ring
    outside it holds geometry artifacts."""
    _, ny, nx = stack.shape
    e = np.zeros((2 * ny, 2 * nx))  # zero padded: the smoothing does not wrap
    for a in stack:
        gy, gx = np.gradient(to_u8(a, window(a)).astype(np.float32))
        e[:ny, :nx] += gx * gx + gy * gy
    yy, xx = np.mgrid[0:ny, 0:nx]
    e[:ny, :nx][np.hypot(yy - ny / 2, xx - nx / 2) >= 0.46 * min(ny, nx)] = 0
    f2 = np.fft.fftfreq(2 * ny)[:, None] ** 2 + np.fft.rfftfreq(2 * nx)[None, :] ** 2
    smooth = np.fft.irfft2(np.fft.rfft2(e) * np.exp(-2 * (np.pi * s / 4) ** 2 * f2), e.shape)[:ny, :nx]
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    return int(np.clip(cy - s // 2, 0, ny - s)), int(np.clip(cx - s // 2, 0, nx - s))


def _tiles(stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, str, str]:
    """A column per slice of `stack`, as the 261003 compare scripts drew them:
    an s×s zoom at full resolution above the whole slice shrunk to s, s as
    wide as one row of them fits `LONG_SIDE`; each panel on its own window.
    The zoom, where `_zoom_at` finds the detail, as text too."""
    n, ny, nx = stack.shape
    s = min(ny, nx, (LONG_SIDE - GAP * (n - 1)) // n)
    y0, x0 = _zoom_at(stack, s)
    k = s / max(ny, nx)
    zooms, wholes, lines = [], [], []
    for a, label in zip(stack, labels):
        z, wz = _own(a[y0 : y0 + s, x0 : x0 + s])
        t, w = _own(a)
        if k < 1:
            t = np.asarray(Image.fromarray(t).resize((round(nx * k), round(ny * k)), Image.Resampling.LANCZOS))
        zooms.append(z)
        wholes.append(t)
        lines.append(f"- {label}, zoom window {_w(wz)}, slice window {_w(w)}")
    zoom = f"the zoom [{y0}:{y0 + s}, {x0}:{x0 + s}] at full resolution above the whole slice"
    return grid([zooms, wholes]), zoom, "\n".join(lines)


def _ortho(sl: np.ndarray, xz: np.ndarray, yz: np.ndarray, ws: list[tuple[float, float]]) -> np.ndarray:
    """The slice top left, xz below it and yz turned to stand right of it."""
    a, b, c = (to_u8(x, w) for x, w in zip((sl, xz, yz.T), ws))
    return grid([[a, c], [b, None]])


@_figure("align.rings")
def _mean(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    t, w = _own(a["mean"])
    return t, f"Mean −log projection [row, column], window {_w(w)}."


@_figure("align.center_sweep")
def _sweep(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    v, c, f = r["verdict"], r["candidates_px"], r["focus"]
    role = {0: "first", len(c) - 1: "last", v["index"]: "best"}
    role.update({x["index"]: "rival" for x in v["rivals"]})
    keep = sorted(role)
    labels = [f"center {c[i]:.2f} px ({role[i]}, focus {f[i]:.4g})" for i in keep]
    img, zoom, lines = _tiles(a["frames"][keep], labels)
    head = f"Slice {r['slice']} at tilt {r['tilt_deg']:g} for the best, rival and edge candidates, left to right"
    return img, f"{head}, {zoom}:\n{lines}"


@_figure("align.tilt_scan")
def _tilts(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    labels = [f"tilt {t['tilt_deg']:g} deg, slice {t['z_peak']} (focus {t['focus']:.4g})" for t in r["tilts"]]
    img, zoom, lines = _tiles(a["best"], labels)
    return img, f"Each tilt's best slice at center {r['center_px']:.2f} px, left to right, {zoom}:\n{lines}"


@_figure("align.center_compare")
def _centers(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    labels = [f"center {c['center_px']:.2f} px, slice {c['z_peak']} (focus {c['focus']:.4g})" for c in r["centers"]]
    img, zoom, lines = _tiles(a["best"], labels)
    return img, f"Each center's best slice at tilt {r['tilt_deg']:g}, left to right, {zoom}:\n{lines}"


@_figure("recon.preview", "volume.ortho")
def _planes(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    z, y, x = r["at"]
    img = _ortho(a["slice"], a["xz"], a["yz"], [tuple(r["window"])] * 3)
    return img, (
        f"Slice z={z} [y, x] top left, the xz plane at y={y} [z, x] below it, the yz plane at x={x} "
        f"turned to [y, z] right of it; all on the slice's window {_w(r['window'])}."
    )


@_figure("volume.slice")
def _slice(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    return to_u8(a["slice"], r["window"]), f"Slice z={r['z']} [y, x], window {_w(r['window'])}."


def _columns(sl: np.ndarray, xz: np.ndarray, ws: list[tuple[float, float]]) -> np.ndarray:
    """One column per stack entry: its slice above its xz plane, both on its window."""
    return grid([[to_u8(s, w) for s, w in zip(sl, ws)], [to_u8(p, w) for p, w in zip(xz, ws)]])


@_figure("recon.compare")
def _variants(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    ws = [tuple(v["window"]) for v in r["variants"]]
    labels = [v["label"] + (f", CNR {v['cnr']:.3g}" if v["cnr"] is not None else "") for v in r["variants"]]
    z, y = r["at"]
    lines = "\n".join(f"- {lb}, window {_w(w)}" for lb, w in zip(labels, ws))
    head = f"Variants left to right: slice z={z} above, the xz plane at y={y} below, on its slice's window:"
    return _columns(a["slice"], a["xz"], ws), f"{head}\n{lines}"


@_figure("recon.split")
def _halves(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    ws = [window(s) for s in a["slice"]]
    labels = [f"{h} projections ({n})" for h, n in zip(("even", "odd"), r["nproj"])]
    z, y = r["at"]
    lines = "\n".join(f"- {lb}, window {_w(w)}" for lb, w in zip(labels, ws))
    head = f"Even half left, odd right: slice z={z} above, the xz plane at y={y} below, on its slice's window:"
    return _columns(a["slice"], a["xz"], ws), f"{head}\n{lines}"


@_figure("export.mip")
def _mip(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    ws = [window(a[k]) for k in ("z", "y", "x")]
    z0, z1 = r["z"]
    return _ortho(a["z"], a["y"], a["x"], ws), (
        f"Maximum intensity projections of slices [{z0}, {z1}): down z [y, x] top left (window {_w(ws[0])}), "
        f"along y [z, x] below it (window {_w(ws[1])}), along x turned to [y, z] right of it (window {_w(ws[2])})."
    )


@_figure("volume.focus_by_z")
def _focus(r: dict[str, Any], a: Arrays) -> tuple[np.ndarray, str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    f = np.asarray(r["focus"])
    z = np.arange(f.size) * r["step"]
    fig = Figure(figsize=(10, 3.5), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.plot(z, f, color="black", lw=1)
    if r["band"] is not None:
        ax.axvspan(*r["band"], color="0.85", lw=0)
    ax.axvline(r["peak_z"], color="0.4", lw=1, ls="--")
    ax.set(xlabel="slice z", ylabel="focus (mean |∇|²)", xlim=(0, r["shape"][0]))
    canvas.draw()
    band = "none" if r["band"] is None else f"[{r['band'][0]}, {r['band'][1]}] (shaded)"
    return np.asarray(canvas.buffer_rgba())[..., :3].copy(), (
        f"Focus of every {r['step']}th slice of {r['shape'][0]}; suggested band {band}, "
        f"peak at z={r['peak_z']} (dashed)."
    )
