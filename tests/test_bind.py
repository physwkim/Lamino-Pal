import json

import h5py
import numpy as np
import pytest

from lamino_pal import LaminoPalError, Refused, _core


@pytest.fixture
def dxchange(tmp_path):
    """4 projections of 6x8 with 2 flats and a nonzero dark."""
    rng = np.random.default_rng(1)
    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = rng.integers(200, 800, (4, 6, 8)).astype("u2")
        f["exchange/data_white"] = rng.integers(900, 1000, (2, 6, 8)).astype("u2")
        f["exchange/data_dark"] = rng.integers(10, 20, (1, 6, 8)).astype("u2")
        f["exchange/theta"] = np.array([0, 90, 180, 270], "f4")
    return path


def test_prep_frames_are_minus_log_of_the_flat_dark_ratio(dxchange):
    with h5py.File(dxchange) as f:
        d, w, k = (f[f"exchange/{n}"][()].astype("f8") for n in ["data", "data_white", "data_dark"])
    want = -np.log((d - k.mean(0)) / (w.mean(0) - k.mean(0)))
    got = _core.prep_frames(dxchange, [2, 0])
    assert (got.shape, got.dtype) == ((2, 6, 8), np.float32)
    np.testing.assert_allclose(got, want[[2, 0]], rtol=1e-5)
    np.testing.assert_allclose(_core.mean_projection(dxchange, 2), want[[0, 2]].mean(0), rtol=1e-5)
    with pytest.raises(Refused, match="projection 4"):
        _core.prep_frames(dxchange, [4])
    with pytest.raises(LaminoPalError):
        _core.prep_frames(dxchange.with_name("none.h5"), [0])


def test_stack_loads_the_whole_file(dxchange):
    s = _core.Stack(dxchange)
    assert s.shape == (4, 6, 8)
    assert s.nbytes == 4 * 6 * 8 * 4


def test_register_shift_moves_moving_onto_reference():
    y, x = np.mgrid[:64, :64]
    blob = lambda cy, cx: np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 20).astype("f4")  # noqa: E731
    dy, dx = _core.register_shift(blob(30, 30), blob(27, 35), 10.0)
    assert (dy, dx) == pytest.approx((3.0, -5.0), abs=0.11)


def test_grid_and_judge_sweep():
    c = _core.grid(10.0, 1.0, 0.5)
    assert c == [9.0, 9.5, 10.0, 10.5, 11.0]
    v = json.loads(_core.judge_sweep(c, [1.0, 2.0, 5.0, 2.0, 1.0]))
    assert (v["kind"], v["index"], v["value"], v["rivals"]) == ("resolved", 2, 10.0, [])
    assert json.loads(_core.judge_sweep(c, [5.0, 4.0, 3.0, 2.0, 1.0]))["kind"] == "railed"
    with pytest.raises(Refused, match="5 candidates and 2 scores"):
        _core.judge_sweep(c, [1.0, 2.0])
