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


def test_recipe_config_refuses_what_a_reconstruction_would():
    c = json.loads(_core.recipe_config(json.dumps({"algorithm": "fourierrec"})))
    assert (c["algorithm"], c["filter_name"], c["dtype"]) == ("fourierrec", "parzen", "float32")
    for recipe, why in [
        ({"filter": "shepp"}, "has no field filter"),
        ({"algorithm": "fbp,sirt"}, "is a chain"),
        ({"dtype": "int8"}, "unknown dtype"),
        ({"num_iter": "x"}, "expected usize"),
        ([], "not a table"),
    ]:
        with pytest.raises(Refused, match=why):
            _core.recipe_config(json.dumps(recipe))


def test_valid_fov_spans_the_reconstruction_depth():
    depth = _core.lamino_depth(1024, 20.0)
    v = json.loads(_core.valid_fov(1024, 1024, 504.0, 20.0))
    assert (v["depth"], len(v["radius"])) == (depth, depth)
    z0, z1 = v["full_z"]
    assert 0 < z0 < z1 < depth - 1
    assert all(r >= v["r_lateral"] for r in v["radius"][z0 : z1 + 1])


def test_focus_by_z_ranks_the_sharp_slice_first():
    rng = np.random.default_rng(2)
    vol = np.zeros((3, 32, 32), "f4")
    vol[1] = rng.normal(size=(32, 32))
    f = _core.focus_by_z(vol)
    assert len(f) == 3 and f[1] > 0 and f[0] == f[2] == 0


@pytest.fixture
def turn(tmp_path):
    """16 projections of 8x32 over a full turn: big enough to reconstruct."""
    rng = np.random.default_rng(3)
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = rng.integers(400, 600, (16, 8, 32)).astype("u2")
        f["exchange/data_white"] = np.full((1, 8, 32), 1000, "u2")
        f["exchange/data_dark"] = np.zeros((1, 8, 32), "u2")
        f["exchange/theta"] = np.arange(16, dtype="f4") * 22.5
    return path


def test_select_keeps_the_named_projections(turn):
    s = _core.Stack(turn)
    assert s.select([1, 3, 5]).shape == (3, 8, 32)
    with pytest.raises(LaminoPalError, match="out of range"):
        s.select([16])


def test_a_fired_cancel_stops_the_center_sweep(turn):
    s = _core.Stack(turn)
    sl = s.default_slice(20.0)
    frames, focus = s.center_sweep(20.0, sl, [15.5, 16.0], _core.CancelToken())
    assert frames.shape[0] == len(focus) == 2
    cancel = _core.CancelToken()
    cancel.cancel()
    with pytest.raises(LaminoPalError, match="cancelled"):
        s.center_sweep(20.0, sl, [15.5, 16.0], cancel)


def test_planes_cross_at_the_voxel(turn):
    s = _core.Stack(turn)
    recipe = json.dumps({"algorithm": "fourierrec"})
    rh = s.depth(20.0)
    z, y, x = rh // 2, 10, 20
    rows = []
    sl, xz, yz = s.planes(recipe, 16.0, 20.0, (z, y, x), lambda done, n: rows.append((done, n)))
    assert (sl.shape, xz.shape, yz.shape) == ((32, 32), (rh, 32), (rh, 32))
    assert rows[-1] == (rh, rh)
    np.testing.assert_array_equal(xz[z], sl[y])
    np.testing.assert_array_equal(yz[z], sl[:, x])
    assert np.abs(sl).max() > 0
    with pytest.raises(Refused, match="outside"):
        s.planes(recipe, 16.0, 20.0, (rh, y, x))

    def stop(done, n):
        raise KeyError("stop")

    with pytest.raises(KeyError, match="stop"):
        s.planes(recipe, 16.0, 20.0, (z, y, x), stop)
