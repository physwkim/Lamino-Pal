import h5py
import numpy as np
import pytest

from lamino_pal import Refused, _core, dispatch

Y, X = np.mgrid[:48, :64]


def _blob(cy, cx):
    return np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / 30)


def _scan(project, tag, theta, shifts=None):
    """A bin-2 scan whose −log projections are a blob at (20, 30), moved by
    `shifts[i]` in projection i."""
    shifts = shifts or {}
    data = [
        1000 * np.exp(-_blob(20 + dy, 30 + dx))
        for dy, dx in (shifts.get(i, (0, 0)) for i in range(len(theta)))
    ]
    path = project.work_dir / f"{tag}.h5"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = np.round(data).astype("u2")
        f["exchange/data_white"] = np.full((1, 48, 64), 1000, "u2")
        f["exchange/data_dark"] = np.zeros((1, 48, 64), "u2")
        f["exchange/theta"] = np.asarray(theta, "f4")
    params = {"tag": tag, "h5": str(path), "bin": 2, "kind": "step"}
    dispatch.run("scan.adopt", params, project=project, caller="cli")
    return path


def _run(project, name, **params):
    return dispatch.run(name, params, project=project, caller="cli")


def test_frame_is_the_prepped_projection_with_its_angle(project):
    path = _scan(project, "s", [0, 90, 180, 270, 360])
    r = _run(project, "raw.frame", tag="s", index=3)
    assert (r.index, r.theta_deg, r.shape) == (3, 270.0, (48, 64))
    a = _core.prep_frames(path, [3])[0]
    np.testing.assert_array_equal(r.arrays["frame"], a)
    assert r.window == pytest.approx(tuple(np.percentile(a, [0.5, 99.5])))
    with pytest.raises(Refused, match="projection 5"):
        _run(project, "raw.frame", tag="s", index=5)
    with pytest.raises(Refused, match="no scan t"):
        _run(project, "raw.frame", tag="t")


def test_mean_projection_counts_what_it_averaged(project):
    _scan(project, "s", [0, 90, 180, 270, 360])
    r = _run(project, "raw.mean_projection", tag="s", step=2)
    assert (r.n, r.shape) == (3, (48, 64))


@pytest.mark.parametrize(
    "theta, kind, turn_index",
    [
        ([0, 90, 180, 270, 360], "full", 4),  # step: closed at 360
        ([0, 90, 180, 270], "full", None),  # fly: endpoint excluded
        ([0, 90, 180], "half", None),
        ([0, 90], "half", None),
        ([0, 90, 180, 270, 360, 450], "partial", 4),
    ],
)
def test_range_check_names_the_turn(project, theta, kind, turn_index):
    _scan(project, "s", theta)
    r = _run(project, "raw.range_check", tag="s")
    assert (r.nproj, r.step_deg, r.kind, r.turn_index) == (len(theta), 90.0, kind, turn_index)


def test_range_check_warns_of_uneven_and_backward_steps(project):
    _scan(project, "s", [0, 90, 180, 330, 300, 360])
    w = _run(project, "raw.range_check", tag="s").warnings
    assert w == [
        "angles do not increase at projection 4",
        "uneven steps: one is 120 deg off the median 90",
    ]


def test_drift_is_where_the_turn_frame_sits_in_full_pixels(project):
    _scan(project, "s", [0, 90, 180, 270, 360], shifts={4: (3, -5)})
    r = _run(project, "raw.drift", tag="s")
    assert (r.first, r.turn, r.bin) == (0, 4, 2)
    assert (r.dy_px, r.dx_px) == pytest.approx((6.0, -10.0), abs=0.25)
    assert set(r.arrays) == {"first", "turn"}


def test_drift_needs_the_frame_one_turn_on(project):
    _scan(project, "s", [0, 90, 180, 270])
    with pytest.raises(Refused, match="no projection at 360.00 deg"):
        _run(project, "raw.drift", tag="s")


@pytest.mark.golden
def test_step1000_turn_and_drift(golden_project):
    r = _run(golden_project, "raw.range_check", tag="step1000")
    assert (r.nproj, r.kind, r.turn_index) == (1001, "full", 1000)
    assert r.step_deg == pytest.approx(0.36, abs=1e-4)
    d = _run(golden_project, "raw.drift", tag="step1000")
    # Measured at bin 2: the 360 deg frame sits (+0.2, −6.4) off the first.
    assert (d.dy_px, d.dx_px) == pytest.approx((0.4, -12.8), abs=0.2)


@pytest.mark.golden
def test_f180_is_a_half_turn_without_drift(golden_project):
    r = _run(golden_project, "raw.range_check", tag="f180_N500")
    assert (r.nproj, r.kind, r.turn_index) == (500, "half", None)
    with pytest.raises(Refused, match="no projection at 360"):
        _run(golden_project, "raw.drift", tag="f180_N500")


@pytest.mark.golden
def test_step1000_tilt_flow(golden_project):
    r = _run(golden_project, "raw.tilt_flow", tag="step1000")
    assert (r.tilt_deg, r.tilt_sigma_deg) == pytest.approx((19.40, 1.46), abs=0.01)
    assert (r.axis_px, r.axis_sigma_px) == pytest.approx((958.18, 149.56), abs=0.05)
    assert (len(r.pairs), r.group_tilt_deg) == (24, None)
