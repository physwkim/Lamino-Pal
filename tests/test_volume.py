import json
import os
import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import DATA
from test_recon import _scan, run

from lamino_pal import Refused, _core, dispatch, locks
from lamino_pal.artifacts import written
from lamino_pal.ops import volume


def _volume(path, vol, **kw):
    with h5py.File(path, "w") as f:
        f.create_dataset("exchange/data", data=vol, **kw)
    return str(path)


def _layered(depth=150, n=48, band=(40, 99)):
    """Noise texture in the band, a smooth ramp elsewhere."""
    rng = np.random.default_rng(1)
    vol = np.tile(np.linspace(0, 1, n, dtype=np.float32), (depth, n, 1))
    vol[band[0] : band[1] + 1] += rng.normal(0, 1, (band[1] - band[0] + 1, n, n)).astype(np.float32)
    return vol


CHUNKED = {"chunks": (2, 12, 12), "compression": "gzip"}


@pytest.mark.parametrize("hold", [True, False])
@pytest.mark.parametrize("layout", [{}, CHUNKED])
def test_ortho_reads_the_planes_through_the_voxel(project, monkeypatch, layout, hold):
    monkeypatch.setattr(volume, "HOLD", hold)
    vol = np.random.default_rng(0).normal(size=(10, 12, 14)).astype(np.float32)
    path = _volume(project.work_dir / "v.h5", vol, **layout)
    # tomoxide's contiguous layout is mapped; a chunked one is read through h5py.
    assert isinstance(volume.data(path), np.memmap) == (hold and not layout)
    assert isinstance(volume.data(path), volume.Unheld) == (not hold)
    r = run(project, "volume.ortho", volume=path, z=3, y=4, x=5)
    assert r.at == (3, 4, 5) and r.shape == (10, 12, 14)
    for k, want in [("slice", vol[3]), ("xz", vol[:, 4, :]), ("yz", vol[:, :, 5])]:
        np.testing.assert_array_equal(r.arrays[k], want)
    r = run(project, "volume.slice", volume=path)
    assert r.z == 5 and r.window == tuple(np.percentile(vol[5], [0.5, 99.5]))
    np.testing.assert_array_equal(r.arrays["slice"], vol[5])
    with pytest.raises(Refused, match="column 14 is outside the 14-column volume"):
        run(project, "volume.ortho", volume=path, x=14)


@pytest.mark.parametrize("layout", [{}, CHUNKED])
def test_a_replaced_volume_is_read_anew(project, layout):
    path = _volume(project.work_dir / "v.h5", np.zeros((4, 12, 12), np.float32), **layout)
    assert run(project, "volume.slice", volume=path).arrays["slice"].max() == 0
    part = _volume(project.work_dir / "v.part.h5", np.ones((4, 12, 12), np.float32), **layout)
    os.replace(part, path)
    assert run(project, "volume.slice", volume=path).arrays["slice"].min() == 1


def test_an_unheld_volume_replaced_by_another_shape_is_refused(project, monkeypatch):
    monkeypatch.setattr(volume, "HOLD", False)
    path = _volume(project.work_dir / "v.h5", np.zeros((4, 8, 8), np.float32))
    d = volume.data(path)
    os.replace(_volume(project.work_dir / "v.part.h5", np.ones((4, 8, 6), np.float32)), path)
    with pytest.raises(Refused, match="is no longer the 4x8x8 float32 volume"):
        d[0]


def _read_open(path, monkeypatch):
    """`path` open as a read holds it; where not on Windows, refused a replace
    while open as Windows refuses it."""
    fh = open(path, "rb")
    if os.name != "nt":
        replace = os.replace

        def refused_while_open(src, dst):
            if not fh.closed and Path(dst) == Path(path):
                raise PermissionError(13, "Access is denied", str(dst))
            replace(src, dst)

        monkeypatch.setattr(os, "replace", refused_while_open)
        monkeypatch.setattr(locks, "REPLACE_WAIT_S", 60.0)
    return fh


def test_a_volume_being_read_is_replaced_once_the_read_ends(project, monkeypatch):
    path = _volume(project.work_dir / "v.h5", np.zeros((4, 8, 8), np.float32))
    threading.Timer(0.3, _read_open(path, monkeypatch).close).start()
    with written(path) as part:
        _volume(part, np.ones((4, 8, 8), np.float32))
    assert run(project, "volume.slice", volume=path).arrays["slice"].min() == 1


def test_a_volume_held_open_past_the_wait_is_left_as_it_was(project, monkeypatch):
    path = _volume(project.work_dir / "v.h5", np.zeros((4, 8, 8), np.float32))
    with _read_open(path, monkeypatch):
        monkeypatch.setattr(locks, "REPLACE_WAIT_S", 0.3)
        with pytest.raises(PermissionError, match="was not replaced in 0.3 s: another program holds it open"):
            with written(path) as part:
                _volume(part, np.ones((4, 8, 8), np.float32))
    assert run(project, "volume.slice", volume=path).arrays["slice"].max() == 0
    assert not list(project.work_dir.glob(".v.part*"))


def test_a_file_without_a_volume_is_refused(project):
    path = project.work_dir / "x.h5"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = np.zeros((4, 4), np.float32)
    with pytest.raises(Refused, match="has no 3-D exchange/data"):
        run(project, "volume.slice", volume=str(path))


@pytest.mark.parametrize("step", [1, 2])
def test_focus_by_z_scores_every_slice_and_finds_the_textured_band(project, step):
    vol = _layered()
    path = _volume(project.work_dir / "v.h5", vol)
    seen = []
    r = dispatch.run(
        "volume.focus_by_z", {"volume": path, "step": step}, project=project, caller="cli",
        progress=lambda f, m: seen.append(m),
    )
    # Chunked reads score each slice as one call over the whole volume does.
    np.testing.assert_array_equal(r.arrays["focus"], _core.focus_by_z(vol[::step].copy()))
    assert "focus" not in r.model_dump()
    assert r.band == (40, 98 + (step == 1)) and 40 <= r.peak_z <= 99
    assert (r.air, r.wrap) == ([(0, 38 + (step == 1)), (100, 148 + (step == 1))], None)
    assert seen[-1] == "slice 150/150"


def test_z_regions_take_the_longest_run_and_none_of_a_blank_volume():
    f = np.r_[np.ones(40), np.zeros(40), np.ones(60), np.zeros(10)]
    # The sharp run at z 0 is not wrap: the band's run reaches neither end.
    assert volume.z_regions(f) == ((80, 139), [(40, 79), (140, 149)], None)
    assert volume.z_regions(np.zeros(20)) == (None, [], None)


def test_z_regions_mark_the_wrap_at_the_end_the_bands_run_does_not_reach():
    f = np.r_[np.full(30, 0.3), np.full(40, 0.01), np.ones(80)]
    assert volume.z_regions(f) == ((70, 149), [(30, 69)], (0, 29))
    assert volume.z_regions(f[::-1]) == ((0, 79), [(80, 119)], (120, 149))
    assert volume.z_regions(f[::2], step=2) == ((70, 148), [(30, 68)], (0, 28))


def test_z_regions_mark_no_air_where_the_least_focus_reaches_a_quarter_of_the_peak():
    f = np.r_[np.full(30, 0.3), np.full(40, 0.25), np.ones(80)]
    assert volume.z_regions(f) == ((70, 149), [], None)


def test_valid_fov_takes_the_scan_center_and_tilt_from_the_provenance(project):
    _scan(project, "s", shape=(32, 64))
    depth = _core.lamino_depth(32, 20.0)
    path = _volume(project.work_dir / "v.h5", np.zeros((depth, 64, 64), np.float32))
    with pytest.raises(Refused, match="has no provenance: pass tag, center_px and tilt_deg"):
        run(project, "volume.valid_fov", volume=path)
    with h5py.File(path, "r+") as f:
        f.require_group("lamino_pal/provenance").attrs.update(tag="s", center_px=64.5, tilt_deg=20.0)
    r = run(project, "volume.valid_fov", volume=path)
    want = json.loads(_core.valid_fov(64, 32, 32.0, 20.0))
    assert (r.tag, r.center_px, r.tilt_deg) == ("s", 64.5, 20.0)
    assert (r.r_lateral, r.full_z) == (want["r_lateral"], tuple(want["full_z"]))
    assert r.arrays["radius"].tolist() == want["radius"] and "radius" not in r.model_dump()
    with pytest.raises(Refused, match=f"is {depth}x64x64; scan s at tilt 30.0 reconstructs"):
        run(project, "volume.valid_fov", volume=path, tilt_deg=30.0)


def test_valid_fov_of_a_binned_volume_is_in_its_voxels(project, run_job):
    _scan(project, "s", shape=(32, 64))
    depth = _core.lamino_depth(32, 20.0)
    path = _volume(project.work_dir / "v.h5", np.zeros((depth, 64, 64), np.float32))
    with h5py.File(path, "r+") as f:
        f.require_group("lamino_pal/provenance").attrs.update(tag="s", center_px=64.5, tilt_deg=20.0, bin=2)
    binned = run_job(project, "export.bin", {"volume": path}).artifacts[0]
    full = json.loads(_core.valid_fov(64, 32, 32.0, 20.0))
    r = run(project, "volume.valid_fov", volume=binned)
    # Whole at full resolution on slices 13..23: binned, the pairs 14..23.
    assert (full["full_z"], r.r_lateral, r.full_z) == ([13, 23], 16.0, (7, 11))
    assert len(r.arrays["radius"]) == depth // 2
    assert r.arrays["radius"][6] == full["radius"][12] / 2 < 16.0
    with h5py.File(binned, "r+") as f:
        f["lamino_pal/provenance"].attrs["bin"] = 3
    with pytest.raises(Refused, match="is bin 3, not a multiple of scan s's bin 2"):
        run(project, "volume.valid_fov", volume=binned)


@pytest.mark.golden
@pytest.mark.parametrize(
    "name, band, air, wrap",
    [
        # README: sample 390..1089, air 150..390, z wrap 0..150 (read by eye).
        ("final_step1000_bin2_c504_t20_fourierrec_parzen.h5", (373, 1076), [(143, 340)], (0, 142)),
        # The f360 mount's sample fills the whole depth.
        ("fly/final_f360_N667_3dps_c500_t20_fourierrec_parzen.h5", (0, 1089), [], None),
    ],
)
def test_focus_by_z_regions_on_the_261003_finals(golden_project, name, band, air, wrap):
    r = run(golden_project, "volume.focus_by_z", volume=str(DATA / "tomoxide" / name))
    assert (r.band, r.air, r.wrap) == (band, air, wrap)


@pytest.mark.golden
def test_valid_fov_of_the_bin2_final(golden_project):
    path = str(DATA / "tomoxide/final_step1000_bin2_c504_t20_fourierrec_parzen.h5")
    r = run(golden_project, "volume.valid_fov", volume=path, tag="step1000", center_px=1008.5, tilt_deg=20.0)
    # Bin-2 column 504 of 1024: the width allows 504 px, full at slices 184..906.
    assert (r.r_lateral, len(r.arrays["radius"]), r.full_z) == (504.0, 1090, (184, 906))
