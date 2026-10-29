import json
import os

import h5py
import numpy as np
import pytest
from conftest import DATA
from test_recon import _scan, run

from lamino_pal import Refused, _core, dispatch
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


@pytest.mark.parametrize("layout", [{}, {"chunks": (2, 12, 12), "compression": "gzip"}])
def test_ortho_reads_the_planes_through_the_voxel(project, layout):
    vol = np.random.default_rng(0).normal(size=(10, 12, 14)).astype(np.float32)
    path = _volume(project.work_dir / "v.h5", vol, **layout)
    # tomoxide's contiguous layout is mapped; a chunked one is read through h5py.
    assert isinstance(volume.data(path), np.memmap) == (not layout)
    r = run(project, "volume.ortho", volume=path, z=3, y=4, x=5)
    assert r.at == (3, 4, 5) and r.shape == (10, 12, 14)
    for k, want in [("slice", vol[3]), ("xz", vol[:, 4, :]), ("yz", vol[:, :, 5])]:
        np.testing.assert_array_equal(r.arrays[k], want)
    r = run(project, "volume.slice", volume=path)
    assert r.z == 5 and r.window == tuple(np.percentile(vol[5], [0.5, 99.5]))
    np.testing.assert_array_equal(r.arrays["slice"], vol[5])
    with pytest.raises(Refused, match="column 14 is outside the 14-column volume"):
        run(project, "volume.ortho", volume=path, x=14)


def test_a_replaced_volume_is_read_anew(project):
    path = _volume(project.work_dir / "v.h5", np.zeros((4, 8, 8), np.float32))
    assert run(project, "volume.slice", volume=path).arrays["slice"].max() == 0
    part = _volume(project.work_dir / "v.part.h5", np.ones((4, 8, 8), np.float32))
    os.replace(part, path)
    assert run(project, "volume.slice", volume=path).arrays["slice"].min() == 1


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
    np.testing.assert_array_equal(r.focus, _core.focus_by_z(vol[::step].copy()))
    assert r.band == (40, 98 + (step == 1)) and 40 <= r.peak_z <= 99
    assert seen[-1] == "slice 150/150"


def test_suggest_band_takes_the_longest_run_and_none_of_a_blank_volume():
    f = np.r_[np.ones(40), np.zeros(40), np.ones(60), np.zeros(10)]
    assert volume.suggest_band(f) == (80, 139)
    assert volume.suggest_band(np.zeros(20)) is None


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
    assert (r.r_lateral, r.radius, r.full_z) == (want["r_lateral"], want["radius"], tuple(want["full_z"]))
    with pytest.raises(Refused, match=f"is {depth}x64x64; scan s at tilt 30.0 reconstructs"):
        run(project, "volume.valid_fov", volume=path, tilt_deg=30.0)


@pytest.mark.golden
@pytest.mark.parametrize(
    "name, band",
    [
        # README: sample 390..1089, air 150..390, z wrap 0..150 (read by eye).
        ("final_step1000_bin2_c504_t20_fourierrec_parzen.h5", (373, 1076)),
        # The f360 mount's sample fills the whole depth.
        ("fly/final_f360_N667_3dps_c500_t20_fourierrec_parzen.h5", (0, 1089)),
    ],
)
def test_focus_by_z_band_on_the_261003_finals(golden_project, name, band):
    r = run(golden_project, "volume.focus_by_z", volume=str(DATA / "tomoxide" / name))
    assert r.band == band


@pytest.mark.golden
def test_valid_fov_of_the_bin2_final(golden_project):
    path = str(DATA / "tomoxide/final_step1000_bin2_c504_t20_fourierrec_parzen.h5")
    r = run(golden_project, "volume.valid_fov", volume=path, tag="step1000", center_px=1008.5, tilt_deg=20.0)
    # Bin-2 column 504 of 1024: the width allows 504 px, full at slices 184..906.
    assert (r.r_lateral, len(r.radius), r.full_z) == (504.0, 1090, (184, 906))
