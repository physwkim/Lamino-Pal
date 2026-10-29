import hashlib
import json

import h5py
import numpy as np
import pytest
import tifffile
from conftest import DATA
from test_align import _group
from test_recon import _scan, run

from lamino_pal import Refused, artifacts
from lamino_pal.dispatch import Cancelled
from lamino_pal.ops import export
from lamino_pal.project import ZBand


def _volume(project, shape=(37, 12, 10), name="v.h5", **prov):
    vol = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    path = project.work_dir / name
    with h5py.File(path, "w") as f:
        f["exchange/data"] = vol
        if prov:
            f.require_group("lamino_pal/provenance").attrs.update(prov)
    return str(path), vol


@pytest.fixture
def small_bands(monkeypatch):
    # A few slices per band, so every op streams more than one.
    monkeypatch.setattr(export, "BAND_BYTES", 4 * 12 * 10 * 5)


def test_tiff16_is_export_tiff_py_streamed(project, run_job, small_bands):
    path, vol = _volume(project)
    seen = []
    r = run_job(project, "export.tiff16", {"volume": path, "z": (3, 30)}, lambda f, m: seen.append(m))
    # export_tiff.py, verbatim.
    v = vol[3:30]
    lo, hi = np.percentile(v[::8], [0.1, 99.9])
    want = np.clip((v - lo) / (hi - lo) * 65535, 0, 65535).astype(np.uint16)
    with tifffile.TiffFile(r.artifacts[0]) as t:
        assert json.loads(t.pages[0].description) == {
            "z_range": [3, 30], "window": [lo, hi], "shape": [27, 12, 10]
        }
        np.testing.assert_array_equal(t.asarray(), want)
    assert r.window == (lo, hi) and r.shape == (27, 12, 10)
    assert r.artifacts[0] == str(project.work_dir / "v_z3-30_16bit.tif")
    assert seen[-1] == "slice 27/27" and len(seen) > 3
    w = run(project, "export.window", volume=path, z=(3, 30))
    assert w.window == r.window and sum(w.hist) <= v[::8].size


def test_a_cancelled_export_leaves_no_file(project, run_job, small_bands):
    path, _ = _volume(project)

    def cancel(f, m):
        if f > 0.5:
            raise Cancelled("cancelled")

    with pytest.raises(Cancelled):
        run_job(project, "export.tiff16", {"volume": path, "z": (0, 37), "out": "t.tif"}, cancel)
    assert not [p.name for p in project.work_dir.iterdir() if "t.tif" in p.name]


def test_z_defaults_to_the_group_band_of_the_volumes_scan(project, run_job):
    _scan(project, "s")
    _group(project, ["s"], z_band=ZBand(z=(4, 20), bin=2, tilt_deg=20.0))
    path, _ = _volume(project, tag="s", bin=2, tilt_deg=20.0)
    assert run(project, "export.window", volume=path).z == (4, 21)
    other, _ = _volume(project, name="o.h5", tag="s", bin=2, tilt_deg=30.0)
    with pytest.raises(Refused, match="read at bin 2, tilt 20.0; .*o.h5 is bin 2, tilt 30.0: pass z"):
        run(project, "export.window", volume=other)
    bare, _ = _volume(project, name="b.h5")
    with pytest.raises(Refused, match="b.h5 has no provenance naming its scan: pass z"):
        run_job(project, "export.mip", {"volume": bare})
    with pytest.raises(Refused, match=r"slices \[30, 40\) are outside the 37-slice volume"):
        run(project, "export.window", volume=bare, z=(30, 40))


def test_bin_means_each_2x2x2_cube_and_drops_an_odd_edge(project, run_job, small_bands):
    path, vol = _volume(project, tag="s", bin=2, tilt_deg=20.0)
    r = run_job(project, "export.bin", {"volume": path})
    assert r.shape == (18, 6, 5) and r.artifacts[0] == str(project.work_dir / "v_binned.h5")
    with h5py.File(r.artifacts[0], "r") as f:
        np.testing.assert_array_equal(
            f["exchange/data"][()], vol[:36].reshape(18, 2, 6, 2, 5, 2).mean((1, 3, 5))
        )
        prov = dict(f["lamino_pal/provenance"].attrs)
    assert (prov["bin"], prov["tilt_deg"], prov["binned_from"]) == (4, 20.0, path)


def test_bin_of_a_16bit_tiff_is_export_bin_py_streamed(project, run_job, small_bands):
    path, _ = _volume(project)
    tif = run_job(project, "export.tiff16", {"volume": path, "z": (0, 37)}).artifacts[0]
    seen = []
    r = run_job(project, "export.bin", {"volume": tif}, lambda f, m: seen.append(m))
    assert r.shape == (18, 6, 5) and r.artifacts[0] == str(project.work_dir / "v_z0-37_16bit_binned.tif")
    # export_bin.py, verbatim.
    a = tifffile.imread(tif)
    z, y, x = (s // 2 * 2 for s in a.shape)
    a = a[:z, :y, :x].astype(np.float32)
    a = a.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean((1, 3, 5))
    with tifffile.TiffFile(r.artifacts[0]) as t:
        assert t.is_imagej and t.pages[0].compression == 1
        np.testing.assert_array_equal(t.asarray(), a.astype(np.uint16))
    assert seen[-1] == "slice 18/18" and len(seen) > 3
    tifffile.imwrite(project.work_dir / "f.tif", np.zeros((4, 3), np.float32))
    with pytest.raises(Refused, match="is a float32 4x3 TIFF, not a 16-bit stack"):
        run_job(project, "export.bin", {"volume": str(project.work_dir / "f.tif")})


def test_mip_takes_each_axis_maximum(project, run_job, small_bands):
    path, vol = _volume(project)
    r = run_job(project, "export.mip", {"volume": path, "z": (2, 33)})
    meta, a = artifacts.load_h5(r.artifacts[0])
    v = vol[2:33]
    for k, axis in [("z", 0), ("y", 1), ("x", 2)]:
        np.testing.assert_array_equal(a[k], v.max(axis))
    assert meta == {"z": [2, 33], "shape": [37, 12, 10]}


@pytest.mark.golden
def test_tiff16_writes_the_261003_export_byte_for_byte(golden_project, run_job):
    d = DATA / "tomoxide"
    params = {"volume": str(d / "final_step1000_bin2_c504_t20_fourierrec_parzen.h5"), "z": (380, 700)}
    r = run_job(golden_project, "export.tiff16", params)
    assert r.window == (-0.0030219246202614156, 0.0030576782301068306)
    want = d / "final_step1000_bin2_c504_t20_z380-700_16bit.tif"
    md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in (want, r.artifacts[0])]
    assert md5[0] == md5[1]


@pytest.mark.golden
def test_bin_of_the_261003_tiff_writes_its_bin4_tiff_byte_for_byte(golden_project, run_job):
    d = DATA / "tomoxide"
    r = run_job(golden_project, "export.bin", {"volume": str(d / "final_step1000_bin2_c504_t20_z380-700_16bit.tif")})
    want = d / "final_step1000_bin4_c504_t20_z380-700_16bit.tif"
    md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in (want, r.artifacts[0])]
    assert md5[0] == md5[1]
