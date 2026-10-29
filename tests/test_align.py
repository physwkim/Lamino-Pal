import h5py
import numpy as np
import pytest
from conftest import DATA

from lamino_pal import Refused, dispatch
from lamino_pal.artifacts import load_h5
from lamino_pal.project import Project, ZBand


def _scan(project, tag, theta, shape=(8, 16)):
    path = project.work_dir / f"{tag}.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["exchange/data"] = rng.integers(400, 600, (len(theta), *shape)).astype("u2")
        f["exchange/data_white"] = np.full((1, *shape), 1000, "u2")
        f["exchange/data_dark"] = np.zeros((1, *shape), "u2")
        f["exchange/theta"] = np.asarray(theta, "f4")
    run(project, "scan.adopt", tag=tag, h5=str(path), bin=2, kind="step")


def run(project, name, caller="cli", **params):
    return dispatch.run(name, params, project=project, caller=caller)


def _group(project, tags, **fields):
    run(project, "scan.assign_group", tags=tags, group="m")

    def set_(f):
        for k, v in fields.items():
            setattr(f.group[0], k, v)

    project.commit(set_)


def test_rings_refuse_a_half_turn(project):
    _scan(project, "h", [0, 45, 90, 135])
    with pytest.raises(Refused, match="h is a half turn: its rings are open arcs"):
        run(project, "align.rings", tag="h")


def test_sweep_refuses_a_z_band_of_another_binning(project):
    _scan(project, "s", [0, 90, 180, 270])
    _group(project, ["s"], z_band=ZBand(z=(2, 6), bin=4, tilt_deg=20.0))
    with pytest.raises(Refused, match="read at bin 4; scan s is bin 2"):
        run(project, "align.center_sweep", tag="s", center_px=15.5)


def test_sweep_refuses_a_z_band_read_at_another_tilt(project):
    _scan(project, "s", [0, 90, 180, 270])
    _group(project, ["s"], z_band=ZBand(z=(2, 6), bin=2, tilt_deg=20.0))
    with pytest.raises(Refused, match="read at tilt 20.0; this sweep is at 18.0"):
        run(project, "align.center_sweep", tag="s", center_px=15.5, tilt_deg=18.0)


def test_tilt_scan_needs_a_decided_center(project, run_job):
    _scan(project, "s", [0, 90, 180, 270])
    with pytest.raises(Refused, match="s has no decided center"):
        run_job(project, "align.tilt_scan", {"tag": "s"})


@pytest.fixture(scope="module")
def mount(tmp_path_factory):
    """step1000 in group m with the 261003 sample band at bin 2, tilt 20."""
    p = Project.create(tmp_path_factory.mktemp("mount") / "project.toml", str(DATA))
    run(p, "scan.adopt", tag="step1000", h5=str(DATA / "tomoxide/step1000_bin2.h5"), bin=2, kind="step")
    return p


@pytest.mark.golden
def test_step1000_rings_are_not_trustworthy(golden_project):
    r = run(golden_project, "align.rings", tag="step1000")
    # Measured at bin 2: 495.805, prominence 3.08; quarter bands 493.45 495.51 494.40 498.17.
    assert (r.center_px, r.prominence, r.trustworthy) == (pytest.approx(992.11, abs=0.01),
                                                          pytest.approx(3.08, abs=0.01), False)
    want = [2 * c + 0.5 for c in [493.45, 495.51, 494.40, 498.17]]
    assert [b.center_px for b in r.bands] == pytest.approx(want, abs=0.02)
    assert [b.rows for b in r.bands] == [(0, 256), (256, 512), (512, 768), (768, 1024)]
    with pytest.raises(Refused, match="f180_N500 is a half turn"):
        run(golden_project, "align.rings", tag="f180_N500")


@pytest.mark.golden
def test_step1000_sweep_from_rings_is_ambiguous(golden_project):
    r = run(golden_project, "align.center_sweep", tag="step1000")
    assert (r.center_from, r.tilt_deg, r.slice, r.depth) == ("rings", 20.0, 545, 1090)
    assert len(r.candidates_px) == 49 and r.arrays["frames"].shape == (49, 1024, 1024)
    v = r.verdict
    # Measured at bin 2: 504.305, rival 487.805 behind by 0.0518.
    assert (v.kind, v.center_px) == ("ambiguous", pytest.approx(1009.11, abs=0.01))
    assert [(x.center_px, round(x.deficit, 4)) for x in v.rivals][0] == (pytest.approx(976.11, abs=0.01), 0.0518)


@pytest.mark.golden
def test_tilt_scan_carries_the_band_to_each_tilt(mount, run_job):
    _group(mount, ["step1000"], center_px=1008.5, z_band=ZBand(z=(420, 600), bin=2, tilt_deg=20.0))
    seen = []
    r = run_job(mount, "align.tilt_scan", {"tag": "step1000", "tilts_deg": [18.0, 20.0, 22.0]},
                lambda frac, msg: seen.append(frac))
    # tomoxide align --tilt_width at 499.31 carried 420..600 at 20 deg to these.
    assert [(t.tilt_deg, t.band, t.depth) for t in r.tilts] == [
        (18.0, (415, 593), 1078), (20.0, (420, 600), 1090), (22.0, (426, 609), 1106)
    ]
    assert seen[-3:] == pytest.approx([1 / 3, 2 / 3, 1.0])
    meta, arrays = load_h5(r.artifacts[0])
    assert r.artifacts[0].endswith("artifacts/step1000_tilts_cgroup_t18-22.h5")
    assert meta["center_px"] == 1008.5 and arrays["best"].shape == (3, 1024, 1024)


@pytest.mark.golden
def test_center_compare_keeps_each_centers_best_slice(mount, run_job):
    _group(mount, ["step1000"], center_px=1008.5, z_band=ZBand(z=(420, 600), bin=2, tilt_deg=20.0))
    r = run_job(mount, "align.center_compare", {"tag": "step1000", "centers_px": [1008.5, 976.11]})
    assert [(c.center_px, c.band) for c in r.centers] == [(1008.5, (420, 600)), (976.11, (420, 600))]
    assert all(420 <= c.z_peak < 600 for c in r.centers)
    meta, arrays = load_h5(r.artifacts[0])
    assert arrays["best"].shape == (2, 1024, 1024) and meta["tilt_deg"] == 20.0
