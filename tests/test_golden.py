"""`scan.import` against the h5 files build_step_h5.py and build_fly_h5.py made
from the 261003 data. Missing data fails: these are the import's proof."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from lamino_pal import dispatch
from lamino_pal.project import Project

pytestmark = pytest.mark.golden

DATA = Path(os.environ.get("LAMINO_PAL_DATA", "~/work/261003_Lamino_test")).expanduser()

# Proposed tag -> the original bin-2 file, and the projection pixels at 65535
# its build log reported.
GOLDEN = {
    "step1000": ("tomoxide/step1000_bin2.h5", 8440686),
    "step800": ("tomoxide/step800_bin2.h5", 7135555),
    **{
        f"f360_{n}": (f"tomoxide/fly/f360_{n}_bin2.h5", None)
        for n in ["N400_5dps", "N444_4p5dps", "N500_4dps", "N571_3p5dps", "N667_3dps"]
    },
    **{
        f"f180_{n.rsplit('_', 2)[0]}": (f"tomoxide/fly/f180_{n}_bin2.h5", None)
        for n in [
            "N250_4dps_0p18s_0p75ET",
            "N286_3p5dps_0p18s_0p86ET",
            "N333_3dps_0p18s_1ET",
            "N400_2p5dps_0p18s_1p2ET",
            "N500_2dps_0p18s_1p5ET",
        ]
    },
}


@pytest.fixture(scope="module")
def found(tmp_path_factory):
    assert DATA.is_dir(), f"LAMINO_PAL_DATA {DATA} is missing"
    p = Project.create(tmp_path_factory.mktemp("golden") / "project.toml", str(DATA))
    return p, {s.tag: s for s in dispatch.run("scan.discover", {}, project=p, caller="cli").scans}


def test_discover_finds_every_261003_scan(found):
    assert sorted(found[1]) == sorted(GOLDEN)


@pytest.mark.parametrize("tag", list(GOLDEN))
def test_import_equals_the_original_h5(found, run_job, tag):
    project, scans = found
    golden, saturated = GOLDEN[tag]
    r = run_job(project, "scan.import", {"raw": scans[tag].raw, "tag": tag})
    out = project.work_dir / r.h5
    try:
        if saturated is not None:
            assert r.saturated == saturated
        with h5py.File(out) as a, h5py.File(DATA / golden) as b:
            for k in ["data", "data_white", "data_dark", "theta"]:
                x, y = a[f"exchange/{k}"], b[f"exchange/{k}"]
                assert (x.shape, x.dtype) == (y.shape, y.dtype), k
                assert np.array_equal(x[()], y[()]), k
    finally:
        out.unlink(missing_ok=True)
