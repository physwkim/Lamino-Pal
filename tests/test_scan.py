import unicodedata

import h5py
import numpy as np
import pytest
import tifffile

from lamino_pal import Refused, dispatch
from lamino_pal.project import Project, Scan

STEP = unicodedata.normalize("NFD", "360도 step")
STEP_RAW = f"{STEP}/S_N4_proj"
FLY_RAW = "fly/F_00001"


def _frames(d, names, h=4, w=6, base=0):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        tifffile.imwrite(d / n, (base + 10 * i + np.arange(h * w)).reshape(h, w).astype("u2"))


@pytest.fixture
def raw(project):
    """A 4-projection step scan and a 4-frame 360 deg fly scan of 4x6 frames."""
    root = project.raw_root
    (root / STEP).mkdir()
    (root / STEP / "S_N4.dat").write_text(
        "# Energy Origin(E0) : 8333.0(eV)\n"
        "# StartAngle : 0.0\tStopAngle : 360.0\tStepAngle : 90.0\n"
        + "".join(f"{a}\t1\n" for a in [0.0, 90.1, 180.0, 270.0, 360.0])
    )
    _frames(root / STEP_RAW, [f"S_N4_{i:04}.tif" for i in range(1, 5)] + ["S_N4_ff_0001.tif"])
    fly = root / FLY_RAW
    _frames(fly, [f"TOMO_{i:05}.tif" for i in range(4)] + ["FF_00000.tif"], base=1)
    (fly / "metadata.txt").write_text(
        '{"exptime": 0.5, "numprj": 4, "scan_speed": 180.0, "current_energy": 8332.9}'
    )
    return root


def _discover(project):
    return dispatch.run("scan.discover", {}, project=project, caller="cli")


def test_discover_lists_both_kinds_and_what_the_project_made_of_them(project, raw, run_job):
    d = _discover(project)
    assert [(s.kind, s.raw, s.tag, s.n, s.imported) for s in d.scans] == [
        ("step", STEP_RAW, "step4", 4, []),
        ("fly", FLY_RAW, "f360_N4_180dps", 4, []),
    ]
    run_job(project, "scan.import", {"raw": STEP_RAW, "tag": "s"})
    assert [s.imported for s in _discover(project).scans] == [["s"], []]


def test_import_writes_the_h5_and_commits_the_scan(project, raw, run_job):
    seen = []
    r = run_job(
        project,
        "scan.import",
        {"raw": STEP_RAW, "tag": "s"},
        lambda frac, msg: seen.append((frac, msg)),
    )
    assert (r.h5, r.shape, r.saturated) == ("h5/s_bin2.h5", (4, 2, 3), 0)
    assert seen[-1] == (1.0, "5/5 frames")
    with h5py.File(project.work_dir / r.h5) as f:
        assert f["exchange/theta"][()].tolist() == pytest.approx([0.0, 90.1, 180.0, 270.0])
        # Frame 1 block (0, 0): 10 + (0 + 1 + 6 + 7)/4 = 13.5 -> 14.
        assert f["exchange/data"][1, 0, 0] == 14
        assert f["lamino_pal"].attrs["source"] == STEP_RAW
    assert Project(project.path).data.scan == [
        Scan(tag="s", kind="step", raw=STEP_RAW, h5="h5/s_bin2.h5", bin=2)
    ]


def test_reimport_needs_overwrite_and_keeps_the_group(project, raw, run_job):
    run_job(project, "scan.import", {"raw": STEP_RAW, "tag": "s"})
    dispatch.run("scan.assign_group", {"tags": ["s"], "group": "m1"}, project=project, caller="cli")
    seen = []
    with pytest.raises(Refused, match="scan s exists"):
        run_job(
            project,
            "scan.import",
            {"raw": STEP_RAW, "tag": "s", "out": "b.h5"},
            lambda frac, msg: seen.append(msg),
        )
    # Refused before reading a frame.
    assert seen == ["waiting for another import"]
    assert not (project.work_dir / "b.h5").exists()
    run_job(project, "scan.import", {"raw": FLY_RAW, "tag": "s", "bin": 1, "overwrite": True})
    (s,) = Project(project.path).data.scan
    assert (s.raw, s.bin, s.h5, s.group) == (FLY_RAW, 1, "h5/s_bin1.h5", "m1")


def test_an_import_that_loses_its_tag_removes_its_file(project, raw, run_job, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f["exchange/data"] = np.zeros((2, 3, 3), "u2")
        f["exchange/theta"] = np.zeros(2, "f4")

    def adopt_meanwhile(frac, msg):
        if msg.endswith("frames") and not project.reload().scan:
            dispatch.run(
                "scan.adopt",
                {"tag": "s", "h5": str(other), "bin": 1, "kind": "fly"},
                project=project,
                caller="mcp",
            )

    with pytest.raises(Refused, match="scan s exists"):
        run_job(project, "scan.import", {"raw": STEP_RAW, "tag": "s"}, adopt_meanwhile)
    assert list((project.work_dir / "h5").iterdir()) == []
    (s,) = Project(project.path).data.scan
    assert s.h5 == str(other)


def test_a_progress_exception_stops_the_import(project, raw, run_job):
    def stop(frac, msg):
        if frac > 0:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_job(project, "scan.import", {"raw": STEP_RAW, "tag": "s"}, stop)
    assert list((project.work_dir / "h5").iterdir()) == []
    assert Project(project.path).data.scan == []


def test_adopt_checks_the_file_against_what_it_is_told(project, raw, run_job, tmp_path):
    run_job(project, "scan.import", {"raw": FLY_RAW, "tag": "f"})
    h5 = "h5/f_bin2.h5"
    adopt = lambda **kw: dispatch.run(  # noqa: E731
        "scan.adopt", {"h5": h5, "bin": 2, "kind": "fly", **kw}, project=project, caller="cli"
    )
    with pytest.raises(Refused, match="binning is 2, not 1"):
        adopt(tag="a", bin=1)
    with pytest.raises(Refused, match="kind is 'fly', not 'step'"):
        adopt(tag="a", kind="step")
    with pytest.raises(Refused, match="source"):
        adopt(tag="a", raw=STEP_RAW)
    with pytest.raises(Refused, match="scan f exists"):
        adopt(tag="f")
    assert adopt(tag="a").shape == (4, 2, 3)
    assert Project(project.path).data.find_scan("a").raw == FLY_RAW

    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f["exchange/data"] = np.zeros((3, 2, 2), "u2")
        f["exchange/theta"] = np.zeros(2, "f4")
    with pytest.raises(Refused, match=r"data \(3, 2, 2\) with theta \(2,\)"):
        adopt(tag="b", h5=str(bad), bin=1)


def test_assign_group_creates_moves_and_unassigns(project, raw, run_job):
    run_job(project, "scan.import", {"raw": STEP_RAW, "tag": "s"})
    run_job(project, "scan.import", {"raw": FLY_RAW, "tag": "f"})
    assign = lambda tags, group: dispatch.run(  # noqa: E731
        "scan.assign_group", {"tags": tags, "group": group}, project=project, caller="gui"
    )
    assert assign(["s", "f"], "m1").created
    assert not assign(["f"], "m1").created
    assign(["f"], "m2")
    assign(["s"], None)
    data = Project(project.path).data
    assert [(s.tag, s.group) for s in data.scan] == [("s", None), ("f", "m2")]
    assert [g.name for g in data.group] == ["m1", "m2"]
    with pytest.raises(Refused, match=r"no scan \['x'\]"):
        assign(["s", "x"], "m1")
    assert Project(project.path).data.find_scan("s").group is None
