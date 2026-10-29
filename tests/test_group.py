import pytest

from lamino_pal import Refused, dispatch
from lamino_pal.project import Group, ZBand


def run(project, name, caller="cli", **params):
    return dispatch.run(name, params, project=project, caller=caller)


def test_decide_sets_the_field_and_records_who_and_why(project):
    project.commit(lambda f: f.group.append(Group(name="m")))
    ev = project.work_dir / "artifacts" / "sweep.png"
    ev.parent.mkdir()
    ev.write_bytes(b"")
    r = run(project, "group.decide", group="m", field="center_px", value=1008.5, evidence=[str(ev)])
    assert (r.previous, r.by) == (None, "user")
    r = run(project, "group.decide", "mcp", group="m", field="center_px", value=1009.0,
            evidence=["artifacts/sweep.png"], method="sweep", note="rival 976 blurred")
    assert (r.previous, r.by) == (1008.5, "agent")
    g = project.reload().group[0]
    assert g.center_px == 1009.0
    assert [(d.value, d.by, d.evidence, d.method, d.note) for d in g.decisions] == [
        (1008.5, "user", ["artifacts/sweep.png"], None, ""),
        (1009.0, "agent", ["artifacts/sweep.png"], "sweep", "rival 976 blurred"),
    ]
    band = {"z": [420, 600], "bin": 2, "tilt_deg": 20.0}
    run(project, "group.decide", group="m", field="z_band", value=band, evidence=["artifacts/sweep.png"])
    assert project.reload().group[0].z_band == ZBand(**band)


@pytest.mark.parametrize(
    "params, why",
    [
        ({"group": "x", "field": "tilt_deg", "value": 20.0}, "no group x"),
        ({"group": "m", "field": "z_band", "value": 20.0}, "z_band takes a z band"),
        ({"group": "m", "field": "tilt_deg", "value": {"z": [1, 2], "bin": 2, "tilt_deg": 20.0}},
         "tilt_deg takes a number"),
        ({"group": "m", "field": "tilt_deg", "value": 20.0, "evidence": ["none.png"]}, "evidence none.png"),
        ({"group": "m", "field": "tilt_deg", "value": 20.0, "evidence": ["/etc/hostname"]},
         "evidence /etc/hostname"),
    ],
)
def test_decide_refuses(project, params, why):
    project.commit(lambda f: f.group.append(Group(name="m")))
    (project.work_dir / "e.png").write_bytes(b"")
    with pytest.raises(Refused, match=why):
        run(project, "group.decide", **{"evidence": ["e.png"], **params})
    assert project.reload().group[0].decisions == []
