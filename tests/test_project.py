import multiprocessing as mp

import pytest

from lamino_pal import Refused
from lamino_pal.project import Group, Project, Scan


def test_commit_roundtrips_and_bumps_revision(project):
    def add(f):
        f.group.append(Group(name="mountA", center_px=1008.5, tilt_deg=20.0))

    project.commit(add)
    again = Project(project.path)
    assert again.revision == 1
    assert again.data.group[0].center_px == 1008.5
    assert again.work_dir == project.path.parent


def test_failed_mutation_writes_nothing(project):
    before = project.path.read_bytes()

    def bad(f):
        f.group.append(Group(name="g"))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        project.commit(bad)
    assert project.path.read_bytes() == before


def test_invalid_mutation_is_rejected(project):
    def bad(f):
        f.group.append(Group(name="g"))
        f.group[0].decisions = [{"field": "center_px", "value": 1.0, "by": "user"}]

    with pytest.raises(ValueError):
        project.commit(bad)
    assert Project(project.path).revision == 0


@pytest.mark.parametrize(
    "bad, why",
    [
        (lambda f: f.scan.extend([_scan("a"), _scan("a")]), r"scan tag \['a'\] repeated"),
        (lambda f: f.group.extend([Group(name="g"), Group(name="g")]), r"group \['g'\] repeated"),
        (lambda f: f.scan.append(_scan("a", group="nope")), "scan a: no group nope"),
        (lambda f: f.scan.append(_scan("../a")), "pattern"),
    ],
)
def test_scan_tags_and_groups_are_unique_and_resolve(project, bad, why):
    with pytest.raises(ValueError, match=why):
        project.commit(bad)
    assert Project(project.path).revision == 0


def _scan(tag, group=None):
    return Scan(tag=tag, kind="step", h5=f"{tag}.h5", bin=2, group=group)


def test_work_dir_under_raw_root_is_refused(tmp_path):
    with pytest.raises(Refused):
        Project.create(tmp_path / "raw" / "w" / "project.toml", str(tmp_path / "raw"))


def _writer(path, tag, n, out):
    p = Project(path)
    seen = []
    for i in range(n):

        def add(f, i=i):
            f.group.append(Group(name=f"{tag}{i}"))

        seen.append(p.commit(add).project.revision)
    out.put((tag, seen))


def test_concurrent_commits_never_repeat_or_lose_a_revision(project):
    n = 25
    # Not fork: an import earlier in the session leaves rayon threads running.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ws = [ctx.Process(target=_writer, args=(project.path, t, n, q)) for t in "ab"]
    for w in ws:
        w.start()
    got = dict(q.get(timeout=60) for _ in ws)
    for w in ws:
        w.join(timeout=60)
        assert w.exitcode == 0
    for seen in got.values():
        assert seen == sorted(seen) and len(set(seen)) == n
    assert sorted(got["a"] + got["b"]) == list(range(1, 2 * n + 1))
    final = Project(project.path).data
    assert final.project.revision == 2 * n
    assert len(final.group) == 2 * n
