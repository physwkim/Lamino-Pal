import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import jobops
import pytest

from lamino_pal import LaminoPalError, Refused, dispatch, jobs
from lamino_pal.ops import REGISTRY


@pytest.fixture(autouse=True)
def test_ops(monkeypatch):
    for o in jobops.OPS:
        monkeypatch.setitem(REGISTRY, o.name, o)
    monkeypatch.setattr(jobs, "COMMAND", [sys.executable, str(Path(__file__).with_name("job_main.py")), "job"])


@pytest.fixture
def sleeper():
    """A process this test owns outside any job, killed at teardown."""
    p = subprocess.Popen(["sleep", "60"])
    yield p
    p.kill()
    p.wait()


def until(f, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while not (v := f()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)
    return v


def gone(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def test_a_job_runs_detached_and_records_its_result(project):
    st = jobs.submit(project, "test.steps", {"n": 4}, "mcp")
    assert (st.status in jobs.ACTIVE, st.gpu, os.getpgid(st.pid)) == (True, None, st.pid)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.progress, st.message, st.error) == ("done", 1.0, "step 4", None)
    out = project.work_dir / "artifacts/steps4.txt"
    assert st.result["artifacts"] == [str(out)] and out.read_text() == "4"
    assert st.params["out"] == str(out)
    assert [s.id for s in jobs.list_(project)] == [st.id]
    entry = json.loads(dispatch.log_path(project).read_text().splitlines()[-1])
    assert (entry["op"], entry["caller"], entry["status"]) == ("test.steps", "mcp", "ok")


def test_a_failing_op_fails_its_job(project):
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"fail_at": 1}, "gui").id, 20)
    assert (st.status, st.error, st.message) == ("failed", "RuntimeError: failed at step 1", "step 1")
    assert "RuntimeError: failed at step 1" in (jobs.jobs_dir(project) / st.id / "out.log").read_text()


@pytest.mark.parametrize(
    "name, params, err, why",
    [
        ("test.nope", {}, LaminoPalError, "unknown op test.nope"),
        ("log.tail", {}, LaminoPalError, "log.tail is not a job op"),
        ("test.steps", {"n": "x"}, Exception, "valid integer"),
    ],
)
def test_submit_refuses_before_starting_anything(project, name, params, err, why):
    with pytest.raises(err, match=why):
        jobs.submit(project, name, params, "cli")
    assert jobs.list_(project) == []


def test_an_existing_or_queued_output_is_refused_at_submit(project):
    first = jobs.submit(project, "test.steps", {"n": 2, "dt": 1.0}, "cli")
    with pytest.raises(Refused, match=f"job {first.id} is writing .*artifacts/steps2.txt"):
        jobs.submit(project, "test.steps", {"n": 2}, "cli")
    assert jobs.wait(project, first.id, 20).status == "done"
    with pytest.raises(Refused, match="steps2.txt exists; pass overwrite"):
        jobs.submit(project, "test.steps", {"n": 2}, "cli")
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"n": 2, "overwrite": True}, "cli").id, 20)
    assert st.status == "done"


def test_cancel_stops_the_job_and_its_process_group_only(project, sleeper):
    st = jobs.submit(project, "test.steps", {"n": 400, "child": True}, "cli")
    pid_file = project.work_dir / "artifacts/steps400.txt.child"
    child = int(until(lambda: pid_file.exists() and pid_file.read_text()))
    until(lambda: jobs.get(project, st.id).progress > 0)
    assert jobs.get(project, st.id).eta_s > 0
    jobs.cancel(project, st.id)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.result) == ("cancelled", None)
    until(lambda: gone(child))
    assert sleeper.poll() is None
    with pytest.raises(Refused, match=f"job {st.id} is cancelled"):
        jobs.cancel(project, st.id)


def test_a_job_deaf_inside_a_call_is_stopped_after_the_grace(project):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until(lambda: jobs.get(project, st.id).status == "running")
    t0 = time.monotonic()
    jobs.cancel(project, st.id)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.message) == ("cancelled", "stopped inside a call")
    assert time.monotonic() - t0 < 5
    until(lambda: gone(st.pid))


def test_cancel_refuses_a_pid_that_is_not_the_jobs(project, sleeper):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until(lambda: jobs.get(project, st.id).status == "running")
    real = jobs.get(project, st.id)
    d = jobs.jobs_dir(project) / st.id
    # The sleeper's pid with the job's start time: a reused pid.
    jobs._write(d, real.model_copy(update={"pid": sleeper.pid}))
    with pytest.raises(Refused, match=f"process {sleeper.pid} is not running"):
        jobs.cancel(project, st.id)
    # Its own start time, but not a process group leader: not a job's.
    jobs._write(d, real.model_copy(update={"pid": sleeper.pid, "proc_start": jobs._proc_start(sleeper.pid)}))
    with pytest.raises(Refused, match=f"process {sleeper.pid} is not running"):
        jobs.cancel(project, st.id)
    assert sleeper.poll() is None
    jobs._write(d, real)
    jobs.cancel(project, st.id)
    assert jobs.wait(project, st.id, 20).status == "cancelled"


def test_a_job_cancelled_before_its_process_runs_it_is_cancelled(project, monkeypatch):
    # A job process that has not reached `jobs.run`, where it hears SIGTERM.
    monkeypatch.setattr(jobs, "COMMAND", [sys.executable, "-c", "import time; time.sleep(60)"])
    st = jobs.submit(project, "test.steps", {"n": 1}, "cli")
    assert jobs.cancel(project, st.id).cancel_sent is not None
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.message, st.error) == ("cancelled", "stopped before it ran", None)


def test_a_job_process_that_dies_is_marked_failed(project):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until(lambda: jobs.get(project, st.id).status == "running")
    os.kill(st.pid, signal.SIGKILL)
    st = jobs.wait(project, st.id, 20)
    assert st.status == "failed" and st.error.startswith("the job process ended without a result")


def test_a_gpu_job_takes_a_free_batch_gpu_and_waits_for_a_busy_one(project):
    project.commit(lambda f: setattr(f.gpu, "batch", [2, 3]))
    with dispatch.gpu_lock(2, 0):
        st = jobs.submit(project, "test.gpu_steps", {"n": 1}, "cli")
        assert st.gpu == 3
        assert jobs.wait(project, st.id, 30).result["gpu"] == 3
    project.commit(lambda f: setattr(f.gpu, "batch", [3]))
    with dispatch.gpu_lock(3, 0):
        st = jobs.submit(project, "test.gpu_steps", {"n": 2}, "cli")
        until(lambda: jobs.get(project, st.id).message == "waiting for GPU 3")
        time.sleep(0.5)
        assert jobs.get(project, st.id).status == "running"
    st = jobs.wait(project, st.id, 30)
    assert (st.status, st.result["gpu"]) == ("done", 3)


def test_a_gpu_job_refuses_a_gpu_outside_the_batch(project, monkeypatch):
    project.commit(lambda f: setattr(f.gpu, "batch", [1]))
    monkeypatch.setattr(jobs, "_pick_gpu", lambda project: 1)
    st = jobs.submit(project, "test.gpu_steps", {"n": 1}, "cli")
    project.commit(lambda f: setattr(f.gpu, "batch", [2]))
    st = jobs.wait(project, st.id, 30)
    assert st.status == "failed" and "runs on a batch GPU [2]; this process's CUDA runs on GPU 1" in st.error


def test_job_ids_do_not_leave_the_jobs_directory(project):
    with pytest.raises(Refused, match="no job ../work"):
        jobs.get(project, "../work")


def test_cli_runs_an_op_and_prints_its_result(project, tmp_path, capsys):
    from lamino_pal.cli import main

    params = tmp_path / "p.json"
    params.write_text('{"n": 2}')
    assert main(["run", "--project", str(project.path), "test.steps", str(params)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["n"], out["artifacts"]) == (2, [str(project.work_dir / "artifacts/steps2.txt")])
    assert main(["run", "--project", str(project.path), "test.steps", str(params)]) == 2
    assert "steps2.txt exists; pass overwrite" in capsys.readouterr().err
