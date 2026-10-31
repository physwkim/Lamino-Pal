import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import jobops
import psutil
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
    p = subprocess.Popen(jobops.SLEEP)
    yield p
    p.kill()
    p.wait()


def until(f, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while not (v := f()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)
    return v


def last_log(project):
    return json.loads(dispatch.log_path(project).read_text().splitlines()[-1])


def gone(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_a_job_runs_detached_and_records_its_result(project):
    st = jobs.submit(project, "test.steps", {"n": 4}, "mcp")
    assert (st.status in jobs.ACTIVE, st.gpu) == (True, None)
    if os.name == "posix":
        assert os.getpgid(st.pid) == st.pid
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.progress, st.message, st.error) == ("done", 1.0, "", None)
    out = project.work_dir / "artifacts/steps4.txt"
    assert st.result["artifacts"] == [str(out)] and out.read_text() == "4"
    assert st.params["out"] == str(out)
    assert [s.id for s in jobs.list_(project)] == [st.id]
    queued, ran = [json.loads(x) for x in dispatch.log_path(project).read_text().splitlines()]
    assert (queued["op"], queued["caller"], queued["status"], queued["job"]) == ("test.steps", "mcp", "queued", st.id)
    assert (ran["op"], ran["caller"], ran["status"]) == ("test.steps", "mcp", "ok")


def test_a_failing_op_fails_its_job(project):
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"fail_at": 1}, "gui").id, 20)
    assert (st.status, st.error, st.message) == ("failed", "RuntimeError: failed at step 1", "step 1")
    assert "RuntimeError: failed at step 1" in (jobs.jobs_dir(project) / st.id / "out.log").read_text()


def test_a_jobs_out_log_is_utf8_whatever_the_locale(project, monkeypatch):
    # Windows' locale encoding, which has no Korean.
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"n": 1, "say": "단면 slice"}, "gui").id, 20)
    assert st.status == "done"
    assert "단면 slice" in (jobs.jobs_dir(project) / st.id / "out.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name, params, err, why, status",
    [
        ("test.nope", {}, LaminoPalError, "unknown op test.nope", "error"),
        ("log.tail", {}, LaminoPalError, "log.tail is not a job op", "error"),
        ("test.steps", {"n": "x"}, Exception, "valid integer", "invalid"),
    ],
)
def test_submit_refuses_before_starting_anything_and_logs_why(project, name, params, err, why, status):
    with pytest.raises(err, match=why):
        jobs.submit(project, name, params, "mcp")
    assert jobs.list_(project) == []
    entry = last_log(project)
    assert (entry["op"], entry["caller"], entry["status"], entry["params"]) == (name, "mcp", status, params)
    assert why in entry["error"]


def test_a_gpu_job_on_a_core_without_cuda_is_refused_at_submit(project, monkeypatch):
    monkeypatch.setattr(dispatch._core, "cuda_enabled", lambda: False)
    with pytest.raises(Refused, match="test.gpu_steps runs on an NVIDIA GPU; this core is built without CUDA"):
        jobs.submit(project, "test.gpu_steps", {}, "mcp")
    assert jobs.list_(project) == []
    assert last_log(project)["status"] == "refused"


def test_an_existing_or_queued_output_is_refused_at_submit(project):
    first = jobs.submit(project, "test.steps", {"n": 2, "dt": 1.0}, "cli")
    out = re.escape(os.path.join("artifacts", "steps2.txt"))
    with pytest.raises(Refused, match=f"job {first.id} is writing .*{out}"):
        jobs.submit(project, "test.steps", {"n": 2}, "cli")
    assert last_log(project)["status"] == "refused"
    assert jobs.wait(project, first.id, 20).status == "done"
    with pytest.raises(Refused, match="steps2.txt exists; pass overwrite"):
        jobs.submit(project, "test.steps", {"n": 2}, "cli")
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"n": 2, "overwrite": True}, "cli").id, 20)
    assert st.status == "done"


def test_a_done_job_answers_a_submit_of_its_params_until_what_it_read_changes(project):
    old = time.time() - 60
    os.utime(project.path, (old, old))
    first = jobs.wait(project, jobs.submit(project, "test.steps", {"n": 2}, "cli").id, 20)
    assert jobs.answering(project, "test.steps", {"n": 2, "overwrite": False}).id == first.id
    assert jobs.answering(project, "test.steps", {"n": 2, "overwrite": True}) is None
    assert jobs.answering(project, "test.steps", {"n": 2, "dt": 0.01}) is None
    assert jobs.answering(project, "test.steps", {"n": 3}) is None
    # The newest job writing the output answers, or none.
    second = jobs.submit(project, "test.steps", {"n": 2, "dt": 0.01, "overwrite": True}, "cli")
    assert jobs.wait(project, second.id, 20).status == "done"
    assert jobs.answering(project, "test.steps", {"n": 2}) is None
    assert jobs.answering(project, "test.steps", {"n": 2, "dt": 0.01}).id == second.id
    project.commit(lambda f: setattr(f.gpu, "batch", [1]))
    assert jobs.answering(project, "test.steps", {"n": 2, "dt": 0.01}) is None
    os.utime(project.path, (old, old))
    assert jobs.answering(project, "test.steps", {"n": 2, "dt": 0.01}).id == second.id
    (project.work_dir / "artifacts/steps2.txt").unlink()
    assert jobs.answering(project, "test.steps", {"n": 2, "dt": 0.01}) is None


def test_the_eta_leaves_out_a_share_reported_at_once(project):
    # Half the job reported before any step ran, then 4 steps of dt: after
    # step 1, 3 steps remain.
    dt = 0.5
    st = jobs.submit(project, "test.steps", {"n": 4, "dt": dt, "reserved": 0.5}, "cli")
    st = until(lambda: (s := jobs.get(project, st.id)).message == "step 1" and s)
    assert st.eta_s >= 3 * dt


def test_cancel_stops_the_job_and_its_children_only(project, sleeper):
    st = jobs.submit(project, "test.steps", {"n": 400, "child": True}, "cli")
    pid_file = project.work_dir / "artifacts/steps400.txt.child"
    child = int(until(lambda: pid_file.exists() and pid_file.read_text()))
    assert until(lambda: jobs.get(project, st.id).eta_s) > 0
    jobs.cancel(project, st.id)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.result) == ("cancelled", None)
    assert (last_log(project)["status"], last_log(project)["error"]) == ("cancelled", "Cancelled: cancelled")
    until(lambda: gone(child))
    assert sleeper.poll() is None
    with pytest.raises(Refused, match=f"job {st.id} is cancelled"):
        jobs.cancel(project, st.id)


def test_a_job_whose_op_fails_because_the_cancel_killed_its_child_is_cancelled(project):
    # As recon.run's build of the tomoxide CLI fails when cargo dies.
    st = jobs.submit(project, "test.steps", {"n": 1, "child_step": True}, "cli")
    until((project.work_dir / "artifacts/steps1.txt.child").exists)
    t0 = time.monotonic()
    jobs.cancel(project, st.id)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.error) == ("cancelled", None)
    assert time.monotonic() - t0 < 5
    assert last_log(project)["status"] == "cancelled"


def test_a_job_deaf_inside_a_call_is_stopped_after_the_grace_and_leaves_no_part(project):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until((project.work_dir / "artifacts/.steps1.part.txt").exists)
    t0 = time.monotonic()
    jobs.cancel(project, st.id)
    st = jobs.wait(project, st.id, 20)
    assert (st.status, st.message) == ("cancelled", "stopped inside a call")
    assert time.monotonic() - t0 < 5
    entry = last_log(project)
    assert (entry["op"], entry["status"], entry["error"]) == ("test.steps", "cancelled", "stopped inside a call")
    assert list((project.work_dir / "artifacts").iterdir()) == []
    until(lambda: gone(st.pid))


def test_cancel_refuses_a_pid_that_is_not_the_jobs(project, sleeper):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until(lambda: jobs.get(project, st.id).status == "running")
    real = jobs.get(project, st.id)
    d = jobs.jobs_dir(project) / st.id
    # The sleeper's pid, as a reused pid would be: the job's process is gone.
    # Queued, a cancel that took it for the job's would terminate it.
    jobs._write(d, real.model_copy(update={"pid": sleeper.pid, "status": "queued"}))
    with pytest.raises(Refused, match=f"job {st.id} is failed"):
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


def test_a_cancel_while_the_job_process_starts_waits_for_its_pid(project, monkeypatch):
    popen, starting, cancelled = subprocess.Popen, threading.Event(), []

    def slow_popen(*a, **kw):
        starting.set()
        time.sleep(0.5)
        return popen(*a, **kw)

    def cancel():
        starting.wait(10)
        jid = next(j.name for j in jobs.jobs_dir(project).iterdir() if j.is_dir())
        cancelled.append(jobs.cancel(project, jid))

    monkeypatch.setattr(jobs.subprocess, "Popen", slow_popen)
    t = threading.Thread(target=cancel)
    t.start()
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    t.join(10)
    assert cancelled[0].pid == st.pid and cancelled[0].cancel_sent is not None
    assert jobs.wait(project, st.id, 20).status == "cancelled"


def test_a_job_whose_submitter_died_before_starting_it_is_failed(project):
    st = jobs.wait(project, jobs.submit(project, "test.steps", {"n": 1}, "cli").id, 20)
    jobs._write(jobs.jobs_dir(project) / st.id, st.model_copy(update={"status": "queued", "pid": None}))
    with pytest.raises(Refused, match=f"job {st.id} is failed"):
        jobs.cancel(project, st.id)
    st = jobs.get(project, st.id)
    assert (st.status, st.error) == ("failed", "the submitting process ended before starting the job")


def test_a_job_process_that_dies_is_marked_failed(project):
    st = jobs.submit(project, "test.steps", {"n": 1, "dt": 60}, "cli")
    until(lambda: jobs.get(project, st.id).status == "running")
    psutil.Process(st.pid).kill()
    st = jobs.wait(project, st.id, 20)
    assert st.status == "failed" and st.error.startswith("the job process ended without a result")


@pytest.mark.gpu
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


@pytest.mark.gpu
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
