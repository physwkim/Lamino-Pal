import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import DATA, band_at
from pydantic import ValidationError

from lamino_pal import LaminoPalError, Refused, _core, dispatch
from lamino_pal.artifacts import load_h5
from lamino_pal.ops import REGISTRY, recon
from lamino_pal.project import RECIPE, ZBand


def _scan(project, tag, nproj=16, shape=(8, 64)):
    path = project.work_dir / f"{tag}.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["exchange/data"] = rng.integers(400, 600, (nproj, *shape)).astype("u2")
        f["exchange/data_white"] = np.full((1, *shape), 1000, "u2")
        f["exchange/data_dark"] = np.zeros((1, *shape), "u2")
        f["exchange/theta"] = np.arange(nproj, dtype="f4") * (360 / nproj)
    run(project, "scan.adopt", tag=tag, h5=str(path), bin=2, kind="step")


def run(project, name, **params):
    return dispatch.run(name, params, project=project, caller="cli")


def validate(project, name, **params):
    return dispatch.validate(REGISTRY[name], params, project)


def _decide(project, tag, **fields):
    run(project, "scan.assign_group", tags=[tag], group="m")

    def set_(f):
        for k, v in fields.items():
            setattr(f.group[0], k, v)

    project.commit(set_)


def test_bind_freezes_the_decisions_and_names_the_volume_by_them(project):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5, tilt_deg=19.5)
    project.commit(lambda f: f.recon.update(filter_name="shepp"))
    p = validate(project, "recon.run", tag="s")
    assert (p.center_px, p.tilt_deg, p.recipe) == (64.5, 19.5, {**RECIPE, "filter_name": "shepp"})
    assert Path(p.out) == project.work_dir / "final_s_bin2_c64p5_t19p5_fourierrec_shepp.h5"
    p = validate(project, "recon.split", tag="s", recipe={"filter_name": "shepp"})
    assert p.recipe == {"filter_name": "shepp"}
    assert Path(p.out) == project.work_dir / "artifacts/s_split_c64p5_t19p5.h5"


def test_a_recipe_may_not_set_what_lamino_pal_sets(project):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5)
    with pytest.raises(Refused, match="Lamino-Pal sets lamino_angle itself"):
        validate(project, "recon.preview", tag="s", recipe={"lamino_angle": 20})
    project.commit(lambda f: f.recon.update(save_format="h5"))
    with pytest.raises(Refused, match="Lamino-Pal sets save_format itself"):
        validate(project, "recon.preview", tag="s")
    with pytest.raises(Refused, match="tomoxide Config has no field filter"):
        validate(project, "recon.preview", tag="s", recipe={"filter": "shepp"})


def test_compare_checks_each_variant_over_the_recipe(project):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5)
    with pytest.raises(Refused, match="unknown filter 'nope'"):
        validate(project, "recon.compare", tag="s", variants=[{}, {"filter_name": "nope"}])
    with pytest.raises(Refused, match="Lamino-Pal sets output itself"):
        validate(project, "recon.compare", tag="s", variants=[{"output": "x"}])
    with pytest.raises(ValidationError, match="signal and background go together"):
        validate(project, "recon.compare", tag="s", signal=(0, 4, 0, 4))


def test_reconstruction_needs_a_decided_center_and_a_final_one_a_decided_tilt(project):
    _scan(project, "s")
    with pytest.raises(Refused, match="s has no decided center"):
        validate(project, "recon.preview", tag="s")
    _decide(project, "s", center_px=64.5)
    assert validate(project, "recon.preview", tag="s").tilt_deg == 20.0
    with pytest.raises(Refused, match="s has no decided tilt"):
        validate(project, "recon.run", tag="s")
    assert validate(project, "recon.run", tag="s", tilt_deg=20.0).tilt_deg == 20.0


def test_preview_crosses_its_planes_at_the_voxel(project):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5)
    seen = []
    r = dispatch.run(
        "recon.preview",
        {"tag": "s", "recipe": {"algorithm": "fourierrec"}, "y": 10, "x": 20},
        project=project,
        caller="cli",
        progress=lambda f, m: seen.append(m),
    )
    z, y, x = r.at
    sl, xz, yz = (r.arrays[k] for k in ("slice", "xz", "yz"))
    assert (y, x, r.depth) == (10, 20, xz.shape[0])
    assert sl.shape == (64, 64) and yz.shape == (r.depth, 64)
    np.testing.assert_array_equal(xz[z], sl[y])
    np.testing.assert_array_equal(yz[z], sl[:, x])
    assert seen[-1] == f"row {r.depth}/{r.depth}"


def test_preview_takes_the_middle_of_the_band_carried_to_its_tilt(project):
    _scan(project, "s", shape=(64, 64))
    _decide(project, "s", center_px=64.5)
    d20, d40 = _core.lamino_depth(64, 20.0), _core.lamino_depth(64, 40.0)
    band = ZBand(z=(10, 30), bin=2, tilt_deg=20.0, depth=d20)
    project.commit(lambda f: setattr(f.group[0], "z_band", band))
    lo, hi = band_at((10, 30), d20, 20.0, 40.0, d40)
    assert (lo, hi) != (10, 30)
    r = run(project, "recon.preview", tag="s", tilt_deg=40.0, recipe={"algorithm": "fourierrec"})
    assert (r.depth, r.at[0]) == (d40, (lo + hi) // 2)


def test_compare_keeps_each_variant_and_its_cnr(project, run_job):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5)
    variants = [{}, {"filter_name": "shepp"}]
    roi = {"signal": (20, 30, 20, 30), "background": (34, 44, 34, 44)}
    params = {"tag": "s", "recipe": {"algorithm": "fourierrec"}, "variants": variants, **roi}
    r = run_job(project, "recon.compare", params)
    assert [v.label for v in r.variants] == ["recipe", "filter_name=shepp"]
    meta, arrays = load_h5(r.artifacts[0])
    assert arrays["slice"].shape == (2, 64, 64) and arrays["xz"].shape[0] == 2
    for v, sl in zip(r.variants, arrays["slice"]):
        s, b = sl[20:30, 20:30], sl[34:44, 34:44]
        assert v.cnr == pytest.approx(abs(s.mean() - b.mean()) / b.std(), rel=1e-5)
    assert meta["variants"][1]["fields"] == {"filter_name": "shepp"}
    with pytest.raises(Refused, match=r"rectangle \(0, 70, 0, 4\) leaves the 64 x 64 slice"):
        params.update(signal=(0, 70, 0, 4), overwrite=True)
        run_job(project, "recon.compare", params)


def test_split_reconstructs_each_half_alone(project, run_job):
    # Tall and wide enough for a disk past the FRC taper.
    _scan(project, "s", shape=(32, 160))
    _decide(project, "s", center_px=160.5)
    seen = []
    params = {"tag": "s", "recipe": {"algorithm": "fourierrec"}}
    r = run_job(project, "recon.split", params, lambda f, m: seen.append(m))
    assert r.nproj == (8, 8) and seen[-1].startswith("odd half: row ")
    assert r.radius_px >= 32
    meta, arrays = load_h5(r.artifacts[0])
    assert arrays["slice"].shape == (2, 160, 160) and arrays["frc"].shape == (80,)
    assert "frc" not in meta
    # The halves differ: neither is the whole scan reconstructed twice.
    assert not np.array_equal(arrays["slice"][0], arrays["slice"][1])
    with pytest.raises(Refused, match="outside the"):
        run_job(project, "recon.split", {"tag": "s", "z": 10_000, "overwrite": True})


def test_frc_is_one_for_the_same_image_and_none_for_independent_noise():
    rng = np.random.default_rng(1)
    n = 128
    # A band-limited image: power only below 8/128 cycles per pixel.
    f = np.fft.fft2(rng.normal(size=(n, n)))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"))
    sig = np.fft.ifft2(np.where(k < 8 / n, f, 0)).real
    sig /= sig.std()
    curve, pearson = recon.frc(sig, sig, 60)
    assert pearson == pytest.approx(1.0) and np.allclose(curve[1:], 1.0)
    a = sig + 0.3 * rng.normal(size=(n, n))
    b = sig + 0.3 * rng.normal(size=(n, n))
    curve, pearson = recon.frc(a, b, 60)
    assert 0.8 < pearson < 1.0 and curve[1:6].min() > 0.9
    # Of the values themselves: the taper is the FFT's alone.
    yy, xx = np.indices((n, n))
    inside = np.hypot(yy - n / 2, xx - n / 2) < 60
    assert pearson == pytest.approx(np.corrcoef(a[inside], b[inside])[0, 1])
    curve, pearson = recon.frc(rng.normal(size=(n, n)), rng.normal(size=(n, n)), 60)
    # Each ring is off zero by about 1/sqrt(its pixels), either way.
    assert abs(pearson) < 0.05 and abs(curve[4:].mean()) < 0.06 and np.sqrt((curve[4:] ** 2).mean()) < 0.2


def _fake_cli(tmp_path, body):
    exe = tmp_path / "tomoxide"
    exe.write_text(f"#!{sys.executable}\n{body}")
    exe.chmod(0o755)
    return exe


FAKE_RECON = """
import json, sys, h5py, numpy as np
a = sys.argv
if a[1:] == ["--version"]:
    sys.exit(print("tomoxide 0.0.0 (fake)"))
out, config = a[a.index("--output") + 1], a[a.index("--config") + 1]
out = out if out.endswith(".h5") else out + ".h5"  # as tomoxide's H5Writer
print("recon: config", open(config).read().strip(), flush=True)
for end in (4, 8):
    print(json.dumps({"start": end - 4, "end": end, "total": 8, "secs": end / 2}), flush=True)
with h5py.File(out, "w") as f:
    f["exchange/data"] = np.zeros((8, 4, 4), "f4")
"""


def test_run_writes_the_volume_with_its_provenance_and_leaves_nothing_else(
    project, run_job, tmp_path, monkeypatch, capsys
):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5, tilt_deg=20.0)
    exe = _fake_cli(tmp_path, FAKE_RECON)
    busy = []
    monkeypatch.setattr(recon, "tomoxide_cli", lambda: busy.append(dispatch.gpu_busy(0)) or exe)
    seen = []
    params = {"tag": "s", "recipe": {"algorithm": "fourierrec"}}
    r = run_job(project, "recon.run", params, lambda f, m: seen.append((f, m)))
    # Built before the GPU is taken, found again with it held.
    assert busy == [False, True]
    out = Path(r.artifacts[0])
    assert out.name == "final_s_bin2_c64p5_t20_fourierrec_parzen.h5"
    assert (r.shape, r.secs) == ((8, 4, 4), 4.0)
    assert seen[:4] == [
        (0.0, "building the tomoxide CLI"),
        (0.0, "waiting for GPU 0"),
        (0.0, "on GPU 0"),
        (0.05, "reconstructing"),
    ]
    assert (0.05 + 0.95 * 4 / 8, "row 4/8") in seen and (1.0, "row 8/8") in seen
    with h5py.File(out) as f:
        prov = dict(f["lamino_pal/provenance"].attrs)
    argv = json.loads(prov["argv"])
    assert argv[argv.index("--center") + 1] == "32.0"  # bin 2 column of full 64.5
    assert (prov["bin"], prov["center_px"], prov["tilt_deg"]) == (2, 64.5, 20.0)
    assert json.loads(prov["recipe"]) == {"algorithm": "fourierrec"}
    assert prov["tomoxide"] == _core.tomoxide_rev()
    assert [p.name for p in out.parent.iterdir() if p.name.startswith(f".{out.stem}")] == []
    assert capsys.readouterr().out.startswith("tomoxide 0.0.0 (fake)\n")


def test_a_failed_run_says_why_and_leaves_nothing(project, run_job, tmp_path, monkeypatch):
    _scan(project, "s")
    _decide(project, "s", center_px=64.5, tilt_deg=20.0)
    body = FAKE_RECON.replace("with h5py", "sys.exit(print('Error: out of memory') or 3)\nwith h5py")
    monkeypatch.setattr(recon, "tomoxide_cli", lambda: _fake_cli(tmp_path, body))
    with pytest.raises(LaminoPalError, match="(?s)exited 3:.*Error: out of memory"):
        run_job(project, "recon.run", {"tag": "s"})
    assert [p.name for p in project.work_dir.iterdir() if "final_s" in p.name] == []


def test_the_cli_is_built_only_from_the_cores_own_commit_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(recon, "SOURCE", tmp_path)
    monkeypatch.setattr(recon, "CLI_CACHE", tmp_path / "cache")
    with pytest.raises(Refused, match="which is missing"):
        recon.tomoxide_cli()
    tox = tmp_path / "lib" / "tomoxide"
    tox.mkdir(parents=True)
    (tox / "Cargo.toml").write_text("[workspace]\n")

    def git(*a):
        return subprocess.run(["git", "-C", str(tox), *a], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    # tomoxide's release tags are lightweight.
    git("tag", "v1.0.0")
    rev = "v1.0.0"
    with pytest.raises(Refused, match=f"is at {rev}: rebuild the core"):
        recon.tomoxide_cli()
    (tox / "Cargo.toml").write_text("[workspace]\n# edit\n")
    with pytest.raises(Refused, match=f"uncommitted changes \\({rev}-dirty\\)"):
        recon.tomoxide_cli()
    git("checkout", "-q", "Cargo.toml")
    monkeypatch.setattr(recon._core, "tomoxide_rev", lambda: rev)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    built = tmp_path / "built"
    built.write_text("#!/bin/sh\n")
    built.chmod(0o755)
    msg = {"reason": "compiler-artifact", "target": {"name": "tomoxide"}, "executable": str(built)}
    (bin_ / "cargo").write_text(f"#!/bin/sh\necho '{json.dumps(msg)}'\n")
    (bin_ / "cargo").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    exe = recon.tomoxide_cli()
    assert exe == tmp_path / "cache" / rev / "tomoxide" and os.access(exe, os.X_OK)
    assert sorted(p.name for p in exe.parent.iterdir()) == [".build.lock", "tomoxide"]
    # Kept: neither the source nor cargo is needed again.
    shutil.rmtree(bin_)
    shutil.rmtree(tox)
    assert recon.tomoxide_cli() == exe


# 261003 step1000 at bin 2, center 504 (full 1008.5), tilt 20: the volumes
# `tomoxide recon --algorithm fourierrec --filter ... --save_format h5` wrote.
GOLDEN_RECIPE = {"algorithm": "fourierrec"}
VARIANT_VOLUMES = [
    "final_step1000_bin2_c504_t20_fourierrec_parzen.h5",
    "rec_bin2_fourierrec_shepp.h5",
    "rec_bin2_fourierrec_ramp.h5",
    "rec_bin2_fourierrec_parzen_voall.h5",
]


def _nrmse(a, b):
    a, b = np.asarray(a, "f8"), np.asarray(b, "f8")
    return np.sqrt(((a - b) ** 2).sum() / (b**2).sum())


@pytest.mark.golden
def test_compare_equals_the_cli_volume_of_each_variant(golden_project, run_job):
    params = {"tag": "step1000", "center_px": 1008.5, "tilt_deg": 20.0, "recipe": GOLDEN_RECIPE}
    r = run_job(golden_project, "recon.compare", params)
    z, y = r.at
    _, arrays = load_h5(r.artifacts[0])
    for i, name in enumerate(VARIANT_VOLUMES):
        with h5py.File(DATA / "tomoxide" / name) as f:
            d = f["exchange/data"]
            # Measured 2.3e-7 to 1.2e-6: the GPU FFTs' run-to-run floor.
            assert _nrmse(arrays["slice"][i], d[z]) < 3e-6, name
            assert _nrmse(arrays["xz"][i], d[:, y, :]) < 3e-6, name


@pytest.mark.golden
def test_split_on_step1000(golden_project, run_job):
    params = {"tag": "step1000", "center_px": 1008.5, "tilt_deg": 20.0, "recipe": GOLDEN_RECIPE}
    r = run_job(golden_project, "recon.split", params)
    assert r.nproj == (501, 500)
    # Measured: the middle slice sees the whole 504 px disk; FRC 0.27 at ring
    # 73, 0.12 at 74.
    assert (r.at, r.radius_px) == ((545, 512), 504.0)
    assert r.correlation == pytest.approx(0.7297, abs=1e-3)
    assert r.resolution_px == pytest.approx(1024 / 74)


@pytest.mark.golden
def test_run_equals_the_cli_volume(golden_project, run_job):
    params = {"tag": "step1000", "center_px": 1008.5, "tilt_deg": 20.0, "recipe": GOLDEN_RECIPE}
    r = run_job(golden_project, "recon.run", params)
    assert Path(r.artifacts[0]).name == "final_step1000_bin2_c1008p5_t20_fourierrec_parzen.h5"
    with h5py.File(r.artifacts[0]) as a, h5py.File(DATA / "tomoxide" / VARIANT_VOLUMES[0]) as b:
        got, want = a["exchange/data"], b["exchange/data"]
        assert got.shape == want.shape
        se = sr = 0.0
        for z0 in range(0, want.shape[0], 64):
            g, w = got[z0 : z0 + 64].astype("f8"), want[z0 : z0 + 64].astype("f8")
            se += ((g - w) ** 2).sum()
            sr += (w**2).sum()
        # Measured 1.7e-7.
        assert np.sqrt(se / sr) < 1e-6
