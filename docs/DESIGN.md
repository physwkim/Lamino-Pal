# Lamino-Pal 설계

BL7C laminography 재구성 전 과정(import → 정렬 → 재구성 → 검증 → export)을
Qt GUI와 MCP 서버로 감싼다. 2D/3D viewer는 napari를 임베딩하고, 1D curve는
silx를 쓴다(§6). 계산은 tomoxide(Rust)를 PyO3로 호출하고, 앱 골격은 TXM-Pal과
같다(PySide6 + Qt Designer `.ui`, maturin 빌드 Rust core).

원본 과정: `~/work/261003_Lamino_test/tomoxide/` (`README.md`,
`PROCESSING.log`, `build_*.py`, `ring_center.py`, `compare_*.py`,
`inspect_volume.py`, `export_*.py`, `view3d.py`, `fly/run_finals.sh`)와
tomoxide `docs/LAMINOGRAPHY_ALIGNMENT.md`.

## 1. 원칙

- **op 하나 = 분석 기능 하나, 구현은 한 곳.** 모든 분석은 `lamino_pal.ops`
  registry의 op로만 존재한다. GUI, MCP, CLI는 같은 dispatcher를 부르는 얇은
  adapter다. MCP tool 목록은 registry에서 생성하므로 "모든 분석 기능을 MCP로"는
  관례가 아니라 구조로 성립한다(§9 coverage test).
- **일반 알고리즘은 tomoxide, BL7C 형식은 `lamino_pal_core`, 표시(window,
  montage, PNG)는 Python.** tomoxide CLI·tomoxide-gui와 공유할 수 있는 커널은
  앱에 두지 않는다.
- **결정은 project 파일 한 곳.** center/tilt/z-band는 mount group의 속성이고,
  GUI·MCP·job 모두 같은 `project.toml`을 읽고 쓴다.
- **단위는 하나.** center는 항상 full-res(bin 1) detector column `center_px`로
  저장한다. bin b 파일용 값 `c_b = (c_full − (b−1)/2)/b`는
  `units.center_for_scan(scan, c_full)` 한 함수만 계산하고, b는 project에 기록된
  scan의 `bin`이다. import가 자기 binning을 기록하고, adopt는 h5의 `/lamino_pal`
  group에 `binning` attribute가 있으면 그 값과 대조한다(README의 "bin-2 504 = full
  1008.5" 수기 환산 제거). tilt는 tomocupy `--lamino_angle` 규약
  `phi = π/2 + tilt·π/180`.

## 2. 원본 과정 → op

| 원본 | op | 비용 | 구현 위치 |
|---|---|---|---|
| metadata.txt / scan `.dat` 해석 | `scan.discover` | instant | core `bl7c` |
| `build_step_bin2.py`, `build_step_h5.py`, `build_fly_h5.py` | `scan.import` | job | core `bl7c` |
| 기존 h5(위 스크립트 산출물) 편입 | `scan.adopt` (bin 명시) | instant | Python |
| mount 분류(f360 ≠ step/f180) | `scan.assign_group` | instant | Python |
| 원본 projection 확인 | `raw.frame` | instant | tomoxide `read_h5_frame` |
| 360° 평균 projection | `raw.mean_projection` | interactive | tomoxide `mean_projection` |
| ALIGNMENT §5 각도 범위 확인 | `raw.range_check` | instant | Python |
| proj0 vs proj@360 drift (−6 px) | `raw.drift` | interactive | tomoxide `register_shift` |
| 인접 projection flow tilt (τ = 19.4 ± 1.5°) | `raw.tilt_flow` | interactive | tomoxide `estimate_tilt_flow` |
| `ring_center.py`, `align --rings_only` | `align.rings` | interactive | tomoxide `find_center_rings_from_mean` |
| `align` center sweep | `align.center_sweep` | interactive | tomoxide align (T1) |
| `align --tilt_width`, `compare_tilts.py` | `align.tilt_scan` | job | tomoxide `lamino_tilt_scan` |
| `compare_centers.py` | `align.center_compare` | job | tomoxide recon + Python montage |
| 단일 slice 미리보기 | `recon.preview` | interactive | tomoxide (T1 params) |
| 알고리즘 비교(parzen/shepp/ramp/Paganin/vo-all) | `recon.compare` | job | tomoxide + Python CNR |
| even/odd projection split | `recon.split` | job | tomoxide |
| `recon`, `fly/run_finals.sh` | `recon.run` | job | `tomoxide` CLI subprocess |
| `inspect_volume.py` focus-by-z | `volume.focus_by_z` | interactive | tomoxide `focus_by_z` |
| slice/ortho 보기 | `volume.slice`, `volume.ortho` | instant | h5py lazy |
| valid FOV | `volume.valid_fov` | instant | tomoxide `Geometry::valid_fov` |
| `export_tiff.py` | `export.tiff16` | job | core |
| `export_bin.py` | `export.bin` | job | core |
| `view3d.py` MIP | `export.mip` + GUI 3D 탭 | job | core |
| PROCESSING.log 수기 기록 | dispatcher 자동 기록, `log.tail` | instant | Python |
| 결정 기록 | `group.decide` | instant | Python |

op 비용 등급: **instant**(<0.2 s, 호출 스레드), **interactive**(수 초, 프로세스
안 GPU, 취소 가능), **job**(분 단위, 별도 프로세스). 등급은 op 선언에 고정한다.

## 3. tomoxide 선행 작업

Lamino-Pal은 tomoxide 커밋에 pin된다(§8 submodule). 아래가 먼저 들어가야 한다.

**T1 — align/recon 조립을 library로 (구조).** 지금 조립 로직이 CLI 바이너리에만
있고 tomoxide-gui가 이미 복제했다. PyO3가 세 번째 사본이 되면 안 된다.

- tilt→phi geometry: `tomoxide-cli/src/main.rs:903` `mk_geom`,
  `tomoxide-gui/src/worker.rs:753`
- Config→`ReconParams`: `main.rs:2045` `recon_params`, `worker.rs:610`
- phase 조립: `main.rs:469` `build_phase`, `views/tune.rs:312`; stripe:
  `main.rs:417` `build_stripe`
- `run_align`(`main.rs:814`–1080), `grid`, 기본 slice 선택

이것들을 `tomoxide::plan`(geometry, params, stripe/phase)과
`tomoxide::align`(`AlignRequest` → `AlignReport{ring, sweep, verdict,
tilt_focus}`, `CancelToken`·progress callback 인자)로 옮기고 CLI, tomoxide-gui,
Lamino-Pal이 모두 이것을 부른다. 검증: 옮기기 전후 `tomoxide align` 출력이
`step1000_bin2.h5`에서 같다.

**T2 — lamino fourierrec y-flip.** USFFT divker2d가 tomocupy의 `n1-ty-1` 행을
그대로 써서 linerec/fbp와 y가 뒤집혀 있었다. tomoxide `15808c6`에서 고쳤고 GUI는
뒤집지 않는다. tomocupy 출력은 이제 y가 반대다. 261003의 `final_*fourierrec*`
볼륨은 고친 빌드로 다시 만들었다(PROCESSING.log 2026-10-29).

**T3 — 일반 커널 공개** (branch `feat/lamino-kernels`).

- `register_shift(reference, moving, backend, upsample) -> (dy, dx)`:
  `find_center_pc` 내부 phase correlation.
- `find_center_rings_from_mean(mean, backend)`: row band는 Python이 mean
  projection에서 잘라 넘긴다(ring_center.py의 quarter-band 일관성 확인).
  `find_center_rings`는 이것의 wrapper.
- `focus_by_z(vol) -> Vec<f64>`: slice별 focus. `lamino_tilt_scan`,
  `PreppedStack::center_sweep`과 공용.
- `Geometry::valid_fov(depth) -> ValidFov { r_lateral, radius[z] }`,
  `full_z()`: `r(h) = min(min(c, W−c), (H/2 − |h|·cos t)/sin t)`.
- `estimate_tilt_flow(tomo, theta, backend) -> TiltFlow`: 인접 projection 쌍의
  vertical flow를 `dz = sx·x + sz·y + c`로 적합해 tilt, roll, axis와 쌍별 값을
  낸다. center는 입력이 아니라 출력. step1000_bin2: 19.40 ± 1.46°, axis
  478.8 ± 74.8(Python 원본과 쌍마다 일치).

## 4. Rust core `lamino_pal_core`

maturin cdylib, `abi3-py312`, feature `cuda`를 tomoxide로 전달. 의존: tomoxide
(path, submodule), pyo3, numpy(rust-numpy), tiff, rayon.

- `bl7c`: scan 탐색(NFD 폴더명 그대로 처리), `metadata.txt` JSON과 `.dat`
  header/column 해석, import. step: theta = `.dat` 앞 N행, flat = 뒤쪽 행. fly:
  `theta = linspace(0, R, N, endpoint=False)`, `R = round(N·speed·exptime)`,
  R ∉ {180, 360}이면 거부. dark = 0, b×b mean binning, saturation 수 보고, h5의
  `/lamino_pal` group에 attribute `binning, source, energy_ev, kind` 기록. 출력
  경로가 raw root 아래면 거부. 출력은 `.<name>.part`에 쓰고 rename하므로 실패나
  취소 뒤에 반쪽 파일이 남지 않는다. `scan.import`는 project당 하나씩
  (`.import.lock`) 돌며, tag 확인부터 commit까지 그 lock 안에 있다.
- `bind`: tomoxide T1/T3 API의 얇은 wrapper. 무거운 호출은 `py.allow_threads`로
  GIL을 놓고, Python 쪽 `Cancel` 객체가 tomoxide `CancelToken`을 감싼다. progress
  callback은 GIL을 다시 잡고 호출하며 10 Hz로 제한한다. 배열은 numpy로 반환한다
  (프로세스 안에서만; MCP로는 나가지 않는다).
- `export`: 16-bit TIFF(z 범위, p0.1/p99.9 window를 TIFF metadata에 기록),
  2×2×2 bin, 축별 MIP. h5 볼륨(bin 1에서 36.5 GB)은 z band 단위로 스트리밍.
- 오류는 `LaminoPalError` 하위 예외로 변환하며, guard 거부는 `Refused`(이유
  문자열 포함)로 구분한다.

## 5. Python 계층

```
lamino_pal/
  ops/registry.py   @op(name, cost, params: pydantic, result: pydantic)
  ops/{scan,raw,align,recon,volume,export,group}.py
  dispatch.py       유일한 실행 경로: 검증 → GPU lock → 실행 → 로그 → artifact
  project.py        project.toml 소유자
  units.py          center/z 단위 변환의 유일한 위치
  jobs.py           job 프로세스 생성/상태/취소
  render.py         window(pct), montage, PNG
  gui/  mcp/  cli.py
```

**dispatcher.** 모든 op 호출은 `dispatch.run(op, params, caller)`를 지난다.
여기서 params 검증, 출력 경로가 `work_dir` 아래인지 확인, 덮어쓰기는
`overwrite=true`일 때만, GPU lock, 실행, `log.jsonl` 기록(op, params, 결과
요약, artifact, 소요 시간, caller=gui|mcp|cli, tomoxide 버전)을 한다. 기록이
dispatcher 안에 있으므로 어떤 op도 로그를 건너뛸 수 없다. 생략할 수 있는 출력
경로(`OptOutPath`)는 op의 `default_out`이 params로 이름을 짓고(예: §5의
`final_…` 규칙), 그 이름도 같은 경로 검사를 거친다.

**project.toml** (work_dir 루트; work_dir은 이 파일이 있는 폴더이며 따로 저장하지
않는다):

```toml
[project]
raw_root = "/home/stevek/mnt/analysis/261003 Lamino test"
revision = 17

[[group]]                       # mount = geometry를 공유하는 scan 집합
name = "mountA"                 # step1000, step800, f180 *
center_px = 1008.5              # full-res
tilt_deg = 20.0
z_band = { z = [390, 1089], bin = 2, tilt_deg = 20.0 }   # tomoxide SampleBand + bin
decisions = [ { field = "center_px", value = 1008.5, by = "user", evidence = ["artifacts/…png"], note = "…" } ]

[[scan]]
tag = "step1000"
kind = "step"
raw = "360도 step/Step_N1000_0p18s_proj"   # 실제 NFD 이름을 그대로 저장
h5 = "h5/step1000_bin2.h5"
bin = 2
group = "mountA"

[recon]                         # tomoxide Config 필드 그대로; job이 top-level TOML로 떼어 CLI --config로 넘김
algorithm = "fourierrec"
filter_name = "parzen"
dtype = "float32"
save_format = "h5"
```

쓰기는 `Project.commit(mutation)` 하나다: flock → 다시 읽기 → 적용 → revision
증가 → atomic rename. GUI는 `QFileSystemWatcher`로 다시 읽으므로 MCP 쪽 결정이
즉시 보인다. `group.decide`에는 evidence artifact가 필수이고, `by = "agent"`
결정은 GUI에서 따로 표시되어 사람 확인을 받는다.

**job.** job은 detached 프로세스다. GUI를 닫아도 batch는 계속 돌고, GUI와
MCP가 같은 job을 본다. 상태는 `work_dir/jobs/<id>/state.json`(pid, pgid, op,
params, progress, eta, result)에 있다.

- `recon.run`은 submodule에서 빌드한 `tomoxide` 바이너리를 절대 경로로 실행한다
  (`--config` = `[recon]`, `--center` = `units.center_for_scan`, `--lamino_angle`,
  `--progress_json`).
- 다른 job op는 `python -m lamino_pal.cli run <op> <params.json>`으로 돈다.
- 취소는 자기 job의 process group에 SIGTERM을 보낸다.
- 완료 시 출력 h5에 `/lamino_pal/provenance`(center_px, tilt, params, source,
  tomoxide 버전, 명령줄)를 쓴다.
- 출력 이름 규칙은 기존 것을 따른다:
  `final_<tag>_bin<b>_c<center>_t<tilt>_<alg>_<filter>.h5`.

**GPU.** GPU마다 `$XDG_RUNTIME_DIR/lamino-pal/gpu<N>.lock`을 exclusive flock으로
잡는다. 기본값은 interactive = GPU 0, batch = GPU 1–3이고, 설정으로 바꾼다.
tomoxide는 CUDA 장치를 호출마다 `TOMOXIDE_CUDA_DEVICES`에서 읽으므로(없으면 GPU
전부), 프로세스의 GPU는 띄우는 쪽이 이 env로 하나만 정하고 실행 중에는 바꾸지
않는다. dispatcher는 그 GPU가 project의 interactive GPU일 때만 GPU op를 돌린다.
`recon.run`은 scan 여러 개면 GPU당 scan 하나(run_finals.sh 방식), scan 하나면
CLI의 multi-GPU shard를 쓴다.

## 6. GUI (napari + silx)

TXM-Pal 구조를 따른다: `gui/ui/main.ui`(Designer), worker thread →
`silx.gui.utils.concurrent.submitToQtMainThread`.

**Viewer.** 이미지와 볼륨은 napari, 1D curve는 silx `Plot1D`가 맡는다.

- napari를 쓰는 이유:
  - 3D 볼륨 렌더링(mip, iso, plane depiction)과 2D slice가 한 viewer에서 전환된다.
  - dask 배열로 h5를 lazy하게 연다.
  - layer별 contrast와 colormap을 따로 둔다.
  - Shapes layer로 ROI와 z-band를 그린다.
  - silx 2.1 `plot3d`에는 cut plane과 isosurface만 있고 MIP 같은 volume
    rendering은 없다(`silx/gui/plot3d` 소스에 MIP/raycast 없음).
- napari에는 1D plot이 없다. 그래서 center score, tilt focus, focus-by-z,
  histogram은 silx `Plot1D`로 그린다.
- 임베딩: `napari.components.ViewerModel` + `napari.qt.QtViewer`(public export,
  `QSplitter`)를 탭 위젯 안에 넣는다. layer list/control은 `QtViewer`의
  `dockLayerList`/`dockLayerControls`를 우리 `QMainWindow`에 붙인다. napari의
  `Window`(자체 메인 창)는 쓰지 않는다.
- 확인한 것(napari 0.6.6, silx 2.1.0, `QT_QPA_PLATFORM=offscreen`): PySide6와
  PyQt5 각각에서 한 `QApplication`에 `QtViewer` + silx `Plot1D` + napari dock을 함께
  만들고, image/shapes layer 추가, `ndisplay` 2↔3 전환, `dims` step 이동이
  동작했다. offscreen에는 GL context가 없어서 실제 렌더링은 확인하지 못했다.
  M3에서 실제 display로 확인한다.
- binding은 PySide6 하나로 고정한다(`QT_API=pyside6`, silx도 같은 binding을
  따른다).
- 3D에는 bin-4 볼륨(545×512×512, float32 0.57 GB)을 만들어 올린다. 최종 볼륨인
  bin-1 2180×2048×2048(36.6 GB)과 bin-2 1090×1024×1024(4.6 GB)는 GL 3D
  texture로 올리기에 너무 크다. 2D slice는 full-res를 dask로 lazy하게 연다.
- viewer는 탭이 처음 보일 때 만든다. GL canvas는 Raw, Align, Preview, Volume에
  하나씩이다.

**레이아웃.**

- 왼쪽: project tree (group → scan → 볼륨/artifact).
- 가운데: workflow 탭 7개.
- 오른쪽 dock: 현재 탭 parameter (project에 binding).
- 아래 dock: job (진행, ETA, 취소)과 처리 로그.

상단 bar는 선택한 group 상태를 보여준다: center ✓(rings/sweep/eye/agent),
tilt ✓, z-band ✓, 재구성 n/m. 상태는 project에서 계산하고 따로 저장하지 않는다.
다음 단계 버튼이 강조된다.

**탭.**

1. **Data**
   - raw root 선택 → `scan.discover` 표: tag, kind, N, R, speed, exptime, E,
     import 여부, group.
   - 다중 선택 → import(bin, 이름) job, 또는 group 지정.
   - 기존 h5 adopt.
2. **Raw**
   - napari: −log projection stack(θ 축 slider)과 θ readout.
   - 평균 projection.
   - range check 배지, drift 버튼, tilt-flow 버튼(결과 수치와 group tilt 비교).
3. **Align**, 세 단계를 세로로:
   - **Rings.** ring center 선과 band별 marker를 그린다. prominence 배지는
     ≥ 8이면 녹색, 미만이면 황색이다. R = 180 scan은 비활성화하고 "open arc —
     ring 무효 (README: 462–467, 40 px 오차)"를 표시한 뒤 같은 group의 center를
     쓰자고 제안한다.
   - **Center.** sweep 결과의 score curve(silx)와 verdict 배지를 보여준다.
     candidate slice는 napari grid mode에 candidate당 layer 하나로 놓는다. layer마다
     자기 contrast를 쓴다. layer를 고르면 center가 정해진다. Ambiguous면 rival 두 개를 나란히 보여주고, Railed면 "범위 넓히기"
     버튼을 낸다. 그 결과 slice가 비어 있으면(flat) z-band 경고를 띄운다.
   - **Tilt.** tilt 목록으로 job을 돌린다. tilt마다 best-focus slice와 xz
     ortho를 grid로 놓는다. focus curve는 "참고용: tilt에 따라 단조 증가하므로
     판단 근거 아님"으로 표시하고, 결정은 panel 클릭(eye)으로 한다.
     ALIGNMENT §6의 실패 방법은 메뉴에 두지 않는다.
4. **Preview**
   - algorithm/filter/stripe/phase로 단일 slice와 ortho를 미리 본다.
   - slider로 바꿀 때 이전 요청은 취소된다(latest-wins).
   - 비교 모드: variant를 추가하면 napari grid와 ROI 표(signal/noise ROI → CNR,
     noise σ; TXM-Pal `roiTableWidget` 방식)를 보여준다. ROI는 Shapes layer
     하나를 모든 variant에 공유한다.
   - even/odd split 버튼.
5. **Batch**
   - group 결정 × scan 선택으로 계획된 명령 목록을 만든다(CLI 그대로 복사 가능).
   - GPU 선택과 lock 상태, 실행/취소.
6. **Volume**
   - h5를 dask로 lazy하게 열어 napari에 올린다(z slider). xz/yz ortho는 보조
     `QtViewer` 두 개로 보여 주고, `dims.events.current_step`으로 crosshair를
     연동한다.
   - focus-by-z curve(silx): z-wrap / air / sample 구간을 음영으로 표시하고,
     sample band를 자동 제안한 뒤 사용자가 확정한다.
   - valid FOV 원과 z 한계는 napari Shapes overlay로 그린다. window 기본값은
     p0.5/p99.5(`contrast_limits`)이다.
   - 3D: 같은 viewer를 `ndisplay = 3`으로 바꾸고 bin-4 볼륨을 올린다. rendering은
     mip / attenuated_mip / iso로, cut plane은 `depiction = "plane"`으로 고른다.
7. **Export**
   - z 범위 기본값은 group z-band이다. 16-bit window histogram 미리보기(silx),
     bin 출력.

쉽게 쓰기 위한 장치:

- 이 데이터에서 검증된 값이 기본이다: 정렬은 bin 2, fourierrec + parzen,
  tilt 시작값 20°.
- center 단위 환산은 자동이다.
- 결정 하나로 group 안의 scan 전부를 batch에 넣는다.
- 각 결정에는 근거 이미지가 붙어 남는다.

## 7. MCP 서버

`lamino-pal-mcp [--project PATH]`, Python `mcp` SDK(FastMCP), stdio.

**tool 생성.** registry를 순회하며 op마다 tool 하나를 만든다. 이름은
`align.center_sweep` → `align_center_sweep`, 설명은 op docstring(영문), 입력
schema는 pydantic 모델이다. 고정 tool로 `project_open`, `project_info`,
`log_tail`, `job_list`, `job_status`, `job_wait`, `job_cancel`, `job_result`를
더한다.

**반환.**

- 구조화 결과(JSON)에 artifact 경로를 담는다. 배열은 넘기지 않고 경로/ID만
  주고받는다.
- 판단이 눈에 달린 op(rings, center_sweep, tilt_scan, center_compare,
  preview, compare, split, focus_by_z, slice, ortho, mip)는 PNG를
  `ImageContent`로 같이 돌려준다. 긴 변 ≤ 1568 px, 8-bit이고, window 값과
  panel 라벨은 캡션 텍스트로 붙인다. 원본 해상도는 artifact로 남는다.

**job 모델.** job op는 즉시 `{job_id, est_s}`를 돌려준다.
`job_wait(job_id, timeout_s ≤ 45)`는 끝나거나 timeout이 될 때까지 기다린 뒤
progress를 돌려준다. Codex 기본 `tool_timeout_sec` 60보다 짧게 잡아 어떤 tool
호출도 timeout에 걸리지 않게 한다. MCP progress notification도 보낸다.

**resource / prompt.**

- resource: `lamino://project`(project.toml), `lamino://log`,
  `lamino://guide/alignment`(정렬 절차와 실패 방법 §6, agent가 재시도하지 않게).
- prompt: `align_mount(group)`(rings → sweep → tilt eye 확인 → decide),
  `reconstruct_group(group)`.

**안전.**

- raw root는 쓰기 금지(dispatcher 경로 검사).
- 덮어쓰기는 명시할 때만 한다.
- 취소는 자기 job state에 기록된 pgid에만 보낸다.
- agent 결정은 `by = "agent"`로 남는다.

**등록.** Claude Code:

```sh
claude mcp add lamino-pal --scope user -- \
  micromamba run -n lamino-pal lamino-pal-mcp \
  --project /home/stevek/work/261003_Lamino_test/lamino_pal/project.toml
```

(또는 저장소 `.mcp.json`). Codex `~/.codex/config.toml`:

```toml
[mcp_servers.lamino-pal]
command = "micromamba"
args = ["run", "-n", "lamino-pal", "lamino-pal-mcp",
        "--project", "/home/stevek/work/261003_Lamino_test/lamino_pal/project.toml"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

## 8. 저장소와 빌드

```
Lamino-Pal/
  Cargo.toml  src/{lib.rs, bl7c/, bind/, export/}   # lamino_pal_core
  lib/tomoxide/                                     # submodule, physwkim/tomoxide pin
  pyproject.toml                                    # maturin, python-source = "python"
  python/lamino_pal/ …                              # §5
  env.yaml  tests/  docs/DESIGN.md
```

```sh
micromamba create -f env.yaml        # python 3.12, numpy 1.26.4, h5py, silx 2.1.0,
                                     # napari 0.6.6, dask, pyside6, pyopengl, matplotlib,
                                     # tifffile, maturin, mcp, pydantic
git submodule update --init
maturin develop --release --features cuda
cargo build --release --features cuda --manifest-path lib/tomoxide/Cargo.toml -p tomoxide-cli
```

core와 CLI가 같은 submodule 커밋에서 빌드되므로, preview(프로세스 안)와
batch(CLI)의 geometry가 같은 코드다. job마다 `tomoxide --version`을 로그에
남긴다.

## 9. 시험

시험 데이터는 `LAMINO_PAL_DATA=~/work/261003_Lamino_test`이다. 없으면 data 시험은
skip이 아니라 실패로 처리한다.

- Rust(`cargo nextest run --no-default-features`): `.dat`/metadata 해석, fly R 판정, binning, units.
- 구조:
  - registry의 op 집합 = MCP `list_tools`(고정 tool 제외) = GUI action 표.
  - 모든 op 호출이 log.jsonl에 남는다.
  - 두 writer가 동시에 `Project.commit`해도 revision이 단조 증가한다.
  - job 취소는 자기 process group만 죽인다.
- golden (기존 산출물 대비):
  - `scan.import` step N1000 bin 2 → `step1000_bin2.h5`와 data/flat/theta 동일.
    fly 10개 → `build_fly_h5.py` 산출물과 동일.
  - `units`: bin-2 504 ↔ 1008.5.
  - `align.rings` step1000_bin2 → `trustworthy = false`(prominence 3.08).
    f180 scan → `Refused`.
  - `align.center_sweep` → Ambiguous, rival {504.8, 487.8}.
  - `recon.run` c = 1008.5, t = 20, bin 1 →
    `final_step1000_bin1_c1008p5_t20_fourierrec_parzen.h5`와 비교.
- MCP e2e: stdio client로 rings → sweep → preview → `recon.run` job →
  `job_wait`/`job_cancel`. 이미지 content가 오는지 확인한다.
- GUI smoke: `QT_QPA_PLATFORM=offscreen`, project 열기, instant op 하나,
  job dock 갱신, napari layer 추가와 `ndisplay` 전환. offscreen에는 GL이 없으므로
  렌더링 결과(screenshot)는 실제 display가 있는 시험에서만 본다.

## 10. 마일스톤

- **M0.** 저장소 골격, env.yaml, 빈 core maturin 빌드, registry + dispatcher +
  project + log.
- **M1.** tomoxide T1(필수), T2, T3.
- **M2.** `bl7c` import/discover/adopt, units, golden import 시험.
- **M3.** raw/align op + GUI Data/Raw/Align 탭.
- **M4.** preview/compare/split, job runner + `recon.run`(4 GPU), Batch/Volume
  탭.
- **M5.** MCP 서버, 등록, e2e. export와 3D 탭.

## 11. 결정 필요

1. **T1을 physwkim/tomoxide에 먼저 넣는지.** Lamino-Pal M3 이후는 T1에 의존한다.
