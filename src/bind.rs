//! tomoxide's alignment steps for `lamino_pal._core`.
//!
//! Arrays cross as numpy, results with fields as JSON (as the bl7c bindings
//! do). Every call that reads a file or computes releases the GIL. CUDA work
//! runs on the devices `TOMOXIDE_CUDA_DEVICES` names, read by tomoxide per call.

use std::path::PathBuf;

use ndarray::{s, Array2, Array3, ArrayBase, Axis, Data, Dimension};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use serde_json::{json, Value};
use tomoxide::align::PreppedStack;
use tomoxide::config::Config;
use tomoxide::recon::center::{SampleBand, SweepVerdict, TiltFocus};
use tomoxide::{Angles, Beam, Center, CpuBackend, CudaBackend, Detector, Error, Geometry};

use crate::prep;
use crate::python::{LaminoPalError, Refused};

fn tomoxide_err(e: Error) -> PyErr {
    match e {
        Error::InvalidParam(why) => Refused::new_err(why),
        e => LaminoPalError::new_err(e.to_string()),
    }
}

// numpy's ndarray is not tomoxide's, so arrays cross by value.
fn numpy<'py, S: Data<Elem = f32>, D: Dimension>(
    py: Python<'py>,
    a: &ArrayBase<S, D>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = a.shape().to_vec();
    Ok(PyArray1::from_vec(py, a.iter().copied().collect())
        .reshape(shape)?
        .into_any())
}

fn array2(a: &PyReadonlyArray2<f32>) -> Array2<f32> {
    let v = a.as_array();
    Array2::from_shape_vec(v.dim(), v.iter().copied().collect()).expect("its own elements")
}

/// Projections `indices` of DXchange `path`, flat/dark corrected and −log,
/// `[len(indices), nz, nx]` float32.
#[pyfunction]
fn prep_frames<'py>(
    py: Python<'py>,
    path: PathBuf,
    indices: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let a = py
        .detach(|| prep::frames(&path, &indices))
        .map_err(tomoxide_err)?;
    numpy(py, &a)
}

/// The mean of every `step`th prepped projection of `path`, `[nz, nx]`.
#[pyfunction]
fn mean_projection<'py>(
    py: Python<'py>,
    path: PathBuf,
    step: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let a = py
        .detach(|| prep::mean_projection(&path, step))
        .map_err(tomoxide_err)?;
    numpy(py, &a)
}

/// JSON `{center, prominence, trustworthy}` of the rings in a mean
/// projection, or in a band of its rows.
#[pyfunction]
fn rings_from_mean(py: Python<'_>, mean: PyReadonlyArray2<f32>) -> PyResult<String> {
    let m = array2(&mean);
    let r = py
        .detach(|| tomoxide::recon::center::find_center_rings_from_mean(m.view(), &CpuBackend))
        .map_err(tomoxide_err)?;
    Ok(
        json!({"center": r.center, "prominence": r.prominence, "trustworthy": r.trustworthy})
            .to_string(),
    )
}

/// `(dy, dx)` that moves `moving` onto `reference`, to `1/upsample` px.
#[pyfunction]
fn register_shift(
    py: Python<'_>,
    reference: PyReadonlyArray2<f32>,
    moving: PyReadonlyArray2<f32>,
    upsample: f64,
) -> PyResult<(f64, f64)> {
    let (r, m) = (array2(&reference), array2(&moving));
    py.detach(|| tomoxide::recon::center::register_shift(&r, &m, &CpuBackend, upsample))
        .map_err(tomoxide_err)
}

fn array3(a: &PyReadonlyArray3<f32>) -> Array3<f32> {
    let v = a.as_array();
    Array3::from_shape_vec(v.dim(), v.iter().copied().collect()).expect("its own elements")
}

/// `recipe`, JSON of tomoxide `Config` fields, read as `tomoxide recon
/// --config` reads its file. A field Config lacks is refused: serde drops it,
/// so a misspelt one would run on its default.
fn config(recipe: &str) -> PyResult<Config> {
    let bad = |e: serde_json::Error| Refused::new_err(format!("recipe: {e}"));
    let v: Value = serde_json::from_str(recipe).map_err(bad)?;
    let known = serde_json::to_value(Config::default()).expect("plain data");
    let (Some(given), Some(known)) = (v.as_object(), known.as_object()) else {
        return Err(Refused::new_err("recipe: not a table of Config fields"));
    };
    if let Some(k) = given.keys().find(|k| !known.contains_key(*k)) {
        return Err(Refused::new_err(format!(
            "recipe: tomoxide Config has no field {k}"
        )));
    }
    let c: Config = serde_json::from_value(v).map_err(bad)?;
    c.algorithm()
        .and(c.recon_params())
        .and(c.prep())
        .map_err(tomoxide_err)?;
    Ok(c)
}

/// Every tomoxide `Config` field as `recipe` (JSON of some of them) sets it,
/// as JSON; refused as a reconstruction would refuse it: an unknown field, an
/// algorithm chain, a bad name or value.
#[pyfunction]
fn recipe_config(recipe: &str) -> PyResult<String> {
    Ok(serde_json::to_string(&config(recipe)?).expect("plain data"))
}

/// Output rows of a laminography reconstruction of `height` detector rows at
/// `tilt_deg`.
#[pyfunction]
fn lamino_depth(height: usize, tilt_deg: f32) -> usize {
    tomoxide::cuda::lamino_recon_height(height, tilt_deg)
}

/// JSON `{depth, r_lateral, radius, full_z}` of the disk every projection saw
/// in a laminography reconstruction of a `width` × `height` detector at
/// `center` and `tilt_deg`: `radius[z]` per output row, `full_z` the
/// inclusive rows whose whole `r_lateral` disk is valid, or null.
#[pyfunction]
fn valid_fov(width: usize, height: usize, center: f32, tilt_deg: f32) -> PyResult<String> {
    let depth = lamino_depth(height, tilt_deg);
    let geom = Geometry {
        angles: Angles(vec![]),
        center: Center::Scalar(center),
        beam: Beam::laminography(tilt_deg),
        detector: Detector {
            width,
            height,
            pixel_size: 1.0,
        },
    };
    let f = geom.valid_fov(depth).map_err(tomoxide_err)?;
    Ok(
        json!({"depth": depth, "r_lateral": f.r_lateral, "radius": f.radius,
               "full_z": f.full_z()})
        .to_string(),
    )
}

/// The focus of every slice of `vol` along axis 0: mean |∇|² inside a
/// 0.92-FOV disk, the score the tilt scan and center sweep rank by.
#[pyfunction]
fn focus_by_z(py: Python<'_>, vol: PyReadonlyArray3<f32>) -> Vec<f64> {
    let v = array3(&vol);
    py.detach(|| tomoxide::recon::center::focus_by_z(v.view()))
}

/// `center ± half_width` in steps of `step`, `center` included.
#[pyfunction]
fn grid(center: f32, half_width: f32, step: f32) -> Vec<f32> {
    tomoxide::align::grid(center, half_width, step)
}

/// JSON `{kind, index, value, rivals: [{index, value, prominence, deficit}]}`
/// of a sweep's focus curve; `kind` is resolved, railed, ambiguous or flat.
#[pyfunction]
fn judge_sweep(cands: Vec<f32>, score: Vec<f64>) -> PyResult<String> {
    let v = tomoxide::recon::center::judge_sweep(&cands, &score).ok_or_else(|| {
        Refused::new_err(format!(
            "a sweep of {} candidates and {} scores",
            cands.len(),
            score.len()
        ))
    })?;
    let (kind, index, value, rivals) = match v {
        SweepVerdict::Resolved { index, value } => ("resolved", index, value, vec![]),
        SweepVerdict::Railed { index, value } => ("railed", index, value, vec![]),
        SweepVerdict::Flat { index, value } => ("flat", index, value, vec![]),
        SweepVerdict::Ambiguous {
            index,
            value,
            rivals,
        } => ("ambiguous", index, value, rivals),
    };
    let rivals: Vec<Value> = rivals
        .iter()
        .map(|r| {
            json!({"index": r.index, "value": r.value, "prominence": r.prominence,
                   "deficit": r.deficit})
        })
        .collect();
    Ok(json!({"kind": kind, "index": index, "value": value, "rivals": rivals}).to_string())
}

fn focus_json(f: &TiltFocus) -> Value {
    json!({"tilt_deg": f.tilt_deg, "focus": f.focus, "z_peak": f.z_peak, "band": f.band,
           "depth": f.depth, "focus_by_z": f.focus_by_z})
}

/// A DXchange file's whole projection stack, prepped, in host memory.
#[pyclass(frozen, module = "lamino_pal._core")]
struct Stack(PreppedStack);

#[pymethods]
impl Stack {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        py.detach(|| PreppedStack::load(&CpuBackend, &path))
            .map(Stack)
            .map_err(tomoxide_err)
    }

    /// `(nproj, nz, nx)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.0.shape()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.0.bytes()
    }

    /// Output rows at `tilt_deg`; `None` is parallel beam.
    #[pyo3(signature = (tilt_deg=None))]
    fn depth(&self, tilt_deg: Option<f32>) -> usize {
        self.0.depth(tilt_deg)
    }

    /// The slice a centre sweep scores: the middle of `band`, else of the
    /// volume at `tilt_deg`.
    #[pyo3(signature = (tilt_deg=None, band=None))]
    fn default_slice(&self, tilt_deg: Option<f32>, band: Option<(usize, usize)>) -> usize {
        self.0.default_slice(tilt_deg, band)
    }

    /// JSON of the tilt read off the flow between neighbouring projections:
    /// `{tilt_deg, tilt_sigma_deg, roll_deg, roll_sigma_deg, axis_col,
    /// axis_col_sigma, pairs: [...]}`.
    fn tilt_flow(&self, py: Python<'_>) -> PyResult<String> {
        let f = py
            .detach(|| self.0.tilt_flow(&CpuBackend))
            .map_err(tomoxide_err)?;
        let pairs: Vec<Value> = f
            .pairs
            .iter()
            .map(|p| {
                json!({"first": p.first, "gap": p.gap, "dtheta_deg": p.dtheta_deg,
                       "tilt_deg": p.tilt_deg, "roll_deg": p.roll_deg, "axis_col": p.axis_col,
                       "blocks": p.blocks, "resid_px": p.resid_px})
            })
            .collect();
        Ok(
            json!({"tilt_deg": f.tilt_deg, "tilt_sigma_deg": f.tilt_sigma_deg,
                  "roll_deg": f.roll_deg, "roll_sigma_deg": f.roll_sigma_deg,
                  "axis_col": f.axis_col, "axis_col_sigma": f.axis_col_sigma, "pairs": pairs})
            .to_string(),
        )
    }

    /// Output slice `slice` reconstructed at every centre of `centers`:
    /// `(frames [len(centers), ny, nx], focus per centre)`.
    fn center_sweep<'py>(
        &self,
        py: Python<'py>,
        tilt_deg: Option<f32>,
        slice: usize,
        centers: Vec<f32>,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<f64>)> {
        let s = py
            .detach(|| self.0.center_sweep(tilt_deg, slice, &centers))
            .map_err(tomoxide_err)?;
        Ok((numpy(py, &s.frames)?, s.focus))
    }

    /// The stack of projections `indices` only, in that order.
    fn select(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Stack> {
        py.detach(|| self.0.select(&indices))
            .map(Stack)
            .map_err(tomoxide_err)
    }

    /// Reconstruct with `recipe` (JSON of tomoxide `Config` fields) at
    /// `center` and `tilt_deg`, keeping the three planes through voxel `(z, y,
    /// x)`: `(slice z [n, n], xz at row y [rh, n], yz at column x [rh, n])`,
    /// as `tomoxide recon` writes them for the same recipe. `on_rows(done,
    /// rh)` runs after each tile; an exception from it stops the
    /// reconstruction and is raised here.
    #[pyo3(signature = (recipe, center, tilt_deg, at, on_rows=None))]
    fn planes<'py>(
        &self,
        py: Python<'py>,
        recipe: &str,
        center: f32,
        tilt_deg: f32,
        at: (usize, usize, usize),
        on_rows: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let cfg = config(recipe)?;
        let (algorithm, params, prep) = (
            cfg.algorithm().map_err(tomoxide_err)?,
            cfg.recon_params().map_err(tomoxide_err)?,
            cfg.prep().map_err(tomoxide_err)?,
        );
        let (z, y, x) = at;
        let (_, _, n) = self.0.shape();
        let rh = self.0.depth(Some(tilt_deg));
        if z >= rh || y >= n || x >= n {
            return Err(Refused::new_err(format!(
                "plane (z {z}, y {y}, x {x}) is outside the {rh} x {n} x {n} volume"
            )));
        }
        let geom = self.0.geometry(center, Some(tilt_deg));
        let (mut slice, mut xz, mut yz) = (
            Array2::zeros((n, n)),
            Array2::zeros((rh, n)),
            Array2::zeros((rh, n)),
        );
        let mut raised = None;
        let r = py.detach(|| {
            let backend = CudaBackend::new()?;
            self.0.reconstruct(
                &backend,
                &geom,
                algorithm,
                &params,
                &prep,
                &mut |rh0, tile| {
                    let t = &tile.array;
                    let rows = rh0..rh0 + t.len_of(Axis(0));
                    if rows.contains(&z) {
                        slice.assign(&t.index_axis(Axis(0), z - rh0));
                    }
                    xz.slice_mut(s![rows.clone(), ..])
                        .assign(&t.slice(s![.., y, ..]));
                    yz.slice_mut(s![rows.clone(), ..])
                        .assign(&t.slice(s![.., .., x]));
                    let Some(cb) = &on_rows else { return Ok(()) };
                    Python::attach(|py| cb.call1(py, (rows.end, rh)).map(drop)).map_err(|e| {
                        raised = Some(e);
                        Error::Backend("on_rows raised".into())
                    })
                },
            )
        });
        if let Some(e) = raised {
            return Err(e);
        }
        let shape = r.map_err(tomoxide_err)?;
        if shape != (rh, n, n) {
            return Err(LaminoPalError::new_err(format!(
                "reconstruction is {shape:?}, not ({rh}, {n}, {n})"
            )));
        }
        Ok((numpy(py, &slice)?, numpy(py, &xz)?, numpy(py, &yz)?))
    }

    /// Reconstruct fully at each tilt of `tilts` and score each by its best
    /// slice; JSON list of `{tilt_deg, focus, z_peak, band, depth,
    /// focus_by_z}`. `band` is `((z0, z1), tilt_deg)`: the sample's rows in a
    /// reconstruction at that tilt. `on_tilt(focus_json, best_slice)` runs
    /// after each tilt; an exception from it stops the scan and is raised here.
    #[pyo3(signature = (center, tilts, band=None, on_tilt=None))]
    fn tilt_scan(
        &self,
        py: Python<'_>,
        center: f32,
        tilts: Vec<f32>,
        band: Option<((usize, usize), f32)>,
        on_tilt: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        let band = band.map(|(z, tilt_deg)| SampleBand { z, tilt_deg });
        let mut raised = None;
        let r = py.detach(|| {
            self.0.tilt_scan(center, &tilts, band, &mut |f, best| {
                let Some(cb) = &on_tilt else { return Ok(()) };
                Python::attach(|py| -> PyResult<()> {
                    cb.call1(py, (focus_json(f).to_string(), numpy(py, &best)?))?;
                    Ok(())
                })
                .map_err(|e| {
                    raised = Some(e);
                    Error::Backend("on_tilt raised".into())
                })
            })
        });
        if let Some(e) = raised {
            return Err(e);
        }
        let foci = r.map_err(tomoxide_err)?;
        Ok(Value::Array(foci.iter().map(focus_json).collect()).to_string())
    }
}

pub(crate) fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Stack>()?;
    m.add_function(wrap_pyfunction!(prep_frames, m)?)?;
    m.add_function(wrap_pyfunction!(mean_projection, m)?)?;
    m.add_function(wrap_pyfunction!(rings_from_mean, m)?)?;
    m.add_function(wrap_pyfunction!(register_shift, m)?)?;
    m.add_function(wrap_pyfunction!(grid, m)?)?;
    m.add_function(wrap_pyfunction!(judge_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(recipe_config, m)?)?;
    m.add_function(wrap_pyfunction!(lamino_depth, m)?)?;
    m.add_function(wrap_pyfunction!(valid_fov, m)?)?;
    m.add_function(wrap_pyfunction!(focus_by_z, m)?)?;
    Ok(())
}
