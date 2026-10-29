//! BL7C formats and tomoxide bindings behind `lamino_pal._core`.

/// tomoxide submodule commit this core was built from.
pub const TOMOXIDE_REV: &str = env!("TOMOXIDE_REV");

pub mod bl7c;
pub mod prep;

#[cfg(feature = "python")]
mod bind;

#[cfg(feature = "python")]
mod python {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;

    use crate::bl7c;

    create_exception!(_core, LaminoPalError, PyException);
    // A guard refused the request; the message says why.
    create_exception!(_core, Refused, LaminoPalError);

    #[pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[pyfunction]
    fn tomoxide_rev() -> &'static str {
        super::TOMOXIDE_REV
    }

    #[pyfunction]
    fn cuda_enabled() -> bool {
        cfg!(feature = "cuda")
    }

    /// The devices tomoxide runs CUDA work on: `TOMOXIDE_CUDA_DEVICES`, else
    /// every visible one; none without the `cuda` feature.
    #[pyfunction]
    fn cuda_devices() -> Vec<i32> {
        tomoxide::cuda::selected_devices()
    }

    fn bl7c_err(e: bl7c::Error) -> PyErr {
        match e {
            bl7c::Error::Refused(why) => Refused::new_err(why),
            e => LaminoPalError::new_err(e.to_string()),
        }
    }

    /// JSON `{"scans": [...], "skipped": [[raw, why], ...]}` under `root`.
    #[pyfunction]
    fn bl7c_discover(py: Python<'_>, root: PathBuf) -> PyResult<String> {
        let d = py.detach(|| bl7c::discover(&root)).map_err(bl7c_err)?;
        Ok(serde_json::to_string(&d).expect("plain data"))
    }

    /// Import the scan at `raw` under `root` to `out`; JSON of
    /// `bl7c::Imported`. `progress(done, total)` runs at most 10 times a
    /// second and once at the end; an exception from it stops the import and
    /// is raised here.
    #[pyfunction]
    #[pyo3(signature = (root, raw, out, bin, progress=None))]
    fn bl7c_import(
        py: Python<'_>,
        root: PathBuf,
        raw: String,
        out: PathBuf,
        bin: usize,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        let mut raised = None;
        let mut last: Option<Instant> = None;
        let r = py.detach(|| {
            bl7c::import(&root, &raw, &out, bin, &mut |done, total| {
                let Some(cb) = &progress else { return true };
                if done < total && last.is_some_and(|t| t.elapsed() < Duration::from_millis(100)) {
                    return true;
                }
                last = Some(Instant::now());
                Python::attach(|py| cb.call1(py, (done, total)))
                    .map_err(|e| raised = Some(e))
                    .is_ok()
            })
        });
        if let Some(e) = raised {
            return Err(e);
        }
        let r = r.map_err(bl7c_err)?;
        Ok(serde_json::to_string(&r).expect("plain data"))
    }

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        m.add("LaminoPalError", py.get_type::<LaminoPalError>())?;
        m.add("Refused", py.get_type::<Refused>())?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(tomoxide_rev, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_enabled, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_devices, m)?)?;
        m.add_function(wrap_pyfunction!(bl7c_discover, m)?)?;
        m.add_function(wrap_pyfunction!(bl7c_import, m)?)?;
        crate::bind::add(m)
    }
}
