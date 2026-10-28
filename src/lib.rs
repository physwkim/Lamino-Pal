//! BL7C formats and tomoxide bindings behind `lamino_pal._core`.

/// tomoxide submodule commit this core was built from.
pub const TOMOXIDE_REV: &str = env!("TOMOXIDE_REV");

#[cfg(feature = "python")]
mod python {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;

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

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        m.add("LaminoPalError", py.get_type::<LaminoPalError>())?;
        m.add("Refused", py.get_type::<Refused>())?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(tomoxide_rev, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_enabled, m)?)?;
        Ok(())
    }
}
