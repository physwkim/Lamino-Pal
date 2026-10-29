//! A few −log projections, prepped as the align stack preps all of them.
//!
//! [`PreppedStack::load`](tomoxide::align::PreppedStack::load) reads and preps
//! the whole stack; a frame viewer, the drift check and the ring mean need a
//! handful of frames, so these read only those and run the same
//! `normalize_dataset` on them, on the CPU.

use std::path::Path;

use ndarray::{Array2, Array3};
use tomoxide::io::{open_dxchange, read_h5_band, read_h5_frame};
use tomoxide::{CpuBackend, Dataset, Error, Frames, Layout, Result, Tomo};

fn shaped(dim: (usize, usize, usize), v: Vec<f32>) -> Result<Array3<f32>> {
    Array3::from_shape_vec(dim, v).map_err(|e| Error::ShapeMismatch {
        expected: format!("{dim:?}"),
        found: e.to_string(),
    })
}

/// Projections `indices` of the DXchange file at `path`, flat/dark corrected
/// and −log, `[indices.len(), nz, nx]`.
pub fn frames(path: &Path, indices: &[usize]) -> Result<Array3<f32>> {
    let p = path.to_string_lossy();
    let (nproj, nz, nx, nflat, ndark) = open_dxchange(&p)?.read_sizes()?;
    if let Some(i) = indices.iter().find(|&&i| i >= nproj) {
        return Err(Error::InvalidParam(format!(
            "projection {i}: {p} has {nproj}"
        )));
    }
    let mut data = Vec::with_capacity(indices.len() * nz * nx);
    for &i in indices {
        data.extend(read_h5_frame(&p, "/exchange/data", i)?.2);
    }
    let stack = |name: &str, n: usize| -> Result<Option<Frames<f32>>> {
        if n == 0 {
            return Ok(None);
        }
        let (k, ny, nx, v) = read_h5_band(&p, name, 0, n)?;
        Ok(Some(Frames::new(shaped((k, ny, nx), v)?)))
    };
    let mut ds = Dataset {
        data: Tomo::new(shaped((indices.len(), nz, nx), data)?, Layout::Projection),
        flat: stack("/exchange/data_white", nflat)?,
        dark: stack("/exchange/data_dark", ndark)?,
        theta: Vec::new(),
    };
    tomoxide::prep::normalize_dataset(&mut ds, &CpuBackend)?;
    // Prep keeps the caller's layout.
    Ok(ds.data.array)
}

/// The mean of every `step`th prepped projection, tomoxide's
/// `mean_projection` read from the file.
pub fn mean_projection(path: &Path, step: usize) -> Result<Array2<f32>> {
    if step == 0 {
        return Err(Error::InvalidParam(
            "mean_projection: step must be ≥1".into(),
        ));
    }
    let nproj = open_dxchange(&path.to_string_lossy())?.read_sizes()?.0;
    let indices: Vec<usize> = (0..nproj).step_by(step).collect();
    let tomo = Tomo::new(frames(path, &indices)?, Layout::Projection);
    tomoxide::recon::center::mean_projection(&tomo, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use rust_hdf5::H5File;
    use tomoxide::align::PreppedStack;

    /// 5 projections of 3x4 with 2 flats and a nonzero dark.
    fn dxchange(name: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("lamino-pal-{name}-{}.h5", std::process::id()));
        let file = H5File::create(&path).unwrap();
        let ex = file.create_group("exchange").unwrap();
        let put = |name: &str, n: usize, px: &dyn Fn(usize) -> u16| {
            let v: Vec<u16> = (0..n * 12).map(px).collect();
            ex.new_dataset::<u16>()
                .shape([n, 3, 4])
                .create(name)
                .and_then(|d| d.write_raw(&v))
                .unwrap();
        };
        put("data", 5, &|j| 200 + (j * 37 % 101) as u16);
        put("data_white", 2, &|j| 900 + (j % 7) as u16);
        put("data_dark", 1, &|j| 10 + (j % 3) as u16);
        ex.new_dataset::<f32>()
            .shape([5])
            .create("theta")
            .and_then(|d| d.write_raw(&[0.0f32, 72.0, 144.0, 216.0, 288.0]))
            .unwrap();
        file.close().unwrap();
        path
    }

    #[test]
    fn frames_equal_the_loaded_stack() {
        let path = dxchange("frames");
        let stack = PreppedStack::load(&CpuBackend, &path).unwrap();
        let all = &stack.data().array;
        let got = frames(&path, &[3, 0]).unwrap();
        assert_eq!(got.index_axis(Axis(0), 0), all.index_axis(Axis(0), 3));
        assert_eq!(got.index_axis(Axis(0), 1), all.index_axis(Axis(0), 0));
        assert!(frames(&path, &[5]).is_err());
        let mean = mean_projection(&path, 2).unwrap();
        let want = tomoxide::recon::center::mean_projection(stack.data(), 2).unwrap();
        assert_eq!(mean, want);
        std::fs::remove_file(path).unwrap();
    }
}
