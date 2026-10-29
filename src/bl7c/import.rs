//! A scan folder to one DXchange h5, as build_step_h5.py / build_fly_h5.py
//! wrote them: u16 `exchange/{data, data_white, data_dark}` with an all-zero
//! dark (the sets have no dark frames) and f32 `exchange/theta` in degrees.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rust_hdf5::H5File;
use serde::Serialize;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

use super::bin::bin_mean;
use super::{open, Error, Layout, Scan};

/// Frames decoded in parallel per h5 write.
const BATCH: usize = 32;

#[derive(Debug, Serialize)]
pub struct Imported {
    pub scan: Scan,
    pub bin: usize,
    /// `exchange/data` shape: projections, binned rows, binned columns.
    pub shape: [usize; 3],
    /// Projection pixels at 65535 before binning; flats are not counted.
    pub saturated: u64,
    pub saturated_frac: f64,
}

/// Import the scan at `raw` under `root` to `out`, `bin`×`bin` mean binned.
///
/// `out` is written in place and left half written on an error: the caller
/// names a scratch file and renames it.
/// `progress(done, total)` runs after every batch of frames; false aborts.
pub fn import(
    root: &Path,
    raw: &str,
    out: &Path,
    bin: usize,
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> Result<Imported, Error> {
    if bin == 0 {
        return Err(Error::Refused("bin must be at least 1".into()));
    }
    let lay = open(root, raw)?;
    let (Some(parent), Some(_)) = (out.parent(), out.file_name()) else {
        return Err(Error::Refused(format!(
            "{} is not a file path",
            out.display()
        )));
    };
    let canon = |p: &Path| p.canonicalize().map_err(|e| Error::Io(p.to_owned(), e));
    if canon(parent)?.starts_with(canon(root)?) {
        return Err(Error::Refused(format!(
            "{} is under the raw root",
            out.display()
        )));
    }
    let (h, w, _) = read_frame(&lay.proj[0])?;
    if h < bin || w < bin {
        return Err(Error::Refused(format!(
            "bin {bin} exceeds the {h}x{w} frame"
        )));
    }
    let shape = [lay.scan.n, h / bin, w / bin];
    let saturated = write(out, &lay, (h, w), bin, shape, progress)?;
    Ok(Imported {
        saturated_frac: saturated as f64 / (lay.scan.n * h * w) as f64,
        scan: lay.scan,
        bin,
        shape,
        saturated,
    })
}

fn write(
    path: &Path,
    lay: &Layout,
    (h, w): (usize, usize),
    bin: usize,
    [n, bh, bw]: [usize; 3],
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> Result<u64, Error> {
    let h5 = |e| Error::H5(path.to_owned(), e);
    let total = n + lay.flat.len();
    let file = H5File::create(path).map_err(h5)?;
    let ex = file.create_group("exchange").map_err(h5)?;
    // Contiguous, so each batch is one `write_slice`.
    let data = ex
        .new_dataset::<u16>()
        .shape([n, bh, bw])
        .create("data")
        .map_err(h5)?;
    let mut saturated = 0;
    for i0 in (0..n).step_by(BATCH) {
        let k = BATCH.min(n - i0);
        let (slab, sat) = binned(&lay.proj[i0..i0 + k], (h, w), bin)?;
        saturated += sat;
        data.write_slice(&[i0, 0, 0], &[k, bh, bw], &slab)
            .map_err(h5)?;
        if !progress(i0 + k, total) {
            return Err(Error::Aborted);
        }
    }
    let (white, _) = binned(&lay.flat, (h, w), bin)?;
    ex.new_dataset::<u16>()
        .shape([lay.flat.len(), bh, bw])
        .create("data_white")
        .and_then(|d| d.write_raw(&white))
        .map_err(h5)?;
    ex.new_dataset::<u16>()
        .shape([1, bh, bw])
        .create("data_dark")
        .and_then(|d| d.write_raw(&vec![0u16; bh * bw]))
        .map_err(h5)?;
    ex.new_dataset::<f32>()
        .shape([n])
        .create("theta")
        .and_then(|d| d.write_raw(&lay.theta))
        .map_err(h5)?;
    let lp = file.create_group("lamino_pal").map_err(h5)?;
    lp.set_attr_numeric("binning", &(bin as u32)).map_err(h5)?;
    lp.set_attr_string("source", &lay.scan.raw).map_err(h5)?;
    lp.set_attr_string("kind", lay.scan.kind.as_str())
        .map_err(h5)?;
    if let Some(e) = lay.scan.energy_ev {
        lp.set_attr_numeric("energy_ev", &e).map_err(h5)?;
    }
    file.close().map_err(h5)?;
    if !progress(total, total) {
        return Err(Error::Aborted);
    }
    Ok(saturated)
}

/// The binned frames back to back, and their pixels at 65535 before binning.
fn binned(paths: &[PathBuf], (h, w): (usize, usize), bin: usize) -> Result<(Vec<u16>, u64), Error> {
    let frames = paths
        .par_iter()
        .map(|p| {
            let (fh, fw, px) = read_frame(p)?;
            if (fh, fw) != (h, w) {
                return Err(Error::Format(
                    p.clone(),
                    format!("{fh}x{fw}, not {h}x{w} as the first projection"),
                ));
            }
            let sat = px.iter().filter(|&&v| v == u16::MAX).count() as u64;
            Ok((bin_mean(&px, h, w, bin), sat))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut slab = Vec::with_capacity(frames.len() * (h / bin) * (w / bin));
    let mut sat = 0;
    for (f, s) in frames {
        slab.extend_from_slice(&f);
        sat += s;
    }
    Ok((slab, sat))
}

/// A 16-bit greyscale TIFF: (rows, columns, pixels).
fn read_frame(path: &Path) -> Result<(usize, usize, Vec<u16>), Error> {
    let tiff = |e| Error::Tiff(path.to_owned(), e);
    let f = File::open(path).map_err(|e| Error::Io(path.to_owned(), e))?;
    let mut dec = Decoder::new(BufReader::new(f)).map_err(tiff)?;
    let (w, h) = dec.dimensions().map_err(tiff)?;
    match dec.colortype().map_err(tiff)? {
        ColorType::Gray(16) => {}
        c => {
            return Err(Error::Format(
                path.to_owned(),
                format!("{c:?}, not 16-bit greyscale"),
            ))
        }
    }
    match dec.read_image().map_err(tiff)? {
        DecodingResult::U16(px) => Ok((h as usize, w as usize, px)),
        _ => Err(Error::Format(path.to_owned(), "not u16 samples".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::tempdir;
    use super::*;
    use std::fs;
    use tiff::encoder::{colortype::Gray16, TiffEncoder};

    fn tif(path: &Path, h: usize, w: usize, px: impl Fn(usize) -> u16) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let data: Vec<u16> = (0..h * w).map(px).collect();
        TiffEncoder::new(File::create(path).unwrap())
            .unwrap()
            .write_image::<Gray16>(w as u32, h as u32, &data)
            .unwrap();
    }

    /// A 4-frame 360 deg fly scan of 5x4 frames; frame i pixel j is 10·i + j,
    /// frame 3 pixel 0 is saturated; one flat of 7s.
    fn fly(root: &Path) {
        let d = root.join("fly/F_00001");
        fs::create_dir_all(&d).unwrap();
        fs::write(
            d.join("metadata.txt"),
            r#"{"exptime": 0.5, "numprj": 4, "scan_speed": 180.0, "current_energy": 8332.9}"#,
        )
        .unwrap();
        for i in 0..4 {
            tif(&d.join(format!("TOMO_{i:05}.tif")), 5, 4, |j| {
                if i == 3 && j == 0 {
                    u16::MAX
                } else {
                    (10 * i + j) as u16
                }
            });
        }
        tif(&d.join("FF_00000.tif"), 5, 4, |_| 7);
    }

    fn read<T: rust_hdf5::H5Type>(f: &H5File, name: &str) -> (Vec<usize>, Vec<T>) {
        let d = f.dataset(name).unwrap();
        (d.shape(), d.read_raw().unwrap())
    }

    #[test]
    fn import_writes_dxchange_and_the_provenance_attrs() {
        let tmp = tempdir();
        let (root, work) = (tmp.join("raw"), tmp.join("work"));
        fly(&root);
        fs::create_dir_all(&work).unwrap();
        let out = work.join("f.h5");
        let mut seen = Vec::new();
        let r = import(&root, "fly/F_00001", &out, 2, &mut |d, t| {
            seen.push((d, t));
            true
        })
        .unwrap();
        assert_eq!((r.shape, r.saturated, r.bin), ([4, 2, 2], 1, 2));
        assert_eq!(r.saturated_frac, 1.0 / 80.0);
        assert_eq!(seen, [(4, 5), (5, 5)]);

        let f = H5File::open(&out).unwrap();
        let (shape, data) = read::<u16>(&f, "exchange/data");
        assert_eq!(shape, [4, 2, 2]);
        // Frame 0 rows 0-1 cols 0-1: (0+1+4+5)/4 = 2.5 -> 2; row 4 is cropped.
        assert_eq!(&data[..4], [2, 4, 10, 12]);
        // Frame 3 block 0: (65535+31+34+35)/4 = 16408.75 -> 16409.
        assert_eq!(data[12], 16409);
        assert_eq!(
            read::<u16>(&f, "exchange/data_white"),
            (vec![1, 2, 2], vec![7; 4])
        );
        assert_eq!(
            read::<u16>(&f, "exchange/data_dark"),
            (vec![1, 2, 2], vec![0; 4])
        );
        assert_eq!(
            read::<f32>(&f, "exchange/theta").1,
            [0.0, 90.0, 180.0, 270.0]
        );
        let lp = f.root_group().group("lamino_pal").unwrap();
        assert_eq!(lp.attr_string("source").unwrap(), "fly/F_00001");
        assert_eq!(lp.attr_string("kind").unwrap(), "fly");
        let mut names = lp.attr_names().unwrap();
        names.sort();
        assert_eq!(names, ["binning", "energy_ev", "kind", "source"]);
    }

    #[test]
    fn an_abort_or_a_bad_frame_is_an_error() {
        let tmp = tempdir();
        let (root, work) = (tmp.join("raw"), tmp.join("work"));
        fly(&root);
        fs::create_dir_all(&work).unwrap();
        let out = work.join("f.h5");
        let e = import(&root, "fly/F_00001", &out, 1, &mut |_, _| false)
            .err()
            .unwrap();
        assert!(matches!(e, Error::Aborted));
        tif(&root.join("fly/F_00001/TOMO_00002.tif"), 4, 4, |_| 0);
        let e = import(&root, "fly/F_00001", &out, 1, &mut |_, _| true)
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .ends_with("4x4, not 5x4 as the first projection"),
            "{e}"
        );
    }

    #[test]
    fn import_refuses_an_output_under_the_raw_root_and_bin_zero() {
        let tmp = tempdir();
        let root = tmp.join("raw");
        fly(&root);
        let under = root.join("fly/x.h5");
        for (out, bin) in [(&under, 2), (&tmp.join("x.h5"), 0), (&tmp.join("x.h5"), 6)] {
            let e = import(&root, "fly/F_00001", out, bin, &mut |_, _| true)
                .err()
                .unwrap();
            assert!(matches!(e, Error::Refused(_)), "{e}");
        }
        assert!(!under.exists());
    }
}
