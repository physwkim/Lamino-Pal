//! BL7C scan folders: discovery and import to DXchange h5.
//!
//! A step scan is `<scan>_proj/` (`<scan>_NNNN.tif` projections, `*_ff_*`
//! flats) beside `<scan>.dat`, whose first rows are the projection angles. A
//! fly scan is a folder with `metadata.txt`, `TOMO_*.tif` projections and
//! `FF_*.tif` flats. Folder names are kept as on disk (NFD Korean).

mod bin;
mod dat;
mod fly;
mod import;

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

pub use import::{import, Imported};

#[derive(Debug)]
pub enum Error {
    /// A guard declined the request; the text says why.
    Refused(String),
    /// The folder is not the layout described above, or its files disagree.
    Format(PathBuf, String),
    Io(PathBuf, std::io::Error),
    Tiff(PathBuf, tiff::TiffError),
    H5(PathBuf, rust_hdf5::Hdf5Error),
    /// The progress callback asked to stop.
    Aborted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Refused(why) => f.write_str(why),
            Error::Format(p, why) => write!(f, "{}: {why}", p.display()),
            Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Error::Tiff(p, e) => write!(f, "{}: {e}", p.display()),
            Error::H5(p, e) => write!(f, "{}: {e}", p.display()),
            Error::Aborted => f.write_str("aborted"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Step,
    Fly,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Step => "step",
            Kind::Fly => "fly",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Scan {
    pub kind: Kind,
    /// The scan folder under the raw root, `/`-separated, as on disk.
    pub raw: String,
    /// Proposed project tag: `step1000`, `f360_N444_4p5dps`.
    pub tag: String,
    /// Projections.
    pub n: usize,
    pub n_flat: usize,
    /// Rotation the projections span: the `.dat` Start→StopAngle, or a fly R.
    pub range_deg: f64,
    pub speed_dps: Option<f64>,
    pub exptime_s: Option<f64>,
    /// The `.dat` E0 of a step scan; the fly metadata `current_energy`.
    pub energy_ev: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Discovery {
    pub scans: Vec<Scan>,
    /// Folders laid out as a scan that cannot be imported: (raw, why).
    pub skipped: Vec<(String, String)>,
}

/// A scan and the files its import reads.
pub(crate) struct Layout {
    pub scan: Scan,
    pub proj: Vec<PathBuf>,
    pub flat: Vec<PathBuf>,
    /// Projection angles, degrees.
    pub theta: Vec<f32>,
}

/// Every scan folder under `root`, in name order. A scan folder is not
/// descended into; hidden folders are skipped.
pub fn discover(root: &Path) -> Result<Discovery, Error> {
    let mut d = Discovery {
        scans: Vec::new(),
        skipped: Vec::new(),
    };
    walk(root, root, &mut d)?;
    Ok(d)
}

fn walk(root: &Path, dir: &Path, d: &mut Discovery) -> Result<(), Error> {
    for name in names(dir)? {
        let path = dir.join(&name);
        if !path.is_dir() {
            continue;
        }
        if !is_scan_dir(&path) {
            walk(root, &path, d)?;
            continue;
        }
        let raw = rel(root, &path);
        match open(root, &raw) {
            Ok(l) => d.scans.push(l.scan),
            Err(e) => d.skipped.push((raw, e.to_string())),
        }
    }
    Ok(())
}

fn is_scan_dir(path: &Path) -> bool {
    path.join("metadata.txt").is_file() || step_dat(path).is_some_and(|p| p.is_file())
}

/// `<dir>/../<scan>.dat` for a `<scan>_proj` folder.
fn step_dat(dir: &Path) -> Option<PathBuf> {
    let stem = dir.file_name()?.to_str()?.strip_suffix("_proj")?;
    Some(dir.with_file_name(format!("{stem}.dat")))
}

fn rel(root: &Path, path: &Path) -> String {
    let r = path.strip_prefix(root).expect("walk stays under the root");
    let parts: Vec<_> = r
        .iter()
        .map(|c| c.to_str().expect("names() is UTF-8"))
        .collect();
    parts.join("/")
}

/// Sorted UTF-8 entry names of `dir`, without dot files: macOS leaves `._*`
/// and `.DS_Store` beside the frames.
fn names(dir: &Path) -> Result<Vec<String>, Error> {
    let io = |e| Error::Io(dir.to_owned(), e);
    let mut v = Vec::new();
    for e in fs::read_dir(dir).map_err(io)? {
        let name = e.map_err(io)?.file_name();
        let Some(name) = name.to_str() else {
            return Err(Error::Format(dir.join(&name), "name is not UTF-8".into()));
        };
        if !name.starts_with('.') {
            v.push(name.to_owned());
        }
    }
    // Byte order of UTF-8 is code-point order: Python's `sorted(os.listdir())`.
    v.sort();
    Ok(v)
}

/// The scan at `raw` under `root`, with its frame lists and angles.
pub(crate) fn open(root: &Path, raw: &str) -> Result<Layout, Error> {
    // Each name as its folder lists it, not as the file system finds it: a
    // Mac's and Windows's also find another normalization or case. "", "."
    // and ".." are never listed.
    let dir = raw.split('/').try_fold(root.to_path_buf(), |dir, name| {
        let mut es = fs::read_dir(&dir).ok()?;
        es.any(|e| e.is_ok_and(|e| e.file_name() == name))
            .then(|| dir.join(name))
    });
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return Err(Error::Refused(format!(
            "raw {raw:?} is no folder under the raw root; pass it as scan.discover \
             lists it: names match byte for byte, and a Mac writes Korean as NFD"
        )));
    };
    if dir.join("metadata.txt").is_file() {
        return open_fly(dir, raw);
    }
    match step_dat(&dir) {
        Some(dat) if dat.is_file() => open_step(dir, dat, raw),
        _ => Err(Error::Format(
            dir,
            "neither a fly scan (metadata.txt) nor a step scan (<scan>_proj beside <scan>.dat)"
                .into(),
        )),
    }
}

fn open_step(dir: PathBuf, dat_path: PathBuf, raw: &str) -> Result<Layout, Error> {
    let text = fs::read(&dat_path).map_err(|e| Error::Io(dat_path.clone(), e))?;
    let dat = dat::Dat::parse(&String::from_utf8_lossy(&text))
        .map_err(|why| Error::Format(dat_path.clone(), why))?;
    let names = names(&dir)?;
    let tif = |n: &&String| n.ends_with(".tif");
    let proj: Vec<_> = names
        .iter()
        .filter(tif)
        .filter(|n| !n.contains("_ff_"))
        .collect();
    let flat: Vec<_> = names
        .iter()
        .filter(tif)
        .filter(|n| n.contains("_ff_"))
        .collect();
    let n = proj.len();
    let format = |why: String| Error::Format(dat_path.clone(), why);
    if n == 0 {
        return Err(Error::Format(dir, "no projections".into()));
    }
    if dat.angles.len() < n {
        return Err(format(format!(
            "{} angle rows for {n} projections",
            dat.angles.len()
        )));
    }
    let header = |k: &str| {
        dat.number(k)
            .ok_or_else(|| format(format!("no {k} in the header")))
    };
    let (start, stop, step) = (
        header("StartAngle")?,
        header("StopAngle")?,
        header("StepAngle")?,
    );
    // The trailing rows are the flats' angles.
    let theta = dat.angles[..n].iter().map(|&a| a as f32).collect();
    let scan = Scan {
        kind: Kind::Step,
        raw: raw.to_owned(),
        tag: format!("step{}", ((stop - start) / step).round()),
        n,
        n_flat: flat.len(),
        range_deg: stop - start,
        speed_dps: None,
        exptime_s: None,
        energy_ev: dat.number("Energy Origin(E0)"),
    };
    Ok(Layout {
        scan,
        proj: proj.into_iter().map(|n| dir.join(n)).collect(),
        flat: flat.into_iter().map(|n| dir.join(n)).collect(),
        theta,
    })
}

fn open_fly(dir: PathBuf, raw: &str) -> Result<Layout, Error> {
    let mpath = dir.join("metadata.txt");
    let text = fs::read(&mpath).map_err(|e| Error::Io(mpath.clone(), e))?;
    let m: fly::Meta =
        serde_json::from_slice(&text).map_err(|e| Error::Format(mpath.clone(), e.to_string()))?;
    let names = names(&dir)?;
    let frames = |pre: &str| -> Vec<PathBuf> {
        names
            .iter()
            .filter(|n| n.starts_with(pre) && n.ends_with(".tif"))
            .map(|n| dir.join(n))
            .collect()
    };
    let (proj, flat) = (frames("TOMO_"), frames("FF_"));
    let n = proj.len();
    if n != m.numprj {
        return Err(Error::Format(
            dir,
            format!("{n} TOMO_*.tif, metadata numprj {}", m.numprj),
        ));
    }
    let r = fly::range_deg(n, m.scan_speed, m.exptime).map_err(Error::Refused)?;
    let scan = Scan {
        kind: Kind::Fly,
        raw: raw.to_owned(),
        // The metadata speed is the exact R/(n·exptime), 4.5045 for a 4.5 deg/s
        // scan; the tag takes the nominal one.
        tag: format!(
            "f{r}_N{n}_{}dps",
            ((m.scan_speed * 10.0).round() / 10.0)
                .to_string()
                .replace('.', "p")
        ),
        n,
        n_flat: flat.len(),
        range_deg: f64::from(r),
        speed_dps: Some(m.scan_speed),
        exptime_s: Some(m.exptime),
        energy_ev: m.current_energy,
    };
    Ok(Layout {
        scan,
        proj,
        flat,
        theta: fly::theta(n, r),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path, body: &str) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    const DAT: &str = "# Energy Origin(E0) : 8333.0(eV)\n\
# StartAngle : 0.0\tStopAngle : 360.0\tStepAngle : 120.0\n\
-0.001\t1\n120.0\t1\n240.0\t1\n360.0\t1\n360.0\t1\n";

    fn meta(numprj: usize, speed: f64) -> String {
        format!(
            r#"{{"exptime": 0.5, "numprj": {numprj}, "scan_speed": {speed}, "numff": 1, "current_energy": 8332.9}}"#
        )
    }

    /// A raw root with one step scan, two fly scans (one a partial turn),
    /// a mislabelled step folder and the dot files a Mac leaves.
    fn tree(root: &Path) {
        let step = root.join("360도 step");
        touch(&step.join("S_N3.dat"), DAT);
        for f in [
            "S_N3_0001.tif",
            "S_N3_0002.tif",
            "S_N3_0003.tif",
            "S_N3_0004.tif",
            "S_N3_ff_0001.tif",
            "settings.ini",
            "._S_N3_0001.tif",
        ] {
            touch(&step.join("S_N3_proj").join(f), "");
        }
        touch(&step.join("S_N3_position.dat"), "");
        touch(&step.join("Lone_proj/x.tif"), "");
        let fly = root.join("fly");
        // 4 frames x 180 deg/s x 0.5 s = 360.
        touch(&fly.join("F_00001/metadata.txt"), &meta(4, 180.0));
        for f in [
            "TOMO_00000.tif",
            "TOMO_00001.tif",
            "TOMO_00002.tif",
            "TOMO_00003.tif",
            "FF_00000.tif",
        ] {
            touch(&fly.join("F_00001").join(f), "");
        }
        // 3 frames at 240.24 deg/s: a 240.2 deg/s scan, 3 x 240.24 x 0.5 = 360.36.
        touch(&fly.join("H_00001/metadata.txt"), &meta(3, 240.24));
        for f in ["TOMO_00000.tif", "TOMO_00001.tif", "TOMO_00002.tif"] {
            touch(&fly.join("H_00001").join(f), "");
        }
        // 2 x 100 x 0.5 = 100: not a whole rotation.
        touch(&fly.join("G_00001/metadata.txt"), &meta(2, 100.0));
        touch(&fly.join("G_00001/TOMO_00000.tif"), "");
        touch(&fly.join("G_00001/TOMO_00001.tif"), "");
        touch(&root.join(".hidden/F_00001/metadata.txt"), &meta(4, 180.0));
        touch(&root.join("._x"), "");
    }

    #[test]
    fn discover_reads_both_kinds_and_skips_the_unimportable() {
        let tmp = tempdir();
        tree(&tmp);
        let d = discover(&tmp).unwrap();
        let tags: Vec<_> = d
            .scans
            .iter()
            .map(|s| (s.kind, s.raw.as_str(), s.tag.as_str(), s.n, s.n_flat))
            .collect();
        assert_eq!(
            tags,
            [
                (Kind::Step, "360도 step/S_N3_proj", "step3", 4, 1),
                (Kind::Fly, "fly/F_00001", "f360_N4_180dps", 4, 1),
                (Kind::Fly, "fly/H_00001", "f360_N3_240p2dps", 3, 0),
            ]
        );
        let s = &d.scans[0];
        assert_eq!(
            (s.range_deg, s.energy_ev, s.speed_dps),
            (360.0, Some(8333.0), None)
        );
        let f = &d.scans[1];
        assert_eq!(
            (f.range_deg, f.energy_ev, f.speed_dps, f.exptime_s),
            (360.0, Some(8332.9), Some(180.0), Some(0.5))
        );
        assert_eq!(d.skipped.len(), 1);
        assert_eq!(d.skipped[0].0, "fly/G_00001");
        assert!(
            d.skipped[0].1.contains("not a 180 or 360 deg scan"),
            "{}",
            d.skipped[0].1
        );
    }

    #[test]
    fn open_takes_the_first_n_dat_rows_and_refuses_paths_out_of_the_root() {
        let tmp = tempdir();
        tree(&tmp);
        let l = open(&tmp, "360도 step/S_N3_proj").unwrap();
        assert_eq!(l.theta, [-0.001, 120.0, 240.0, 360.0]);
        assert_eq!(l.proj.last().unwrap().file_name().unwrap(), "S_N3_0004.tif");
        assert_eq!(l.flat.len(), 1);
        let f = open(&tmp, "fly/F_00001").unwrap();
        assert_eq!(f.theta, [0.0, 90.0, 180.0, 270.0]);
        for raw in [
            "",
            "../x",
            "/etc",
            "fly/../fly/F_00001",
            "fly/F_00001/",
            "FLY/F_00001",
            "fly\\F_00001",
        ] {
            assert!(matches!(open(&tmp, raw), Err(Error::Refused(_))), "{raw}");
        }
        assert!(matches!(
            open(&tmp, "360도 step/Lone_proj"),
            Err(Error::Format(..))
        ));
        assert!(matches!(open(&tmp, "fly/G_00001"), Err(Error::Refused(_))));
        // The fixture's NFC "도" spelled NFD: another name on disk.
        let e = open(&tmp, "360\u{1103}\u{1169} step/S_N3_proj")
            .err()
            .unwrap();
        assert!(
            matches!(&e, Error::Refused(why) if why.contains("is no folder under the raw root")),
            "{e}"
        );
    }

    #[test]
    fn a_dat_shorter_than_the_projections_is_a_format_error() {
        let tmp = tempdir();
        touch(
            &tmp.join("A.dat"),
            "# StartAngle : 0\tStopAngle : 1\tStepAngle : 1\n0\t1\n",
        );
        touch(&tmp.join("A_proj/A_0001.tif"), "");
        touch(&tmp.join("A_proj/A_0002.tif"), "");
        let e = open(&tmp, "A_proj").err().unwrap();
        assert!(
            e.to_string().ends_with("1 angle rows for 2 projections"),
            "{e}"
        );
    }

    pub(super) fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "lamino-pal-bl7c-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
}
