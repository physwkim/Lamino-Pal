//! Fly scan angles. The frames carry no angle: `metadata.txt` gives the
//! speed and exposure, and `n·speed·exptime` is the whole rotation.

use serde::Deserialize;

/// The `metadata.txt` fields the import reads.
#[derive(Deserialize)]
pub(crate) struct Meta {
    pub exptime: f64,
    pub numprj: usize,
    pub scan_speed: f64,
    pub current_energy: Option<f64>,
}

/// `round(n·speed·exptime)` as build_fly_h5.py takes it (Python rounds half
/// to even), refused unless it is a 180° or 360° scan.
pub(crate) fn range_deg(n: usize, speed: f64, exptime: f64) -> Result<u32, String> {
    let r = (n as f64 * speed * exptime).round_ties_even();
    match r {
        180.0 | 360.0 => Ok(r as u32),
        _ => Err(format!(
            "{n} frames x {speed} deg/s x {exptime} s = {} deg, not a 180 or 360 deg scan",
            n as f64 * speed * exptime
        )),
    }
}

/// numpy `linspace(0, r, n, endpoint=False)` cast to f32: `i·(r/n)` in f64.
pub(crate) fn theta(n: usize, r: u32) -> Vec<f32> {
    let step = f64::from(r) / n as f64;
    (0..n).map(|i| (i as f64 * step) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_261003_scans_are_whole_rotations() {
        // (numprj, scan_speed) of every 261003 fly scan; exptime 0.18 s.
        for (n, v, r) in [
            (400, 5.0, 360),
            (444, 4.504504504504505, 360),
            (500, 4.0, 360),
            (571, 3.502626970227671, 360),
            (667, 2.998500749625187, 360),
            (250, 4.0, 180),
            (286, 3.4965034965034967, 180),
            (333, 3.0030030030030033, 180),
            (400, 2.5, 180),
            (500, 2.0, 180),
        ] {
            assert_eq!(range_deg(n, v, 0.18), Ok(r), "N{n} {v} deg/s");
        }
    }

    #[test]
    fn range_rounds_half_to_even_then_refuses_partial_turns() {
        // 0.5 steps either side of 180: half to even keeps 180 for 180.5 and
        // takes 182 for 181.5.
        assert_eq!(range_deg(361, 1.0, 0.5), Ok(180));
        assert!(range_deg(363, 1.0, 0.5).is_err());
        assert!(range_deg(100, 1.0, 1.0).is_err());
    }

    #[test]
    fn theta_matches_numpy_linspace() {
        // np.linspace(0, r, n, endpoint=False).astype("float32")[[1, n//3, n-1]]
        for (n, r, want) in [
            (444, 360, [0.8108108, 120.0, 359.18918]),
            (286, 180, [0.6293706, 59.79021, 179.37064]),
            (667, 360, [0.53973013, 119.82009, 359.46027]),
        ] {
            let t = theta(n, r);
            assert_eq!(t.len(), n);
            assert_eq!(t[0], 0.0);
            assert_eq!([t[1], t[n / 3], t[n - 1]], want, "N{n} R{r}");
        }
    }
}
