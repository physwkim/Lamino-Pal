//! The step scan `.dat`: `#` header lines of tab-separated `key : value`
//! fields, then numeric rows whose first column is the measured angle.

pub(crate) struct Dat {
    fields: Vec<(String, String)>,
    /// First column of every data row, in file order.
    pub angles: Vec<f64>,
}

impl Dat {
    pub fn parse(text: &str) -> Result<Dat, String> {
        let mut fields = Vec::new();
        let mut angles = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if let Some(head) = line.strip_prefix('#') {
                for f in head.split('\t') {
                    if let Some((k, v)) = f.split_once(" : ") {
                        fields.push((k.trim().to_owned(), v.trim().to_owned()));
                    }
                }
                continue;
            }
            let Some(first) = line.split_whitespace().next() else {
                continue;
            };
            let a = first
                .parse()
                .map_err(|_| format!("line {}: {first:?} is not a number", i + 1))?;
            angles.push(a);
        }
        Ok(Dat { fields, angles })
    }

    /// The number a header value starts with: `8333.0(eV)` reads 8333.0.
    pub fn number(&self, key: &str) -> Option<f64> {
        let v = &self.fields.iter().find(|(k, _)| k == key)?.1;
        let end = v
            .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
            .unwrap_or(v.len());
        v[..end].parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Head and tail of `360도 step/Step_N1000_0p18s.dat`.
    const STEP: &str = "# Data were collected from BL7C in Pohang Light Source(PLS-II) by : bl7c\tNumber of Points : 1011\tScanning Mode : 3D CT\tEnergy Origin(E0) : 8333.0(eV)\tScanning Time : 46.9963 (min)\n\
# SR E-beam energy : 3.0 (GeV)\tSR current (begin) : 301.507 (mA)\n\
# Description : \n\
# StartAngle : 0.0\tStopAngle : 360.0\tStepAngle : 0.36\n\
# Angle(deg.)\tROI1\n\
-5.000e-04\t1.454e+11\n\
3.605e-01\t1.454e+11\n\
3.600e+02\t1.454e+11\n";

    #[test]
    fn header_numbers_and_angle_column() {
        let d = Dat::parse(STEP).unwrap();
        assert_eq!(d.number("Energy Origin(E0)"), Some(8333.0));
        assert_eq!(d.number("Scanning Time"), Some(46.9963));
        assert_eq!(d.number("StepAngle"), Some(0.36));
        assert_eq!(d.number("Description"), None);
        assert_eq!(d.number("No such key"), None);
        assert_eq!(d.angles, [-5.0e-4, 0.3605, 360.0]);
    }

    #[test]
    fn a_non_numeric_row_is_an_error() {
        let e = Dat::parse("# a : 1\n0.5\t1\nnan?\t2\n").err().unwrap();
        assert_eq!(e, "line 3: \"nan?\" is not a number");
    }
}
