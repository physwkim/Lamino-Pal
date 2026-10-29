//! Detector binning.

/// `b`×`b` block mean of a row-major `h`×`w` frame, rounded half to even and
/// cropped to whole blocks: `(h/b)`×`(w/b)`.
///
/// For b = 2 this is build_step_bin2.py's f32 `0.5·(0.5·(a+b) + 0.5·(c+d))`
/// then `np.rint`: every one of those intermediates is exact in f32, so both
/// are the sum over 4 rounded half to even.
pub(crate) fn bin_mean(src: &[u16], h: usize, w: usize, b: usize) -> Vec<u16> {
    assert_eq!(src.len(), h * w);
    if b == 1 {
        return src.to_vec();
    }
    let (bh, bw) = (h / b, w / b);
    let d = (b * b) as u64;
    let mut out = Vec::with_capacity(bh * bw);
    let mut acc = vec![0u64; bw];
    for oy in 0..bh {
        acc.fill(0);
        for row in src[oy * b * w..].chunks(w).take(b) {
            for (a, block) in acc.iter_mut().zip(row.chunks_exact(b)) {
                *a += block.iter().map(|&v| u64::from(v)).sum::<u64>();
            }
        }
        out.extend(acc.iter().map(|&s| {
            let (q, r) = (s / d, s % d);
            (q + u64::from(2 * r > d || (2 * r == d && q % 2 == 1))) as u16
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// build_step_bin2.py's `bin2_mean`, op for op in f32.
    fn bin2_f32(src: &[u16], h: usize, w: usize) -> Vec<u16> {
        let a = |y: usize, x: usize| f32::from(src[y * w + x]);
        let mut out = Vec::new();
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                let top = 0.5 * (a(y, x) + a(y, x + 1));
                let bot = 0.5 * (a(y + 1, x) + a(y + 1, x + 1));
                out.push((0.5 * (top + bot)).round_ties_even() as u16);
            }
        }
        out
    }

    #[test]
    fn bin2_equals_the_f32_recipe() {
        // 2x2 blocks with every sum mod 4 and the extremes, side by side in a
        // 2-row frame, then a pseudo-random 64x64 field.
        let blocks: [[u16; 4]; 8] = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 3],
            [65535, 65535, 65535, 65535],
            [65535, 65535, 65535, 65534],
            [65535, 0, 65535, 0],
        ];
        let mut src = vec![0u16; 2 * 16];
        for (i, [a, b, c, d]) in blocks.into_iter().enumerate() {
            src[2 * i..2 * i + 2].copy_from_slice(&[a, b]);
            src[16 + 2 * i..16 + 2 * i + 2].copy_from_slice(&[c, d]);
        }
        // Sums 1, 2, 3, 4, 9, 4·65535, 4·65535−1, 2·65535 over 4.
        let want = [0, 0, 1, 1, 2, 65535, 65535, 32768];
        assert_eq!(bin_mean(&src, 2, 16, 2), want);
        assert_eq!(bin2_f32(&src, 2, 16), want);

        let mut x = 0x2545_f491_u32;
        let field: Vec<u16> = (0..64 * 64)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u16
            })
            .collect();
        assert_eq!(bin_mean(&field, 64, 64, 2), bin2_f32(&field, 64, 64));
    }

    #[test]
    fn odd_sizes_crop_and_ties_go_to_even() {
        // 3x3 blocks of a 7x4 frame: one block column, two block rows.
        let mut src = vec![0u16; 7 * 4];
        src[0] = 4;
        src[1] = 5;
        src[3 * 4] = 13;
        src[3] = 9000; // column 3 and row 6 are cropped
        src[6 * 4] = 9000;
        assert_eq!(bin_mean(&src, 7, 4, 3), [1, 1]);
        // 4x4: sums 8, 24, 40 are means 0.5, 1.5, 2.5 → 0, 2, 2.
        let tie = |s: u16| {
            let mut f = vec![0u16; 16];
            f[5] = s;
            bin_mean(&f, 4, 4, 4)[0]
        };
        assert_eq!((tie(8), tie(24), tie(40), tie(41)), (0, 2, 2, 3));
        assert_eq!(bin_mean(&[3, 4, 5], 1, 3, 1), [3, 4, 5]);
    }
}
