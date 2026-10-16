use nalgebra::{DMatrix, DVector};
/// Damped BFGS with an existing library's Cholesky check. Returns false if a
/// reset is required, true for an update or a negligible displacement.
pub(crate) fn update(b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
    if s.iter().fold(0.0_f64, |v, s| v.max(s.abs())) < 1e-14 {
        return true;
    }
    let s = DVector::from_column_slice(s);
    let y = DVector::from_column_slice(y);
    let bs = &*b * &s;
    let t = s.dot(&bs);
    let sy = s.dot(&y);
    if !t.is_finite() || !sy.is_finite() || t <= 0.0 {
        return false;
    }
    let theta = if sy >= 0.2 * t {
        1.0
    } else {
        0.8 * t / (t - sy)
    };
    let r = theta * y + (1.0 - theta) * &bs;
    let sr = s.dot(&r);
    if !sr.is_finite() || sr <= 0.0 {
        return false;
    }
    // B − (Bs)(Bs)ᵀ/t + rrᵀ/sr, entry by entry, without the five n×n
    // temporaries of the matrix expression. B is exactly symmetric (it starts
    // as a multiple of I, and every term below is symmetric in i, j because
    // floating-point products commute), so no symmetrization is needed and
    // the result is bit-identical to `0.5·(N + Nᵀ)` of the matrix expression.
    let n = b.nrows();
    let mut next = DMatrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            next[(i, j)] = (b[(i, j)] - (bs[i] * bs[j]) / t) + (r[i] * r[j]) / sr;
        }
    }
    if next.iter().any(|v| !v.is_finite()) || next.clone().cholesky().is_none() {
        return false;
    }
    *b = next;
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn damping_preserves_positive_definiteness() {
        let mut b = DMatrix::identity(2, 2);
        assert!(update(&mut b, &[1.0, 2.0], &[-4.0, -3.0]));
        assert!(b.clone().cholesky().is_some());
        assert!(update(&mut b, &[0.0, 0.0], &[1.0, 2.0]));
        assert!(!update(&mut b, &[1e308, 1e308], &[1.0, 1.0]));
    }

    #[test]
    fn entrywise_update_is_bit_identical_to_the_matrix_expression() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for n in [1, 2, 5, 17, 64] {
            let mut b = DMatrix::identity(n, n) * 1.7;
            for step in 0..40 {
                let s: Vec<f64> = (0..n).map(|_| uniform()).collect();
                // Every fourth step has negative curvature, so damping applies.
                let sign = if step % 4 == 3 { -1.0 } else { 1.0 };
                let y: Vec<f64> = s.iter().map(|v| sign * v * 2.0 + 0.1 * uniform()).collect();
                let (sv, yv) = (
                    DVector::from_column_slice(&s),
                    DVector::from_column_slice(&y),
                );
                let bs = &b * &sv;
                let t = sv.dot(&bs);
                let sy = sv.dot(&yv);
                let theta = if sy >= 0.2 * t {
                    1.0
                } else {
                    0.8 * t / (t - sy)
                };
                let r = theta * yv + (1.0 - theta) * &bs;
                let sr = sv.dot(&r);
                let reference = &b - (&bs * bs.transpose()) / t + (&r * r.transpose()) / sr;
                let reference = 0.5 * (&reference + reference.transpose());
                assert!(update(&mut b, &s, &y), "n {n}, step {step}");
                assert!(
                    b.iter()
                        .zip(reference.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "n {n}, step {step}"
                );
            }
        }
    }
}
