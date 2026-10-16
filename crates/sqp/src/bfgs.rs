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
    let next = &*b - (&bs * bs.transpose()) / t + (&r * r.transpose()) / sr;
    let next = 0.5 * (&next + next.transpose());
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
}
