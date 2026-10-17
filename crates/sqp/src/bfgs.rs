use nalgebra::{DMatrix, DVector, DVectorView};

/// Per-solve scratch for damped BFGS; callbacks and unrelated solves never
/// share it.
///
/// Full mode does not factorize B. Damping keeps every update positive
/// definite in exact arithmetic, and the QP backend checks the B it receives
/// exactly: the Clarabel adapter runs a Cholesky gate on every model and
/// reports [`sqp_craft::qp::QpOutcome::NotPositiveDefinite`], on which the
/// driver resets B and re-solves, as it does after a failed update. A
/// factorization here would repeat that O(n³) check once per iteration.
pub(crate) struct Workspace {
    next: DMatrix<f64>,
    bs: DVector<f64>,
    r: DVector<f64>,
    /// Explicit blocks check their own update, so a failure can switch the
    /// solve to full BFGS at the update, before any QP sees the block.
    check: Option<DMatrix<f64>>,
}

impl Workspace {
    pub fn new(n: usize, check: bool) -> Self {
        Self {
            next: DMatrix::zeros(n, n),
            bs: DVector::zeros(n),
            r: DVector::zeros(n),
            check: check.then(|| DMatrix::zeros(n, n)),
        }
    }

    /// Damped BFGS update of `b`. Returns false if a reset is required, true
    /// for an update or a negligible displacement.
    pub fn update(&mut self, b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
        if s.iter().fold(0.0_f64, |v, s| v.max(s.abs())) < 1e-14 {
            return true;
        }
        let sv = DVectorView::from_slice(s, s.len());
        let yv = DVectorView::from_slice(y, y.len());
        b.mul_to(&sv, &mut self.bs);
        let t = sv.dot(&self.bs);
        let sy = sv.dot(&yv);
        if !t.is_finite() || !sy.is_finite() || t <= 0. {
            return false;
        }
        let theta = if sy >= 0.2 * t {
            1.
        } else {
            0.8 * t / (t - sy)
        };
        for (i, &yi) in y.iter().enumerate() {
            self.r[i] = theta * yi + (1. - theta) * self.bs[i];
        }
        let sr = sv.dot(&self.r);
        if !sr.is_finite() || sr <= 0. {
            return false;
        }
        // B − (Bs)(Bs)ᵀ/t + rrᵀ/sr. B is exactly symmetric and every term is
        // symmetric in i, j (floating-point products commute), so one
        // triangle is computed and mirrored; the result is bit-identical to
        // `0.5·(N + Nᵀ)` of the matrix expression.
        let n = b.nrows();
        for j in 0..n {
            for i in j..n {
                let v = (b[(i, j)] - (self.bs[i] * self.bs[j]) / t) + (self.r[i] * self.r[j]) / sr;
                self.next[(i, j)] = v;
                self.next[(j, i)] = v;
            }
        }
        if self.next.iter().any(|v| !v.is_finite()) {
            return false;
        }
        if let Some(buffer) = self.check.as_mut() {
            let mut work = std::mem::take(buffer);
            work.copy_from(&self.next);
            match work.cholesky() {
                Some(factor) => *buffer = factor.unpack_dirty(),
                None => {
                    *buffer = DMatrix::zeros(n, n);
                    return false;
                }
            }
        }
        std::mem::swap(b, &mut self.next);
        true
    }
}

struct Block {
    start: usize,
    matrix: DMatrix<f64>,
    work: Workspace,
}

/// Full-mode behavior is preserved; blocks are an explicit approximation.
pub(crate) struct Curvature {
    full: Option<Workspace>,
    blocks: Vec<Block>,
}

impl Curvature {
    pub fn new(n: usize, size: usize) -> Self {
        if size == 0 || size >= n {
            return Self {
                full: Some(Workspace::new(n, false)),
                blocks: vec![],
            };
        }
        let blocks = (0..n)
            .step_by(size)
            .map(|start| {
                let k = size.min(n - start);
                Block {
                    start,
                    matrix: DMatrix::zeros(k, k),
                    work: Workspace::new(k, true),
                }
            })
            .collect();
        Self { full: None, blocks }
    }

    pub fn update(&mut self, b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
        if let Some(w) = self.full.as_mut() {
            return w.update(b, s, y);
        }
        for block in &mut self.blocks {
            let (start, k) = (block.start, block.matrix.nrows());
            block.matrix.copy_from(&b.view((start, start), (k, k)));
            if !block.work.update(
                &mut block.matrix,
                &s[start..start + k],
                &y[start..start + k],
            ) {
                return false;
            }
            b.view_mut((start, start), (k, k)).copy_from(&block.matrix);
        }
        true
    }
}

#[cfg(test)]
fn update(b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
    Workspace::new(s.len(), false).update(b, s, y)
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
    fn only_checked_blocks_factorize_and_reject_an_indefinite_result() {
        // diag(1, -5) updated along e0 gives diag(2, -5): an update can only
        // be indefinite from rounding or an indefinite start; this exercises
        // the check itself deterministically.
        let start = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -5.0]);
        let (s, y) = ([1.0, 0.0], [2.0, 0.0]);
        let mut unchecked = start.clone();
        assert!(Workspace::new(2, false).update(&mut unchecked, &s, &y));
        assert_eq!(unchecked.as_slice(), &[2.0, 0.0, 0.0, -5.0]);
        let mut checked = start.clone();
        let mut work = Workspace::new(2, true);
        assert!(!work.update(&mut checked, &s, &y));
        assert_eq!(checked, start, "a rejected update leaves B unchanged");
        // The buffer is restored after a rejection; later updates still work.
        let mut fine = DMatrix::identity(2, 2);
        assert!(work.update(&mut fine, &[1.0, 2.0], &[2.0, 3.0]));
        assert!(work.update(&mut fine, &[0.5, -1.0], &[1.0, -1.5]));
        assert!(fine.clone().cholesky().is_some());
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
            let mut work = Workspace::new(n, false);
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
                assert!(work.update(&mut b, &s, &y), "n {n}, step {step}");
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
