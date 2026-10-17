use nalgebra::{DMatrix, DVector, DVectorView, Dyn, linalg::Cholesky};
/// Per-solve scratch; callbacks and unrelated solves never share it.
pub(crate) struct Workspace {
    next: DMatrix<f64>,
    check: DMatrix<f64>,
    bs: DVector<f64>,
    r: DVector<f64>,
    factor: Option<Cholesky<f64, Dyn>>,
    updates: usize,
    fresh_each_update: bool,
}
impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            next: DMatrix::zeros(n, n),
            check: DMatrix::zeros(n, n),
            bs: DVector::zeros(n),
            r: DVector::zeros(n),
            factor: None,
            updates: 0,
            fresh_each_update: n <= 16,
        }
    }
    pub fn reset(&mut self) {
        self.factor = None;
        self.updates = 0;
    }
    fn rebuild(&mut self, b: &DMatrix<f64>, candidate: bool) -> bool {
        let matrix = if candidate { &self.next } else { b };
        let mut work = self
            .factor
            .take()
            .map(|f| f.unpack_dirty())
            .unwrap_or_else(|| std::mem::take(&mut self.check));
        work.resize_mut(matrix.nrows(), matrix.ncols(), 0.);
        work.copy_from(matrix);
        self.factor = work.cholesky();
        self.updates = 0;
        self.factor.is_some()
    }
    pub fn update(&mut self, b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
        if s.iter().fold(0.0_f64, |v, s| v.max(s.abs())) < 1e-14 {
            return true;
        }
        if self.factor.is_none() && !self.rebuild(b, false) {
            return false;
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
        let rebuild = if self.fresh_each_update || self.updates >= 31 {
            true
        } else {
            let factor = self.factor.as_mut().expect("factor initialized");
            // Positive update first: B - uu' alone is singular in exact arithmetic.
            factor.rank_one_update(&self.r, 1. / sr);
            factor.rank_one_update(&self.bs, -1. / t);
            self.updates += 1;
            let l = factor.l_dirty();
            (0..n).any(|i| {
                !l[(i, i)].is_finite() || l[(i, i)] <= 0. || (0..=i).any(|j| !l[(i, j)].is_finite())
            })
        };
        if rebuild && !self.rebuild(b, true) {
            return false;
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
                full: Some(Workspace::new(n)),
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
                    work: {
                        let mut work = Workspace::new(k);
                        work.fresh_each_update = true;
                        work
                    },
                }
            })
            .collect();
        Self { full: None, blocks }
    }
    pub fn reset(&mut self) {
        if let Some(w) = self.full.as_mut() {
            w.reset();
        }
        for block in &mut self.blocks {
            block.work.reset();
        }
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
    Workspace::new(s.len()).update(b, s, y)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_updates_track_the_independent_matrix_and_reset() {
        let n = 25;
        let mut b = DMatrix::identity(n, n);
        let mut w = Workspace::new(n);
        for k in 0..120 {
            let s: Vec<_> = (0..n)
                .map(|j| ((j * 17 + k * 7) % 31) as f64 / 31. - 0.4)
                .collect();
            let y: Vec<_> = s
                .iter()
                .enumerate()
                .map(|(j, &v)| v * (1. + (j % 4) as f64))
                .collect();
            assert!(w.update(&mut b, &s, &y));
            let l = w.factor.as_ref().unwrap().l();
            let actual = &l * l.transpose();
            let scale = 1. + b.amax();
            assert!((&actual - &b).amax() < 1e-8 * scale);
            if k == 60 {
                b = DMatrix::identity(n, n) * 2.;
                w.reset();
            }
        }
    }
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
