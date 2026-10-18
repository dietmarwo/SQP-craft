use nalgebra::{DMatrix, DVector, DVectorView};
use sqp_craft::{CsrPattern, CurvatureUpdate};

/// The largest repair `Δy` that is applied, relative to `‖Bs‖₂`. SNOPT gives
/// up on the modification "if ω is very large" without giving a number
/// (Gill, Murray and Saunders 2005, §2.10). The natural size of `y` is that
/// of `Bs`; a repair far beyond it would add a huge rank-one term in a
/// direction unrelated to the step.
const MAX_REPAIR: f64 = 1e2;

/// Data for SNOPT's second modification of `y` (Gill, Murray and Saunders
/// 2005, §2.10, eqs. 2.7–2.8): `y` becomes the gradient difference of the
/// augmented modified Lagrangian, `y + (J(x⁺) − J(x))ᵀ Ω d_L`, with the
/// minimum-norm weights `Ω = diag(ω) ≥ 0` that raise `sᵀy` to the threshold.
pub(crate) struct Repair<'a> {
    /// Jacobian pattern.
    pub pattern: &'a CsrPattern,
    /// `J(x⁺) − J(x)` in pattern order.
    pub change: &'a [f64],
    /// Departure from linearity `d_L = c(x⁺) − c(x) − J(x)s`, one per row.
    pub departure: &'a [f64],
}

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
    /// Repair scratch: `Δy` (block length) and per-row weights.
    dy: DVector<f64>,
    rows: Vec<f64>,
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
            dy: DVector::zeros(n),
            rows: Vec::new(),
            check: check.then(|| DMatrix::zeros(n, n)),
        }
    }

    /// BFGS update of `b`. Below the curvature threshold `sᵀy < 0.2·sᵀBs`
    /// it damps `y` (Powell), or, with `repair`, applies SNOPT's
    /// modification and damps only if no repair exists. `offset` is the
    /// first column of this block in the Jacobian. Returns `None` if a reset
    /// is required.
    pub fn update(
        &mut self,
        b: &mut DMatrix<f64>,
        s: &[f64],
        y: &[f64],
        repair: Option<(&Repair<'_>, usize)>,
    ) -> Option<CurvatureUpdate> {
        if s.iter().all(|v| v.abs() < 1e-14) {
            return Some(CurvatureUpdate::Negligible);
        }
        let sv = DVectorView::from_slice(s, s.len());
        let yv = DVectorView::from_slice(y, y.len());
        b.mul_to(&sv, &mut self.bs);
        let t = sv.dot(&self.bs);
        let sy = sv.dot(&yv);
        if !t.is_finite() || !sy.is_finite() || t <= 0. {
            return None;
        }
        let mut kind = if sy >= 0.2 * t {
            CurvatureUpdate::Exact
        } else {
            CurvatureUpdate::Damped
        };
        let repaired = match repair {
            Some((repair, offset)) if kind == CurvatureUpdate::Damped => {
                self.repaired(repair, offset, s, y, 0.2 * t - sy)
            }
            _ => false,
        };
        if repaired {
            kind = CurvatureUpdate::Modified;
        } else {
            // Powell's damping, also when no repair exists: skipping the
            // update instead keeps a B that overestimates the curvature
            // (see `HessianUpdate::Augmented`).
            let theta = if sy >= 0.2 * t {
                1.
            } else {
                0.8 * t / (t - sy)
            };
            if theta == 1.0 {
                self.r.as_mut_slice().copy_from_slice(y);
            } else {
                for (i, &yi) in y.iter().enumerate() {
                    self.r[i] = theta * yi + (1. - theta) * self.bs[i];
                }
            }
        }
        let sr = sv.dot(&self.r);
        if !sr.is_finite() || sr <= 0. {
            return None;
        }
        // B − (Bs)(Bs)ᵀ/t + rrᵀ/sr. B is exactly symmetric and every term is
        // symmetric in i, j (floating-point products commute), so one
        // triangle is computed and mirrored; the result is bit-identical to
        // `0.5·(N + Nᵀ)` of the matrix expression.
        let n = b.nrows();
        let mut finite = true;
        for j in 0..n {
            let source = &b.as_slice()[j * n..(j + 1) * n];
            let target = &mut self.next.as_mut_slice()[j * n..(j + 1) * n];
            let (bsj, rj) = (self.bs[j], self.r[j]);
            for (((out, &old), &bs), &r) in target[j..]
                .iter_mut()
                .zip(&source[j..])
                .zip(&self.bs.as_slice()[j..])
                .zip(&self.r.as_slice()[j..])
            {
                let v = (old - (bs * bsj) / t) + (r * rj) / sr;
                finite &= v.is_finite();
                *out = v;
            }
        }
        if !finite {
            return None;
        }
        let next = self.next.as_mut_slice();
        for j in 0..n {
            for i in j + 1..n {
                next[i * n + j] = next[j * n + i];
            }
        }
        if let Some(buffer) = self.check.as_mut() {
            let mut work = std::mem::take(buffer);
            work.copy_from(&self.next);
            match work.cholesky() {
                Some(factor) => *buffer = factor.unpack_dirty(),
                None => {
                    *buffer = DMatrix::zeros(n, n);
                    return None;
                }
            }
        }
        std::mem::swap(b, &mut self.next);
        Some(kind)
    }

    /// Writes `y + Δy` to `r`, with `Δy = (J⁺ − J)ᵀ Ω d_L` restricted to the
    /// columns `offset..offset + s.len()` and `Ω` the minimum-norm solution
    /// of `aᵀω = deficit, ω ≥ 0`, `aᵢ = ((J⁺ − J)s)ᵢ · d_L,ᵢ`. Then
    /// `sᵀ(y + Δy) = sᵀy + deficit`. Returns false when no such `ω` exists
    /// (no row is curved along `s` in the right sense, for example only
    /// linear rows) or the repair is too large or not finite; `r` is then
    /// overwritten by the caller.
    fn repaired(
        &mut self,
        repair: &Repair<'_>,
        offset: usize,
        s: &[f64],
        y: &[f64],
        deficit: f64,
    ) -> bool {
        let (pattern, k) = (repair.pattern, s.len());
        let in_block = |j: usize| j >= offset && j < offset + k;
        self.rows.clear();
        let mut norm2 = 0.0_f64;
        for (i, &w) in repair.departure.iter().enumerate() {
            let mut v = 0.0;
            for a in pattern.indptr[i]..pattern.indptr[i + 1] {
                let j = pattern.indices[a];
                if in_block(j) {
                    v += repair.change[a] * s[j - offset];
                }
            }
            let plus = (v * w).max(0.0);
            norm2 += plus * plus;
            self.rows.push(plus);
        }
        if !(norm2.is_finite() && norm2 > 0.0 && deficit.is_finite()) {
            return false;
        }
        // Row i contributes ωᵢ·d_L,ᵢ·(J⁺ − J)ᵢ with ωᵢ = deficit·aᵢ⁺/‖a⁺‖².
        for (i, factor) in self.rows.iter_mut().enumerate() {
            *factor = deficit * *factor / norm2 * repair.departure[i];
        }
        self.dy.fill(0.0);
        for (i, &factor) in self.rows.iter().enumerate() {
            if factor == 0.0 {
                continue;
            }
            for a in pattern.indptr[i]..pattern.indptr[i + 1] {
                let j = pattern.indices[a];
                if in_block(j) {
                    self.dy[j - offset] += repair.change[a] * factor;
                }
            }
        }
        let size = self.dy.norm();
        if !(size.is_finite() && size <= MAX_REPAIR * self.bs.norm()) {
            return false;
        }
        let mut sr = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            self.r[i] = yi + self.dy[i];
            sr += s[i] * self.r[i];
        }
        // sᵀ(y + Δy) = 0.2·sᵀBs > 0 up to rounding.
        sr.is_finite() && sr > 0.0
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

    /// Updates `b` (per block in block mode). Returns `None` if a reset is
    /// required, otherwise the most significant kind of update applied to
    /// any block.
    pub fn update(
        &mut self,
        b: &mut DMatrix<f64>,
        s: &[f64],
        y: &[f64],
        repair: Option<&Repair<'_>>,
    ) -> Option<CurvatureUpdate> {
        if let Some(w) = self.full.as_mut() {
            return w.update(b, s, y, repair.map(|r| (r, 0)));
        }
        let rank = |k: CurvatureUpdate| match k {
            CurvatureUpdate::Negligible => 0,
            CurvatureUpdate::Exact => 1,
            CurvatureUpdate::Damped => 2,
            _ => 3,
        };
        let mut kind = CurvatureUpdate::Negligible;
        for block in &mut self.blocks {
            let (start, k) = (block.start, block.matrix.nrows());
            block.matrix.copy_from(&b.view((start, start), (k, k)));
            let done = block.work.update(
                &mut block.matrix,
                &s[start..start + k],
                &y[start..start + k],
                repair.map(|r| (r, start)),
            )?;
            if rank(done) > rank(kind) {
                kind = done;
            }
            b.view_mut((start, start), (k, k)).copy_from(&block.matrix);
        }
        Some(kind)
    }
}

#[cfg(test)]
fn update(b: &mut DMatrix<f64>, s: &[f64], y: &[f64]) -> bool {
    Workspace::new(s.len(), false)
        .update(b, s, y, None)
        .is_some()
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
        assert!(
            Workspace::new(2, false)
                .update(&mut unchecked, &s, &y, None)
                .is_some()
        );
        assert_eq!(unchecked.as_slice(), &[2.0, 0.0, 0.0, -5.0]);
        let mut checked = start.clone();
        let mut work = Workspace::new(2, true);
        assert!(work.update(&mut checked, &s, &y, None).is_none());
        assert_eq!(checked, start, "a rejected update leaves B unchanged");
        // The buffer is restored after a rejection; later updates still work.
        let mut fine = DMatrix::identity(2, 2);
        assert!(
            work.update(&mut fine, &[1.0, 2.0], &[2.0, 3.0], None)
                .is_some()
        );
        assert!(
            work.update(&mut fine, &[0.5, -1.0], &[1.0, -1.5], None)
                .is_some()
        );
        assert!(fine.clone().cholesky().is_some());
    }

    fn repair_of<'a>(
        pattern: &'a CsrPattern,
        change: &'a [f64],
        departure: &'a [f64],
    ) -> Repair<'a> {
        Repair {
            pattern,
            change,
            departure,
        }
    }

    #[test]
    fn the_repair_raises_the_curvature_to_the_threshold_in_closed_form() {
        // One row with J⁺ − J = (1, 0) and d_L = 0.5; s = e₀, y = −e₀, B = I:
        // deficit = 0.2 + 1, a = 1·0.5, ω = 1.2·0.5/0.25 = 2.4 and
        // Δy = 2.4·0.5·(1, 0), so y⁺ = (0.2, 0) and sᵀy⁺ = 0.2·sᵀBs.
        let pattern = CsrPattern::dense(1, 2);
        let repair = repair_of(&pattern, &[1.0, 0.0], &[0.5]);
        let mut b = DMatrix::identity(2, 2);
        let mut work = Workspace::new(2, false);
        let kind = work.update(&mut b, &[1.0, 0.0], &[-1.0, 0.0], Some((&repair, 0)));
        assert_eq!(kind, Some(CurvatureUpdate::Modified));
        let expected = DMatrix::from_row_slice(2, 2, &[0.2, 0.0, 0.0, 1.0]);
        assert!((&b - expected).norm() < 1e-15, "{b}");
        // Damping would have blended y with Bs instead.
        let mut damped = DMatrix::identity(2, 2);
        let kind = work.update(&mut damped, &[1.0, 0.0], &[-1.0, 0.0], None);
        assert_eq!(kind, Some(CurvatureUpdate::Damped));
    }

    #[test]
    fn without_a_usable_repair_the_update_is_damped() {
        let pattern = CsrPattern::dense(1, 2);
        let start = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let (s, y) = ([1.0, 0.0], [-1.0, 0.3]);
        let mut damped = start.clone();
        Workspace::new(2, false).update(&mut damped, &s, &y, None);
        for (change, departure, why) in [
            ([0.0, 0.0], 0.0, "a linear row: J⁺ = J and d_L = 0"),
            ([1.0, 0.0], -0.5, "curvature of the wrong sign: a < 0"),
            ([1e-6, 1.0], 1.0, "a repair far beyond ‖Bs‖"),
        ] {
            let d = [departure];
            let repair = repair_of(&pattern, &change, &d);
            let mut b = start.clone();
            let kind = Workspace::new(2, false).update(&mut b, &s, &y, Some((&repair, 0)));
            assert_eq!(kind, Some(CurvatureUpdate::Damped), "{why}");
            assert_eq!(b, damped, "{why}");
        }
        // Enough curvature: the repair is not consulted.
        let repair = repair_of(&pattern, &[0.0, 0.0], &[0.0]);
        let mut b = start.clone();
        let kind = Workspace::new(2, false).update(&mut b, &s, &[2.0, 0.5], Some((&repair, 0)));
        assert_eq!(kind, Some(CurvatureUpdate::Exact));
    }

    #[test]
    fn block_repairs_use_only_their_own_columns() {
        // Row 0 is curved in x₁ only and row 1 is linear; blocks {0, 1}, {2, 3}.
        let pattern = CsrPattern::dense(2, 4);
        let change = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let repair = repair_of(&pattern, &change, &[1.0, 1.0]);
        let mut b = DMatrix::identity(4, 4);
        let mut curvature = Curvature::new(4, 2);
        let s = [0.0, 1.0, 0.0, 1.0];
        let y = [0.0, -1.0, 0.0, -1.0];
        // Block 0 is repaired; block 1 has no curved row and is damped.
        assert_eq!(
            curvature.update(&mut b, &s, &y, Some(&repair)),
            Some(CurvatureUpdate::Modified)
        );
        assert!((b[(1, 1)] - 0.2).abs() < 1e-15, "{b}");
        let mut damped = DMatrix::identity(2, 2);
        Workspace::new(2, false).update(&mut damped, &[0.0, 1.0], &[0.0, -1.0], None);
        assert_eq!(b.view((2, 2), (2, 2)), damped);
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
                assert!(
                    work.update(&mut b, &s, &y, None).is_some(),
                    "n {n}, step {step}"
                );
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

#[cfg(test)]
#[test]
fn nonfinite_displacement_is_not_a_negligible_bfgs_step() {
    let mut b = DMatrix::identity(2, 2);
    let before = b.clone();
    assert!(
        Workspace::new(2, false)
            .update(&mut b, &[f64::NAN, 0.0], &[1.0, 1.0], None)
            .is_none()
    );
    assert_eq!(b, before);
}
