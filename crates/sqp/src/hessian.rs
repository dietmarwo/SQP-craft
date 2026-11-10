//! Curvature models for the step QP. The legacy path preserves its arithmetic.

use super::*;
use nalgebra::{DMatrix, DVectorView, DVectorViewMut};

pub(super) struct Update<'a> {
    pub spec: &'a ProblemSpec,
    pub previous: &'a Evaluation,
    pub next: &'a Evaluation,
    pub multipliers: &'a Multipliers,
    pub step: &'a [f64],
    pub scales: &'a Scales,
    pub scaled_identity: bool,
    pub strategy: HessianUpdate,
}

pub(super) trait HessianModel {
    fn lower_csr(&self) -> CsrMatrix;
    fn refresh_lower(&self, lower: &mut CsrMatrix);
    fn product(&self, p: &[f64], out: &mut [f64]);
    fn curvature_along(&self, p: &[f64]) -> Result<f64> {
        let mut hp = vec![0.0; p.len()];
        self.product(p, &mut hp);
        dot(p, &hp)
    }
    fn update(&mut self, data: &Update<'_>) -> Result<Option<CurvatureUpdate>>;
    fn reset(&mut self);
    fn can_fallback(&self) -> bool;
    fn exact_at(
        &mut self,
        _model: &mut dyn NlpProblem,
        _x: &[f64],
        _lambda: &[f64],
        _scales: &Scales,
    ) -> Result<()> {
        unreachable!("exact mode must supply a Hessian callback")
    }
    fn shift_diagonal(&mut self) -> Result<()> {
        unreachable!("only the exact Hessian can be shifted")
    }
    fn structured_pattern(&self) -> Option<&CsrPattern> {
        None
    }
    fn structure_report(&self) -> Option<sqp_craft::StructureReport> {
        None
    }
}

pub(super) struct Legacy {
    pub matrix: DMatrix<f64>,
    pub block: usize,
    scale: f64,
    work: bfgs::Curvature,
}

impl Legacy {
    pub fn new(n: usize, block: usize, scale: f64) -> Self {
        let block = if block >= n { 0 } else { block };
        Self {
            matrix: DMatrix::identity(n, n) * scale,
            block,
            scale,
            work: bfgs::Curvature::new(n, block),
        }
    }
}

impl HessianModel for Legacy {
    fn lower_csr(&self) -> CsrMatrix {
        let n = self.matrix.nrows();
        let mut indptr = vec![0];
        let capacity = n * (if self.block == 0 { n } else { self.block } + 1) / 2;
        let mut indices = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity);
        for i in 0..n {
            let first = i.checked_div(self.block).unwrap_or(0) * self.block;
            for j in first..=i {
                indices.push(j);
                values.push(self.matrix[(i, j)]);
            }
            indptr.push(indices.len());
        }
        CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr,
                indices,
            },
            values,
        }
    }

    fn refresh_lower(&self, lower: &mut CsrMatrix) {
        for i in 0..lower.pattern.rows {
            for k in lower.pattern.indptr[i]..lower.pattern.indptr[i + 1] {
                lower.values[k] = self.matrix[(i, lower.pattern.indices[k])];
            }
        }
    }

    fn product(&self, p: &[f64], out: &mut [f64]) {
        let pv = DVectorView::from_slice(p, p.len());
        let n = out.len();
        self.matrix
            .mul_to(&pv, &mut DVectorViewMut::from_slice(out, n));
    }

    fn curvature_along(&self, p: &[f64]) -> Result<f64> {
        let pv = DVectorView::from_slice(p, p.len());
        finite((&self.matrix * pv).dot(&pv), "step curvature")
    }

    fn update(&mut self, data: &Update<'_>) -> Result<Option<CurvatureUpdate>> {
        let Update {
            spec,
            previous,
            next,
            multipliers,
            step,
            scales,
            scaled_identity,
            strategy,
        } = *data;
        let old_l = lagrangian_gradient(spec, previous, multipliers)?;
        let new_l = lagrangian_gradient(spec, next, multipliers)?;
        let y: Vec<f64> = new_l.iter().zip(old_l).map(|(&a, b)| a - b).collect();
        let (mut change, mut departure) = match strategy {
            HessianUpdate::Damped => (Vec::new(), Vec::new()),
            HessianUpdate::Augmented => curvature_repair(spec, previous, next, step)?,
        };
        // B lives in scaled coordinates: s̃ = s/D, ỹ = D y, J̃ = R J D.
        let (s_scaled, y) = if scaled_identity {
            (step.to_vec(), y)
        } else {
            let jacobian = spec.jacobian();
            for i in 0..spec.m() {
                for a in jacobian.indptr[i]..jacobian.indptr[i + 1] {
                    if let Some(c) = change.get_mut(a) {
                        *c *= scales.rows[i] * scales.variables[jacobian.indices[a]];
                    }
                }
                if let Some(d) = departure.get_mut(i) {
                    *d *= scales.rows[i];
                }
            }
            let y = y
                .iter()
                .zip(&scales.variables)
                .map(|(v, d)| v * d)
                .collect();
            (scales.to_scaled(step), y)
        };
        let repair = (strategy == HessianUpdate::Augmented).then(|| bfgs::Repair {
            pattern: spec.jacobian(),
            change: &change,
            departure: &departure,
        });
        Ok(self
            .work
            .update(&mut self.matrix, &s_scaled, &y, repair.as_ref()))
    }

    fn reset(&mut self) {
        let n = self.matrix.nrows();
        self.matrix = DMatrix::identity(n, n) * self.scale;
    }

    fn can_fallback(&self) -> bool {
        self.block != 0
    }
}

/// Discards the fixed QP template when curvature switches to full BFGS.
pub(super) fn fallback_to_full(
    hessian: &mut Box<dyn HessianModel>,
    qp_work: &mut Option<QpModel>,
    report: &mut sqp_craft::RunReport,
    n: usize,
    scale: f64,
) {
    if hessian.can_fallback() {
        if let Some(mut structure) = hessian.structure_report() {
            structure.fell_back_to_full = true;
            report.structure = Some(structure);
        } else {
            report.block_hessian_fallbacks += 1;
        }
        *hessian = Box::new(Legacy::new(n, 0, scale));
        *qp_work = None;
    }
}
