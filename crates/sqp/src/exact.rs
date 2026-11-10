//! Optional sparse Lagrangian Hessian with backend-driven convexification.

use super::*;
use hessian::{HessianModel, Update};
use sqp_craft::{StructureReport, sparse};

pub(super) struct Exact {
    lower: CsrMatrix,
    restoration: CsrPattern,
    initial_scale: f64,
    shift: f64,
    radii: Vec<f64>,
    report: StructureReport,
}

impl Exact {
    pub fn new(spec: &ProblemSpec, settings: &SqpOptions) -> Result<Self> {
        let pattern = spec.lagrangian_hessian_pattern().ok_or_else(|| {
            Error::InvalidInput("exact mode requires a Lagrangian Hessian pattern".into())
        })?;
        let n = spec.n();
        if n > settings.dense_size_limit && pattern.nnz() == n.saturating_mul(n + 1) / 2 {
            return Err(Error::Unsupported("exact Hessian pattern is dense".into()));
        }
        let limit = settings.structured_memory_limit_bytes;
        let fixed = spec
            .nnz()
            .saturating_mul(64)
            .saturating_add(n.saturating_add(spec.m()).saturating_mul(192))
            .saturating_add(
                settings.record_qp_limit.saturating_mul(
                    spec.nnz()
                        .saturating_add(pattern.nnz())
                        .saturating_add(n)
                        .saturating_add(spec.m())
                        .saturating_mul(24),
                ),
            );
        sparse::check_memory(fixed, limit)?;
        let budget = limit - fixed;
        let hessian_factor_nnz = sparse::Symbolic::of(pattern, budget)?.nnz();
        let kkt_factor_nnz = sparse::kkt_fill(
            pattern,
            spec.jacobian(),
            spec.constraints(),
            spec.variables(),
            budget,
        )?;
        let mut graph = sparse::Graph::new(n, budget)?;
        for i in 0..spec.m() {
            if !spec.linear_rows()[i]
                && (spec.constraints().lower[i].is_finite()
                    || spec.constraints().upper[i].is_finite())
            {
                graph.clique(spec.jacobian().row(i))?;
            }
        }
        let mut exact = Self {
            lower: CsrMatrix {
                pattern: pattern.clone(),
                values: vec![0.0; pattern.nnz()],
            },
            restoration: graph.into_pattern(),
            initial_scale: settings.initial_hessian_scale,
            shift: 0.0,
            radii: vec![0.0; n],
            report: StructureReport {
                hessian_nnz: pattern.nnz(),
                hessian_factor_nnz,
                kkt_factor_nnz,
                estimated_memory_bytes: fixed
                    .saturating_add(sparse::factor_bytes(hessian_factor_nnz.max(kkt_factor_nnz))),
                exact_hessian_calls: Some(0),
                maximum_diagonal_shift: Some(0.0),
                diagonal_shifts: Some(0),
                ..Default::default()
            },
        };
        exact.reset();
        Ok(exact)
    }
}

impl HessianModel for Exact {
    fn lower_csr(&self) -> CsrMatrix {
        self.lower.clone()
    }

    fn refresh_lower(&self, lower: &mut CsrMatrix) {
        lower.values.copy_from_slice(&self.lower.values);
    }

    fn product(&self, p: &[f64], out: &mut [f64]) {
        self.lower.symmetric_lower_product(p, out);
    }

    fn update(&mut self, _data: &Update<'_>) -> Result<Option<CurvatureUpdate>> {
        Ok(Some(CurvatureUpdate::Negligible))
    }

    fn reset(&mut self) {
        self.lower.values.fill(0.0);
        for &end in &self.lower.pattern.indptr[1..] {
            self.lower.values[end - 1] = self.initial_scale;
        }
        self.shift = 0.0;
    }

    fn can_fallback(&self) -> bool {
        false
    }

    fn structured_pattern(&self) -> Option<&CsrPattern> {
        Some(&self.restoration)
    }

    fn structure_report(&self) -> Option<StructureReport> {
        Some(self.report.clone())
    }

    fn exact_at(
        &mut self,
        model: &mut dyn NlpProblem,
        x: &[f64],
        lambda: &[f64],
        scales: &Scales,
    ) -> Result<()> {
        self.lower.values.fill(f64::NAN);
        model.lagrangian_hessian(x, lambda, &mut self.lower.values)?;
        self.report.exact_hessian_calls = self.report.exact_hessian_calls.map(|n| n + 1);
        self.shift = 0.0;
        for i in 0..self.lower.pattern.rows {
            for k in self.lower.pattern.indptr[i]..self.lower.pattern.indptr[i + 1] {
                let j = self.lower.pattern.indices[k];
                self.lower.values[k] = finite(
                    self.lower.values[k] * scales.variables[i] * scales.variables[j],
                    "scaled exact Hessian entry",
                )?;
            }
        }
        Ok(())
    }

    fn shift_diagonal(&mut self) -> Result<()> {
        // A backend rejection triggers a sufficient Gershgorin shift. The
        // margin keeps it strictly dominant despite rounding in the row sums.
        self.radii.fill(0.0);
        let pattern = &self.lower.pattern;
        for i in 0..pattern.rows {
            for k in pattern.indptr[i]..pattern.indptr[i + 1] - 1 {
                let j = pattern.indices[k];
                let v = self.lower.values[k].abs();
                self.radii[i] += v;
                self.radii[j] += v;
            }
        }
        let mut deficit = 0.0_f64;
        let mut norm = 0.0_f64;
        for (i, &radius) in self.radii.iter().enumerate() {
            let diagonal = self.lower.values[pattern.indptr[i + 1] - 1];
            norm = norm.max(finite(diagonal.abs() + radius, "exact Hessian row norm")?);
            deficit = deficit.max(radius - diagonal);
        }
        let extra = finite(
            deficit + (1e-8 * (1.0 + norm)).max(self.shift),
            "exact Hessian shift",
        )?;
        self.shift = finite(self.shift + extra, "total exact Hessian shift")?;
        for &end in &pattern.indptr[1..] {
            self.lower.values[end - 1] = finite(
                self.lower.values[end - 1] + extra,
                "shifted exact Hessian diagonal",
            )?;
        }
        self.report.maximum_diagonal_shift = self
            .report
            .maximum_diagonal_shift
            .map(|v| v.max(self.shift));
        self.report.diagonal_shifts = self.report.diagonal_shifts.map(|n| n + 1);
        Ok(())
    }
}
