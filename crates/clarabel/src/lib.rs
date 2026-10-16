//! Cold-start, single-thread convex QP adapter. Strictly positive-definite
//! Hessians, n <= 256. Unsupported warm-start components are rejected.
#![forbid(unsafe_code)]
use clarabel::{algebra::CscMatrix, solver::*};
use nalgebra::DMatrix;
use sqp_craft::{CancellationToken, Error, Multipliers, Result, qp::*};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct ClarabelQp {
    settings: QpOptions,
    last: QpReport,
    #[cfg(test)]
    cancel_at_iteration: Option<u32>,
}
impl QpBackend for ClarabelQp {
    fn configure(&mut self, options: &QpOptions) -> Result<()> {
        options.validate()?;
        self.settings = options.clone();
        Ok(())
    }
    fn report(&self) -> QpReport {
        self.last.clone()
    }
    fn solve(
        &mut self,
        model: &QpModel,
        warm: Option<&QpWarmStart>,
        cancel: &CancellationToken,
    ) -> Result<QpOutcome> {
        model.validate()?;
        self.settings.validate()?;
        if let Some(w) = warm {
            w.validate(model.n(), model.m())?;
            if w.step.is_some() || w.multipliers.is_some() || w.active_set.is_some() {
                return Err(Error::Unsupported(
                    "Clarabel adapter supports cold starts only".into(),
                ));
            }
        }
        if model.n() > 256 {
            return Err(Error::Unsupported(
                "Clarabel dense pilot is limited to 256 variables".into(),
            ));
        }
        self.last = QpReport {
            backend: "clarabel 0.11.1".into(),
            ..Default::default()
        };
        let began = Instant::now();
        let limit = self.settings.time_limit_seconds;
        if cancel.is_cancelled() {
            return Ok(QpOutcome::Cancelled);
        }
        if limit == Some(0.0) {
            self.last.limit_reason = Some("time".into());
            return Ok(QpOutcome::LimitReached(None));
        }
        let n = model.n();
        let mut dense = DMatrix::zeros(n, n);
        let mut hi = vec![];
        let mut hj = vec![];
        let mut hv = vec![];
        let h = &model.hessian_lower;
        for i in 0..n {
            for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                let j = h.pattern.indices[k];
                let v = h.values[k];
                dense[(i, j)] = v;
                dense[(j, i)] = v;
                hi.push(j);
                hj.push(i);
                hv.push(v);
            }
        }
        if dense.cholesky().is_none() {
            return Err(Error::Unsupported(
                "QP pilot requires a positive-definite Hessian".into(),
            ));
        }
        let p = CscMatrix::new_from_triplets(n, n, hi, hj, hv);
        // Map each conic row to (variable?, original index, signed-dual factor).
        let mut mapping = Vec::new();
        let mut ai = vec![];
        let mut aj = vec![];
        let mut av = vec![];
        let mut b = vec![];
        let mut cones = vec![];
        let mut add =
            |variable: bool, index: usize, entries: Vec<(usize, f64)>, lower: f64, upper: f64| {
                let sides = if lower == upper {
                    vec![(1.0, upper, -1.0, true)]
                } else {
                    let mut sides = vec![];
                    if lower.is_finite() {
                        sides.push((-1.0, -lower, 1.0, false));
                    }
                    if upper.is_finite() {
                        sides.push((1.0, upper, -1.0, false));
                    }
                    sides
                };
                for (scale, rhs, dual_sign, equality) in sides {
                    let row = b.len();
                    for &(j, v) in &entries {
                        ai.push(row);
                        aj.push(j);
                        av.push(scale * v);
                    }
                    b.push(rhs);
                    mapping.push((variable, index, dual_sign));
                    cones.push(if equality {
                        ZeroConeT(1)
                    } else {
                        NonnegativeConeT(1)
                    });
                }
            };
        for i in 0..model.m() {
            let a = &model.matrix;
            let entries = (a.pattern.indptr[i]..a.pattern.indptr[i + 1])
                .map(|k| (a.pattern.indices[k], a.values[k]))
                .collect();
            add(false, i, entries, model.rows.lower[i], model.rows.upper[i]);
        }
        for j in 0..n {
            add(
                true,
                j,
                vec![(j, 1.0)],
                model.variables.lower[j],
                model.variables.upper[j],
            );
        }
        let a = CscMatrix::new_from_triplets(b.len(), n, ai, aj, av);
        let remaining = limit.map(|s| (s - began.elapsed().as_secs_f64()).max(0.0));
        if remaining == Some(0.0) {
            self.last.limit_reason = Some("time".into());
            return Ok(QpOutcome::LimitReached(None));
        }
        let tol = self.settings.tolerance;
        let settings = DefaultSettings {
            verbose: false,
            max_iter: self.settings.max_iterations,
            time_limit: remaining.unwrap_or(f64::INFINITY),
            tol_gap_abs: tol,
            tol_gap_rel: tol,
            tol_feas: tol,
            tol_infeas_abs: tol,
            tol_infeas_rel: tol,
            ..Default::default()
        };
        let mut solver = DefaultSolver::new(&p, &model.gradient, &a, &b, &cones, settings)
            .map_err(|e| Error::Backend(format!("Clarabel setup: {e}")))?;
        let token = cancel.clone();
        #[cfg(test)]
        let cancel_at_iteration = self.cancel_at_iteration;
        solver.set_termination_callback(move |_info: &DefaultInfo<f64>| {
            #[cfg(test)]
            if cancel_at_iteration.is_some_and(|i| _info.iterations >= i) {
                token.cancel();
            }
            token.is_cancelled() || limit.is_some_and(|s| began.elapsed().as_secs_f64() >= s)
        });
        solver.solve();
        let raw = solver.solution.status;
        self.last.raw_status = Some(format!("{raw:?}"));
        self.last.iterations = u64::from(solver.solution.iterations);
        if cancel.is_cancelled() {
            return Ok(QpOutcome::Cancelled);
        }
        if limit.is_some_and(|s| began.elapsed().as_secs_f64() >= s) || raw == SolverStatus::MaxTime
        {
            self.last.limit_reason = Some("time".into());
            return Ok(QpOutcome::LimitReached(None));
        }
        if raw == SolverStatus::MaxIterations {
            self.last.limit_reason = Some("iterations".into());
            return Ok(QpOutcome::LimitReached(None));
        }
        match raw {
            SolverStatus::PrimalInfeasible => Ok(QpOutcome::Infeasible),
            SolverStatus::DualInfeasible => Ok(QpOutcome::Unbounded),
            SolverStatus::Solved => {
                let mut multipliers = Multipliers {
                    constraints: vec![0.0; model.m()],
                    variables: vec![0.0; n],
                };
                for (&dual, &(variable, index, sign)) in solver.solution.z.iter().zip(&mapping) {
                    if variable {
                        multipliers.variables[index] += sign * dual;
                    } else {
                        multipliers.constraints[index] += sign * dual;
                    }
                }
                let solution = QpSolution {
                    step: solver.solution.x.clone(),
                    multipliers,
                    active_set: None,
                    iterations: self.last.iterations,
                };
                let kkt = model.residuals(&solution)?;
                if kkt.max() > self.settings.acceptance_tolerance {
                    return Err(Error::Backend(format!(
                        "Clarabel returned unacceptable original-unit residuals: {kkt:?}"
                    )));
                }
                Ok(QpOutcome::Optimal(solution))
            }
            _ => Err(Error::Backend(format!(
                "Clarabel did not produce an accepted QP solution: {raw:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqp_craft::{Bounds, CsrMatrix, CsrPattern};

    #[test]
    fn cancels_from_inside_the_running_qp_callback() {
        let model = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern::dense(1, 1),
                values: vec![1.0],
            },
            gradient: vec![1.0],
            variables: Bounds::new(vec![0.0], vec![2.0]),
            matrix: CsrMatrix {
                pattern: CsrPattern::dense(0, 1),
                values: vec![],
            },
            rows: Bounds::free(0),
        };
        // Deterministic test-only injection avoids a platform-dependent sleep.
        // The real Clarabel callback must run after one completed iteration.
        let mut backend = ClarabelQp {
            cancel_at_iteration: Some(1),
            ..Default::default()
        };
        let token = CancellationToken::default();
        assert!(matches!(
            backend.solve(&model, None, &token).unwrap(),
            QpOutcome::Cancelled
        ));
        assert_eq!(
            backend.report().raw_status.as_deref(),
            Some("CallbackTerminated")
        );
        assert!(backend.report().iterations >= 1);
    }
}
