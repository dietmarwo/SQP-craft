//! Clarabel 0.11.1 QP adapter with optional compatible workspace reuse.
//!
//! Every solve starts from Clarabel's default primal/dual point. Workspace
//! reuse keeps allocation, symbolic structure and initial equilibration, not
//! numerical factors or iterates. Nonempty warm starts remain unsupported.
//! An updated solve must pass the same original-coordinate independent checks
//! as a fresh one; failures get at most one cold retry within the call budget.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
use clarabel::{algebra::CscMatrix, solver::*};
use nalgebra::DMatrix;
use sqp_craft::{CancellationToken, Error, Multipliers, Result, qp::*};
use std::time::Instant;

/// Clarabel QP backend; configure through [`QpBackend::configure`].
/// Clones copy settings/reporting and begin with an empty workspace cache.
#[derive(Default)]
pub struct ClarabelQp {
    settings: QpOptions,
    last: QpReport,
    workspace: Option<Workspace>,
    #[cfg(test)]
    cancel_at_iteration: Option<u32>,
    #[cfg(test)]
    reject_reused_solution: bool,
}

impl Clone for ClarabelQp {
    fn clone(&self) -> Self {
        Self {
            settings: self.settings.clone(),
            last: self.last.clone(),
            workspace: None,
            #[cfg(test)]
            cancel_at_iteration: self.cancel_at_iteration,
            #[cfg(test)]
            reject_reused_solution: self.reject_reused_solution,
        }
    }
}
impl std::fmt::Debug for ClarabelQp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClarabelQp")
            .field("settings", &self.settings)
            .field("last", &self.last)
            .field("has_workspace", &self.workspace.is_some())
            .finish()
    }
}

#[derive(PartialEq)]
struct Structure {
    n: usize,
    original_rows: usize,
    p_columns: Vec<usize>,
    p_rows: Vec<usize>,
    a_columns: Vec<usize>,
    a_rows: Vec<usize>,
    mapping: Vec<(bool, usize, f64)>,
    equalities: Vec<bool>,
}
struct Workspace {
    solver: DefaultSolver<f64>,
    key: Structure,
}
struct Converted {
    original_rows: usize,
    p: CscMatrix<f64>,
    a: CscMatrix<f64>,
    b: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
    mapping: Vec<(bool, usize, f64)>,
}
impl Converted {
    fn structure(&self) -> Structure {
        Structure {
            n: self.p.n,
            original_rows: self.original_rows,
            p_columns: self.p.colptr.clone(),
            p_rows: self.p.rowval.clone(),
            a_columns: self.a.colptr.clone(),
            a_rows: self.a.rowval.clone(),
            mapping: self.mapping.clone(),
            equalities: self
                .cones
                .iter()
                .map(|c| matches!(c, ZeroConeT(_)))
                .collect(),
        }
    }
}

impl Structure {
    /// Same comparison as `*self == data.structure()`, without the copies.
    fn matches(&self, data: &Converted) -> bool {
        self.n == data.p.n
            && self.original_rows == data.original_rows
            && self.p_columns == data.p.colptr
            && self.p_rows == data.p.rowval
            && self.a_columns == data.a.colptr
            && self.a_rows == data.a.rowval
            && self.mapping == data.mapping
            && self.equalities.len() == data.cones.len()
            && self
                .equalities
                .iter()
                .zip(&data.cones)
                .all(|(&e, c)| e == matches!(c, ZeroConeT(_)))
    }
}

/// The conic rows of one bounded row or variable: `(scale, rhs, dual sign,
/// equality)` per finite side, an equal pair as one equality row.
fn sides(lower: f64, upper: f64) -> impl Iterator<Item = (f64, f64, f64, bool)> {
    let equal = lower == upper;
    let lower_side = (!equal && lower.is_finite()).then_some((-1.0, -lower, 1.0, false));
    let upper_side = (equal || upper.is_finite()).then_some((1.0, upper, -1.0, equal));
    lower_side.into_iter().chain(upper_side)
}

fn positive_definite(model: &QpModel) -> bool {
    let n = model.n();
    let h = &model.hessian_lower;
    let mut dense = DMatrix::zeros(n, n);
    for i in 0..n {
        for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
            let j = h.pattern.indices[k];
            dense[(i, j)] = h.values[k];
            dense[(j, i)] = h.values[k];
        }
    }
    dense.cholesky().is_some()
}

/// Builds Clarabel's data directly in canonical CSC form, in O(nnz).
///
/// The validated lower-triangle CSR Hessian (strictly increasing columns per
/// row, none above the diagonal) is exactly the upper-triangle CSC that
/// Clarabel expects. `A` is filled column by column in conic-row order, so
/// row indices are sorted without a sort. Explicit zeros are kept, as before.
fn convert(model: &QpModel) -> Result<Converted> {
    if !positive_definite(model) {
        return Err(Error::Unsupported(
            "QP pilot requires a positive-definite Hessian".into(),
        ));
    }
    let n = model.n();
    let h = &model.hessian_lower;
    let p = CscMatrix::new(
        n,
        n,
        h.pattern.indptr.clone(),
        h.pattern.indices.clone(),
        h.values.clone(),
    );
    // Conic rows: each original row's sides, then each variable's sides.
    // `mapping` sends a conic dual to (variable?, original index, sign).
    let mut mapping = Vec::new();
    let mut b = Vec::new();
    let mut cones = Vec::new();
    let mut row_start = Vec::with_capacity(model.m() + n + 1);
    let mut scales = Vec::new();
    let mut push = |variable: bool, index: usize, lower: f64, upper: f64| {
        row_start.push(b.len());
        for (scale, rhs, dual_sign, equality) in sides(lower, upper) {
            scales.push(scale);
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
        push(false, i, model.rows.lower[i], model.rows.upper[i]);
    }
    for j in 0..n {
        push(true, j, model.variables.lower[j], model.variables.upper[j]);
    }
    row_start.push(b.len());
    // (first conic row, number of conic rows) of each original row, then of
    // each variable.
    let spans: Vec<(usize, usize)> = row_start.windows(2).map(|w| (w[0], w[1] - w[0])).collect();
    let (row_spans, variable_spans) = spans.split_at(model.m());
    let a = &model.matrix;
    let row_entries = |i: usize| a.pattern.indptr[i]..a.pattern.indptr[i + 1];
    let mut colptr = vec![0; n + 1];
    for (i, &(_, width)) in row_spans.iter().enumerate() {
        for &j in &a.pattern.indices[row_entries(i)] {
            colptr[j + 1] += width;
        }
    }
    for (j, &(_, width)) in variable_spans.iter().enumerate() {
        colptr[j + 1] += width;
    }
    for j in 0..n {
        colptr[j + 1] += colptr[j];
    }
    let nnz = colptr[n];
    let mut next = colptr.clone();
    let mut rowval = vec![0; nnz];
    let mut nzval = vec![0.0; nnz];
    let mut place = |j: usize, (first, width): (usize, usize), value: f64| {
        for (r, scale) in scales.iter().enumerate().skip(first).take(width) {
            rowval[next[j]] = r;
            nzval[next[j]] = scale * value;
            next[j] += 1;
        }
    };
    for (i, &span) in row_spans.iter().enumerate() {
        for k in row_entries(i) {
            place(a.pattern.indices[k], span, a.values[k]);
        }
    }
    for (j, &span) in variable_spans.iter().enumerate() {
        place(j, span, 1.0);
    }
    Ok(Converted {
        original_rows: model.m(),
        p,
        a: CscMatrix::new(b.len(), n, colptr, rowval, nzval),
        b,
        cones,
        mapping,
    })
}

impl ClarabelQp {
    fn engine_settings(&self, began: Instant) -> DefaultSettings<f64> {
        let tol = self.settings.tolerance;
        DefaultSettings {
            verbose: false,
            max_iter: self
                .settings
                .max_iterations
                .saturating_sub(self.last.iterations as u32),
            time_limit: self
                .settings
                .time_limit_seconds
                .map(|s| (s - began.elapsed().as_secs_f64()).max(0.0))
                .unwrap_or(f64::INFINITY),
            tol_gap_abs: tol,
            tol_gap_rel: tol,
            tol_feas: tol,
            tol_infeas_abs: tol,
            tol_infeas_rel: tol,
            ..Default::default()
        }
    }

    fn stop(&mut self, began: Instant, cancel: &CancellationToken) -> Option<QpOutcome> {
        if cancel.is_cancelled() {
            Some(QpOutcome::Cancelled)
        } else if self
            .settings
            .time_limit_seconds
            .is_some_and(|s| began.elapsed().as_secs_f64() >= s)
        {
            self.last.limit_reason = Some("time".into());
            Some(QpOutcome::LimitReached(None))
        } else {
            None
        }
    }

    fn cold_retry(&mut self, reason: String) {
        self.last.retry_reason = Some(reason);
    }
}

impl QpBackend for ClarabelQp {
    fn configure(&mut self, options: &QpOptions) -> Result<()> {
        options.validate()?;
        self.settings = options.clone();
        if !options.reuse_workspace {
            self.workspace = None;
        }
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
        let previous = self.workspace.take();
        if let Some(outcome) = self.stop(began, cancel) {
            return Ok(outcome);
        }
        let t = Instant::now();
        let converted = convert(model);
        self.last.conversion_seconds = t.elapsed().as_secs_f64();
        let data = converted?;
        let mut reusable = previous.filter(|w| {
            self.settings.reuse_workspace
                && w.key.matches(&data)
                && w.solver.is_data_update_allowed()
        });
        if let Some(w) = reusable.as_mut() {
            let t = Instant::now();
            let updated = w
                .solver
                .update_data(&data.p, &model.gradient, &data.a, &data.b);
            self.last.update_seconds += t.elapsed().as_secs_f64();
            match updated {
                Ok(()) => self.last.updates += 1,
                Err(err) => {
                    self.cold_retry(format!("data update failed: {err}"));
                    reusable = None; // update_data may have partially overwritten P/q/A/b.
                }
            }
        }
        // At most two engine attempts: an updated solve and one fresh retry.
        loop {
            if let Some(outcome) = self.stop(began, cancel) {
                return Ok(outcome);
            }
            let reused = reusable.is_some();
            let mut workspace = if let Some(w) = reusable.take() {
                w
            } else {
                if self.last.retry_reason.is_some() {
                    self.last.cold_retries += 1;
                }
                let t = Instant::now();
                let solver = DefaultSolver::new(
                    &data.p,
                    &model.gradient,
                    &data.a,
                    &data.b,
                    &data.cones,
                    self.engine_settings(began),
                );
                self.last.setup_seconds += t.elapsed().as_secs_f64();
                self.last.setups += 1;
                Workspace {
                    solver: solver.map_err(|e| Error::Backend(format!("Clarabel setup: {e}")))?,
                    key: data.structure(),
                }
            };
            if let Some(outcome) = self.stop(began, cancel) {
                return Ok(outcome);
            }
            // Settings (especially remaining time/iterations) and callback are
            // refreshed even for a cached workspace and before every retry.
            workspace
                .solver
                .update_settings(self.engine_settings(began))
                .map_err(|e| Error::Backend(format!("Clarabel settings: {e}")))?;
            // Clarabel's time-limit check uses the sum of its root timers:
            // "setup", "solve" and "post-process". It resets only "solve"
            // per call. Our remaining budget already deducts setup/update
            // time, so reset the other two: otherwise setup is charged twice,
            // and a reused workspace carries every earlier call's
            // post-processing time into its time-limit check.
            if let Some(timers) = workspace.solver.timers.as_mut() {
                timers.reset_timer("setup");
                timers.reset_timer("post-process");
            }
            let token = cancel.clone();
            let limit = self.settings.time_limit_seconds;
            #[cfg(test)]
            let cancel_at_iteration = self.cancel_at_iteration;
            workspace
                .solver
                .set_termination_callback(move |_info: &DefaultInfo<f64>| {
                    #[cfg(test)]
                    if cancel_at_iteration.is_some_and(|i| _info.iterations >= i) {
                        token.cancel();
                    }
                    token.is_cancelled()
                        || limit.is_some_and(|s| began.elapsed().as_secs_f64() >= s)
                });
            let t = Instant::now();
            workspace.solver.solve();
            self.last.solve_seconds += t.elapsed().as_secs_f64();
            workspace.solver.unset_termination_callback();
            let raw = workspace.solver.solution.status;
            self.last.raw_status = Some(format!("{raw:?}"));
            self.last.iterations += u64::from(workspace.solver.solution.iterations);
            if raw != SolverStatus::Solved {
                if let Some(outcome) = self.stop(began, cancel) {
                    return Ok(outcome);
                }
                if raw == SolverStatus::MaxTime {
                    self.last.limit_reason = Some("time".into());
                    return Ok(QpOutcome::LimitReached(None));
                }
                if raw == SolverStatus::MaxIterations {
                    self.last.limit_reason = Some("iterations".into());
                    return Ok(QpOutcome::LimitReached(None));
                }
            }
            let t = Instant::now();
            let outcome = extract(
                &workspace.solver,
                model,
                &data.mapping,
                self.settings.acceptance_tolerance,
            );
            self.last.check_seconds += t.elapsed().as_secs_f64();
            #[cfg(test)]
            let outcome = if reused && self.reject_reused_solution {
                Err(Error::Backend(
                    "test injection: reject reused solution".into(),
                ))
            } else {
                outcome
            };
            if reused && !matches!(outcome, Ok(QpOutcome::Optimal(_))) {
                // Never turn an unreliable reused infeasibility finding into
                // an NLP restoration decision without a cold confirmation.
                self.cold_retry(format!("reused solve not accepted: {outcome:?}"));
                if let Some(outcome) = self.stop(began, cancel) {
                    return Ok(outcome);
                }
                if self.last.iterations >= u64::from(self.settings.max_iterations) {
                    self.last.limit_reason = Some("iterations".into());
                    return Ok(QpOutcome::LimitReached(None));
                }
                continue;
            }
            if matches!(outcome, Ok(QpOutcome::Optimal(_)))
                && self.settings.reuse_workspace
                && workspace.solver.is_data_update_allowed()
            {
                self.workspace = Some(workspace);
            }
            return outcome.map(|outcome| match outcome {
                QpOutcome::Optimal(mut solution) => {
                    solution.iterations = self.last.iterations;
                    QpOutcome::Optimal(solution)
                }
                other => other,
            });
        }
    }
}

fn extract(
    solver: &DefaultSolver<f64>,
    model: &QpModel,
    mapping: &[(bool, usize, f64)],
    tolerance: f64,
) -> Result<QpOutcome> {
    match solver.solution.status {
        SolverStatus::PrimalInfeasible => Ok(QpOutcome::Infeasible),
        SolverStatus::DualInfeasible => Ok(QpOutcome::Unbounded),
        SolverStatus::Solved => {
            let mut multipliers = Multipliers {
                constraints: vec![0.0; model.m()],
                variables: vec![0.0; model.n()],
            };
            for (&dual, &(variable, index, sign)) in solver.solution.z.iter().zip(mapping) {
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
                iterations: u64::from(solver.solution.iterations),
            };
            let check = model.accepts(&solution, tolerance)?;
            if !check.accepted {
                return Err(Error::Backend(format!(
                    "Clarabel solution failed the independent check (scale {:.3e}): {:?}",
                    check.scale, check.residuals
                )));
            }
            Ok(QpOutcome::Optimal(solution))
        }
        raw => Err(Error::Backend(format!(
            "Clarabel did not produce an accepted QP solution: {raw:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::TriangularMatrixChecks;
    use sqp_craft::{Bounds, CsrMatrix, CsrPattern};

    fn reusable() -> (ClarabelQp, QpModel) {
        let model = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern::dense(1, 1),
                values: vec![1.0],
            },
            gradient: vec![-1.0],
            variables: Bounds::new(vec![0.0], vec![2.0]),
            matrix: CsrMatrix {
                pattern: CsrPattern::dense(0, 1),
                values: vec![],
            },
            rows: Bounds::free(0),
        };
        let mut b = ClarabelQp::default();
        b.configure(&QpOptions {
            reuse_workspace: true,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            b.solve(&model, None, &CancellationToken::default())
                .unwrap(),
            QpOutcome::Optimal(_)
        ));
        (b, model)
    }

    /// The previous conversion through Clarabel's sorting triplet constructor.
    fn convert_by_triplets(model: &QpModel) -> Converted {
        let n = model.n();
        let h = &model.hessian_lower;
        let (mut hi, mut hj, mut hv) = (vec![], vec![], vec![]);
        for i in 0..n {
            for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                hi.push(h.pattern.indices[k]);
                hj.push(i);
                hv.push(h.values[k]);
            }
        }
        let (mut mapping, mut b, mut cones) = (vec![], vec![], vec![]);
        let (mut ai, mut aj, mut av) = (vec![], vec![], vec![]);
        let mut add = |variable: bool, index: usize, entries: Vec<(usize, f64)>, lo, up| {
            for (scale, rhs, dual_sign, equality) in sides(lo, up) {
                for &(j, v) in &entries {
                    ai.push(b.len());
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
        let a = &model.matrix;
        for i in 0..model.m() {
            let entries = (a.pattern.indptr[i]..a.pattern.indptr[i + 1])
                .map(|k| (a.pattern.indices[k], a.values[k]))
                .collect();
            add(false, i, entries, model.rows.lower[i], model.rows.upper[i]);
        }
        for j in 0..n {
            let (lo, up) = (model.variables.lower[j], model.variables.upper[j]);
            add(true, j, vec![(j, 1.0)], lo, up);
        }
        Converted {
            original_rows: model.m(),
            p: CscMatrix::new_from_triplets(n, n, hi, hj, hv),
            a: CscMatrix::new_from_triplets(b.len(), n, ai, aj, av),
            b,
            cones,
            mapping,
        }
    }

    #[test]
    fn direct_conversion_equals_the_sorted_triplet_conversion() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |k: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % k
        };
        let inf = f64::INFINITY;
        for _ in 0..300 {
            let (n, m) = (1 + next(9) as usize, next(7) as usize);
            let mut side = |lower: bool| match next(5) {
                0 => -1.5,
                1 => 2.0,
                2 => 0.0,
                _ if lower => -inf,
                _ => inf,
            };
            let bounds = |count: usize, side: &mut dyn FnMut(bool) -> f64| {
                let (mut lower, mut upper) = (vec![], vec![]);
                for _ in 0..count {
                    let (l, u) = (side(true), side(false));
                    let (l, u) = if l <= u { (l, u) } else { (u, u) };
                    lower.push(l);
                    upper.push(u);
                }
                Bounds::new(lower, upper)
            };
            let rows = bounds(m, &mut side);
            let variables = bounds(n, &mut side);
            // Dense, positive-definite lower triangle with one explicit zero.
            let (mut indptr, mut indices, mut values) = (vec![0], vec![], vec![]);
            for i in 0..n {
                for j in 0..=i {
                    indices.push(j);
                    values.push(if i == j {
                        2.0 + n as f64
                    } else if j == 0 {
                        0.0
                    } else {
                        0.5
                    });
                }
                indptr.push(indices.len());
            }
            let (mut aptr, mut aidx, mut aval) = (vec![0], vec![], vec![]);
            for _ in 0..m {
                for j in 0..n {
                    if next(2) == 0 {
                        aidx.push(j);
                        aval.push([0.0, -1.0, 3.5][next(3) as usize]);
                    }
                }
                aptr.push(aidx.len());
            }
            let model = QpModel {
                hessian_lower: CsrMatrix {
                    pattern: CsrPattern {
                        rows: n,
                        cols: n,
                        indptr,
                        indices,
                    },
                    values,
                },
                gradient: vec![1.0; n],
                variables,
                matrix: CsrMatrix {
                    pattern: CsrPattern {
                        rows: m,
                        cols: n,
                        indptr: aptr,
                        indices: aidx,
                    },
                    values: aval,
                },
                rows,
            };
            model.validate().unwrap();
            let (direct, sorted) = (convert(&model).unwrap(), convert_by_triplets(&model));
            for (x, y) in [(&direct.p, &sorted.p), (&direct.a, &sorted.a)] {
                assert_eq!((x.m, x.n), (y.m, y.n));
                assert_eq!((&x.colptr, &x.rowval), (&y.colptr, &y.rowval));
                assert_eq!(
                    x.nzval.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    y.nzval.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            assert!(direct.a.check_format().is_ok() && direct.p.is_triu());
            assert_eq!((&direct.b, &direct.mapping), (&sorted.b, &sorted.mapping));
            assert!(direct.structure() == sorted.structure());
            assert!(direct.structure().matches(&sorted));
        }
    }

    #[test]
    fn reused_workspace_time_limit_counts_only_the_current_call() {
        // Clarabel compares the sum of its root timers with the time limit
        // and resets only "solve" per call. Every call's own engine time is
        // measured around `solve()`, so with the stale timers reset their sum
        // cannot exceed it; accumulated post-processing would.
        let (mut b, model) = reusable();
        for _ in 0..300 {
            b.solve(&model, None, &CancellationToken::default())
                .unwrap();
            assert_eq!(b.last.updates, 1);
            let timers = b
                .workspace
                .as_ref()
                .unwrap()
                .solver
                .timers
                .as_ref()
                .unwrap();
            assert!(
                timers.total_time().as_secs_f64() <= b.last.solve_seconds,
                "{:?} > {}",
                timers.total_time(),
                b.last.solve_seconds
            );
        }
    }

    #[test]
    fn rejected_reused_solution_gets_one_checked_cold_retry() {
        let (mut b, model) = reusable();
        b.reject_reused_solution = true;
        let QpOutcome::Optimal(s) = b
            .solve(&model, None, &CancellationToken::default())
            .unwrap()
        else {
            panic!()
        };
        assert!(model.accepts(&s, 1e-7).unwrap().accepted);
        assert_eq!(
            (b.last.setups, b.last.updates, b.last.cold_retries),
            (1, 1, 1)
        );
        assert_eq!(s.iterations, b.last.iterations);
        assert!(b.last.iterations <= u64::from(b.settings.max_iterations));
        assert!(b.last.check_seconds >= 0.0);
    }

    #[test]
    fn partial_update_failure_discards_workspace() {
        let (mut b, mut model) = reusable();
        // P and q update first; then this deliberately incompatible internal A
        // makes the real update_data call fail after partial mutation.
        b.workspace
            .as_mut()
            .unwrap()
            .solver
            .data
            .A
            .rowval
            .swap(0, 1);
        model.gradient[0] = -0.5;
        let QpOutcome::Optimal(s) = b
            .solve(&model, None, &CancellationToken::default())
            .unwrap()
        else {
            panic!()
        };
        assert!((s.step[0] - 0.5).abs() < 1e-7);
        assert_eq!(
            (b.last.setups, b.last.updates, b.last.cold_retries),
            (1, 0, 1)
        );
        assert!(
            b.last
                .retry_reason
                .as_ref()
                .unwrap()
                .contains("data update failed")
        );
    }

    #[test]
    fn rejection_with_exhausted_iteration_budget_does_not_retry() {
        let (mut b, model) = reusable();
        let iterations = b.last.iterations as u32;
        b.settings.max_iterations = iterations;
        b.reject_reused_solution = true;
        assert!(matches!(
            b.solve(&model, None, &CancellationToken::default())
                .unwrap(),
            QpOutcome::LimitReached(_)
        ));
        assert_eq!(b.last.iterations, u64::from(iterations));
        assert_eq!(b.last.cold_retries, 0);
        assert!(b.workspace.is_none());
    }

    #[test]
    fn cancellation_inside_reused_engine_does_not_retry() {
        let (mut b, model) = reusable();
        b.cancel_at_iteration = Some(1);
        let token = CancellationToken::default();
        assert!(matches!(
            b.solve(&model, None, &token).unwrap(),
            QpOutcome::Cancelled
        ));
        assert_eq!(b.last.updates, 1);
        assert_eq!(b.last.cold_retries, 0);
        assert!(b.workspace.is_none());
    }

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
