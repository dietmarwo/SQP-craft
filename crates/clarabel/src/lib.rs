//! Clarabel 0.11.1 QP adapter with optional compatible workspace reuse.
//!
//! Every solve starts from Clarabel's default primal/dual point. Workspace
//! reuse keeps allocation, symbolic structure and initial equilibration, not
//! numerical factors or iterates. Nonempty warm starts remain unsupported.
//! An updated solve must pass the same original-coordinate independent checks
//! as a fresh one. A rejected updated solve gets a cold retry; a rejected
//! equilibrated cold solve can be polished or retry without equilibration. All attempts share
//! the original call's time and iteration budgets and independent checks.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
use clarabel::{algebra::CscMatrix, solver::*};
use nalgebra::DMatrix;
use sqp_craft::{CancellationToken, CsrPattern, Error, Multipliers, Result, qp::*, sparse};
use std::time::Instant;
mod polish;

/// Clarabel QP backend; configure through [`QpBackend::configure`].
/// Clones copy settings/reporting and begin with an empty workspace cache.
#[derive(Default)]
pub struct ClarabelQp {
    settings: QpOptions,
    last: QpReport,
    workspace: Option<Workspace>,
    checks: QpResidualWorkspace,
    conversion: Option<ConversionCache>,
    pd: PdScratch,
    sparse_pd: Option<SparsePd>,
    threads: usize,
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
            checks: QpResidualWorkspace::default(),
            conversion: None,
            pd: PdScratch::default(),
            sparse_pd: None,
            threads: self.threads,
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
    equilibrated: bool,
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

/// Reused buffers of [`positive_definite`]: the dense copy alone is 215 KB at
/// n = 164, above glibc's mmap threshold, so a fresh one per QP costs page
/// faults on every call.
#[derive(Default)]
struct PdScratch {
    dense: DMatrix<f64>,
    diagonal: Vec<f64>,
    radii: Vec<f64>,
}

const DENSE_PD_LIMIT: usize = 256;

struct SparsePd {
    pattern: CsrPattern,
    symbolic: sparse::Symbolic,
    factor: Vec<f64>,
}

/// Exact positive-definiteness gate for the Hessian.
///
/// A strictly diagonally dominant matrix with positive diagonal is positive
/// definite (Gershgorin); the radii are summed with upward rounding, so the
/// sufficient test is safe in floating point. A stored pattern that splits
/// into disconnected components is factored per component. Everything else
/// gets a dense Cholesky factorization of the lower triangle, which is all
/// nalgebra reads; for a full lower triangle the copy and the dominance sums
/// share one pass.
fn positive_definite(model: &QpModel, scratch: &mut PdScratch) -> bool {
    let n = model.n();
    let h = &model.hessian_lower;
    let full = h.pattern.nnz() == n * (n + 1) / 2;
    let PdScratch {
        dense,
        diagonal,
        radii,
    } = scratch;
    diagonal.clear();
    diagonal.resize(n, 0.);
    radii.clear();
    radii.resize(n, 0.);
    if full && dense.shape() != (n, n) {
        *dense = DMatrix::zeros(n, n);
    }
    let add_up = |a: f64, b: f64| {
        let sum = a + b;
        if !sum.is_finite() {
            f64::INFINITY
        } else {
            f64::from_bits(sum.to_bits() + 1)
        }
    };
    // Radii only grow, so the dominance sums stop at the first row that
    // already fails (for a dense BFGS matrix, within the first rows); the copy
    // for the Cholesky test continues. A row's diagonal is the last entry of
    // its lower-triangle row, so diagonal[j] is final whenever j < i.
    let mut dominant = true;
    for i in 0..n {
        for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
            let j = h.pattern.indices[k];
            let v = h.values[k];
            if full {
                dense[(i, j)] = v;
            }
            if !dominant {
                continue;
            }
            if i == j {
                diagonal[i] = v;
                dominant = v.is_finite() && v > radii[i];
            } else if v != 0. {
                radii[i] = add_up(radii[i], v.abs());
                radii[j] = add_up(radii[j], v.abs());
                dominant = radii[j] < diagonal[j];
            }
        }
    }
    // A sufficient SPD proof; inconclusive cases keep a Cholesky test.
    if dominant
        && diagonal
            .iter()
            .zip(radii.iter())
            .all(|(&d, &r)| d.is_finite() && d > r)
    {
        return true;
    }
    // Missing stored off-block entries prove exact structural disconnection.
    if !full {
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut parents: Vec<_> = (0..n).collect();
        for i in 0..n {
            for &j in h.pattern.row(i) {
                let a = root(&mut parents, i);
                let b = root(&mut parents, j);
                parents[b] = a;
            }
        }
        let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for i in 0..n {
            groups.entry(root(&mut parents, i)).or_default().push(i);
        }
        if groups.len() > 1 {
            let mut local = vec![0; n];
            for group in groups.values() {
                for (k, &i) in group.iter().enumerate() {
                    local[i] = k;
                }
            }
            for group in groups.values() {
                let mut part = DMatrix::zeros(group.len(), group.len());
                for &i in group {
                    for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                        part[(local[i], local[h.pattern.indices[k]])] = h.values[k];
                    }
                }
                if part.cholesky().is_none() {
                    return false;
                }
            }
            return true;
        }
        *dense = DMatrix::zeros(n, n);
        for i in 0..n {
            for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                dense[(i, h.pattern.indices[k])] = h.values[k];
            }
        }
    }
    match std::mem::take(dense).cholesky() {
        Some(factor) => {
            *dense = factor.unpack_dirty();
            true
        }
        None => false,
    }
}

/// Builds Clarabel's data directly in canonical CSC form, in O(nnz).
///
/// The validated lower-triangle CSR Hessian (strictly increasing columns per
/// row, none above the diagonal) is exactly the upper-triangle CSC that
/// Clarabel expects. `A` is filled column by column in conic-row order, so
/// row indices are sorted without a sort. Explicit zeros are kept, as before.
/// The caller checks positive definiteness first ([`positive_definite`]).
fn convert(model: &QpModel) -> Result<Converted> {
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

fn side_class(lower: f64, upper: f64) -> u8 {
    if lower == upper {
        4
    } else {
        u8::from(lower.is_finite()) | u8::from(upper.is_finite()) << 1
    }
}
struct ConversionCache {
    hessian: sqp_craft::CsrPattern,
    matrix: sqp_craft::CsrPattern,
    sides: Vec<u8>,
    sources: Vec<(usize, f64)>,
    data: Converted,
}
impl ConversionCache {
    fn matches(&self, q: &QpModel) -> bool {
        self.hessian == q.hessian_lower.pattern
            && self.matrix == q.matrix.pattern
            && self.sides.len() == q.m() + q.n()
            && self.sides.iter().copied().eq(q
                .rows
                .lower
                .iter()
                .zip(&q.rows.upper)
                .chain(q.variables.lower.iter().zip(&q.variables.upper))
                .map(|(&l, &u)| side_class(l, u)))
    }
    fn new(q: &QpModel) -> Result<Self> {
        let data = convert(q)?;
        let mut sources = Vec::with_capacity(data.a.nzval.len());
        for j in 0..q.n() {
            for k in data.a.colptr[j]..data.a.colptr[j + 1] {
                let (variable, i, dual_sign) = data.mapping[data.a.rowval[k]];
                let source = if variable {
                    usize::MAX
                } else {
                    q.matrix.pattern.indptr[i]
                        + q.matrix
                            .pattern
                            .row(i)
                            .binary_search(&j)
                            .expect("validated conversion source")
                };
                sources.push((source, -dual_sign));
            }
        }
        Ok(Self {
            hessian: q.hessian_lower.pattern.clone(),
            matrix: q.matrix.pattern.clone(),
            sides: q
                .rows
                .lower
                .iter()
                .zip(&q.rows.upper)
                .chain(q.variables.lower.iter().zip(&q.variables.upper))
                .map(|(&l, &u)| side_class(l, u))
                .collect(),
            sources,
            data,
        })
    }
    fn refresh(&mut self, q: &QpModel) {
        self.data.p.nzval.clone_from(&q.hessian_lower.values);
        for (value, &(source, scale)) in self.data.a.nzval.iter_mut().zip(&self.sources) {
            *value = scale
                * if source == usize::MAX {
                    1.
                } else {
                    q.matrix.values[source]
                };
        }
        for (value, &(variable, i, sign)) in self.data.b.iter_mut().zip(&self.data.mapping) {
            let bounds = if variable { &q.variables } else { &q.rows };
            *value = if sign > 0. {
                -bounds.lower[i]
            } else {
                bounds.upper[i]
            };
        }
    }
}

impl ClarabelQp {
    fn engine_settings(
        &self,
        began: Instant,
        equilibrated: bool,
        accuracy_scale: f64,
    ) -> DefaultSettings<f64> {
        let tol = (self.settings.tolerance / accuracy_scale).max(f64::MIN_POSITIVE);
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
            equilibrate_enable: equilibrated,
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
        self.last.retry_reason = Some(match self.last.retry_reason.take() {
            Some(previous) => format!("{previous}; {reason}"),
            None => reason,
        });
    }
}

impl QpBackend for ClarabelQp {
    fn set_threads(&mut self, threads: usize) -> Result<()> {
        sqp_craft_parallel::validate_threads(threads).map_err(Error::Unsupported)?;
        self.threads = threads;
        Ok(())
    }

    fn configure(&mut self, options: &QpOptions) -> Result<()> {
        options.validate()?;
        if options.elastic_weight.is_some() {
            return Err(Error::Unsupported(
                "the Clarabel adapter does not solve elastic QPs".into(),
            ));
        }
        self.settings = options.clone();
        if !options.reuse_workspace {
            self.workspace = None;
        }
        Ok(())
    }
    fn supports_large_sparse(&self) -> bool {
        true
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
        let pd = if model.n() > DENSE_PD_LIMIT {
            let h = &model.hessian_lower;
            let n = model.n();
            if h.pattern.nnz() == n.saturating_mul(n + 1) / 2 {
                return Err(Error::Unsupported(
                    "Clarabel limits dense Hessians to 256 variables".into(),
                ));
            }
            let fixed = n
                .saturating_add(model.m())
                .saturating_mul(192)
                .saturating_add(model.matrix.pattern.nnz().saturating_mul(64));
            sparse::check_memory(fixed, sparse::MAX_MEMORY_BYTES)?;
            if !self.conversion.as_ref().is_some_and(|c| c.matches(model)) {
                sparse::kkt_fill(
                    &h.pattern,
                    &model.matrix.pattern,
                    &model.rows,
                    &model.variables,
                    sparse::MAX_MEMORY_BYTES - fixed,
                )?;
            }
            if !self
                .sparse_pd
                .as_ref()
                .is_some_and(|pd| pd.pattern == h.pattern)
            {
                self.sparse_pd = Some(SparsePd {
                    symbolic: sparse::Symbolic::of(&h.pattern, sparse::MAX_MEMORY_BYTES)?,
                    pattern: h.pattern.clone(),
                    factor: Vec::new(),
                });
            }
            let pd = self.sparse_pd.as_mut().expect("sparse Hessian pattern");
            pd.symbolic
                .positive_definite(&h.pattern, &h.values, &mut pd.factor)
        } else {
            positive_definite(model, &mut self.pd)
        };
        if !pd {
            // Nothing was solved or changed: keep both caches, so the
            // caller's re-solve (with a reset Hessian) reuses them exactly
            // as if this call had not happened.
            self.workspace = previous;
            self.last.conversion_seconds = t.elapsed().as_secs_f64();
            return Ok(QpOutcome::NotPositiveDefinite);
        }
        let converted =
            if let Some(mut cached) = self.conversion.take().filter(|c| c.matches(model)) {
                cached.refresh(model);
                Ok(cached)
            } else {
                ConversionCache::new(model)
            };
        self.last.conversion_seconds = t.elapsed().as_secs_f64();
        let conversion = converted?;
        let data = &conversion.data;
        let mut reusable = previous.filter(|w| {
            self.settings.reuse_workspace
                && w.key.matches(data)
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
        // At most three attempts: update, fresh equilibrated, then fresh
        // unequilibrated if the original-coordinate check still rejects it.
        let mut unequilibrated_retry = false;
        loop {
            if let Some(outcome) = self.stop(began, cancel) {
                return Ok(outcome);
            }
            let reused = reusable.is_some();
            let original_scale = data
                .b
                .iter()
                .chain(&model.gradient)
                .fold(1.0_f64, |a, b| a.max(b.abs()));
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
                    self.engine_settings(
                        began,
                        !unequilibrated_retry,
                        if unequilibrated_retry {
                            original_scale
                        } else {
                            1.0
                        },
                    ),
                );
                self.last.setup_seconds += t.elapsed().as_secs_f64();
                self.last.setups += 1;
                Workspace {
                    solver: solver.map_err(|e| Error::Backend(format!("Clarabel setup: {e}")))?,
                    key: data.structure(),
                    equilibrated: !unequilibrated_retry,
                }
            };
            if let Some(outcome) = self.stop(began, cancel) {
                return Ok(outcome);
            }
            // Settings (especially remaining time/iterations) and callback are
            // refreshed even for a cached workspace and before every retry.
            workspace
                .solver
                .update_settings(self.engine_settings(
                    began,
                    workspace.equilibrated,
                    if workspace.equilibrated {
                        1.0
                    } else {
                        original_scale
                    },
                ))
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
            self.last.factor_nnz = Some(workspace.solver.info.linsolver.nnzL);
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
                &mut self.checks,
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
            let equilibrated = workspace.equilibrated;
            if outcome.is_err() && matches!(raw, SolverStatus::Solved | SolverStatus::AlmostSolved)
            {
                let candidate = candidate(&workspace.solver, model, &data.mapping);
                // Keep only conversion data while allocating the bounded
                // polish factor, rather than holding two solver factors.
                drop(workspace);
                let t = Instant::now();
                let before_checks = self.last.check_seconds;
                let polished = polish::solve(
                    model,
                    &candidate,
                    &self.settings,
                    began,
                    cancel,
                    &mut self.last,
                    &mut self.checks,
                );
                self.last.solve_seconds += (t.elapsed().as_secs_f64()
                    - (self.last.check_seconds - before_checks))
                    .max(0.0);
                if let Some(outcome) = self.stop(began, cancel) {
                    return Ok(outcome);
                }
                if let Some(solution) = polished {
                    self.cold_retry(
                        "active-set polish accepted by original-coordinate checks".into(),
                    );
                    self.conversion = Some(conversion);
                    return Ok(QpOutcome::Optimal(solution));
                }
                if self.last.iterations >= u64::from(self.settings.max_iterations) {
                    self.last.limit_reason = Some("iterations".into());
                    return Ok(QpOutcome::LimitReached(None));
                }
            } else {
                if matches!(outcome, Ok(QpOutcome::Optimal(_)))
                    && self.settings.reuse_workspace
                    && workspace.solver.is_data_update_allowed()
                {
                    self.workspace = Some(workspace);
                }
            }
            if !reused && equilibrated && outcome.is_err() {
                // Equilibrated stopping measures can understate residuals in
                // the original model. Retry the identical QP without that
                // transformation; never weaken the independent acceptance.
                self.cold_retry(format!("unequilibrated retry after {outcome:?}"));
                if let Some(outcome) = self.stop(began, cancel) {
                    return Ok(outcome);
                }
                if self.last.iterations >= u64::from(self.settings.max_iterations) {
                    self.last.limit_reason = Some("iterations".into());
                    return Ok(QpOutcome::LimitReached(None));
                }
                unequilibrated_retry = true;
                continue;
            }
            if matches!(outcome, Ok(QpOutcome::Optimal(_))) {
                self.conversion = Some(conversion);
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

fn candidate(
    solver: &DefaultSolver<f64>,
    model: &QpModel,
    mapping: &[(bool, usize, f64)],
) -> QpSolution {
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
    QpSolution {
        step: solver.solution.x.clone(),
        multipliers,
        active_set: None,
        iterations: u64::from(solver.solution.iterations),
    }
}

fn extract(
    solver: &DefaultSolver<f64>,
    model: &QpModel,
    mapping: &[(bool, usize, f64)],
    tolerance: f64,
    workspace: &mut QpResidualWorkspace,
) -> Result<QpOutcome> {
    match solver.solution.status {
        SolverStatus::PrimalInfeasible => Ok(QpOutcome::Infeasible),
        SolverStatus::DualInfeasible => Ok(QpOutcome::Unbounded),
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let solution = candidate(solver, model, mapping);
            let check = model.accepts_with_workspace(&solution, tolerance, workspace)?;
            if !check.accepted {
                return Err(Error::Backend(format!(
                    "Clarabel solution failed the independent check (scale {:.3e}): {:?}",
                    check.scale, check.residuals
                )));
            }
            // SQP rounds tolerance-sized bound overshoots before using a step.
            // Certify that version too: clipping can amplify row/H residuals.
            let needs_snap = solution
                .step
                .iter()
                .enumerate()
                .any(|(j, &v)| v < model.variables.lower[j] || v > model.variables.upper[j]);
            if needs_snap {
                let mut rounded = solution.clone();
                for (j, v) in rounded.step.iter_mut().enumerate() {
                    *v = v.clamp(model.variables.lower[j], model.variables.upper[j]);
                }
                let check = model.accepts_with_workspace(&rounded, tolerance, workspace)?;
                if !check.accepted {
                    return Err(Error::Backend(format!(
                        "Clarabel rounded step failed the independent check (scale {:.3e}): {:?}",
                        check.scale, check.residuals
                    )));
                }
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

    fn power_qp(data: &str) -> QpModel {
        let tokens: Vec<&str> = data
            .lines()
            .filter(|l| !l.starts_with('#'))
            .flat_map(str::split_whitespace)
            .collect();
        let mut cursor = 0;
        let mut take = |count: usize| {
            let v = tokens[cursor..cursor + count]
                .iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            cursor += count;
            v
        };
        let dims = take(2);
        let (n, m) = (dims[0] as usize, dims[1] as usize);
        let gradient = take(n);
        let variables = Bounds::new(take(n), take(n));
        let rows = Bounds::new(take(m), take(m));
        let mut matrix = |rows, cols| {
            let nnz = take(1)[0] as usize;
            CsrMatrix {
                pattern: CsrPattern {
                    rows,
                    cols,
                    indptr: take(rows + 1).iter().map(|v| *v as usize).collect(),
                    indices: take(nnz).iter().map(|v| *v as usize).collect(),
                },
                values: take(nnz),
            }
        };
        let hessian_lower = matrix(n, n);
        let matrix = matrix(m, n);
        QpModel {
            hessian_lower,
            gradient,
            variables,
            matrix,
            rows,
        }
    }

    fn captured_power_qp() -> QpModel {
        power_qp(include_str!("../tests/data/epigrids-reference-step.txt"))
    }

    #[test]
    fn later_power_step_recovers_from_a_wrong_active_bound() {
        let model = power_qp(include_str!("../tests/data/epigrids-later-step.txt"));
        let mut backend = ClarabelQp::default();
        backend
            .configure(&QpOptions {
                tolerance: 1e-10,
                acceptance_tolerance: 1e-10,
                max_iterations: 200,
                ..Default::default()
            })
            .unwrap();
        let QpOutcome::Optimal(solution) = backend
            .solve(&model, None, &CancellationToken::default())
            .unwrap()
        else {
            panic!("later QP not accepted")
        };
        assert!(model.accepts(&solution, 1e-10).unwrap().accepted);
        assert!(backend.report().iterations <= 200);
        assert!(
            backend
                .report()
                .retry_reason
                .as_ref()
                .unwrap()
                .contains("active-set polish accepted")
        );
    }

    #[test]
    fn captured_power_step_meets_original_coordinate_checks() {
        let model = captured_power_qp();
        let mut backend = ClarabelQp::default();
        backend
            .configure(&QpOptions {
                tolerance: 1e-10,
                acceptance_tolerance: 1e-10,
                max_iterations: 200,
                reuse_workspace: true,
                ..Default::default()
            })
            .unwrap();
        let outcome = backend
            .solve(&model, None, &CancellationToken::default())
            .unwrap();
        let QpOutcome::Optimal(solution) = outcome else {
            panic!("{outcome:?}");
        };
        assert!(model.accepts(&solution, 1e-10).unwrap().accepted);
        assert!(backend.report().iterations <= 200);
        assert_eq!(backend.report().cold_retries, 0);
        assert!(
            backend
                .report()
                .retry_reason
                .as_ref()
                .unwrap()
                .contains("active-set polish accepted")
        );
        assert!(backend.workspace.is_none());
        let QpOutcome::Optimal(repeated) = backend
            .solve(&model, None, &CancellationToken::default())
            .unwrap()
        else {
            panic!("cached unscaled solve failed");
        };
        assert!(model.accepts(&repeated, 1e-10).unwrap().accepted);
        assert_eq!(backend.report().updates, 0);
        assert_eq!(backend.report().setups, 1);
    }

    #[test]
    fn power_accuracy_retry_shares_the_original_iteration_budget() {
        let model = captured_power_qp();
        let mut backend = ClarabelQp::default();
        backend
            .configure(&QpOptions {
                tolerance: 1e-10,
                acceptance_tolerance: 1e-10,
                max_iterations: 20,
                ..Default::default()
            })
            .unwrap();
        assert!(matches!(
            backend
                .solve(&model, None, &CancellationToken::default())
                .unwrap(),
            QpOutcome::LimitReached(_)
        ));
        assert!(backend.report().iterations <= 20);
        assert_eq!(backend.report().setups, 1);
        assert_eq!(backend.report().cold_retries, 0);
    }

    #[test]
    fn almost_solved_requires_the_same_independent_certificate() {
        let (_, q) = reusable();
        let data = convert(&q).unwrap();
        let mut solver = DefaultSolver::new(
            &data.p,
            &q.gradient,
            &data.a,
            &data.b,
            &data.cones,
            DefaultSettings {
                verbose: false,
                ..Default::default()
            },
        )
        .unwrap();
        solver.solution.status = SolverStatus::AlmostSolved;
        solver.solution.x = vec![1.0];
        solver.solution.z.fill(0.0);
        assert!(matches!(
            extract(
                &solver,
                &q,
                &data.mapping,
                1e-10,
                &mut QpResidualWorkspace::default()
            )
            .unwrap(),
            QpOutcome::Optimal(_)
        ));
        solver.solution.x[0] += 1e-5;
        assert!(
            extract(
                &solver,
                &q,
                &data.mapping,
                1e-10,
                &mut QpResidualWorkspace::default()
            )
            .is_err()
        );
    }

    #[test]
    fn polish_corrects_wrong_active_guesses_and_honors_cancellation() {
        let q = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern {
                    rows: 2,
                    cols: 2,
                    indptr: vec![0, 1, 2],
                    indices: vec![0, 1],
                },
                values: vec![1.0, 1.0],
            },
            gradient: vec![-1.0, -1.0],
            variables: Bounds::new(vec![0.0, 0.0], vec![2.0, 2.0]),
            matrix: CsrMatrix {
                pattern: CsrPattern::dense(0, 2),
                values: vec![],
            },
            rows: Bounds::free(0),
        };
        let mut candidate = QpSolution {
            step: vec![2.0, 1.0],
            multipliers: Multipliers {
                constraints: vec![],
                variables: vec![-1.0, 0.0],
            },
            active_set: None,
            iterations: 0,
        };
        let mut report = QpReport::default();
        let mut checks = QpResidualWorkspace::default();
        let options = QpOptions::default();
        let corrected = polish::solve(
            &q,
            &candidate,
            &options,
            Instant::now(),
            &CancellationToken::default(),
            &mut report,
            &mut checks,
        )
        .unwrap();
        assert_eq!(corrected.step, vec![1.0, 1.0]);
        assert!(
            q.accepts(&corrected, options.acceptance_tolerance)
                .unwrap()
                .accepted
        );
        assert!(report.iterations <= 27);
        let one = QpOptions {
            max_iterations: 1,
            ..options.clone()
        };
        let mut limited = QpReport::default();
        assert!(
            polish::solve(
                &q,
                &candidate,
                &one,
                Instant::now(),
                &CancellationToken::default(),
                &mut limited,
                &mut checks
            )
            .is_none()
        );
        assert_eq!(limited.iterations, 1);
        candidate.step[0] = 1.0;
        candidate.multipliers.variables[0] = 0.0;
        let token = CancellationToken::default();
        token.cancel();
        let mut report = QpReport::default();
        assert!(
            polish::solve(
                &q,
                &candidate,
                &options,
                Instant::now(),
                &token,
                &mut report,
                &mut checks
            )
            .is_none()
        );
        assert_eq!(report.iterations, 0);
        let options = QpOptions {
            max_iterations: 0,
            ..options
        };
        assert!(
            polish::solve(
                &q,
                &candidate,
                &options,
                Instant::now(),
                &CancellationToken::default(),
                &mut report,
                &mut checks
            )
            .is_none()
        );
        assert_eq!(report.iterations, 0);
    }

    #[test]
    fn polish_never_ignores_an_inactive_row() {
        let (_, mut q) = reusable();
        q.matrix = CsrMatrix {
            pattern: CsrPattern::dense(1, 1),
            values: vec![1.0],
        };
        q.rows = Bounds::new(vec![f64::NEG_INFINITY], vec![0.5]);
        let candidate = QpSolution {
            step: vec![0.25],
            multipliers: Multipliers {
                variables: vec![0.0],
                constraints: vec![0.0],
            },
            active_set: None,
            iterations: 0,
        };
        assert!(
            polish::solve(
                &q,
                &candidate,
                &QpOptions::default(),
                Instant::now(),
                &CancellationToken::default(),
                &mut QpReport::default(),
                &mut QpResidualWorkspace::default()
            )
            .is_none()
        );
    }

    #[test]
    fn cached_unequilibrated_workspace_keeps_its_mode() {
        let (mut backend, q) = reusable();
        let data = convert(&q).unwrap();
        let solver = DefaultSolver::new(
            &data.p,
            &q.gradient,
            &data.a,
            &data.b,
            &data.cones,
            DefaultSettings {
                verbose: false,
                equilibrate_enable: false,
                ..Default::default()
            },
        )
        .unwrap();
        backend.workspace = Some(Workspace {
            solver,
            key: data.structure(),
            equilibrated: false,
        });
        let QpOutcome::Optimal(solution) = backend
            .solve(&q, None, &CancellationToken::default())
            .unwrap()
        else {
            panic!("unscaled cache failed")
        };
        assert!(
            q.accepts(&solution, backend.settings.acceptance_tolerance)
                .unwrap()
                .accepted
        );
        assert_eq!(backend.report().updates, 1);
        assert_eq!(backend.report().setups, 0);
        assert!(!backend.workspace.as_ref().unwrap().equilibrated);
    }

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

    #[test]
    fn conversion_cache_refreshes_every_value_and_checks_structure() {
        let (_, mut q) = reusable();
        let mut cached = ConversionCache::new(&q).unwrap();
        for k in 1..10 {
            q.hessian_lower.values[0] = 1. + k as f64 / 10.;
            q.gradient[0] = -(k as f64);
            q.variables.lower[0] = -(k as f64);
            q.variables.upper[0] = k as f64 + 2.;
            assert!(cached.matches(&q));
            cached.refresh(&q);
            let fresh = convert(&q).unwrap();
            assert_eq!(cached.data.p.nzval, fresh.p.nzval);
            assert_eq!(cached.data.a.nzval, fresh.a.nzval);
            assert_eq!(cached.data.b, fresh.b);
            assert_eq!(cached.data.mapping, fresh.mapping);
        }
        q.variables.lower[0] = f64::NEG_INFINITY;
        assert!(!cached.matches(&q));
        q.variables.lower[0] = q.variables.upper[0];
        assert!(!cached.matches(&q));
        q.variables.lower[0] = -9.;
        q.matrix.pattern = sqp_craft::CsrPattern::dense(1, 1);
        q.matrix.values = vec![1.];
        q.rows = sqp_craft::Bounds::fixed(vec![0.]);
        assert!(!cached.matches(&q));
    }

    #[test]
    fn conversion_cache_tracks_signed_sparse_jacobian_and_rhs_changes() {
        let mut q = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern {
                    rows: 2,
                    cols: 2,
                    indptr: vec![0, 1, 3],
                    indices: vec![0, 0, 1],
                },
                values: vec![2., 0.5, 2.],
            },
            gradient: vec![0.; 2],
            variables: Bounds::new(vec![-2., f64::NEG_INFINITY], vec![3., 4.]),
            matrix: CsrMatrix {
                pattern: CsrPattern::dense(2, 2),
                values: vec![0., -2., 3., 4.],
            },
            rows: Bounds::new(vec![-3., 2.], vec![5., 2.]),
        };
        let mut cache = ConversionCache::new(&q).unwrap();
        let pointers = (
            cache.data.a.rowval.as_ptr(),
            cache.data.a.colptr.as_ptr(),
            cache.data.a.nzval.as_ptr(),
        );
        for k in 1..8 {
            q.matrix.values = vec![(k - 1) as f64, -(k as f64), 3. * k as f64, 4. - k as f64];
            q.rows.lower[0] = -3. * k as f64;
            q.rows.upper[0] = 5. * k as f64;
            q.rows.lower[1] = 2. * k as f64;
            q.rows.upper[1] = q.rows.lower[1];
            assert!(cache.matches(&q));
            cache.refresh(&q);
            let fresh = convert(&q).unwrap();
            for (a, b) in [
                (&cache.data.p.nzval, &fresh.p.nzval),
                (&cache.data.a.nzval, &fresh.a.nzval),
                (&cache.data.b, &fresh.b),
            ] {
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            assert_eq!(
                pointers,
                (
                    cache.data.a.rowval.as_ptr(),
                    cache.data.a.colptr.as_ptr(),
                    cache.data.a.nzval.as_ptr()
                )
            );
        }
    }

    #[test]
    fn dominance_gate_and_cholesky_fallback_cover_boundary_cases() {
        let mut q = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern {
                    rows: 2,
                    cols: 2,
                    indptr: vec![0, 1, 3],
                    indices: vec![0, 0, 1],
                },
                values: vec![1., 0., 1.],
            },
            gradient: vec![0.; 2],
            variables: Bounds::free(2),
            matrix: CsrMatrix {
                pattern: CsrPattern::dense(0, 2),
                values: vec![],
            },
            rows: Bounds::free(0),
        };
        for (values, expected) in [
            (vec![1., 0., 2.], true),
            (vec![0.1, 0.2, 2.], true),
            (vec![1., 2., 1.], false),
            (vec![1., 1., 1.], false),
            (vec![1., 0.9999999999999999, 1.], true),
        ] {
            q.hessian_lower.values = values;
            assert_eq!(positive_definite(&q, &mut PdScratch::default()), expected);
        }
    }

    #[test]
    fn a_hessian_that_is_not_positive_definite_is_an_outcome_and_keeps_both_caches() {
        let (mut b, mut q) = reusable();
        assert!(b.workspace.is_some() && b.conversion.is_some());
        q.hessian_lower.values[0] = -1.0;
        assert_eq!(
            b.solve(&q, None, &CancellationToken::default()).unwrap(),
            QpOutcome::NotPositiveDefinite
        );
        assert!(b.workspace.is_some() && b.conversion.is_some());
        assert_eq!(
            (b.last.setups, b.last.updates, b.last.iterations),
            (0, 0, 0)
        );
        q.hessian_lower.values[0] = 2.0;
        assert!(matches!(
            b.solve(&q, None, &CancellationToken::default()).unwrap(),
            QpOutcome::Optimal(_)
        ));
        assert_eq!((b.last.setups, b.last.updates), (0, 1));
    }

    #[test]
    fn the_positive_definiteness_scratch_follows_dimension_and_pattern_changes() {
        let mut scratch = PdScratch::default();
        for n in [3, 1, 6, 6, 2] {
            for (dense, indefinite) in [(true, false), (true, true), (false, false), (false, true)]
            {
                let (mut indptr, mut indices, mut values) = (vec![0], vec![], vec![]);
                for i in 0..n {
                    for j in 0..=i {
                        if dense || j == i || j + 1 == i {
                            indices.push(j);
                            // 1 on the diagonal, 0.9 next to it: never
                            // diagonally dominant for n > 2, positive definite
                            // for small n and indefinite at n = 6, so the
                            // Cholesky paths decide; compared with a reference.
                            values.push(if i == j {
                                1.0
                            } else if j + 1 == i {
                                0.9
                            } else {
                                0.0
                            });
                        }
                    }
                    indptr.push(indices.len());
                }
                if indefinite {
                    let last = values.len() - 1;
                    values[last] = -1.0;
                }
                let q = QpModel {
                    hessian_lower: CsrMatrix {
                        pattern: CsrPattern {
                            rows: n,
                            cols: n,
                            indptr,
                            indices,
                        },
                        values,
                    },
                    gradient: vec![0.; n],
                    variables: Bounds::free(n),
                    matrix: CsrMatrix {
                        pattern: CsrPattern::dense(0, n),
                        values: vec![],
                    },
                    rows: Bounds::free(0),
                };
                let mut reference = DMatrix::zeros(n, n);
                for i in 0..n {
                    for k in
                        q.hessian_lower.pattern.indptr[i]..q.hessian_lower.pattern.indptr[i + 1]
                    {
                        let j = q.hessian_lower.pattern.indices[k];
                        reference[(i, j)] = q.hessian_lower.values[k];
                        reference[(j, i)] = q.hessian_lower.values[k];
                    }
                }
                assert_eq!(
                    positive_definite(&q, &mut scratch),
                    reference.cholesky().is_some(),
                    "n {n}, dense {dense}, indefinite {indefinite}"
                );
            }
        }
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
