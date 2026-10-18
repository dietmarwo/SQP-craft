//! Dense dual active-set QP backend: the method of Goldfarb and Idnani (1983).
//!
//! The QP of [`QpModel`] is solved on a Cholesky factor `H = L Lᵀ`, which
//! also decides positive definiteness exactly: a failed factorization is
//! [`QpOutcome::NotPositiveDefinite`]. The method keeps `J = L⁻ᵀ Q` and the
//! upper-triangular `R` of `L⁻¹ N = Q [R; 0]`, where `N` holds the normals of
//! the working set. Each change of the working set updates `J` and `R` in
//! `O(n²)` operations: an entering constraint by one Householder reflector on
//! the columns of `J` outside the working set, a leaving one by Givens
//! rotations.
//!
//! The iterates stay dual feasible (`x` minimizes the objective on the
//! working set with nonnegative inequality multipliers) and become primal
//! feasible at the end, so the method needs no feasible starting point, and
//! a working set from [`QpWarmStart::active_set`] is a valid start. That
//! working set is the main reason for this backend: near SQP convergence the
//! active set of consecutive QPs rarely changes.
//!
//! Every bound is a constraint `σ aᵀp ≥ σ b`: the lower side (`σ = 1`), the
//! upper side (`σ = -1`) or an equality (both bounds equal; never dropped,
//! multiplier of either sign). Multipliers are reported in the shared
//! convention `H p + g = Aᵀλ + z`, and the final working set is reported as
//! [`QpSolution::active_set`].
//!
//! [`QpReport::iterations`] counts the working-set changes of a call after the
//! initial working set (equalities and warm-start constraints). The budget
//! [`QpOptions::max_iterations`] applies to the changes that are not the
//! first entry of a constraint in the attempt, that is to drops and
//! re-entries: a cold start needs one entry per active inequality, which can
//! exceed any fixed number for large `n + m`, while repeated changes are
//! what a degenerate or badly conditioned QP costs. A warm-started solve
//! that ends infeasible or fails the independent check [`QpModel::accepts`]
//! is repeated once from a cold start within the same budget.
//!
//! Dense storage: three working `n × n` matrices, plus an exact Hessian key.
//! Repeated Hessians may also retain one unrotated inverse matrix. All caches
//! are backend-local, owned, and invalidated by exact structure/value changes.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod dense;

use sqp_craft::qp::{
    ActiveSet, Activity, QpBackend, QpModel, QpOptions, QpOutcome, QpReport, QpResidualWorkspace,
    QpSolution, QpWarmStart,
};
use sqp_craft::{CancellationToken, Error, Multipliers, Result};
use std::time::Instant;

/// Engine name in [`QpReport::backend`].
pub const BACKEND: &str = concat!("goldfarb-idnani ", env!("CARGO_PKG_VERSION"));

/// A constraint is linearly dependent on the working set when the part of its
/// transformed normal `d = Jᵀn` outside the working set, `‖d₂‖`, is at most
/// `DEPENDENCE · ε · √n · ‖J‖_F · ‖n‖`, the rounding level of `d`. The
/// Frobenius norm of `J = L⁻ᵀQ` is computed once per attempt, since the
/// orthogonal updates keep it.
///
/// The test is absolute on purpose. With a nearly singular `H`, `L⁻¹` maps
/// independent normals to nearly parallel ones, so `‖d₂‖ / ‖d‖` can be tiny
/// for a constraint that is far from dependent; the large primal step that
/// follows is then the right one.
const DEPENDENCE: f64 = 1e2;
/// Entries of the multiplier direction `R⁻¹d₁` below
/// `NOISE · ε · √n · κ · max|R⁻¹d₁|` are rounding and taken as zero. `κ` is
/// the ratio of the largest to the smallest diagonal entry of `L`, a cheap
/// lower estimate of its condition number.
const NOISE: f64 = 1e3;
/// Cap on the relative noise level.
const MAX_NOISE: f64 = 1e-6;
/// Iterative-refinement steps whenever `x` and `u` are recomputed.
const REFINEMENT_STEPS: usize = 1;

/// Dense dual active-set QP backend. Configure it through
/// [`QpBackend::configure`]; [`QpOptions::reuse_workspace`] has no effect,
/// since buffers and exactly matching factors are always reused. Factors are
/// reused only under a complete bitwise Hessian key; independent acceptance
/// checks remain mandatory.
#[derive(Clone, Debug, Default)]
pub struct ActiveSetQp {
    settings: QpOptions,
    last: QpReport,
    work: Work,
    norms: Vec<f64>,
    checks: QpResidualWorkspace,
    execution: sqp_craft_parallel::Execution,
}

impl QpBackend for ActiveSetQp {
    fn set_threads(&mut self, threads: usize) -> Result<()> {
        if self.execution.threads() != threads {
            self.execution =
                sqp_craft_parallel::Execution::new(threads).map_err(Error::Unsupported)?;
        }
        Ok(())
    }

    fn configure(&mut self, options: &QpOptions) -> Result<()> {
        options.validate()?;
        self.settings = options.clone();
        Ok(())
    }

    fn report(&self) -> QpReport {
        self.last.clone()
    }

    fn supports_elastic(&self) -> bool {
        true
    }

    /// Uses [`QpWarmStart::active_set`] and ignores a starting step or
    /// multipliers, which a dual method cannot exploit.
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
        }
        let hard = self.settings.hard_rows.len();
        if hard != 0 && hard != model.m() {
            return Err(Error::InvalidInput(format!(
                "QP hard rows: {hard} flags for {} rows",
                model.m()
            )));
        }
        self.last = QpReport {
            backend: BACKEND.into(),
            ..Default::default()
        };
        let mut budget = Budget {
            began: Instant::now(),
            time_limit: self.settings.time_limit_seconds,
            cancel,
            max: u64::from(self.settings.max_iterations),
            changes: 0,
            charged: 0,
        };
        let outcome = self.run(model, warm.and_then(|w| w.active_set.as_ref()), &mut budget);
        self.last.iterations = budget.changes;
        self.last.raw_status = Some(
            match &outcome {
                Ok(QpOutcome::Optimal(_)) => "optimal",
                Ok(QpOutcome::Infeasible) => "infeasible",
                Ok(QpOutcome::LimitReached(_)) => "limit",
                Ok(QpOutcome::Cancelled) => "cancelled",
                Ok(QpOutcome::NotPositiveDefinite) => "not positive definite",
                Ok(_) => "other",
                Err(_) => "failed",
            }
            .into(),
        );
        outcome
    }
}

impl ActiveSetQp {
    fn run(
        &mut self,
        model: &QpModel,
        mut warm: Option<&ActiveSet>,
        budget: &mut Budget,
    ) -> Result<QpOutcome> {
        if let Err(stop) = budget.check() {
            return Ok(self.stopped(stop));
        }
        let t = Instant::now();
        self.work.load(
            model,
            self.settings.tolerance,
            self.settings.elastic_weight,
            &self.settings.hard_rows,
        );
        self.norms.clear();
        self.norms.extend((0..model.m()).map(|i| {
            if model.rows.lower[i] == f64::NEG_INFINITY && model.rows.upper[i] == f64::INFINITY {
                0.0
            } else {
                let range = model.matrix.pattern.indptr[i]..model.matrix.pattern.indptr[i + 1];
                dense::norm(&model.matrix.values[range])
            }
        }));
        self.last.conversion_seconds = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let definite = self.work.factor();
        if self.work.factor_reused {
            self.last.updates = 1;
            self.last.update_seconds = t.elapsed().as_secs_f64();
        } else {
            self.last.setups = 1;
            self.last.setup_seconds = t.elapsed().as_secs_f64();
        }
        if !definite {
            return Ok(QpOutcome::NotPositiveDefinite);
        }
        if let Some(outcome) = self.direct(model, budget)? {
            return Ok(outcome);
        }
        let sources = Sources {
            model,
            norms: &self.norms,
        };
        loop {
            let t = Instant::now();
            let end = self.work.solve(&sources, warm, budget);
            self.last.solve_seconds += t.elapsed().as_secs_f64();
            let reason = match end {
                Err(Stop::Infeasible) if warm.is_some() => {
                    "the warm-started solve reported infeasibility".to_string()
                }
                Err(stop) => return Ok(self.stopped(stop)),
                Ok(()) => {
                    let t = Instant::now();
                    let solution = self.work.solution(&sources, budget.changes);
                    let tolerance = self.settings.acceptance_tolerance;
                    let check = match self.settings.elastic_weight {
                        Some(weight) => model.accepts_elastic_with_hard_rows(
                            &solution,
                            tolerance,
                            weight,
                            &self.settings.hard_rows,
                            &mut self.checks,
                        )?,
                        None => {
                            model.accepts_with_workspace(&solution, tolerance, &mut self.checks)?
                        }
                    };
                    self.last.check_seconds += t.elapsed().as_secs_f64();
                    if check.accepted {
                        return Ok(QpOutcome::Optimal(solution));
                    }
                    let reason = format!(
                        "active-set solution failed the independent check (scale {:.3e}): {:?}",
                        check.scale, check.residuals
                    );
                    if warm.is_none() {
                        return Err(Error::Backend(reason));
                    }
                    reason
                }
            };
            self.last.retry_reason = Some(reason);
            self.last.cold_retries += 1;
            warm = None;
        }
    }

    /// A checked unconstrained/diagonal-box proposal; failure preserves the general fallback.
    fn direct(&mut self, model: &QpModel, budget: &Budget) -> Result<Option<QpOutcome>> {
        if self.work.weight.is_some() {
            return Ok(None);
        }
        let boxed = !self.work.bounded.is_empty();
        if boxed {
            let h = &model.hessian_lower;
            for i in 0..model.n() {
                for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                    if h.pattern.indices[k] != i && h.values[k] != 0.0 {
                        return Ok(None);
                    }
                }
            }
        }
        if let Err(stop) = budget.check() {
            return Ok(Some(self.stopped(stop)));
        }
        let t = Instant::now();
        let n = model.n();
        let mut variables = vec![0.0; n];
        let mut activity = vec![Activity::Inactive; n];
        if boxed {
            let h = &model.hessian_lower;
            for i in 0..n {
                // A positive-definite diagonal matrix has a positive stored diagonal.
                let diagonal = h.values[h.pattern.indptr[i + 1] - 1];
                let (lo, hi) = (model.variables.lower[i], model.variables.upper[i]);
                let value = (-model.gradient[i] / diagonal).clamp(lo, hi);
                self.work.x[i] = value;
                let side = if value == lo && lo.is_finite() {
                    Activity::AtLower
                } else if value == hi && hi.is_finite() {
                    Activity::AtUpper
                } else {
                    Activity::Inactive
                };
                if side != Activity::Inactive {
                    let mu = diagonal * value + model.gradient[i];
                    variables[i] = mu;
                    if mu != 0.0 || lo == hi {
                        activity[i] = side;
                    }
                }
            }
        } else {
            for (x, &g) in self.work.x.iter_mut().zip(&model.gradient) {
                *x = -g;
            }
            dense::solve_factored(&self.work.l, &mut self.work.x, n);
            model.hessian_product(&self.work.x, &mut self.work.rd);
            for (r, &g) in self.work.rd.iter_mut().zip(&model.gradient) {
                *r = -(*r + g);
            }
            dense::solve_factored(&self.work.l, &mut self.work.rd, n);
            dense::axpy(1.0, &self.work.rd, &mut self.work.x);
        }
        let solution = QpSolution {
            step: self.work.x.clone(),
            multipliers: Multipliers {
                variables,
                constraints: vec![0.0; model.m()],
            },
            active_set: Some(ActiveSet {
                variables: activity,
                rows: vec![Activity::Inactive; model.m()],
            }),
            iterations: 0,
        };
        self.last.solve_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let accepted = model
            .accepts_with_workspace(
                &solution,
                self.settings.acceptance_tolerance,
                &mut self.checks,
            )
            .is_ok_and(|v| v.accepted);
        self.last.check_seconds += t.elapsed().as_secs_f64();
        if let Err(stop) = budget.check() {
            return Ok(Some(self.stopped(stop)));
        }
        Ok(accepted.then_some(QpOutcome::Optimal(solution)))
    }

    fn stopped(&mut self, stop: Stop) -> QpOutcome {
        match stop {
            Stop::Infeasible => QpOutcome::Infeasible,
            Stop::Cancelled => QpOutcome::Cancelled,
            Stop::Limit(reason) => {
                self.last.limit_reason = Some(reason.into());
                QpOutcome::LimitReached(None)
            }
        }
    }
}

/// Why an attempt ended without a solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Infeasible,
    Limit(&'static str),
    Cancelled,
}

/// Working-set change budget, time limit and cancellation of one call.
struct Budget<'a> {
    began: Instant,
    time_limit: Option<f64>,
    cancel: &'a CancellationToken,
    /// Limit on `charged`.
    max: u64,
    /// All working-set changes after the initial working set.
    changes: u64,
    /// Drops and re-entries.
    charged: u64,
}

impl Budget<'_> {
    fn check(&self) -> std::result::Result<(), Stop> {
        if self.cancel.is_cancelled() {
            Err(Stop::Cancelled)
        } else if self
            .time_limit
            .is_some_and(|s| self.began.elapsed().as_secs_f64() >= s)
        {
            Err(Stop::Limit("time"))
        } else {
            Ok(())
        }
    }

    /// Records one change; `charged` for a drop or a re-entry.
    fn spend(&mut self, charged: bool) -> std::result::Result<(), Stop> {
        self.changes += 1;
        self.charged += u64::from(charged);
        if self.charged > self.max {
            Err(Stop::Limit("iterations"))
        } else {
            Ok(())
        }
    }
}

/// The `n + m` bound sources of a model: variables first, then rows.
struct Sources<'a> {
    model: &'a QpModel,
    norms: &'a [f64],
}

impl Sources<'_> {
    fn n(&self) -> usize {
        self.model.n()
    }

    fn bounds(&self, s: usize) -> (f64, f64) {
        let n = self.n();
        let (bounds, i) = if s < n {
            (&self.model.variables, s)
        } else {
            (&self.model.rows, s - n)
        };
        (bounds.lower[i], bounds.upper[i])
    }

    fn norm(&self, s: usize) -> f64 {
        if s < self.n() {
            1.0
        } else {
            self.norms[s - self.n()]
        }
    }

    fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let a = &self.model.matrix;
        let range = a.pattern.indptr[i]..a.pattern.indptr[i + 1];
        (&a.pattern.indices[range.clone()], &a.values[range])
    }

    /// `(aᵀx, Σ |a_i x_i|)`; the second value scales the rounding of the first.
    fn value(&self, s: usize, x: &[f64]) -> (f64, f64) {
        let n = self.n();
        if s < n {
            return (x[s], x[s].abs());
        }
        let (indices, values) = self.row(s - n);
        indices
            .iter()
            .zip(values)
            .fold((0.0, 0.0), |(v, m), (&i, &a)| {
                (v + a * x[i], m + (a * x[i]).abs())
            })
    }

    /// Activity without the magnitude used only by tolerance decisions.
    fn activity(&self, s: usize, x: &[f64]) -> f64 {
        if s < self.n() {
            return x[s];
        }
        let (indices, values) = self.row(s - self.n());
        indices
            .iter()
            .zip(values)
            .fold(0.0, |sum, (&i, &a)| sum + a * x[i])
    }

    /// `out += alpha a_s`.
    fn scatter(&self, s: usize, alpha: f64, out: &mut [f64]) {
        let n = self.n();
        if s < n {
            out[s] += alpha;
            return;
        }
        let (indices, values) = self.row(s - n);
        for (&i, &a) in indices.iter().zip(values) {
            out[i] += alpha * a;
        }
    }

    /// `d = σ Jᵀ a_s` for column-major `j`.
    fn project(&self, s: usize, sigma: f64, j: &[f64], d: &mut [f64]) {
        let n = self.n();
        if s < n {
            for (k, dk) in d.iter_mut().enumerate() {
                *dk = sigma * j[k * n + s];
            }
            return;
        }
        let (indices, values) = self.row(s - n);
        if indices.len() == 1 {
            let (i, a) = (indices[0], values[0]);
            for (k, dk) in d.iter_mut().enumerate() {
                *dk = sigma * (0.0 + a * j[k * n + i]);
            }
            return;
        }
        for (dk, column) in d.iter_mut().zip(j.chunks_exact(n)) {
            let sum: f64 = indices
                .iter()
                .zip(values)
                .map(|(&i, &a)| a * column[i])
                .sum();
            *dk = sigma * sum;
        }
    }
}

/// One oriented constraint `σ aᵀp ≥ rhs` (`rhs = σ b`) with multiplier
/// bounds `low <= u <= high`: `[0, ∞)` for an inequality, `(−∞, ∞)` for an
/// equality, and `[0, γ]` or `[−γ, γ]` for rows of an elastic QP with weight
/// `γ`. A multiplier that reaches a finite bound leaves the working set; at
/// `±γ` the row becomes elastic (see [`Work::elastic`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Member {
    source: usize,
    sigma: f64,
    rhs: f64,
    low: f64,
    high: f64,
}

/// Factors and iterate of the dual method, reused across calls.
#[derive(Clone, Debug, Default)]
struct Work {
    n: usize,
    /// Owned exact input key for a successful factorization.
    hessian_key: Option<sqp_craft::CsrMatrix>,
    factor_valid: bool,
    factor_reused: bool,
    /// Unrotated inverse, promoted only after confirmed cross-call H reuse.
    initial_j: Vec<f64>,
    initial_dependence: f64,
    initial_valid: bool,
    /// Cholesky factor `L` in the lower triangle (column-major).
    l: Vec<f64>,
    /// `J = L⁻ᵀ Q`: columns `0..q` span the working set, `q..n` its complement.
    j: Vec<f64>,
    /// Upper-triangular `R`, `q × q`, in the leading columns (stride `n`).
    r: Vec<f64>,
    q: usize,
    x: Vec<f64>,
    /// `Jᵀ n⁺` of the constraint being added.
    d: Vec<f64>,
    /// `R⁻¹ d₁`, the multiplier change per unit step.
    y: Vec<f64>,
    /// Reflector scratch.
    w: Vec<f64>,
    /// Polishing scratch: `Jᵀ g`, the dual and primal residuals and the
    /// refinement correction.
    c: Vec<f64>,
    rd: Vec<f64>,
    rp: Vec<f64>,
    dx: Vec<f64>,
    du: Vec<f64>,
    members: Vec<Member>,
    /// Multipliers of `members` for their oriented normals.
    u: Vec<f64>,
    /// Whether each source is in the working set.
    active: Vec<bool>,
    /// Whether each source has been in the working set during the attempt.
    entered: Vec<bool>,
    /// Sources with at least one finite bound, rebuilt for each model.
    bounded: Vec<usize>,
    /// Equality sources in original order, rebuilt alongside bounded.
    equalities: Vec<usize>,
    /// Elastic weight `γ` of an elastic QP; rows are then soft.
    weight: Option<f64>,
    /// Rows that stay hard in an elastic QP (empty: none).
    hard: Vec<bool>,
    /// Per source, the signed count of elastic sides: `σ` when the row may
    /// violate its side `σ` at a fixed multiplier `σγ` (a linear term
    /// `−γσa` in the objective). Opposite entries cancel.
    elastic: Vec<f64>,
    /// `g − γ Σ elastic_s a_s`, the gradient with the elastic terms.
    ge: Vec<f64>,
    tolerance: f64,
    /// Absolute dependence threshold per unit normal (see [`DEPENDENCE`]).
    dependence: f64,
    /// Relative noise level of multiplier directions (see [`NOISE`]).
    noise: f64,
    /// `x` and `u` were recomputed from the factors after the last change.
    polished: bool,
}

impl Work {
    fn load(&mut self, model: &QpModel, tolerance: f64, weight: Option<f64>, hard: &[bool]) {
        let n = model.n();
        self.n = n;
        let h = &model.hessian_lower;
        self.factor_reused = self.factor_valid
            && self.hessian_key.as_ref().is_some_and(|key| {
                key.pattern == h.pattern
                    && key.values.len() == h.values.len()
                    && key
                        .values
                        .iter()
                        .zip(&h.values)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            });
        if !self.factor_reused {
            // Invalidate before loading any changed or unverified Hessian.
            self.factor_valid = false;
            self.initial_valid = false;
            match self.hessian_key.as_mut() {
                Some(key) => {
                    key.pattern.rows = h.pattern.rows;
                    key.pattern.cols = h.pattern.cols;
                    key.pattern.indptr.clone_from(&h.pattern.indptr);
                    key.pattern.indices.clone_from(&h.pattern.indices);
                    key.values.clone_from(&h.values);
                }
                None => self.hessian_key = Some(h.clone()),
            }
        }
        self.members.reserve(n.saturating_sub(self.members.len()));
        self.u.reserve(n.saturating_sub(self.u.len()));
        self.l.resize(n * n, 0.0);
        let full = n.checked_mul(n + 1).map(|v| v / 2) == Some(model.hessian_lower.values.len());
        if !self.factor_reused && !full {
            for (k, column) in self.l.chunks_exact_mut(n).enumerate() {
                column[k..].fill(0.0);
            }
        }
        // `j` and `r` are fully written before they are read.
        self.j.resize(n * n, 0.0);
        self.r.resize(n * n, 0.0);
        for v in [
            &mut self.x,
            &mut self.d,
            &mut self.y,
            &mut self.w,
            &mut self.c,
            &mut self.rd,
            &mut self.rp,
            &mut self.dx,
            &mut self.du,
            &mut self.ge,
        ] {
            v.resize(n, 0.0);
        }
        if !self.factor_reused {
            for i in 0..n {
                for a in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                    self.l[h.pattern.indices[a] * n + i] = h.values[a];
                }
            }
        }
        for flags in [&mut self.active, &mut self.entered] {
            flags.clear();
            flags.resize(n + model.m(), false);
        }
        self.elastic.clear();
        self.elastic.resize(n + model.m(), 0.0);
        self.weight = weight;
        self.hard.clear();
        self.hard.extend_from_slice(hard);
        self.bounded.clear();
        self.equalities.clear();
        for (i, (&lo, &hi)) in model
            .variables
            .lower
            .iter()
            .zip(&model.variables.upper)
            .chain(model.rows.lower.iter().zip(&model.rows.upper))
            .enumerate()
        {
            if lo != f64::NEG_INFINITY || hi != f64::INFINITY {
                self.bounded.push(i);
                if lo == hi {
                    self.equalities.push(i);
                }
            }
        }
        self.tolerance = tolerance;
    }

    fn factor(&mut self) -> bool {
        if self.factor_reused {
            return true;
        }
        let n = self.n;
        if !dense::cholesky(&mut self.l, n) {
            return false;
        }
        let (low, high) = (0..n)
            .map(|k| self.l[k * n + k])
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), d| {
                (lo.min(d), hi.max(d))
            });
        self.noise = (NOISE * f64::EPSILON * (n as f64).sqrt() * (high / low)).min(MAX_NOISE);
        self.factor_valid = true;
        true
    }

    /// One attempt from the equalities plus the warm-start working set.
    fn solve(
        &mut self,
        sources: &Sources,
        warm: Option<&ActiveSet>,
        budget: &mut Budget,
    ) -> std::result::Result<(), Stop> {
        if self.initial_valid {
            self.j.copy_from_slice(&self.initial_j);
            self.dependence = self.initial_dependence;
        } else {
            dense::inverse_transpose(&self.l, &mut self.j, self.n);
            self.dependence =
                DEPENDENCE * f64::EPSILON * (self.n as f64).sqrt() * dense::norm(&self.j);
            if self.factor_reused
                && self.dependence.is_finite()
                && self.j.iter().all(|v| v.is_finite())
            {
                self.initial_j.clone_from(&self.j);
                self.initial_dependence = self.dependence;
                self.initial_valid = true;
            }
        }
        self.q = 0;
        self.members.clear();
        self.u.clear();
        self.active.fill(false);
        self.entered.fill(false);
        self.elastic.fill(0.0);
        self.polished = false;
        for k in 0..self.equalities.len() {
            let s = self.equalities[k];
            let (lower, _) = sources.bounds(s);
            let member = self.member(sources, s, 1.0, lower, true);
            self.force(sources, member);
        }
        if let Some(set) = warm {
            for (s, activity) in set.variables.iter().chain(&set.rows).enumerate() {
                let (lower, upper) = sources.bounds(s);
                let (sigma, rhs) = match activity {
                    _ if lower == upper => continue,
                    Activity::AtLower if lower.is_finite() => (1.0, lower),
                    Activity::AtUpper if upper.is_finite() => (-1.0, -upper),
                    _ => continue,
                };
                let member = self.member(sources, s, sigma, rhs, false);
                self.force(sources, member);
            }
        }
        self.settle(sources, budget)?;
        loop {
            budget.check()?;
            match self.most_violated(sources) {
                Some(p) => self.add(sources, p, budget)?,
                None if self.polished => return Ok(()),
                None => self.settle(sources, budget)?,
            }
        }
    }

    /// The member for side `sigma` of source `s`, with the multiplier bounds
    /// of its kind.
    fn member(&self, sources: &Sources, s: usize, sigma: f64, rhs: f64, equality: bool) -> Member {
        let soft = self.weight.filter(|_| {
            s >= sources.n() && !self.hard.get(s - sources.n()).copied().unwrap_or(false)
        });
        let (low, high) = match (soft, equality) {
            (None, false) => (0.0, f64::INFINITY),
            (None, true) => (f64::NEG_INFINITY, f64::INFINITY),
            (Some(w), false) => (0.0, w),
            (Some(w), true) => (-w, w),
        };
        Member {
            source: s,
            sigma,
            rhs,
            low,
            high,
        }
    }

    /// Removes member `k`, which left at its lower bound (`at_high` false) or
    /// upper bound; at `±γ` its row becomes elastic on that side.
    fn release(&mut self, k: usize, at_high: bool) {
        let member = self.members[k];
        self.drop(k);
        if at_high && member.high.is_finite() {
            self.elastic[member.source] += member.sigma;
        } else if !at_high && member.low < 0.0 && member.low.is_finite() {
            self.elastic[member.source] -= member.sigma;
        }
    }

    /// Adds `member` to the initial working set unless it is dependent.
    fn force(&mut self, sources: &Sources, member: Member) {
        if self.q == self.n {
            return;
        }
        sources.project(member.source, member.sigma, &self.j, &mut self.d);
        if self.free_part(sources.norm(member.source)).is_some() {
            self.push(member, 0.0);
        }
    }

    /// `‖d₂‖`, the part of `d = Jᵀn` outside the working set, or `None` when
    /// the constraint (normal norm `norm`) is dependent on the working set.
    fn free_part(&self, norm: f64) -> Option<f64> {
        let free = dense::norm(&self.d[self.q..]);
        (free > self.dependence * norm).then_some(free)
    }

    /// Recomputes `x` and `u` on the working set, then releases the member
    /// whose multiplier is furthest outside its bounds until none is.
    fn settle(&mut self, sources: &Sources, budget: &mut Budget) -> std::result::Result<(), Stop> {
        loop {
            self.polish(sources);
            let largest = self.u.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let mut excess = self.tolerance * (1.0 + largest);
            let mut worst = None;
            for (k, (member, &u)) in self.members.iter().zip(&self.u).enumerate() {
                if member.low - u > excess {
                    excess = member.low - u;
                    worst = Some((k, false));
                }
                if u - member.high > excess {
                    excess = u - member.high;
                    worst = Some((k, true));
                }
            }
            let Some((k, at_high)) = worst else {
                self.polished = true;
                return Ok(());
            };
            budget.check()?;
            budget.spend(true)?;
            self.release(k, at_high);
        }
    }

    /// The minimizer on the working set and its multipliers (`G x + g = N u`,
    /// `Nᵀx = b`) from the factors, followed by iterative refinement against
    /// the original data, which recovers the accuracy that an ill-conditioned
    /// `H` or many rotations cost.
    fn polish(&mut self, sources: &Sources) {
        let (n, q) = (self.n, self.q);
        let model = sources.model;
        if let Some(weight) = self.weight {
            self.ge.copy_from_slice(&model.gradient);
            for (s, &e) in self.elastic.iter().enumerate().skip(n) {
                if e != 0.0 {
                    sources.scatter(s, -weight * e, &mut self.ge);
                }
            }
        }
        let ge = if self.weight.is_some() {
            self.ge.as_slice()
        } else {
            model.gradient.as_slice()
        };
        for (b, member) in self.rp.iter_mut().zip(&self.members) {
            *b = member.rhs;
        }
        kkt_solve(
            (&self.j, &self.r, n, q),
            ge,
            &self.rp[..q],
            &mut self.c,
            &mut self.x,
            &mut self.u,
        );
        for _ in 0..REFINEMENT_STEPS {
            model.hessian_product(&self.x, &mut self.rd);
            for (r, &g) in self.rd.iter_mut().zip(ge) {
                *r += g;
            }
            for (k, member) in self.members.iter().enumerate() {
                sources.scatter(member.source, -member.sigma * self.u[k], &mut self.rd);
                let v = sources.activity(member.source, &self.x);
                self.rp[k] = member.rhs - member.sigma * v;
            }
            kkt_solve(
                (&self.j, &self.r, n, q),
                &self.rd,
                &self.rp[..q],
                &mut self.c,
                &mut self.dx,
                &mut self.du[..q],
            );
            dense::axpy(1.0, &self.dx, &mut self.x);
            dense::axpy(1.0, &self.du[..q], &mut self.u);
        }
    }

    /// The most violated constraint outside the working set, by violation
    /// divided by the normal's norm, oriented so that it reads `σ aᵀp ≥ rhs`.
    fn most_violated(&self, sources: &Sources) -> Option<Member> {
        let mut best: Option<(f64, Member)> = None;
        for &s in &self.bounded {
            if self.active[s] {
                continue;
            }
            let (lower, upper) = sources.bounds(s);
            let (v, magnitude) = sources.value(s, &self.x);
            let tau = |b: f64| self.tolerance * (1.0 + b.abs() + magnitude);
            let member =
                |sigma: f64, rhs: f64, equality| self.member(sources, s, sigma, rhs, equality);
            let e = self.elastic[s];
            let found = if e != 0.0 {
                // An elastic row may violate side e. Once it satisfies that
                // side strictly, its fixed multiplier is wrong: re-enter the
                // opposite side with multiplier bounds [0, γ], which together
                // with the elastic term gives the row a multiplier in
                // σ·[0, γ] on its bound, or cancels it at γ.
                let rhs = if e > 0.0 { lower } else { -upper };
                let slack = e * v - rhs;
                (slack > tau(rhs)).then(|| {
                    let mut back = member(-e, -rhs, false);
                    back.low = 0.0;
                    (slack, back)
                })
            } else if lower == upper {
                let e = v - lower;
                (e.abs() > tau(lower)).then(|| {
                    let sigma = if e > 0.0 { -1.0 } else { 1.0 };
                    (e.abs(), member(sigma, sigma * lower, true))
                })
            } else if lower - v > tau(lower) {
                Some((lower - v, member(1.0, lower, false)))
            } else if v - upper > tau(upper) {
                Some((v - upper, member(-1.0, -upper, false)))
            } else {
                None
            };
            if let Some((violation, member)) = found {
                let norm = sources.norm(s);
                let measure = if norm > 0.0 {
                    violation / norm
                } else {
                    violation
                };
                if best.is_none_or(|(b, _)| measure > b) {
                    best = Some((measure, member));
                }
            }
        }
        best.map(|(_, member)| member)
    }

    /// Whether `x` violates `p` beyond the tolerance of [`Self::most_violated`].
    fn violates(&self, sources: &Sources, p: &Member) -> bool {
        let (v, magnitude) = sources.value(p.source, &self.x);
        p.rhs - p.sigma * v > self.tolerance * (1.0 + p.rhs.abs() + magnitude)
    }

    /// Moves toward satisfying `p`, dropping blocking constraints, until `p`
    /// enters the working set (or the attempt stops).
    fn add(
        &mut self,
        sources: &Sources,
        p: Member,
        budget: &mut Budget,
    ) -> std::result::Result<(), Stop> {
        let mut multiplier = 0.0;
        loop {
            budget.check()?;
            let (n, q) = (self.n, self.q);
            sources.project(p.source, p.sigma, &self.j, &mut self.d);
            let free = self.free_part(sources.norm(p.source));
            self.back_substitute();
            // Partial step: the first working-set multiplier that reaches a
            // finite bound (u_k(t) = u_k − t y_k). Entries of y at rounding
            // level are zero: for a dependent p they would allow absurd dual
            // steps.
            let floor = self.noise * self.y[..q].iter().fold(0.0_f64, |a, b| a.max(b.abs()));
            let mut partial = f64::INFINITY;
            let mut leaving = None;
            for (k, member) in self.members.iter().enumerate() {
                let (u, y) = (self.u[k], self.y[k]);
                let (ratio, at_high) = if y > floor && member.low.is_finite() {
                    ((u - member.low) / y, false)
                } else if y < -floor && member.high.is_finite() {
                    ((member.high - u) / -y, true)
                } else {
                    continue;
                };
                if ratio.max(0.0) < partial {
                    partial = ratio.max(0.0);
                    leaving = Some((k, at_high));
                }
            }
            // An elastic row's own multiplier stops at its upper bound γ.
            let cap = (p.high - multiplier).max(0.0);
            // Full step: the primal step that satisfies p.
            let full = free.map_or(f64::INFINITY, |free| {
                let v = sources.activity(p.source, &self.x);
                ((p.rhs - p.sigma * v) / (free * free)).max(0.0)
            });
            if partial == f64::INFINITY && full == f64::INFINITY && cap == f64::INFINITY {
                if self.polished {
                    return Err(Stop::Infeasible);
                }
                // Rule out drift of x before claiming infeasibility, then
                // continue with p from the polished point (selecting anew
                // could cycle).
                self.settle(sources, budget)?;
                multiplier = 0.0;
                if !self.violates(sources, &p) {
                    return Ok(());
                }
                continue;
            }
            let t = partial.min(full).min(cap);
            for (u, &y) in self.u.iter_mut().zip(&self.y[..q]) {
                *u -= t * y;
            }
            multiplier += t;
            if free.is_some() && t > 0.0 {
                for (k, column) in self.j.chunks_exact(n).enumerate().skip(q) {
                    dense::axpy(t * self.d[k], column, &mut self.x);
                }
                self.polished = false;
            }
            if full <= partial && full <= cap {
                let reentry = self.entered[p.source];
                self.push(p, multiplier);
                return budget.spend(reentry);
            }
            if cap <= partial {
                // p stays violated at multiplier γ: it becomes elastic.
                let reentry = self.entered[p.source];
                self.entered[p.source] = true;
                self.elastic[p.source] += p.sigma;
                self.polished = false;
                return budget.spend(reentry);
            }
            let (k, at_high) = leaving.expect("a finite partial step has a leaving constraint");
            self.release(k, at_high);
            budget.spend(true)?;
        }
    }

    /// `y[..q] = R⁻¹ d[..q]`.
    fn back_substitute(&mut self) {
        let (n, q) = (self.n, self.q);
        self.y[..q].copy_from_slice(&self.d[..q]);
        for k in (0..q).rev() {
            let column = &self.r[k * n..k * n + k + 1];
            let v = self.y[k] / column[k];
            self.y[k] = v;
            dense::axpy(-v, &column[..k], &mut self.y[..k]);
        }
    }

    /// Appends `member`, whose `Jᵀn` is in `d` and has a nonzero part
    /// outside the working set: reflects that part into `d[q]` and makes
    /// `d[..=q]` the new column of `R`.
    fn push(&mut self, member: Member, multiplier: f64) {
        let (n, q) = (self.n, self.q);
        debug_assert!(q < n);
        dense::reflect(&mut self.j, n, q, &mut self.d[q..], &mut self.w);
        self.r[q * n..q * n + q + 1].copy_from_slice(&self.d[..=q]);
        self.members.push(member);
        self.u.push(multiplier);
        self.active[member.source] = true;
        self.entered[member.source] = true;
        self.q += 1;
        self.polished = false;
    }

    /// Removes working-set entry `k` and restores the triangular `R`.
    fn drop(&mut self, k: usize) {
        let (n, q) = (self.n, self.q);
        let member = self.members.remove(k);
        self.u.remove(k);
        self.active[member.source] = false;
        for c in k + 1..q {
            self.r.copy_within(c * n..c * n + c + 1, (c - 1) * n);
        }
        // Columns k..q-1 now have one entry below the diagonal.
        for c in k..q - 1 {
            let (cs, sn, h) = dense::givens(self.r[c * n + c], self.r[c * n + c + 1]);
            self.r[c * n + c] = h;
            self.r[c * n + c + 1] = 0.0;
            for l in c + 1..q - 1 {
                let (a, b) = (self.r[l * n + c], self.r[l * n + c + 1]);
                self.r[l * n + c] = cs * a + sn * b;
                self.r[l * n + c + 1] = cs * b - sn * a;
            }
            let (x, y) = dense::column_pair(&mut self.j, n, c, c + 1);
            dense::rotate(x, y, cs, sn);
        }
        self.q -= 1;
        self.polished = false;
    }

    fn solution(&self, sources: &Sources, iterations: u64) -> QpSolution {
        let (n, m) = (sources.n(), sources.model.m());
        let mut multipliers = Multipliers {
            constraints: vec![0.0; m],
            variables: vec![0.0; n],
        };
        let mut set = ActiveSet {
            variables: vec![Activity::Inactive; n],
            rows: vec![Activity::Inactive; m],
        };
        for (member, &u) in self.members.iter().zip(&self.u) {
            // A re-entered elastic row sits on its elastic side.
            let e = self.elastic[member.source];
            let sigma = if e != 0.0 && member.sigma == -e {
                e
            } else {
                member.sigma
            };
            let side = if sigma > 0.0 {
                Activity::AtLower
            } else {
                Activity::AtUpper
            };
            let (values, activity, i) = if member.source < n {
                (
                    &mut multipliers.variables,
                    &mut set.variables,
                    member.source,
                )
            } else {
                (
                    &mut multipliers.constraints,
                    &mut set.rows,
                    member.source - n,
                )
            };
            values[i] += member.sigma * u;
            activity[i] = side;
        }
        if let Some(weight) = self.weight {
            for (i, &e) in self.elastic[n..].iter().enumerate() {
                multipliers.constraints[i] += weight * e;
            }
        }
        QpSolution {
            step: self.x.clone(),
            multipliers,
            active_set: Some(set),
            iterations,
        }
    }
}

/// Solves `G x − N u = −gd`, `Nᵀx = bp` for the working set of the factors
/// `(J, R, n, q)`: `x = J₁ R⁻ᵀ bp − J₂ J₂ᵀ gd` and
/// `u = R⁻¹ (R⁻ᵀ bp + J₁ᵀ gd)`. `c` is scratch of length `n`.
fn kkt_solve(
    (j, r, n, q): (&[f64], &[f64], usize, usize),
    gd: &[f64],
    bp: &[f64],
    c: &mut [f64],
    x: &mut [f64],
    u: &mut [f64],
) {
    for (ck, column) in c.iter_mut().zip(j.chunks_exact(n)) {
        *ck = dense::dot(column, gd);
    }
    for k in 0..q {
        let column = &r[k * n..k * n + k + 1];
        let v = (bp[k] - dense::dot(&column[..k], &u[..k])) / column[k];
        u[k] = v;
    }
    x.fill(0.0);
    for (k, column) in j.chunks_exact(n).enumerate() {
        let f = if k < q { u[k] } else { -c[k] };
        if f != 0.0 {
            dense::axpy(f, column, x);
        }
    }
    for k in 0..q {
        u[k] += c[k];
    }
    for k in (0..q).rev() {
        let column = &r[k * n..k * n + k + 1];
        let v = u[k] / column[k];
        u[k] = v;
        dense::axpy(-v, &column[..k], &mut u[..k]);
    }
}

#[cfg(test)]
mod tests;
