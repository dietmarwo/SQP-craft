//! Native SQP driver: dense damped BFGS, convex QP subproblems behind
//! [`QpBackend`], and an L1 merit line search with per-constraint penalties.
//!
//! A bounded research pilot, not a general solver: at most 256 variables,
//! local Gauss–Newton restoration for inconsistent linearizations (disable it
//! with `max_restoration_steps = 0`), no second-order correction and no
//! automatic scaling. Convergence means the first-order KKT residuals at the
//! returned point are within the declared tolerances; it is not a global or
//! second-order optimality certificate.
//!
//! Each iteration at `x_k`:
//!
//! 1. Solve the step QP `min ½pᵀBp + gᵀp` with bounds shifted to step
//!    coordinates, and accept it only through [`QpModel::accepts`].
//! 2. Compute KKT residuals at `x_k` with the QP's multipliers; stop with
//!    [`Status::Converged`] if all four are within tolerance.
//! 3. Update the penalty weights by Powell's rule
//!    `μ_i ← max(t_i, ½(μ_i + t_i))` with `t_i = 1.1·|λ_i|`, and backtrack on the merit
//!    `φ(x) = f(x) + Σ μ_i · violation_i(c(x))` with an Armijo test against the
//!    predicted decrease of the linearized merit.
//! 4. At the accepted point call `derivatives`, then update `B` by damped
//!    BFGS on the Lagrangian gradient with the new multipliers at both ends.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bfgs;
mod restoration;

use std::time::Instant;

use nalgebra::DMatrix;
use sqp_craft::kkt::nlp_residuals;
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome, QpSolution};
use sqp_craft::{
    Bounds, CancellationToken, Control, CsrMatrix, CsrPattern, Error, Evaluation, EvaluationCounts,
    IterationRecord, Multipliers, NlpProblem, Observer, Options, ProblemSpec, Progress, Result,
    SolveResult, SqpSolver, Status,
};

/// Largest number of variables the dense pilot accepts.
pub const DENSE_SIZE_CAP: usize = 256;
/// Largest number of QP models [`SqpOptions::record_qp_limit`] may keep.
pub const QP_TRACE_CAP: usize = 16;
/// Armijo sufficient-decrease constant.
const ARMIJO: f64 = 1e-4;
/// Penalty weights target `PENALTY_MARGIN · |λ_i|`. The margin above `|λ_i|`
/// keeps the predicted merit decrease positive at infeasible iterates although
/// interior-point QP solutions stop slightly short of active bounds.
const PENALTY_MARGIN: f64 = 1.1;

/// Settings of [`NativeSqp`]. Every field is used; invalid values are rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct SqpOptions {
    /// Maximum number of accepted steps.
    pub max_iterations: u64,
    /// Maximum step halvings in one line search after the first trial.
    pub max_backtracks: u32,
    /// Maximum `functions` plus `derivatives` calls, including rejected and
    /// undefined trials.
    pub max_callback_calls: u64,
    /// Cooperative wall-time budget for the whole solve, in seconds.
    pub time_limit_seconds: Option<f64>,
    /// Absolute stationarity tolerance `‖∇f − Jᵀλ − z‖∞`.
    pub stationarity_tolerance: f64,
    /// Absolute complementarity tolerance.
    pub complementarity_tolerance: f64,
    /// Absolute tolerance on multipliers whose sign selects an infinite bound.
    pub dual_sign_tolerance: f64,
    /// Variables allowed for this solve, `1..=DENSE_SIZE_CAP`.
    pub dense_size_limit: usize,
    /// Initial and reset Hessian approximation `B = scale · I`.
    pub initial_hessian_scale: f64,
    /// Total Hessian resets allowed (failed BFGS updates and non-descent retries).
    pub max_hessian_resets: u32,
    /// Native tolerance passed to the QP backend.
    pub qp_tolerance: f64,
    /// Iteration cap per QP.
    pub qp_max_iterations: u32,
    /// Number of step QPs to keep in the report, `0..=QP_TRACE_CAP`.
    pub record_qp_limit: usize,
    /// Keep one [`IterationRecord`] per iteration in the report.
    pub record_history: bool,
    /// Accepted restoration steps allowed across the solve; zero disables it.
    pub max_restoration_steps: u64,
    /// Infinity-norm trust bound on each restoration direction, in x units.
    pub restoration_step_limit: f64,
    /// Positive diagonal regularization of the restoration Gauss–Newton model.
    pub restoration_regularization: f64,
}

impl Default for SqpOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_backtracks: 25,
            max_callback_calls: 2000,
            time_limit_seconds: None,
            stationarity_tolerance: 1e-6,
            complementarity_tolerance: 1e-6,
            dual_sign_tolerance: 1e-8,
            dense_size_limit: DENSE_SIZE_CAP,
            initial_hessian_scale: 1.0,
            max_hessian_resets: 3,
            qp_tolerance: 1e-10,
            qp_max_iterations: 200,
            record_qp_limit: 0,
            record_history: false,
            max_restoration_steps: 10,
            restoration_step_limit: 1.0,
            restoration_regularization: 1e-4,
        }
    }
}

impl SqpOptions {
    /// Rejects non-positive or non-finite tolerances and out-of-range caps.
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("stationarity_tolerance", self.stationarity_tolerance),
            ("complementarity_tolerance", self.complementarity_tolerance),
            ("dual_sign_tolerance", self.dual_sign_tolerance),
            ("initial_hessian_scale", self.initial_hessian_scale),
            ("qp_tolerance", self.qp_tolerance),
            ("restoration_step_limit", self.restoration_step_limit),
            (
                "restoration_regularization",
                self.restoration_regularization,
            ),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidInput(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if let Some(s) = self.time_limit_seconds {
            if !(s.is_finite() && s >= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "time_limit_seconds must be finite and nonnegative, got {s}"
                )));
            }
        }
        if !(1..=DENSE_SIZE_CAP).contains(&self.dense_size_limit) {
            return Err(Error::InvalidInput(format!(
                "dense_size_limit must be in 1..={DENSE_SIZE_CAP}, got {}",
                self.dense_size_limit
            )));
        }
        if self.record_qp_limit > QP_TRACE_CAP {
            return Err(Error::InvalidInput(format!(
                "record_qp_limit must be at most {QP_TRACE_CAP}, got {}",
                self.record_qp_limit
            )));
        }
        Ok(())
    }
}

/// The native SQP solver with QP backend `B`.
#[derive(Clone, Debug, Default)]
pub struct NativeSqp<B> {
    /// QP subproblem backend.
    pub qp: B,
    /// Driver settings.
    pub settings: SqpOptions,
}

fn finite(v: f64, what: &str) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::Evaluation(format!("{what} became nonfinite")))
    }
}

fn dot(a: &[f64], b: &[f64]) -> Result<f64> {
    let mut v = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        v = finite(v + finite(a * b, "dot product")?, "dot product")?;
    }
    Ok(v)
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

/// `Σ weights_i · (max(0, lower_i − v_i) + max(0, v_i − upper_i))` over finite sides.
fn weighted_violation(bounds: &Bounds, values: &[f64], weights: &[f64]) -> Result<f64> {
    let mut total = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let mut row = 0.0;
        if bounds.lower[i].is_finite() {
            row += finite(bounds.lower[i] - v, "merit violation")?.max(0.0);
        }
        if bounds.upper[i].is_finite() {
            row += finite(v - bounds.upper[i], "merit violation")?.max(0.0);
        }
        total = finite(
            total + finite(weights[i] * row, "merit violation")?,
            "merit",
        )?;
    }
    Ok(total)
}

/// Model calls with budget, cancellation and timing bookkeeping.
struct Calls<'a> {
    began: Instant,
    settings: &'a SqpOptions,
    cancel: &'a CancellationToken,
    counts: EvaluationCounts,
    seconds: f64,
}

impl Calls<'_> {
    fn stop(&self) -> Option<Status> {
        if self.cancel.is_cancelled() {
            Some(Status::Cancelled)
        } else if self
            .settings
            .time_limit_seconds
            .is_some_and(|s| self.began.elapsed().as_secs_f64() >= s)
        {
            Some(Status::TimeLimit)
        } else {
            None
        }
    }

    fn before_call(&self) -> Option<Status> {
        self.stop().or_else(|| {
            (self.counts.functions + self.counts.derivatives >= self.settings.max_callback_calls)
                .then_some(Status::EvaluationLimit)
        })
    }

    fn functions(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        spec.check_point(x, "callback point")?;
        let mut c = vec![f64::NAN; spec.m()];
        self.counts.functions += 1;
        let t = Instant::now();
        let f = model.functions(x, &mut c);
        self.seconds += t.elapsed().as_secs_f64();
        let f = finite(f?, "objective")?;
        for &v in &c {
            finite(v, "constraint value")?;
        }
        Ok((f, c))
    }

    fn derivatives(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x: &[f64],
        (f, c): (f64, Vec<f64>),
        tolerance: f64,
    ) -> Result<Evaluation> {
        spec.check_point(x, "callback point")?;
        let mut g = vec![f64::NAN; spec.n()];
        let mut j = vec![f64::NAN; spec.nnz()];
        self.counts.derivatives += 1;
        let t = Instant::now();
        let result = model.derivatives(x, &mut g, &mut j);
        self.seconds += t.elapsed().as_secs_f64();
        result?;
        for &v in g.iter().chain(&j) {
            finite(v, "derivative value")?;
        }
        let primal = spec
            .variables()
            .checked_violation(x)?
            .max(spec.constraints().checked_violation(&c)?);
        Ok(Evaluation {
            objective: f,
            constraints: c,
            gradient: g,
            jacobian_values: j,
            primal_violation: primal,
            feasible_candidate: primal <= tolerance,
        })
    }
}

/// Bounds shifted to step coordinates: `bound − value` on finite sides.
fn shifted(bounds: &Bounds, values: &[f64]) -> Result<Bounds> {
    let shift = |side: &[f64]| -> Result<Vec<f64>> {
        side.iter()
            .zip(values)
            .map(|(&b, &v)| {
                if b.is_finite() {
                    finite(b - v, "shifted bound")
                } else {
                    Ok(b)
                }
            })
            .collect()
    };
    Ok(Bounds::new(shift(&bounds.lower)?, shift(&bounds.upper)?))
}

/// The step QP at `x` with model values `e` and Hessian approximation `b`.
fn qp_model(spec: &ProblemSpec, x: &[f64], e: &Evaluation, b: &DMatrix<f64>) -> Result<QpModel> {
    let n = spec.n();
    let mut indptr = vec![0];
    let mut indices = Vec::with_capacity(n * (n + 1) / 2);
    let mut values = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            indices.push(j);
            values.push(b[(i, j)]);
        }
        indptr.push(indices.len());
    }
    Ok(QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr,
                indices,
            },
            values,
        },
        gradient: e.gradient.clone(),
        variables: shifted(spec.variables(), x)?,
        matrix: CsrMatrix {
            pattern: spec.jacobian().clone(),
            values: e.jacobian_values.clone(),
        },
        rows: shifted(spec.constraints(), &e.constraints)?,
    })
}

/// `∇f − Jᵀλ − z`.
fn lagrangian_gradient(spec: &ProblemSpec, e: &Evaluation, m: &Multipliers) -> Result<Vec<f64>> {
    let mut jt = vec![0.0; spec.n()];
    spec.jacobian()
        .multiply_transpose(&e.jacobian_values, &m.constraints, &mut jt);
    e.gradient
        .iter()
        .zip(jt)
        .zip(&m.variables)
        .map(|((&g, j), &z)| finite(g - j - z, "Lagrangian gradient"))
        .collect()
}

/// Stores KKT residuals of the current candidate for multipliers `m`.
///
/// Any multiplier vector gives an upper bound on the best first-order residual
/// at the point, so this is meaningful even when `m` comes from the QP solved
/// at the previous iterate.
fn store_residuals(spec: &ProblemSpec, r: &mut SolveResult, m: Multipliers) -> Result<()> {
    let (Some(x), Some(e)) = (r.candidate.as_ref(), r.evaluation.as_ref()) else {
        return Ok(());
    };
    let k = nlp_residuals(spec, x, e, &m)?;
    r.stationarity = Some(k.stationarity);
    r.complementarity = Some(k.complementarity);
    r.dual_sign = Some(k.dual_sign);
    r.multipliers = Some(m);
    Ok(())
}

fn status_message(status: Status) -> &'static str {
    match status {
        Status::Converged => "first-order KKT residuals within tolerance at the returned point",
        Status::IterationLimit => "iteration limit reached",
        Status::EvaluationLimit => "callback budget exhausted; returning the last complete point",
        Status::TimeLimit => "time limit reached; returning the last complete point",
        Status::Cancelled => "cancelled",
        Status::LinearizationInfeasible => {
            "the linearized constraints are inconsistent at this point; the nonlinear problem \
             may still be feasible (restoration is disabled)"
        }
        Status::RestorationFailed => {
            "local restoration failed; nonlinear infeasibility is not established"
        }
        Status::RestorationLimit => "feasibility restoration step budget exhausted",
        Status::LineSearchFailure => "no acceptable step length within the backtracking limit",
        Status::NoDescent => "QP step is not a descent direction for the merit function",
        Status::Stalled => "QP step too small to change the iterate",
        Status::NumericalFailure => "Hessian reset budget exhausted",
        Status::QpFailure => "QP subproblem not solved acceptably",
        _ => "",
    }
}

impl<B: QpBackend> SqpSolver for NativeSqp<B> {
    fn solve(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x0: &[f64],
        options: &Options,
        cancel: &CancellationToken,
        mut observer: Option<&mut dyn Observer>,
    ) -> Result<SolveResult> {
        options.validate()?;
        self.settings.validate()?;
        spec.check_point(x0, "start point")?;
        if spec.n() > self.settings.dense_size_limit {
            return Err(Error::Unsupported(format!(
                "n = {} exceeds the dense pilot limit of {} variables",
                spec.n(),
                self.settings.dense_size_limit
            )));
        }
        let settings = &self.settings;
        let qp_acceptance = options
            .feasibility_tolerance
            .min(settings.stationarity_tolerance)
            .min(settings.complementarity_tolerance);
        let mut calls = Calls {
            began: Instant::now(),
            settings,
            cancel,
            counts: EvaluationCounts::default(),
            seconds: 0.0,
        };
        let mut r = SolveResult::empty(Status::IterationLimit, "");
        r.report.method = "sqp".into();

        // Start point: a complete evaluation or nothing.
        let mut status = 'start: {
            if let Some(status) = calls.before_call() {
                break 'start Some(status);
            }
            let values = calls.functions(spec, model, x0)?;
            if let Some(status) = calls.before_call() {
                break 'start Some(status);
            }
            let evaluation =
                calls.derivatives(spec, model, x0, values, options.feasibility_tolerance)?;
            r.candidate = Some(x0.to_vec());
            r.evaluation = Some(evaluation);
            None
        };

        let n = spec.n();
        let mut b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
        let mut penalties = vec![0.0; spec.m()];
        let mut have_penalties = false;
        let mut retry_descent = false;
        let mut notify = true;
        while status.is_none() {
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            if notify {
                if let Some(obs) = observer.as_deref_mut() {
                    let e = r.evaluation.as_ref().expect("evaluated");
                    let progress = Progress {
                        iteration: r.iterations,
                        objective: e.objective,
                        primal_violation: e.primal_violation,
                    };
                    if obs.observe(&progress)? == Control::Stop {
                        status = Some(Status::Cancelled);
                        r.message = "stopped by the observer".into();
                        break;
                    }
                }
                if let Some(s) = calls.stop() {
                    status = Some(s);
                    break;
                }
            }
            if r.iterations >= settings.max_iterations {
                status = Some(Status::IterationLimit);
                break;
            }

            // 1. Step QP.
            let x = r.candidate.as_ref().expect("evaluated");
            let e = r.evaluation.as_ref().expect("evaluated");
            let qp = qp_model(spec, x, e, &b)?;
            if r.report.qp_trace.len() < settings.record_qp_limit {
                r.report.qp_trace.push(qp.clone());
            }
            let remaining = settings
                .time_limit_seconds
                .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0));
            self.qp.configure(&QpOptions {
                tolerance: settings.qp_tolerance,
                acceptance_tolerance: qp_acceptance,
                max_iterations: settings.qp_max_iterations,
                time_limit_seconds: remaining,
            })?;
            let t = Instant::now();
            r.report.qp_solves += 1;
            let outcome = self.qp.solve(&qp, None, cancel);
            r.report.qp_seconds += t.elapsed().as_secs_f64();
            let report = self.qp.report();
            r.report.qp_backend = report.backend;
            r.report.qp_status = report.raw_status;
            r.report.qp_iterations += report.iterations;
            let qp_iterations = report.iterations;
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            let mut solution: QpSolution = match outcome {
                Ok(QpOutcome::Optimal(s)) => s,
                Ok(QpOutcome::Infeasible) => {
                    if settings.max_restoration_steps == 0 {
                        status = Some(Status::LinearizationInfeasible);
                        break;
                    }
                    if r.report.restoration_steps >= settings.max_restoration_steps {
                        status = Some(Status::RestorationLimit);
                        break;
                    }
                    let began = Instant::now();
                    let restored =
                        restoration::step(&mut self.qp, spec, model, &mut calls, &mut r, options);
                    r.report.restoration_seconds += began.elapsed().as_secs_f64();
                    status = restored?;
                    if status.is_some() {
                        break;
                    }
                    // Feasibility steps do not carry an NLP multiplier estimate.
                    // Restart curvature/penalties without spending the BFGS failure budget.
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    penalties.fill(0.0);
                    have_penalties = false;
                    retry_descent = false;
                    notify = true;
                    continue;
                }
                Ok(QpOutcome::Cancelled) => {
                    status = Some(Status::Cancelled);
                    break;
                }
                Ok(QpOutcome::LimitReached(_)) => {
                    let reason = report.limit_reason.unwrap_or_else(|| "unknown".into());
                    status = Some(if reason == "time" {
                        Status::TimeLimit
                    } else {
                        Status::QpFailure
                    });
                    r.message = format!("QP subproblem stopped at its {reason} limit");
                    break;
                }
                Ok(QpOutcome::Unbounded) => {
                    status = Some(Status::QpFailure);
                    r.message = "QP subproblem reported unbounded although B is positive \
                                 definite; numerical trouble"
                        .into();
                    break;
                }
                Ok(other) => {
                    status = Some(Status::QpFailure);
                    r.message = format!("unexpected QP outcome {other:?}");
                    break;
                }
                Err(Error::Backend(message)) => {
                    status = Some(Status::QpFailure);
                    r.message = message;
                    break;
                }
                Err(err) => return Err(err),
            };
            // A fixed variable's step bounds are exactly zero. Remove roundoff of
            // at most the feasibility tolerance, count it, and recheck below.
            for j in 0..n {
                let fixed = spec.variables().lower[j] == spec.variables().upper[j];
                let p = solution.step[j];
                if fixed && p != 0.0 && p.abs() <= options.feasibility_tolerance {
                    solution.step[j] = 0.0;
                    r.report.fixed_step_corrections += 1;
                }
            }
            let check = qp.accepts(&solution, qp_acceptance)?;
            if !check.accepted {
                status = Some(Status::QpFailure);
                r.message = format!(
                    "QP solution failed the independent check (scale {:.3e}): {:?}",
                    check.scale, check.residuals
                );
                break;
            }

            // 2. Convergence test at x_k with the fresh multipliers.
            store_residuals(spec, &mut r, solution.multipliers.clone())?;
            let e = r.evaluation.as_ref().expect("evaluated");
            let (stationarity, complementarity) = (
                r.stationarity.expect("stored"),
                r.complementarity.expect("stored"),
            );
            let mut record = IterationRecord {
                iteration: r.iterations,
                objective: e.objective,
                primal_violation: e.primal_violation,
                stationarity,
                complementarity,
                qp_step_norm: norm_inf(&solution.step),
                step_length: None,
                trials: 0,
                max_penalty: 0.0,
                qp_iterations,
            };
            if e.primal_violation <= options.feasibility_tolerance
                && stationarity <= settings.stationarity_tolerance
                && complementarity <= settings.complementarity_tolerance
                && r.dual_sign.expect("stored") <= settings.dual_sign_tolerance
            {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::Converged);
                break;
            }

            // 3. Merit line search.
            let x = r.candidate.as_ref().expect("evaluated");
            let p = &solution.step;
            for (mu, &lambda) in penalties.iter_mut().zip(&solution.multipliers.constraints) {
                let target = PENALTY_MARGIN * lambda.abs();
                *mu = if have_penalties {
                    target.max(0.5 * (*mu + target))
                } else {
                    target
                };
                finite(*mu, "penalty")?;
            }
            have_penalties = true;
            record.max_penalty = norm_inf(&penalties);
            let violation = weighted_violation(spec.constraints(), &e.constraints, &penalties)?;
            let mut linear = vec![0.0; spec.m()];
            spec.jacobian().multiply(&e.jacobian_values, p, &mut linear);
            for (l, &c) in linear.iter_mut().zip(&e.constraints) {
                *l = finite(*l + c, "linearized constraints")?;
            }
            let linear_violation = weighted_violation(spec.constraints(), &linear, &penalties)?;
            let predicted = finite(
                -dot(&e.gradient, p)? + (violation - linear_violation),
                "predicted decrease",
            )?;
            if predicted <= 0.0 {
                // First retry once from B = scale·I: a badly scaled B (for
                // example nearly zero curvature) can make an accurate step tiny.
                if !retry_descent
                    && r.report.hessian_resets < u64::from(settings.max_hessian_resets)
                {
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    r.report.hessian_resets += 1;
                    retry_descent = true;
                    notify = false;
                    continue;
                }
                if settings.record_history {
                    r.report.history.push(record);
                }
                // With exact QP solutions the predicted decrease is at least
                // pᵀBp > 0. An interior-point QP leaves active rows a little
                // short of their bounds, which costs up to the QP's
                // complementarity residual per entry. A non-positive value
                // within that accuracy means the step is QP noise.
                let qp = check.residuals;
                let noise = (spec.m() + n) as f64 * qp.complementarity
                    + qp.stationarity * p.iter().map(|v| v.abs()).sum::<f64>()
                    + record.max_penalty * spec.m() as f64 * qp.primal;
                if -predicted <= noise {
                    status = Some(Status::Stalled);
                    r.message = format!(
                        "the QP step is below the QP solution accuracy (predicted merit \
                         decrease {predicted:.3e}, QP accuracy {noise:.3e}); residuals at the \
                         returned point are reported"
                    );
                } else {
                    status = Some(Status::NoDescent);
                }
                break;
            }
            if norm_inf(p) <= 1e-14 * (1.0 + norm_inf(x)) {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::Stalled);
                break;
            }
            let merit = finite(e.objective + violation, "merit")?;
            let mut alpha = 1.0_f64;
            for j in 0..n {
                let (lo, up) = (spec.variables().lower[j], spec.variables().upper[j]);
                if p[j] > 0.0 && up.is_finite() {
                    alpha = alpha.min((up - x[j]) / p[j]);
                }
                if p[j] < 0.0 && lo.is_finite() {
                    alpha = alpha.min((lo - x[j]) / p[j]);
                }
            }
            if !(alpha.is_finite() && alpha > 0.0) {
                status = Some(Status::Stalled);
                r.message = "the QP step leaves the variable box immediately".into();
                break;
            }
            let mut accepted = None;
            for _ in 0..=settings.max_backtracks {
                if let Some(s) = calls.before_call() {
                    status = Some(s);
                    break;
                }
                let trial: Vec<f64> = x.iter().zip(p).map(|(&x, &p)| x + alpha * p).collect();
                if spec.check_point(&trial, "trial").is_err() {
                    // Roundoff pushed the trial out of the box: shorten without a call.
                    alpha *= 0.5;
                    r.report.rejected_trials += 1;
                    continue;
                }
                record.trials += 1;
                let values = match calls.functions(spec, model, &trial) {
                    Ok(v) => v,
                    Err(Error::UndefinedPoint(_)) => {
                        r.report.undefined_trials += 1;
                        r.report.rejected_trials += 1;
                        alpha *= 0.5;
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                let trial_merit = finite(
                    values.0 + weighted_violation(spec.constraints(), &values.1, &penalties)?,
                    "trial merit",
                )?;
                if trial_merit > merit - ARMIJO * alpha * predicted {
                    r.report.rejected_trials += 1;
                    alpha *= 0.5;
                    continue;
                }
                if let Some(s) = calls.before_call() {
                    status = Some(s);
                    break;
                }
                match calls.derivatives(spec, model, &trial, values, options.feasibility_tolerance)
                {
                    Ok(evaluation) => {
                        accepted = Some((trial, evaluation));
                        break;
                    }
                    Err(Error::UndefinedPoint(_)) => {
                        r.report.undefined_trials += 1;
                        r.report.rejected_trials += 1;
                        alpha *= 0.5;
                    }
                    Err(err) => return Err(err),
                }
            }
            if status.is_some() {
                break;
            }
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            let Some((next, next_evaluation)) = accepted else {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::LineSearchFailure);
                break;
            };
            record.step_length = Some(alpha);
            if settings.record_history {
                r.report.history.push(record);
            }

            // 4. Accept the step and update B.
            let s: Vec<f64> = next.iter().zip(x).map(|(&a, &b)| a - b).collect();
            let old_l = lagrangian_gradient(spec, e, &solution.multipliers)?;
            let new_l = lagrangian_gradient(spec, &next_evaluation, &solution.multipliers)?;
            let y: Vec<f64> = new_l.iter().zip(old_l).map(|(&a, b)| a - b).collect();
            let t = Instant::now();
            let updated = bfgs::update(&mut b, &s, &y);
            r.report.bfgs_seconds += t.elapsed().as_secs_f64();
            r.candidate = Some(next);
            r.evaluation = Some(next_evaluation);
            r.iterations += 1;
            r.report.step_norm = Some(norm_inf(&s));
            store_residuals(spec, &mut r, solution.multipliers)?;
            if !updated {
                if r.report.hessian_resets >= u64::from(settings.max_hessian_resets) {
                    status = Some(Status::NumericalFailure);
                    r.message =
                        "BFGS update failed and the Hessian reset budget is exhausted".into();
                    break;
                }
                b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                r.report.hessian_resets += 1;
            }
            retry_descent = false;
            notify = true;
        }

        r.status = status.unwrap_or(Status::IterationLimit);
        if r.message.is_empty() {
            r.message = status_message(r.status).into();
        }
        r.counts = calls.counts;
        r.report.total_seconds = calls.began.elapsed().as_secs_f64();
        r.report.callback_seconds = calls.seconds;
        Ok(r)
    }
}
