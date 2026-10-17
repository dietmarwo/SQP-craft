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
/// Consecutive steps allowed whose predicted merit decrease is non-positive
/// only within the QP's accuracy (see the line search).
const MAX_NOISE_STEPS: u32 = 3;

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
    /// Contiguous curvature block width; zero keeps full BFGS (default).
    pub hessian_block_size: usize,
    /// Total Hessian resets allowed (failed BFGS updates and non-descent retries).
    pub max_hessian_resets: u32,
    /// Native tolerance passed to the QP backend.
    pub qp_tolerance: f64,
    /// Iteration cap per QP.
    pub qp_max_iterations: u32,
    /// Opt in to compatible QP workspace reuse (not a primal/dual warm start).
    pub qp_reuse_workspace: bool,
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
            hessian_block_size: 0,
            max_hessian_resets: 3,
            qp_tolerance: 1e-10,
            qp_max_iterations: 200,
            qp_reuse_workspace: false,
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
        if self.hessian_block_size > DENSE_SIZE_CAP {
            return Err(Error::InvalidInput(
                "hessian_block_size must lie in 0..=256".into(),
            ));
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

fn record_qp_work(run: &mut sqp_craft::RunReport, qp: &sqp_craft::qp::QpReport) {
    run.qp_setups += qp.setups;
    run.qp_updates += qp.updates;
    run.qp_cold_retries += qp.cold_retries;
    if qp.retry_reason.is_some() {
        run.qp_retry_reason.clone_from(&qp.retry_reason);
    }
    run.qp_conversion_seconds += qp.conversion_seconds;
    run.qp_setup_seconds += qp.setup_seconds;
    run.qp_update_seconds += qp.update_seconds;
    run.qp_solve_seconds += qp.solve_seconds;
    run.qp_check_seconds += qp.check_seconds;
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
    gradient_scratch: Vec<f64>,
    jacobian_scratch: Vec<f64>,
}

impl Calls<'_> {
    fn recycle(&mut self, e: Evaluation) {
        self.gradient_scratch = e.gradient;
        self.jacobian_scratch = e.jacobian_values;
    }
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
        let mut g = std::mem::take(&mut self.gradient_scratch);
        let mut j = std::mem::take(&mut self.jacobian_scratch);
        g.resize(spec.n(), f64::NAN);
        g.fill(f64::NAN);
        j.resize(spec.nnz(), f64::NAN);
        j.fill(f64::NAN);
        self.counts.derivatives += 1;
        let t = Instant::now();
        let result = model.derivatives(x, &mut g, &mut j);
        self.seconds += t.elapsed().as_secs_f64();
        if let Err(err) = result {
            self.gradient_scratch = g;
            self.jacobian_scratch = j;
            return Err(err);
        }
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

/// Moves step components that exceed their QP step bounds by at most
/// `tolerance` onto the bound, counting each move.
///
/// Interior-point solutions satisfy bounds only to their tolerance. Near an
/// active variable bound the remaining room `u − x` can be far smaller than
/// that tolerance (it was 3e-14 in the case that exposed this), so an
/// overshoot of 1e-11 would cut the step length for every variable to almost
/// zero. Larger violations are left for the acceptance check to reject.
fn snap_step(
    step: &mut [f64],
    step_bounds: &Bounds,
    variables: &Bounds,
    tolerance: f64,
    report: &mut sqp_craft::RunReport,
) {
    for (j, p) in step.iter_mut().enumerate() {
        let (lo, up) = (step_bounds.lower[j], step_bounds.upper[j]);
        let target = if *p < lo && lo - *p <= tolerance {
            lo
        } else if *p > up && *p - up <= tolerance {
            up
        } else {
            continue;
        };
        *p = target;
        if variables.lower[j] == variables.upper[j] {
            report.fixed_step_corrections += 1;
        } else {
            report.bound_step_corrections += 1;
        }
    }
}

/// Largest `α ≤ 1` with `x + αp` inside the variable box.
fn box_step_limit(bounds: &Bounds, x: &[f64], p: &[f64]) -> f64 {
    let mut alpha = 1.0_f64;
    for j in 0..x.len() {
        if p[j] > 0.0 && bounds.upper[j].is_finite() {
            alpha = alpha.min((bounds.upper[j] - x[j]) / p[j]);
        }
        if p[j] < 0.0 && bounds.lower[j].is_finite() {
            alpha = alpha.min((bounds.lower[j] - x[j]) / p[j]);
        }
    }
    alpha
}

/// `x + αp`, where a component that floating-point rounding of this sum
/// would place beyond a bound (by at most a few ulps) is set to the bound.
/// A step that lands exactly on a bound, `p_j = u_j − x_j` with `α = 1`, can
/// otherwise round one ulp outside and fail the box contract.
fn trial_point(bounds: &Bounds, x: &[f64], p: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(p)
        .enumerate()
        .map(|(j, (&x, &p))| {
            let t = x + alpha * p;
            let rounding = 4.0 * f64::EPSILON * (x.abs() + (alpha * p).abs());
            let (lo, up) = (bounds.lower[j], bounds.upper[j]);
            if t > up && t - up <= rounding {
                up
            } else if t < lo && lo - t <= rounding {
                lo
            } else {
                t
            }
        })
        .collect()
}

/// The step QP at `x` with model values `e` and Hessian approximation `b`.
#[cfg(test)]
fn qp_model(spec: &ProblemSpec, x: &[f64], e: &Evaluation, b: &DMatrix<f64>) -> Result<QpModel> {
    qp_model_with_blocks(spec, x, e, b, 0)
}
fn qp_model_with_blocks(
    spec: &ProblemSpec,
    x: &[f64],
    e: &Evaluation,
    b: &DMatrix<f64>,
    block: usize,
) -> Result<QpModel> {
    let n = spec.n();
    let mut indptr = vec![0];
    let capacity = n * (if block == 0 { n } else { block.min(n) } + 1) / 2;
    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    for i in 0..n {
        let first = i.checked_div(block).unwrap_or(0) * block;
        for j in first..=i {
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

/// Refresh numeric data in a per-solve, fixed-structure QP template.
fn refresh_qp(
    q: &mut QpModel,
    spec: &ProblemSpec,
    x: &[f64],
    e: &Evaluation,
    b: &DMatrix<f64>,
) -> Result<()> {
    let h = &mut q.hessian_lower;
    for i in 0..spec.n() {
        for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
            h.values[k] = b[(i, h.pattern.indices[k])];
        }
    }
    q.gradient.clone_from(&e.gradient);
    q.matrix.values.clone_from(&e.jacobian_values);
    for (dst, src, values) in [
        (&mut q.variables, spec.variables(), x),
        (&mut q.rows, spec.constraints(), e.constraints.as_slice()),
    ] {
        for (out, input) in [(&mut dst.lower, &src.lower), (&mut dst.upper, &src.upper)] {
            for (i, v) in out.iter_mut().enumerate() {
                *v = if input[i].is_finite() {
                    finite(input[i] - values[i], "shifted bound")?
                } else {
                    input[i]
                };
            }
        }
    }
    Ok(())
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

fn noise_message(predicted: f64, noise: f64) -> String {
    format!(
        "the QP step is below the QP solution accuracy (predicted merit decrease \
         {predicted:.3e}, QP accuracy {noise:.3e}); residuals at the returned point are reported"
    )
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
            gradient_scratch: Vec::new(),
            jacobian_scratch: Vec::new(),
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
        let mut effective_block = if settings.hessian_block_size >= n {
            0
        } else {
            settings.hessian_block_size
        };
        let mut bfgs_work = bfgs::Curvature::new(n, effective_block);
        let mut qp_work: Option<QpModel> = None;
        let mut qp_checks = sqp_craft::qp::QpResidualWorkspace::default();
        let mut penalties = vec![0.0; spec.m()];
        let mut have_penalties = false;
        let mut retry_descent = false;
        let mut noise_steps = 0_u32;
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
            if let Some(q) = qp_work.as_mut() {
                refresh_qp(q, spec, x, e, &b)?;
            } else {
                qp_work = Some(qp_model_with_blocks(spec, x, e, &b, effective_block)?);
            }
            let qp = qp_work.as_ref().expect("QP initialized");
            if r.report.qp_trace.len() < settings.record_qp_limit {
                r.report.qp_trace.push((*qp).clone());
            }
            let remaining = settings
                .time_limit_seconds
                .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0));
            self.qp.configure(&QpOptions {
                reuse_workspace: settings.qp_reuse_workspace,
                tolerance: settings.qp_tolerance,
                acceptance_tolerance: qp_acceptance,
                max_iterations: settings.qp_max_iterations,
                time_limit_seconds: remaining,
            })?;
            let t = Instant::now();
            r.report.qp_solves += 1;
            let outcome = self.qp.solve(qp, None, cancel);
            r.report.qp_seconds += t.elapsed().as_secs_f64();
            let report = self.qp.report();
            record_qp_work(&mut r.report, &report);
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
                    noise_steps = 0;
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
                Ok(QpOutcome::NotPositiveDefinite) => {
                    r.report.qp_not_positive_definite += 1;
                    // Damping keeps B positive definite in exact arithmetic,
                    // but rounding in the explicit update can leave it
                    // indefinite. The backend's exact check is the only
                    // positive-definiteness test of B: treat its rejection
                    // like a failed update (reset, or stop without budget)
                    // and re-solve this QP without counting an iteration.
                    if r.report.hessian_resets >= u64::from(settings.max_hessian_resets) {
                        status = Some(Status::NumericalFailure);
                        r.message = "the Hessian approximation is not positive definite and \
                                     the Hessian reset budget is exhausted"
                            .into();
                        break;
                    }
                    if effective_block > 0 {
                        effective_block = 0;
                        bfgs_work = bfgs::Curvature::new(n, 0);
                        qp_work = None;
                        r.report.block_hessian_fallbacks += 1;
                    }
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    r.report.hessian_resets += 1;
                    retry_descent = false;
                    notify = false;
                    continue;
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
            // Put tolerance-level overshoots of the step bounds (including the
            // zero-width bounds of fixed variables) on the bound; recheck below.
            snap_step(
                &mut solution.step,
                &qp.variables,
                spec.variables(),
                options.feasibility_tolerance,
                &mut r.report,
            );
            let check = qp.accepts_with_workspace(&solution, qp_acceptance, &mut qp_checks)?;
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
            if norm_inf(p) <= 1e-14 * (1.0 + norm_inf(x)) {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::Stalled);
                break;
            }
            // `Some(accuracy)` when this step is taken although its predicted
            // decrease is non-positive within the QP's accuracy.
            let mut noise_step = None;
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
                // With exact QP solutions the predicted decrease is at least
                // pᵀBp > 0. An interior-point QP leaves active rows a little
                // short of their bounds, which costs up to the QP's
                // complementarity residual per entry. A non-positive value
                // within that accuracy means the step is at the QP's noise
                // level. Near a solution the step is then still the right one
                // (pᵀBp ~ 1e-12 while stationarity is just above tolerance),
                // so take it if the merit does not rise by more than that
                // accuracy, a bounded number of times in a row.
                let qp = check.residuals;
                let noise = (spec.m() + n) as f64 * qp.complementarity
                    + qp.stationarity * p.iter().map(|v| v.abs()).sum::<f64>()
                    + record.max_penalty * spec.m() as f64 * qp.primal;
                if -predicted <= noise && noise_steps < MAX_NOISE_STEPS {
                    noise_step = Some(noise);
                } else {
                    if settings.record_history {
                        r.report.history.push(record);
                    }
                    if -predicted <= noise {
                        status = Some(Status::Stalled);
                        r.message = noise_message(predicted, noise);
                    } else {
                        status = Some(Status::NoDescent);
                    }
                    break;
                }
            }
            let merit = finite(e.objective + violation, "merit")?;
            let mut alpha = box_step_limit(spec.variables(), x, p);
            if !(alpha.is_finite() && alpha > 0.0) {
                status = Some(Status::Stalled);
                r.message = "the QP step leaves the variable box immediately".into();
                break;
            }
            let mut accepted = None;
            let max_backtracks = if noise_step.is_some() {
                0
            } else {
                settings.max_backtracks
            };
            for _ in 0..=max_backtracks {
                if let Some(s) = calls.before_call() {
                    status = Some(s);
                    break;
                }
                let trial = trial_point(spec.variables(), x, p, alpha);
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
                let threshold = match noise_step {
                    Some(noise) => merit + noise,
                    None => merit - ARMIJO * alpha * predicted,
                };
                if trial_merit > threshold {
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
                if let Some(noise) = noise_step {
                    status = Some(Status::Stalled);
                    r.message = noise_message(predicted, noise);
                } else {
                    status = Some(Status::LineSearchFailure);
                }
                break;
            };
            noise_steps = if noise_step.is_some() {
                noise_steps + 1
            } else {
                0
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
            let updated = bfgs_work.update(&mut b, &s, &y);
            r.report.bfgs_seconds += t.elapsed().as_secs_f64();
            r.candidate = Some(next);
            if let Some(old) = r.evaluation.replace(next_evaluation) {
                calls.recycle(old);
            }
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
                if effective_block > 0 {
                    effective_block = 0;
                    bfgs_work = bfgs::Curvature::new(n, 0);
                    qp_work = None;
                    r.report.block_hessian_fallbacks += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_curvature_blocks_preserve_spd_and_sparse_pattern() {
        let n = 7;
        let spec =
            ProblemSpec::new(Bounds::free(n), Bounds::free(0), CsrPattern::dense(0, n)).unwrap();
        let e = Evaluation {
            objective: 0.,
            constraints: vec![],
            gradient: vec![0.; n],
            jacobian_values: vec![],
            primal_violation: 0.,
            feasible_candidate: true,
        };
        let mut b = DMatrix::identity(n, n);
        let mut work = bfgs::Curvature::new(n, 3);
        assert!(work.update(&mut b, &[1.; 7], &[2., 3., 4., 1., 2., 3., 4.]));
        assert!(b.clone().cholesky().is_some());
        let q = qp_model_with_blocks(&spec, &[0.; 7], &e, &b, 3).unwrap();
        q.validate().unwrap();
        assert_eq!(q.hessian_lower.values.len(), 13);
        for i in 0..n {
            for j in 0..n {
                if i / 3 != j / 3 {
                    assert_eq!(b[(i, j)], 0.);
                }
            }
        }
        for i in 0..n {
            assert!(
                q.hessian_lower
                    .pattern
                    .row(i)
                    .iter()
                    .all(|&j| i / 3 == j / 3)
            );
        }
        assert!(
            SqpOptions {
                hessian_block_size: 257,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn recycled_callback_buffers_still_reject_missing_entries() {
        struct Partial {
            missing: bool,
        }
        impl NlpProblem for Partial {
            fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
                c[0] = x[0] + x[1];
                Ok(0.)
            }
            fn derivatives(&mut self, _: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
                if self.missing {
                    g[0] = 1.;
                } else {
                    g.fill(1.);
                    j.fill(1.);
                }
                Ok(())
            }
        }
        let spec =
            ProblemSpec::new(Bounds::free(2), Bounds::free(1), CsrPattern::dense(1, 2)).unwrap();
        let settings = SqpOptions::default();
        let cancel = CancellationToken::default();
        let mut calls = Calls {
            began: Instant::now(),
            settings: &settings,
            cancel: &cancel,
            counts: EvaluationCounts::default(),
            seconds: 0.,
            gradient_scratch: vec![],
            jacobian_scratch: vec![],
        };
        let mut model = Partial { missing: false };
        let x = [1., 1.];
        let v = calls.functions(&spec, &mut model, &x).unwrap();
        let e = calls.derivatives(&spec, &mut model, &x, v, 1e-7).unwrap();
        calls.recycle(e);
        model.missing = true;
        let v = calls.functions(&spec, &mut model, &x).unwrap();
        assert!(calls.derivatives(&spec, &mut model, &x, v, 1e-7).is_err());
    }
    #[test]
    fn reused_qp_template_matches_fresh_numeric_data() {
        let spec = ProblemSpec::new(
            Bounds::new(vec![0.; 3], vec![4.; 3]),
            Bounds::new(vec![-1., 0.], vec![2., 0.]),
            CsrPattern::dense(2, 3),
        )
        .unwrap();
        let mut e = Evaluation {
            objective: 0.,
            constraints: vec![0., 0.],
            gradient: vec![1., 2., 3.],
            jacobian_values: vec![1.; 6],
            primal_violation: 0.,
            feasible_candidate: true,
        };
        let mut b = DMatrix::identity(3, 3);
        let mut q = qp_model(&spec, &[1.; 3], &e, &b).unwrap();
        let pointers = (
            q.hessian_lower.pattern.indices.as_ptr(),
            q.matrix.pattern.indices.as_ptr(),
            q.hessian_lower.values.as_ptr(),
            q.rows.lower.as_ptr(),
        );
        for step in 1..5 {
            e.constraints[0] = step as f64 / 10.;
            e.jacobian_values[2] = step as f64;
            b[(1, 1)] = step as f64;
            let x = [0.5 + step as f64 / 10.; 3];
            refresh_qp(&mut q, &spec, &x, &e, &b).unwrap();
            assert_eq!(q, qp_model(&spec, &x, &e, &b).unwrap());
            assert_eq!(
                pointers,
                (
                    q.hessian_lower.pattern.indices.as_ptr(),
                    q.matrix.pattern.indices.as_ptr(),
                    q.hessian_lower.values.as_ptr(),
                    q.rows.lower.as_ptr()
                )
            );
        }
    }
    #[test]
    fn a_trial_capped_at_a_bound_stays_inside_the_box() {
        // With α = (u − x)/p, x + αp rounds to one ulp above u here (about 2% of
        // random cases do); the delivered line search then halved α.
        let (x, u, p) = (0.11408985987451992, 0.9812160792571979, 2.958067622106532);
        let bounds = Bounds::new(vec![-1.0], vec![u]);
        let alpha = box_step_limit(&bounds, &[x], &[p]);
        assert!(x + alpha * p > u);
        assert_eq!(trial_point(&bounds, &[x], &[p], alpha), vec![u]);
        // Interior trials are untouched, and a real overshoot is not hidden.
        assert_eq!(
            trial_point(&bounds, &[x], &[p], 0.5 * alpha),
            vec![x + 0.5 * alpha * p]
        );
        assert_eq!(trial_point(&bounds, &[x], &[p], 1.0), vec![x + p]);
    }

    #[test]
    fn snapping_moves_only_tolerance_level_overshoots() {
        let variables = Bounds::new(vec![0.0, 1.0, -5.0], vec![2.0, 1.0, 5.0]);
        let x = [2.0 - 3e-14, 1.0, 0.0];
        let room = 2.0 - x[0];
        let step_bounds = Bounds::new(vec![-1.0, 0.0, -1.0], vec![room, 0.0, 1.0]);
        let mut report = sqp_craft::RunReport::default();
        let mut step = vec![room + 1e-11, -2e-12, 1.5];
        assert!(box_step_limit(&variables, &x, &step) < 0.01);
        snap_step(&mut step, &step_bounds, &variables, 1e-7, &mut report);
        assert_eq!(
            step,
            vec![room, 0.0, 1.5],
            "a 0.5 overshoot is left for the QP check"
        );
        assert_eq!(
            (report.bound_step_corrections, report.fixed_step_corrections),
            (1, 1)
        );
        assert_eq!(box_step_limit(&variables, &x, &step), 1.0);
    }
}
