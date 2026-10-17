//! Solver trait, options, progress reporting and results.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result, ensure};
use crate::problem::NlpProblem;
use crate::spec::ProblemSpec;

/// How a solve ended. More variants will be added with real backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    /// The dummy backend evaluated the start point and stopped. Not a
    /// convergence claim of any kind.
    Stub,
    /// Stopped by the cancellation token or by an observer.
    Cancelled,
    /// Native SQP termination: converged.
    Converged,
    /// Native SQP termination: iteration_limit.
    IterationLimit,
    /// Native SQP termination: evaluation_limit.
    EvaluationLimit,
    /// Native SQP termination: time_limit.
    TimeLimit,
    /// Native SQP termination: qp_failure.
    QpFailure,
    /// Native SQP termination: linearization_infeasible.
    LinearizationInfeasible,
    /// Local feasibility restoration could not find an acceptable step.
    RestorationFailed,
    /// The enabled restoration step budget was exhausted.
    RestorationLimit,
    /// Elastic mode reached a stationary point of the ℓ1 penalty function at
    /// which constraints are still violated, with the elastic weight at its
    /// largest value. Not an infeasibility certificate.
    LocallyInfeasible,
    /// Native SQP termination: line_search_failure.
    LineSearchFailure,
    /// Native SQP termination: no_descent.
    NoDescent,
    /// Native SQP termination: stalled.
    Stalled,
    /// Native SQP termination: numerical_failure.
    NumericalFailure,
}

impl Status {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Stub => "stub",
            Status::Cancelled => "cancelled",
            Status::Converged => "converged",
            Status::IterationLimit => "iteration_limit",
            Status::EvaluationLimit => "evaluation_limit",
            Status::TimeLimit => "time_limit",
            Status::QpFailure => "qp_failure",
            Status::LinearizationInfeasible => "linearization_infeasible",
            Status::RestorationFailed => "restoration_failed",
            Status::RestorationLimit => "restoration_limit",
            Status::LocallyInfeasible => "locally_infeasible",
            Status::LineSearchFailure => "line_search_failure",
            Status::NoDescent => "no_descent",
            Status::Stalled => "stalled",
            Status::NumericalFailure => "numerical_failure",
        }
    }
}

/// Settings accepted by the solvers in this crate.
///
/// Only settings that a backend actually honours are exposed. Scaling,
/// iteration and evaluation limits, Hessian choices and similar knobs will be
/// added together with the backend that implements them.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Absolute tolerance, in original unscaled units, on the largest bound
    /// violation of variables and constraints for
    /// [`Evaluation::feasible_candidate`].
    pub feasibility_tolerance: f64,
    /// Threads used inside one solve. Only `1` is supported; use independent
    /// solves (with independent model instances) for parallelism.
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            feasibility_tolerance: 1e-7,
            threads: 1,
        }
    }
}

impl Options {
    /// Rejects malformed values (`InvalidInput`) and settings no backend
    /// supports yet (`Unsupported`).
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.feasibility_tolerance.is_finite() && self.feasibility_tolerance > 0.0,
            || {
                format!(
                    "feasibility_tolerance must be positive and finite, got {}",
                    self.feasibility_tolerance
                )
            },
        )?;
        ensure(self.threads > 0, || "threads must be at least 1".into())?;
        if self.threads != 1 {
            return Err(Error::Unsupported(format!(
                "threads = {}: only single-threaded solves are implemented; \
                 run independent solves for parallelism",
                self.threads
            )));
        }
        Ok(())
    }
}

/// A cooperative cancellation flag that can be shared across threads.
///
/// Solvers check it between model calls; it cannot interrupt a running call.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Requests cancellation. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Number of model calls made during a solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationCounts {
    /// Calls of [`NlpProblem::functions`].
    pub functions: u64,
    /// Calls of [`NlpProblem::derivatives`].
    pub derivatives: u64,
}

/// Model values at the returned candidate, plus its feasibility measure.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// `f(x)`.
    pub objective: f64,
    /// `c(x)`, length `m`.
    pub constraints: Vec<f64>,
    /// `grad f(x)`, length `n`.
    pub gradient: Vec<f64>,
    /// Jacobian values in pattern order, length `nnz`.
    pub jacobian_values: Vec<f64>,
    /// Largest bound violation over variables and constraints, absolute and
    /// unscaled; `0.0` if none.
    pub primal_violation: f64,
    /// `primal_violation <= feasibility_tolerance`. A numerical statement about
    /// this point only: neither optimality nor (if false) infeasibility of the
    /// problem follows from it.
    pub feasible_candidate: bool,
}

/// Lagrange multipliers.
///
/// Sign convention, shared by the NLP results and the QP seam: at a first-order
/// point
///
/// ```text
/// grad f(x) = J(x)^T * constraints + variables
/// ```
///
/// (the Lagrangian is `f - constraints^T c(x) - variables^T x`). A positive
/// entry belongs to an active lower bound, a negative entry to an active upper
/// bound, zero to an inactive one. This is the convention of HiGHS, the likely
/// first QP backend. [`kkt::residuals`](crate::kkt::residuals) checks
/// multipliers in exactly this convention.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Multipliers {
    /// One per constraint row (length `m`).
    pub constraints: Vec<f64>,
    /// One per variable bound pair (length `n`).
    pub variables: Vec<f64>,
}

/// Information passed to an [`Observer`].
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    /// Iteration number; `0` reports the start point, not an optimization step.
    pub iteration: u64,
    /// Objective at the current iterate.
    pub objective: f64,
    /// Largest bound violation at the current iterate.
    pub primal_violation: f64,
}

/// An observer's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// Keep going.
    Continue,
    /// End the solve with [`Status::Cancelled`], keeping the current candidate.
    Stop,
}

/// Receives progress reports. An error aborts the solve.
pub trait Observer {
    /// Called once per iteration, starting with iteration `0`.
    fn observe(&mut self, progress: &Progress) -> Result<Control>;
}

impl<F> Observer for F
where
    F: FnMut(&Progress) -> Result<Control>,
{
    fn observe(&mut self, progress: &Progress) -> Result<Control> {
        self(progress)
    }
}

/// Outcome of a solve that was not aborted by an error.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Run metadata, diagnostics and optional bounded QP trace.
    pub report: RunReport,
    /// Sign residual of the current multipliers.
    pub dual_sign: Option<f64>,
    /// How the solve ended.
    pub status: Status,
    /// The returned point. `None` if no complete evaluation happened.
    pub candidate: Option<Vec<f64>>,
    /// Model values at `candidate`; present exactly when `candidate` is.
    pub evaluation: Option<Evaluation>,
    /// Multiplier estimates, when the backend computes them.
    pub multipliers: Option<Multipliers>,
    /// Model calls made, including calls whose results were discarded.
    pub counts: EvaluationCounts,
    /// Accepted steps, including feasibility-restoration steps.
    pub iterations: u64,
    /// `‖grad f - J^T λ - z‖∞` at the candidate, when multipliers exist.
    pub stationarity: Option<f64>,
    /// Complementarity residual at the candidate, when multipliers exist.
    pub complementarity: Option<f64>,
    /// Human-readable explanation of `status`.
    pub message: String,
}

/// A nonlinear programming solver.
///
/// `&mut self` lets a solver keep reusable workspace across solves; a solver
/// instance serves one solve at a time.
pub trait SqpSolver {
    /// Solves `spec` with `problem` from the start point `x0`.
    ///
    /// Malformed input is rejected with [`Error::InvalidInput`] before any
    /// model call; `x0` must be finite and within the variable bounds.
    fn solve(
        &mut self,
        spec: &ProblemSpec,
        problem: &mut dyn NlpProblem,
        x0: &[f64],
        options: &Options,
        cancel: &CancellationToken,
        observer: Option<&mut dyn Observer>,
    ) -> Result<SolveResult>;
}

/// Native-loop statistics. Timings are elapsed inclusive phase times, not pure GIL wait.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunReport {
    /// Selected method.
    pub method: String,
    /// QP engine and version.
    pub qp_backend: String,
    /// Last raw QP status.
    pub qp_status: Option<String>,
    /// Number of QP calls including retries.
    pub qp_solves: u64,
    /// Total iterations of all QP calls.
    pub qp_iterations: u64,
    /// Number of rejected merit/domain trials.
    pub rejected_trials: u64,
    /// Recoverable undefined evaluations.
    pub undefined_trials: u64,
    /// Positive-diagonal Hessian resets.
    pub hessian_resets: u64,
    /// BFGS updates that used Powell's damping (`sᵀy < 0.2·sᵀBs` and, with
    /// the augmented-Lagrangian repair, no repair available).
    pub bfgs_damped_updates: u64,
    /// BFGS updates whose `y` was repaired by SNOPT's augmented-Lagrangian
    /// modification.
    pub bfgs_modified_updates: u64,
    /// Switches from requested block curvature to full BFGS after a failed update.
    pub block_hessian_fallbacks: u64,
    /// Step QPs the backend rejected as `QpOutcome::NotPositiveDefinite`; each
    /// is followed by a Hessian reset (counted in `hessian_resets`) or a stop.
    pub qp_not_positive_definite: u64,
    /// Accepted steps whose elastic QP left at least one row violated.
    pub elastic_steps: u64,
    /// Elastic weight at the end of an elastic-mode solve, `None` otherwise.
    pub elastic_weight: Option<f64>,
    /// Times the elastic weight was multiplied by ten at a stationary point of
    /// the ℓ1 penalty with violated constraints.
    pub elastic_weight_increases: u64,
    /// Second-order corrections evaluated after a rejected full step.
    pub second_order_corrections: u64,
    /// Second-order corrections accepted as the step.
    pub second_order_steps: u64,
    /// Explicit corrections of tiny fixed-variable step components.
    pub fixed_step_corrections: u64,
    /// Step components moved onto a variable bound they exceeded by at most the
    /// feasibility tolerance (interior-point roundoff near an active bound).
    pub bound_step_corrections: u64,
    /// Last accepted displacement infinity norm.
    pub step_norm: Option<f64>,
    /// End-to-end elapsed solve time after validation.
    pub total_seconds: f64,
    /// Callback boundary execution and conversion time.
    pub callback_seconds: f64,
    /// QP conversion, solve and check time.
    pub qp_seconds: f64,
    /// Fresh QP engine setup attempts.
    pub qp_setups: u64,
    /// Successful numeric QP workspace updates.
    pub qp_updates: u64,
    /// Fresh retries after a failed update or reused solve.
    pub qp_cold_retries: u64,
    /// Last reason requesting a QP cold retry, even if its budget was exhausted.
    pub qp_retry_reason: Option<String>,
    /// QP conversion and positive-definiteness validation time.
    pub qp_conversion_seconds: f64,
    /// Fresh QP engine setup time.
    pub qp_setup_seconds: f64,
    /// QP numeric data update time.
    pub qp_update_seconds: f64,
    /// QP engine solve time, including retries.
    pub qp_solve_seconds: f64,
    /// Independent backend QP acceptance-check time.
    pub qp_check_seconds: f64,
    /// BFGS and positive-definiteness-check time.
    pub bfgs_seconds: f64,
    /// Optional bounded history in original step coordinates.
    pub qp_trace: Vec<crate::qp::QpModel>,
    /// Optional per-iteration record (empty unless the backend was asked for it).
    pub history: Vec<IterationRecord>,
    /// Number of entries into feasibility restoration, including failures.
    pub restoration_attempts: u64,
    /// Accepted feasibility steps; also included in `SolveResult::iterations`.
    pub restoration_steps: u64,
    /// Inclusive restoration time, overlapping QP and callback phase timings.
    pub restoration_seconds: f64,
    /// Accepted restoration steps, populated when iteration history is enabled.
    pub restoration_history: Vec<RestorationRecord>,
}

/// One SQP iteration: the iterate where a QP was solved and the step taken
/// from it. Recorded by backends on request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationRecord {
    /// Number of accepted steps before this iterate (`0` is the start point).
    pub iteration: u64,
    /// Objective at the iterate.
    pub objective: f64,
    /// Largest bound violation at the iterate.
    pub primal_violation: f64,
    /// Stationarity at the iterate with the QP's multipliers: the value tested
    /// for convergence.
    pub stationarity: f64,
    /// Complementarity at the iterate with the QP's multipliers.
    pub complementarity: f64,
    /// `‖p‖∞` of the QP step.
    pub qp_step_norm: f64,
    /// Step length `α` of the accepted step; `None` if no step was taken.
    pub step_length: Option<f64>,
    /// Trial points evaluated by the line search, including the accepted one.
    pub trials: u32,
    /// Largest merit-penalty weight used for the line search (`0` without rows).
    pub max_penalty: f64,
    /// Backend iterations of the QP solved at this iterate.
    pub qp_iterations: u64,
    /// The accepted step is a second-order correction of the QP step.
    pub second_order_correction: bool,
    /// How the Hessian approximation was updated after the step; `None` if
    /// no step was taken.
    pub curvature: Option<CurvatureUpdate>,
}

/// How the quasi-Newton matrix `B` was updated after an accepted step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CurvatureUpdate {
    /// Plain BFGS: the curvature `sᵀy` was at least `0.2·sᵀBs`.
    Exact,
    /// Powell's damped BFGS: `y` was blended with `Bs`.
    Damped,
    /// `y` was repaired by the augmented-Lagrangian modification (SNOPT).
    Modified,
    /// The displacement was negligible, so `B` was kept unchanged.
    Negligible,
    /// The update failed numerically and `B` was reset (or the solve stopped).
    Reset,
}

impl CurvatureUpdate {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            CurvatureUpdate::Exact => "exact",
            CurvatureUpdate::Damped => "damped",
            CurvatureUpdate::Modified => "modified",
            CurvatureUpdate::Negligible => "negligible",
            CurvatureUpdate::Reset => "reset",
        }
    }
}
impl SolveResult {
    /// Empty result before a complete evaluation exists.
    pub fn empty(status: Status, message: impl Into<String>) -> Self {
        Self {
            status,
            candidate: None,
            evaluation: None,
            multipliers: None,
            counts: EvaluationCounts::default(),
            iterations: 0,
            stationarity: None,
            complementarity: None,
            dual_sign: None,
            message: message.into(),
            report: RunReport::default(),
        }
    }
}

/// One accepted feasibility-restoration step. Multipliers are intentionally
/// absent: restoration-QP duals are not NLP multipliers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestorationRecord {
    /// Accepted-step count before this restoration step.
    pub iteration: u64,
    /// Objective at the preceding complete point.
    pub objective_before: f64,
    /// Objective at the accepted point; it may increase during restoration.
    pub objective_after: f64,
    /// Original-unit maximum violation before the step.
    pub primal_before: f64,
    /// Original-unit maximum violation after the step.
    pub primal_after: f64,
    /// Half the squared row-violation norm before the step.
    pub merit_before: f64,
    /// Half the squared row-violation norm after the step.
    pub merit_after: f64,
    /// Accepted step length.
    pub step_length: f64,
    /// Number of trial function evaluations, including undefined ones.
    pub trials: u32,
    /// Iterations of this restoration QP.
    pub qp_iterations: u64,
}
