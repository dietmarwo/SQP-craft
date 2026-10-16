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
}

impl Status {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Stub => "stub",
            Status::Cancelled => "cancelled",
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
    /// Completed SQP iterations.
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
