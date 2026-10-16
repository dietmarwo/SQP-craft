//! The model callback contract.

use crate::error::Result;

/// A smooth nonlinear model evaluated by a solver.
///
/// Evaluations come in two groups, because that is how SQP uses them: a line
/// search needs `f` and `c` together at every trial point, and the QP
/// subproblem and quasi-Newton update need `grad f` and `J` together at every
/// accepted point. Trajectory models usually compute `f` and `c` in one
/// propagation; grouping them avoids recomputation or ad-hoc caching and halves
/// the number of calls across a Python boundary.
///
/// # Contract kept by every solver in this crate
///
/// * `x` has length `n`, is finite, and lies within the variable bounds of the
///   [`ProblemSpec`](crate::ProblemSpec). The model is never evaluated outside
///   the box, so it may assume, for example, a positive mass or time of flight.
/// * [`derivatives`](Self::derivatives) is only called at the `x` of the
///   immediately preceding [`functions`](Self::functions) call. A model may
///   therefore reuse intermediate results (a propagated trajectory, say)
///   computed by `functions`.
/// * Output buffers have exactly the required length. Their prior content is
///   unspecified (solvers here fill them with NaN), so the model must write
///   every entry; an entry left unwritten is reported as non-finite.
///
/// # Model obligations
///
/// * Jacobian values are written in the order of the fixed
///   [`CsrPattern`](crate::CsrPattern) of the spec, including entries whose
///   value happens to be zero at `x`.
/// * All returned values must be finite. Return
///   [`Error::UndefinedPoint`](crate::Error::UndefinedPoint) when the model is
///   not defined at `x`, and [`Error::Evaluation`](crate::Error::Evaluation)
///   for any other failure.
///
/// The trait deliberately requires neither `Send` nor `Sync`: a model instance
/// is mutable and local to one solve. Parallel strategies run independent
/// solves with independent model instances.
pub trait NlpProblem {
    /// A line search consumed an UndefinedPoint error; release its pending exception state.
    fn discard_undefined(&mut self) {}

    /// Evaluates the objective `f(x)`, returned, and the constraints `c(x)`,
    /// written to `constraints` (length `m`).
    fn functions(&mut self, x: &[f64], constraints: &mut [f64]) -> Result<f64>;

    /// Evaluates the objective gradient (length `n`) and the Jacobian values
    /// (length `nnz`, in pattern order).
    fn derivatives(&mut self, x: &[f64], gradient: &mut [f64], jacobian: &mut [f64]) -> Result<()>;
}
