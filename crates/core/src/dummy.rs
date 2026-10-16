//! The placeholder backend.

use crate::error::{Error, Result, first_non_finite};
use crate::problem::NlpProblem;
use crate::solver::{
    CancellationToken, Control, Evaluation, EvaluationCounts, Observer, Options, Progress,
    SolveResult, SqpSolver, Status,
};
use crate::spec::ProblemSpec;

/// Evaluates the start point once and stops. **It does not optimize.**
///
/// Exact behavior, which the tests pin down:
///
/// 1. Validate options and `x0` (finite, length `n`, within the variable bounds).
/// 2. Call `functions(x0)` once, then `derivatives(x0)` once, checking
///    cancellation before each call, after the last one and after the
///    observer (which is skipped once cancellation is requested).
/// 3. Return `x0` unchanged as the candidate with [`Status::Stub`],
///    `iterations = 0` and no multipliers, stationarity or complementarity.
///
/// A candidate is returned exactly when the start point was completely
/// evaluated: cancellation before that yields `candidate = None`; an observer
/// stop or a cancellation after it keeps the evaluation. A zero gradient or a
/// feasible start does not change the status: the dummy never claims
/// convergence.
#[derive(Clone, Copy, Debug, Default)]
pub struct DummySolver;

const STUB_MESSAGE: &str =
    "DUMMY backend: start point evaluated once; no QP, step, convergence or KKT claim";

impl SqpSolver for DummySolver {
    fn solve(
        &mut self,
        spec: &ProblemSpec,
        problem: &mut dyn NlpProblem,
        x0: &[f64],
        options: &Options,
        cancel: &CancellationToken,
        observer: Option<&mut dyn Observer>,
    ) -> Result<SolveResult> {
        options.validate()?;
        spec.check_point(x0, "start point")?;

        let mut result = SolveResult {
            report: crate::RunReport {
                method: "dummy".into(),
                ..Default::default()
            },
            dual_sign: None,
            status: Status::Cancelled,
            candidate: None,
            evaluation: None,
            multipliers: None,
            counts: EvaluationCounts::default(),
            iterations: 0,
            stationarity: None,
            complementarity: None,
            message: "cancelled before the start point was completely evaluated".into(),
        };

        if cancel.is_cancelled() {
            return Ok(result);
        }
        let mut constraints = vec![f64::NAN; spec.m()];
        result.counts.functions += 1;
        let objective = problem.functions(x0, &mut constraints)?;
        check_output("objective", &[objective])?;
        check_output("constraint values", &constraints)?;

        if cancel.is_cancelled() {
            return Ok(result);
        }
        let mut gradient = vec![f64::NAN; spec.n()];
        let mut jacobian_values = vec![f64::NAN; spec.nnz()];
        result.counts.derivatives += 1;
        problem.derivatives(x0, &mut gradient, &mut jacobian_values)?;
        check_output("gradient", &gradient)?;
        check_output("Jacobian values", &jacobian_values)?;

        let primal_violation = spec
            .variables()
            .checked_violation(x0)?
            .max(spec.constraints().checked_violation(&constraints)?);
        result.candidate = Some(x0.to_vec());
        result.evaluation = Some(Evaluation {
            objective,
            constraints,
            gradient,
            jacobian_values,
            primal_violation,
            feasible_candidate: primal_violation <= options.feasibility_tolerance,
        });

        if cancel.is_cancelled() {
            result.message = "cancelled after evaluating the start point".into();
            return Ok(result);
        }
        if let Some(observer) = observer {
            let progress = Progress {
                iteration: 0,
                objective,
                primal_violation,
            };
            if observer.observe(&progress)? == Control::Stop {
                result.message = "stopped by the observer after evaluating the start point".into();
                return Ok(result);
            }
        }
        if cancel.is_cancelled() {
            result.message = "cancelled after evaluating the start point".into();
            return Ok(result);
        }
        result.status = Status::Stub;
        result.message = STUB_MESSAGE.into();
        Ok(result)
    }
}

fn check_output(what: &str, values: &[f64]) -> Result<()> {
    match first_non_finite(values) {
        None => Ok(()),
        Some((_, v)) if values.len() == 1 && what == "objective" => {
            Err(Error::Evaluation(format!("objective is {v}")))
        }
        Some((i, v)) => Err(Error::Evaluation(format!(
            "{what}: entry {i} is {v}; every entry must be written and finite"
        ))),
    }
}
