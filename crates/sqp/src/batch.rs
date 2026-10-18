//! Ordered independent solves with explicit outer/inner worker budgets.
use crate::{NativeSqp, SqpOptions};
use sqp_craft::{qp::QpBackend, *};
use sqp_craft_parallel::Execution;

/// Convenience batch API for an infallible worker-local model factory.
/// See [`try_solve_many`] for fallible construction and budget/error semantics.
pub fn solve_many<B, M, F>(
    spec: &ProblemSpec,
    starts: &[Vec<f64>],
    settings: &SqpOptions,
    options: &Options,
    workers: usize,
    factory: F,
    cancel: &CancellationToken,
) -> Result<Vec<Result<SolveResult>>>
where
    B: QpBackend + Default,
    M: NlpProblem,
    F: Fn(usize) -> M + Send + Sync,
{
    try_solve_many::<B, M, _>(
        spec,
        starts,
        settings,
        options,
        workers,
        |i| Ok(factory(i)),
        cancel,
    )
}

/// Solves independent starts using models created on their owning worker.
/// The factory is shared, but the model/backend need neither `Send` nor `Sync`.
/// Results preserve start order, including input, factory and evaluation errors.
/// `workers * options.threads` must not exceed eight. For small NLPs use
/// eight outer workers with single-threaded solves. Observers are not shared.
pub fn try_solve_many<B, M, F>(
    spec: &ProblemSpec,
    starts: &[Vec<f64>],
    settings: &SqpOptions,
    options: &Options,
    workers: usize,
    factory: F,
    cancel: &CancellationToken,
) -> Result<Vec<Result<SolveResult>>>
where
    B: QpBackend + Default,
    M: NlpProblem,
    F: Fn(usize) -> Result<M> + Send + Sync,
{
    options.validate()?;
    settings.validate()?;
    if workers == 0 || workers > 8 || workers.saturating_mul(options.threads) > 8 {
        return Err(Error::InvalidInput(
            "batch outer workers times inner threads must lie in 1..=8".into(),
        ));
    }
    if starts.is_empty() {
        return Ok(Vec::new());
    }
    if cancel.is_cancelled() {
        return Ok(starts
            .iter()
            .map(|x| spec.check_point(x, "start point").map(|_| not_started()))
            .collect());
    }
    let execution = Execution::new(workers).map_err(Error::Unsupported)?;
    Ok(execution.map_with_init(
        starts.len(),
        |_| NativeSqp {
            qp: B::default(),
            settings: settings.clone(),
        },
        |solver, index| {
            spec.check_point(&starts[index], "start point")?;
            if cancel.is_cancelled() {
                return Ok(not_started());
            }
            let mut model = factory(index)?;
            solver.solve(spec, &mut model, &starts[index], options, cancel, None)
        },
    ))
}

fn not_started() -> SolveResult {
    let mut result = SolveResult::empty(Status::Cancelled, "cancelled before batch task started");
    result.report.method = "sqp".into();
    result
}
