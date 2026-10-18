//! Ordered independent solves with explicit outer/inner worker budgets.
use crate::{NativeSqp, SqpOptions};
use sqp_craft::{qp::QpBackend, *};
use sqp_craft_parallel::Execution;

/// Solves independent starts using models created on their owning worker.
/// The factory is shared, but the model/backend need neither `Send` nor `Sync`.
/// Results preserve start order, including individual input/evaluation errors.
/// `workers * options.threads` must not exceed eight. For small NLPs use
/// eight outer workers with single-threaded solves. Observers are not shared.
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
    let execution = Execution::new(workers).map_err(Error::Unsupported)?;
    Ok(execution.map(starts.len(), |index| {
        let mut model = factory(index);
        let mut solver = NativeSqp {
            qp: B::default(),
            settings: settings.clone(),
        };
        solver.solve(spec, &mut model, &starts[index], options, cancel, None)
    }))
}
