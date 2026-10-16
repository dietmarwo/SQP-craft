//! Restoration control paths use a deliberately inconsistent nonlinear problem:
//! the solver may improve violation but must never claim feasibility.
use sqp_craft::*;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

struct Inconsistent {
    last: Vec<f64>,
    token: Option<CancellationToken>,
}
impl NlpProblem for Inconsistent {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        assert!((0.0..=3.0).contains(&x[0]));
        assert_eq!(x[1], 4.0);
        self.last = x.to_vec();
        c.copy_from_slice(&[x[0], x[0]]);
        if x[0] > 0.0 {
            if let Some(token) = &self.token {
                token.cancel();
            }
        }
        Ok(x[0] * x[0])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        assert_eq!(x, self.last);
        g.copy_from_slice(&[2.0 * x[0], 0.0]);
        j.copy_from_slice(&[1.0, 0.0, 1.0, 0.0]);
        Ok(())
    }
}
fn spec() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![0.0, 4.0], vec![3.0, 4.0]),
        Bounds::fixed(vec![1.0, 2.0]),
        CsrPattern::dense(2, 2),
    )
    .unwrap()
}
fn run(
    settings: SqpOptions,
    cancel_on_trial: bool,
    observer: Option<&mut dyn Observer>,
) -> SolveResult {
    let token = CancellationToken::default();
    let mut model = Inconsistent {
        last: vec![],
        token: cancel_on_trial.then(|| token.clone()),
    };
    NativeSqp {
        qp: ClarabelQp::default(),
        settings,
    }
    .solve(
        &spec(),
        &mut model,
        &[0.0, 4.0],
        &Options::default(),
        &token,
        observer,
    )
    .unwrap()
}
#[test]
fn inconsistent_problem_is_not_certified_and_fixed_coordinate_never_moves() {
    let r = run(
        SqpOptions {
            record_history: true,
            ..Default::default()
        },
        false,
        None,
    );
    assert!(
        matches!(
            r.status,
            Status::RestorationFailed | Status::RestorationLimit
        ),
        "{}",
        r.message
    );
    assert!(!r.evaluation.unwrap().feasible_candidate);
    assert!(r.report.restoration_steps >= 1);
    assert!(
        r.report
            .restoration_history
            .iter()
            .all(|h| h.merit_after < h.merit_before)
    );
    assert_eq!(r.candidate.unwrap()[1], 4.0);
    assert!(r.multipliers.is_none() && r.stationarity.is_none() && r.dual_sign.is_none());
}
#[test]
fn disabled_restoration_and_distinct_outer_and_restoration_limits() {
    let r = run(
        SqpOptions {
            max_restoration_steps: 0,
            ..Default::default()
        },
        false,
        None,
    );
    assert_eq!(r.status, Status::LinearizationInfeasible);
    assert_eq!(r.report.restoration_attempts, 0);
    let r = run(
        SqpOptions {
            max_restoration_steps: 1,
            ..Default::default()
        },
        false,
        None,
    );
    assert_eq!(r.status, Status::RestorationLimit);
    assert_eq!(r.report.restoration_steps, 1);
    assert_eq!(r.iterations, 1);
    let r = run(
        SqpOptions {
            max_iterations: 1,
            ..Default::default()
        },
        false,
        None,
    );
    assert_eq!(r.status, Status::IterationLimit);
    assert_eq!(r.report.restoration_steps, 1);
}
#[test]
fn partial_trial_budget_and_cancellation_retain_the_last_complete_point() {
    let r = run(
        SqpOptions {
            max_callback_calls: 3,
            ..Default::default()
        },
        false,
        None,
    );
    assert_eq!(r.status, Status::EvaluationLimit);
    assert_eq!(r.counts.functions, 2);
    assert_eq!(r.counts.derivatives, 1);
    assert_eq!(r.candidate, Some(vec![0.0, 4.0]));
    assert_eq!(r.evaluation.unwrap().constraints, vec![0.0, 0.0]);
    let r = run(SqpOptions::default(), true, None);
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!(r.candidate, Some(vec![0.0, 4.0]));
    assert_eq!(r.report.restoration_steps, 0);
}
#[test]
fn observer_stop_after_restoration_has_no_stale_multipliers() {
    let mut events = vec![];
    let mut observer = |p: &Progress| {
        events.push(p.iteration);
        Ok(if p.iteration == 1 {
            Control::Stop
        } else {
            Control::Continue
        })
    };
    let r = run(SqpOptions::default(), false, Some(&mut observer));
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!(events, vec![0, 1]);
    assert!(r.candidate.unwrap()[0] > 0.0);
    assert!(r.multipliers.is_none() && r.stationarity.is_none());
    assert_eq!(r.report.qp_solves, 2);
}
#[test]
fn restoration_options_are_validated_before_callbacks() {
    for invalid in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(
            SqpOptions {
                restoration_step_limit: invalid,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SqpOptions {
                restoration_regularization: invalid,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
