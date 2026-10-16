//! The exact behavior of the dummy backend and the callback contract.

mod common;

use common::{Toy, solve, spec};
use sqp_craft::{
    CancellationToken, Control, DummySolver, Error, NlpProblem, Options, Progress, Result,
    SqpSolver, Status,
};

fn no_cancel() -> CancellationToken {
    CancellationToken::default()
}

#[test]
fn evaluates_start_once_in_order_and_claims_nothing() {
    let mut toy = Toy::default();
    // The start is the exact minimizer: the dummy must still not claim convergence.
    let r = solve(&mut toy, &[1.0, 2.0], &no_cancel(), None).unwrap();
    assert_eq!(r.status, Status::Stub);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.multipliers, None);
    assert_eq!(r.stationarity, None);
    assert_eq!(r.complementarity, None);
    assert_eq!((r.counts.functions, r.counts.derivatives), (1, 1));
    assert_eq!(r.candidate.as_deref(), Some(&[1.0, 2.0][..]));
    let e = r.evaluation.unwrap();
    assert_eq!(e.objective, 0.0);
    assert_eq!(e.constraints, [3.0]);
    assert_eq!(e.gradient, [0.0, 0.0]);
    assert_eq!(e.jacobian_values, [1.0, 1.0]);
    assert!(e.feasible_candidate);
    // derivatives() follows functions() at the same point.
    let names: Vec<_> = toy.calls.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["functions", "derivatives"]);
    assert_eq!(toy.calls[0].1, toy.calls[1].1);
}

#[test]
fn constraint_violation_is_reported_not_repaired() {
    let r = solve(&mut Toy::default(), &[0.0, 0.0], &no_cancel(), None).unwrap();
    assert_eq!(r.status, Status::Stub);
    assert_eq!(r.candidate.as_deref(), Some(&[0.0, 0.0][..]));
    let e = r.evaluation.unwrap();
    assert_eq!(e.primal_violation, 3.0);
    assert!(!e.feasible_candidate);
}

#[test]
fn start_outside_variable_bounds_is_rejected_before_any_call() {
    let mut toy = Toy::default();
    let err = solve(&mut toy, &[-1.0, 0.0], &no_cancel(), None).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidInput(m) if m.contains("entry 0 = -1")),
        "{err}"
    );
    assert!(toy.calls.is_empty());
}

#[test]
fn malformed_start_and_options_are_rejected() {
    let cases: [(&[f64], Options); 4] = [
        (&[1.0], Options::default()),
        (&[f64::NAN, 2.0], Options::default()),
        (
            &[1.0, 2.0],
            Options {
                feasibility_tolerance: 0.0,
                ..Options::default()
            },
        ),
        (
            &[1.0, 2.0],
            Options {
                threads: 0,
                ..Options::default()
            },
        ),
    ];
    for (x0, options) in cases {
        let r = DummySolver.solve(
            &spec(),
            &mut Toy::default(),
            x0,
            &options,
            &no_cancel(),
            None,
        );
        assert!(
            matches!(r, Err(Error::InvalidInput(_))),
            "{x0:?} {options:?}"
        );
    }
    let threads = Options {
        threads: 16,
        ..Options::default()
    };
    let r = DummySolver.solve(
        &spec(),
        &mut Toy::default(),
        &[1.0, 2.0],
        &threads,
        &no_cancel(),
        None,
    );
    assert!(matches!(r, Err(Error::Unsupported(_))));
}

#[test]
fn input_errors_take_precedence_over_cancellation() {
    let cancel = no_cancel();
    cancel.cancel();
    assert!(solve(&mut Toy::default(), &[9.0, 9.0], &cancel, None).is_err());
}

#[test]
fn cancellation_before_start_makes_no_call() {
    let cancel = no_cancel();
    cancel.cancel();
    let mut toy = Toy::default();
    let r = solve(&mut toy, &[1.0, 2.0], &cancel, None).unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!(r.candidate, None);
    assert_eq!(r.evaluation, None);
    assert_eq!(r.counts.functions, 0);
    assert!(toy.calls.is_empty());
}

/// Cancels inside `functions`, as a callback or another thread may do.
struct CancelInFunctions(CancellationToken, Toy);

impl NlpProblem for CancelInFunctions {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.0.cancel();
        self.1.functions(x, c)
    }
    fn derivatives(&mut self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<()> {
        panic!("cancellation checkpoint between the two calls was skipped");
    }
}

#[test]
fn cancellation_between_calls_returns_no_partial_candidate() {
    let cancel = no_cancel();
    let mut model = CancelInFunctions(cancel.clone(), Toy::default());
    let r = solve(&mut model, &[1.0, 2.0], &cancel, None).unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!((r.counts.functions, r.counts.derivatives), (1, 0));
    assert_eq!(r.candidate, None);
}

/// Cancels inside `derivatives`, after which the evaluation is complete.
struct CancelInDerivatives(CancellationToken, Toy);

impl NlpProblem for CancelInDerivatives {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.1.functions(x, c)
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        self.0.cancel();
        self.1.derivatives(x, g, jac)
    }
}

#[test]
fn cancellation_after_complete_evaluation_keeps_it_and_skips_observer() {
    let cancel = no_cancel();
    let mut model = CancelInDerivatives(cancel.clone(), Toy::default());
    let mut never = |_: &Progress| -> Result<Control> { panic!("observer called after cancel") };
    let r = solve(&mut model, &[1.0, 2.0], &cancel, Some(&mut never)).unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!(r.candidate.as_deref(), Some(&[1.0, 2.0][..]));
    assert!(r.evaluation.is_some());
}

#[test]
fn observer_sees_start_and_can_stop() {
    let mut seen = Vec::new();
    let mut stop = |p: &Progress| -> Result<Control> {
        seen.push(p.clone());
        Ok(Control::Stop)
    };
    let r = solve(
        &mut Toy::default(),
        &[0.0, 0.0],
        &no_cancel(),
        Some(&mut stop),
    )
    .unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert!(
        r.evaluation.is_some(),
        "observer stop keeps the complete evaluation"
    );
    assert_eq!(
        seen,
        [Progress {
            iteration: 0,
            objective: 5.0,
            primal_violation: 3.0
        }]
    );
}

#[test]
fn observer_error_aborts() {
    let mut fail = |_: &Progress| -> Result<Control> { Err(Error::Evaluation("observer".into())) };
    let r = solve(
        &mut Toy::default(),
        &[1.0, 2.0],
        &no_cancel(),
        Some(&mut fail),
    );
    assert_eq!(r.unwrap_err(), Error::Evaluation("observer".into()));
}

/// Leaves the second gradient entry unwritten.
struct Forgetful;

impl NlpProblem for Forgetful {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        Toy::default().functions(x, c)
    }
    fn derivatives(&mut self, _: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        g[0] = 0.0;
        jac.fill(1.0);
        Ok(())
    }
}

#[test]
fn unwritten_output_entries_are_detected() {
    let err = solve(&mut Forgetful, &[1.0, 2.0], &no_cancel(), None).unwrap_err();
    assert!(
        matches!(&err, Error::Evaluation(m) if m.contains("gradient: entry 1")),
        "{err}"
    );
}

/// Returns a non-finite objective or reports an undefined point.
struct Broken(Error);

impl NlpProblem for Broken {
    fn functions(&mut self, _: &[f64], c: &mut [f64]) -> Result<f64> {
        c.fill(0.0);
        match &self.0 {
            Error::Evaluation(m) if m == "nan" => Ok(f64::NAN),
            e => Err(e.clone()),
        }
    }
    fn derivatives(&mut self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn model_failures_abort_with_their_kind() {
    let nan = solve(
        &mut Broken(Error::Evaluation("nan".into())),
        &[1.0, 2.0],
        &no_cancel(),
        None,
    );
    assert!(matches!(nan, Err(Error::Evaluation(m)) if m == "objective is NaN"));
    let undefined = Error::UndefinedPoint("propagation failed".into());
    let r = solve(
        &mut Broken(undefined.clone()),
        &[1.0, 2.0],
        &no_cancel(),
        None,
    );
    assert_eq!(r.unwrap_err(), undefined);
}
