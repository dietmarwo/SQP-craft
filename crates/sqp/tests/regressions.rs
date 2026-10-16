//! Cases added while reviewing iteration 1. The QP-acceptance test failed on
//! the delivered code (false `QpFailure`); HS071 and the history are new coverage.

use sqp_craft::qp::QpOutcome;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SolveResult,
    SqpSolver, Status,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

fn run(spec: &ProblemSpec, model: &mut dyn NlpProblem, x0: &[f64]) -> SolveResult {
    run_with(spec, model, x0, SqpOptions::default())
}

fn run_with(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x0: &[f64],
    settings: SqpOptions,
) -> SolveResult {
    NativeSqp::<ClarabelQp> {
        settings,
        ..NativeSqp::default()
    }
    .solve(
        spec,
        model,
        x0,
        &Options::default(),
        &CancellationToken::default(),
        None,
    )
    .unwrap()
}

/// f = (x0 + 1)² + (x1 − 3)⁴ + 0.1·x0·x1 on x0 ∈ [0, 5], x1 ∈ [−10, 10]:
/// a large gradient at the start and an active lower bound at the solution.
struct QuarticWithBound;

impl NlpProblem for QuarticWithBound {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> Result<f64> {
        assert!((0.0..=5.0).contains(&x[0]), "outside the box: {x:?}");
        Ok((x[0] + 1.0).powi(2) + (x[1] - 3.0).powi(4) + 0.1 * x[0] * x[1])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] + 1.0) + 0.1 * x[1];
        g[1] = 4.0 * (x[1] - 3.0).powi(3) + 0.1 * x[0];
        Ok(())
    }
}

#[test]
fn interior_point_complementarity_with_large_multipliers_is_accepted() {
    // At x1 = -3 the gradient is ~ -864, so the first QP's bound multiplier is
    // ~ 2 and its step multipliers are large. The delivered adapter rejected
    // the correct Clarabel solution because complementarity |z|·gap ≈ 5e-7
    // exceeded an absolute 1e-7.
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0, -10.0], vec![5.0, 10.0]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    for x0 in [[0.0, -3.0], [0.5, -2.0], [1.0, 0.0]] {
        let r = run(&spec, &mut QuarticWithBound, &x0);
        assert_eq!(r.status, Status::Converged, "{x0:?}: {}", r.message);
        let x = r.candidate.unwrap();
        assert!(x[0].abs() < 1e-8 && (x[1] - 3.0).abs() < 1e-2, "{x:?}");
        assert!(r.multipliers.unwrap().variables[0] > 0.0);
    }
}

#[test]
fn qp_acceptance_scales_with_multipliers_but_catches_sign_errors() {
    use sqp_craft::qp::{QpBackend, QpModel};
    use sqp_craft::{CsrMatrix, Multipliers};
    // minimize 0.5·p² − 1000·p on p <= 1: p = 1, z = −999.
    let model = QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern::dense(1, 1),
            values: vec![1.0],
        },
        gradient: vec![-1000.0],
        variables: Bounds::new(vec![f64::NEG_INFINITY], vec![1.0]),
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(0, 1),
            values: vec![],
        },
        rows: Bounds::free(0),
    };
    let outcome = ClarabelQp::default()
        .solve(&model, None, &CancellationToken::default())
        .unwrap();
    let QpOutcome::Optimal(solution) = outcome else {
        panic!("{outcome:?}");
    };
    assert!((solution.multipliers.variables[0] + 999.0).abs() < 1e-6);
    let check = model.accepts(&solution, 1e-7).unwrap();
    assert!(check.accepted, "{check:?}");
    let mut flipped = solution.clone();
    flipped.multipliers = Multipliers {
        constraints: vec![],
        variables: vec![999.0],
    };
    assert!(!model.accepts(&flipped, 1e-7).unwrap().accepted);
}

/// Hock–Schittkowski 71: f* = 17.0140173 from (1, 5, 5, 1).
struct Hs071;

impl NlpProblem for Hs071 {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x.iter().product();
        c[1] = x.iter().map(|v| v * v).sum();
        Ok(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]);
        let p: f64 = x.iter().product();
        for i in 0..4 {
            j[i] = p / x[i];
            j[4 + i] = 2.0 * x[i];
        }
        Ok(())
    }
}

fn hs071() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![1.0; 4], vec![5.0; 4]),
        Bounds::new(vec![25.0, 40.0], vec![f64::INFINITY, 40.0]),
        CsrPattern::dense(2, 4),
    )
    .unwrap()
}

#[test]
fn hs071_converges_and_reports_an_inconsistent_linearization_honestly() {
    for x0 in [[1.0, 5.0, 5.0, 1.0], [2.0, 2.0, 2.0, 2.0]] {
        let r = run(&hs071(), &mut Hs071, &x0);
        assert_eq!(r.status, Status::Converged, "{x0:?}: {}", r.message);
        let f = r.evaluation.unwrap().objective;
        assert!((f - 17.0140173).abs() < 1e-6, "{x0:?}: f = {f}");
        let lambda = r.multipliers.unwrap().constraints;
        assert!(
            lambda[0] > 0.0,
            "active lower bound needs a positive multiplier"
        );
    }
    // At (1,1,1,1) the linearized rows need Σp ≥ 24 and Σp = 18: inconsistent,
    // although the nonlinear problem is feasible. Without a restoration phase
    // this must be reported as such, never as infeasibility or convergence.
    let r = run_with(
        &hs071(),
        &mut Hs071,
        &[1.0; 4],
        SqpOptions {
            max_restoration_steps: 0,
            ..Default::default()
        },
    );
    assert_eq!(r.status, Status::LinearizationInfeasible);
    assert!(r.message.contains("may still be feasible"), "{}", r.message);
    assert_eq!(r.candidate, Some(vec![1.0; 4]));
}

#[test]
fn history_records_each_iterate() {
    let r = run_with(
        &hs071(),
        &mut Hs071,
        &[1.0, 5.0, 5.0, 1.0],
        SqpOptions {
            record_history: true,
            ..SqpOptions::default()
        },
    );
    assert_eq!(r.status, Status::Converged);
    let history = &r.report.history;
    assert_eq!(history.len() as u64, r.iterations + 1);
    for (i, h) in history.iter().enumerate() {
        assert_eq!(h.iteration, i as u64);
    }
    let last = history.last().unwrap();
    assert_eq!(last.step_length, None);
    assert!(last.stationarity <= 1e-6);
    assert!(
        history[..history.len() - 1]
            .iter()
            .all(|h| h.step_length.is_some() && h.trials >= 1)
    );
    // Without the option nothing is recorded.
    let r = run(&hs071(), &mut Hs071, &[1.0, 5.0, 5.0, 1.0]);
    assert!(r.report.history.is_empty());
}

#[test]
fn hs071_restoration_recovers_the_previously_inconsistent_start() {
    let spec = hs071();
    let r = run_with(
        &spec,
        &mut Hs071,
        &[1.0; 4],
        SqpOptions {
            record_history: true,
            ..Default::default()
        },
    );
    assert_eq!(r.status, Status::Converged, "{}: {:?}", r.message, r.report);
    assert!((r.evaluation.as_ref().unwrap().objective - 17.0140173).abs() < 1e-6);
    let kkt = sqp_craft::kkt::nlp_residuals(
        &spec,
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        r.multipliers.as_ref().unwrap(),
    )
    .unwrap();
    assert!(
        kkt.primal <= 1e-7
            && kkt.stationarity <= 1e-6
            && kkt.complementarity <= 1e-6
            && kkt.dual_sign <= 1e-8,
        "{kkt:?}"
    );
    assert!(r.report.restoration_steps > 0);
    assert_eq!(
        r.report.restoration_history.len() as u64,
        r.report.restoration_steps
    );
    assert!(
        r.report
            .restoration_history
            .iter()
            .all(|h| h.merit_after < h.merit_before)
    );
    assert_eq!(
        (r.report.history.len() + r.report.restoration_history.len()) as u64,
        r.iterations + 1
    );
}
