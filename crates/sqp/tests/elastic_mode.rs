//! Elastic mode: every step QP is the ℓ1-elastic QP of the active-set backend.
#[path = "../examples/support/orbit.rs"]
mod orbit;

use sqp_craft::kkt::nlp_residuals;
use sqp_craft::qp::QpBackend;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, Error, NlpProblem, Options, ProblemSpec, Result,
    SolveResult, SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

fn run<B: QpBackend + Default>(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x0: &[f64],
    settings: SqpOptions,
) -> Result<SolveResult> {
    NativeSqp::<B> {
        settings,
        qp: B::default(),
    }
    .solve(
        spec,
        model,
        x0,
        &Options::default(),
        &CancellationToken::default(),
        None,
    )
}

fn elastic() -> SqpOptions {
    SqpOptions {
        elastic_mode: true,
        ..SqpOptions::default()
    }
}

/// (x − 2)² subject to x² = 1 on [−3, 3], from x = 0, where J = 0 and the
/// linearization 0 = 1 is inconsistent.
struct Square {
    calls: usize,
}

impl NlpProblem for Square {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.calls += 1;
        c[0] = x[0] * x[0];
        Ok((x[0] - 2.0).powi(2))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] - 2.0);
        j[0] = 2.0 * x[0];
        Ok(())
    }
}

fn square_spec() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![-3.0], vec![3.0]),
        Bounds::fixed(vec![1.0]),
        CsrPattern::dense(1, 1),
    )
    .unwrap()
}

#[test]
fn a_zero_jacobian_start_is_left_by_an_elastic_step() {
    // Gauss–Newton restoration sees a stationary point of the violation at
    // x = 0. The elastic QP still takes the objective's step, and the merit
    // line search with γ on the violated row stops at the feasible x = 1.
    let spec = square_spec();
    let restored = run::<ActiveSetQp>(
        &spec,
        &mut Square { calls: 0 },
        &[0.0],
        SqpOptions::default(),
    )
    .unwrap();
    assert_eq!(restored.status, Status::RestorationFailed);
    let r = run::<ActiveSetQp>(&spec, &mut Square { calls: 0 }, &[0.0], elastic()).unwrap();
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    let x = r.candidate.as_ref().unwrap()[0];
    assert!((x - 1.0).abs() < 1e-9, "{x}");
    assert_eq!(r.report.elastic_steps, 1);
    assert_eq!(r.report.restoration_steps, 0);
    // γ = 10⁴ · max(1, |∇f(0)|) = 4·10⁴, never increased.
    assert_eq!(r.report.elastic_weight, Some(4e4));
    assert_eq!(r.report.elastic_weight_increases, 0);
}

/// (x − 0.3)² subject to x ≥ 1 and x ≤ 0 as two rows: infeasible.
struct Contradiction;

impl NlpProblem for Contradiction {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c.copy_from_slice(&[x[0], x[0]]);
        Ok((x[0] - 0.3).powi(2))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] - 0.3);
        j.copy_from_slice(&[1.0, 1.0]);
        Ok(())
    }
}

#[test]
fn an_infeasible_problem_ends_locally_infeasible_at_the_largest_weight() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![-5.0], vec![5.0]),
        Bounds::new(vec![1.0, f64::NEG_INFINITY], vec![f64::INFINITY, 0.0]),
        CsrPattern::dense(2, 1),
    )
    .unwrap();
    let r = run::<ActiveSetQp>(&spec, &mut Contradiction, &[2.0], elastic()).unwrap();
    assert_eq!(r.status, Status::LocallyInfeasible, "{}", r.message);
    assert!(r.message.contains("not established"), "{}", r.message);
    // Between the rows the total violation is constant: the objective
    // decides, and both rows are violated at multipliers ±γ.
    let x = r.candidate.as_ref().unwrap()[0];
    assert!((x - 0.3).abs() < 1e-9, "{x}");
    assert_eq!(r.report.elastic_weight_increases, 6);
    let start_gradient = 2.0 * (2.0 - 0.3);
    let weight = r.report.elastic_weight.unwrap();
    assert!(
        (weight / (1e4 * start_gradient * 1e6) - 1.0).abs() < 1e-12,
        "{weight}"
    );
    let lambda = &r.multipliers.as_ref().unwrap().constraints;
    assert!(
        (lambda[0] - weight).abs() <= 1e-9 * weight && (lambda[1] + weight).abs() <= 1e-9 * weight
    );
    // Restoration reaches the same conclusion its own way.
    let restored =
        run::<ActiveSetQp>(&spec, &mut Contradiction, &[2.0], SqpOptions::default()).unwrap();
    assert_eq!(restored.status, Status::RestorationFailed);
}

#[test]
fn with_a_large_weight_the_orbit_takes_the_ordinary_path() {
    // The orbit's QPs are consistent with multipliers far below γ: the
    // elastic QP is the ordinary one, bit for bit.
    let (spec, mut model, x0) = orbit::OrbitTransfer::new(20);
    let settings = |elastic_mode| SqpOptions {
        max_iterations: 200,
        elastic_mode,
        ..SqpOptions::default()
    };
    let plain = run::<ActiveSetQp>(&spec, &mut model, &x0, settings(false)).unwrap();
    let r = run::<ActiveSetQp>(&spec, &mut model, &x0, settings(true)).unwrap();
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    assert_eq!(r.candidate, plain.candidate);
    assert_eq!(r.iterations, plain.iterations);
    assert_eq!(r.report.elastic_steps, 0);
    let k = nlp_residuals(
        &spec,
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        r.multipliers.as_ref().unwrap(),
    )
    .unwrap();
    assert!(k.primal <= 1e-7 && k.stationarity <= 1e-6, "{k:?}");
}

#[test]
fn elastic_mode_needs_a_backend_with_elastic_support() {
    let mut model = Square { calls: 0 };
    let error = run::<ClarabelQp>(&square_spec(), &mut model, &[0.0], elastic()).unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert_eq!(model.calls, 0, "rejected before any model call");
    let bad = SqpOptions {
        elastic_weight: 0.0,
        ..elastic()
    };
    assert!(matches!(
        run::<ActiveSetQp>(&square_spec(), &mut model, &[0.0], bad),
        Err(Error::InvalidInput(_))
    ));
}
