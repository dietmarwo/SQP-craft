//! The SQP driver with the dense active-set QP backend, and the warm start
//! from the previous working set.
#[path = "../examples/support/orbit.rs"]
mod orbit;

use sqp_craft::kkt::nlp_residuals;
use sqp_craft::qp::QpBackend;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SolveResult,
    SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

fn run<B: QpBackend + Default>(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x0: &[f64],
    settings: SqpOptions,
) -> SolveResult {
    let r = NativeSqp::<B> {
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
    .unwrap();
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    let k = nlp_residuals(
        spec,
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        r.multipliers.as_ref().unwrap(),
    )
    .unwrap();
    assert!(
        k.primal <= 1e-7 && k.stationarity <= 1e-6 && k.complementarity <= 1e-6,
        "{k:?}"
    );
    r
}

#[test]
fn the_orbit_transfer_takes_the_same_path_as_with_clarabel() {
    let (spec, mut model, x0) = orbit::OrbitTransfer::new(20);
    let settings = SqpOptions {
        max_iterations: 200,
        ..SqpOptions::default()
    };
    let exact = run::<ActiveSetQp>(&spec, &mut model, &x0, settings.clone());
    let interior = run::<ClarabelQp>(&spec, &mut model, &x0, settings);
    let objective = |r: &SolveResult| r.evaluation.as_ref().unwrap().objective;
    assert!((objective(&exact) - objective(&interior)).abs() < 1e-9);
    assert!((-objective(&exact) - 1.52206).abs() < 5e-5);
    assert_eq!(exact.iterations, interior.iterations);
    assert!(exact.report.qp_backend.starts_with("goldfarb-idnani"));
    // One factorization per QP, no engine retries.
    assert_eq!(exact.report.qp_setups, exact.report.qp_solves);
    assert_eq!(exact.report.qp_cold_retries, 0);
}

/// Σ (i+1)(x_i − 3(−1)^i)² + 0.1 x_i⁴ on [−1, 1]⁶: every bound ends up active.
struct Pull;

impl NlpProblem for Pull {
    fn functions(&mut self, x: &[f64], _c: &mut [f64]) -> Result<f64> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let t = if i % 2 == 0 { 3.0 } else { -3.0 };
                (i + 1) as f64 * (v - t).powi(2) + 0.1 * v.powi(4)
            })
            .sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _j: &mut [f64]) -> Result<()> {
        for (i, (gi, &v)) in g.iter_mut().zip(x).enumerate() {
            let t = if i % 2 == 0 { 3.0 } else { -3.0 };
            *gi = 2.0 * (i + 1) as f64 * (v - t) + 0.4 * v.powi(3);
        }
        Ok(())
    }
}

#[test]
fn the_previous_working_set_warm_starts_the_next_qp() {
    let n = 6;
    let spec = ProblemSpec::new(
        Bounds::new(vec![-1.0; n], vec![1.0; n]),
        Bounds::free(0),
        CsrPattern::dense(0, n),
    )
    .unwrap();
    let with = |qp_warm_start| SqpOptions {
        qp_warm_start,
        ..SqpOptions::default()
    };
    let warm = run::<ActiveSetQp>(&spec, &mut Pull, &[0.0; 6], with(true));
    let cold = run::<ActiveSetQp>(&spec, &mut Pull, &[0.0; 6], with(false));
    assert_eq!(warm.candidate, cold.candidate);
    assert_eq!(warm.iterations, cold.iterations);
    // A cold QP enters all six bounds; a warm one starts with them.
    assert_eq!(cold.report.qp_iterations, n as u64 * cold.report.qp_solves);
    assert_eq!(warm.report.qp_iterations, n as u64);
}

#[test]
fn clarabel_runs_do_not_depend_on_the_warm_start_setting() {
    // Clarabel reports no working set, so the driver never passes one.
    let (spec, mut model, x0) = orbit::OrbitTransfer::new(10);
    let with = |qp_warm_start| SqpOptions {
        qp_warm_start,
        ..SqpOptions::default()
    };
    let on = run::<ClarabelQp>(&spec, &mut model, &x0, with(true));
    let off = run::<ClarabelQp>(&spec, &mut model, &x0, with(false));
    assert_eq!(on.candidate, off.candidate);
    assert_eq!(on.multipliers, off.multipliers);
    assert_eq!(on.report.qp_iterations, off.report.qp_iterations);
}
