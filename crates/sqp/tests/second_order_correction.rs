//! The second-order correction against the Maratos effect.
use sqp_craft::qp::QpBackend;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SolveResult,
    SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

/// Nocedal & Wright, Example 15.4: min 2(x₁² + x₂² − 1) − x₁ subject to
/// x₁² + x₂² = 1, solved by x = (1, 0). From a feasible point the QP step is
/// tangent, so the full step leaves the circle and the ℓ1 merit rejects it.
struct Maratos;

impl NlpProblem for Maratos {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        let q = x[0] * x[0] + x[1] * x[1];
        c[0] = q;
        Ok(2.0 * (q - 1.0) - x[0])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[4.0 * x[0] - 1.0, 4.0 * x[1]]);
        j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        Ok(())
    }
}

fn spec(row: Bounds) -> ProblemSpec {
    ProblemSpec::new(Bounds::free(2), row, CsrPattern::dense(1, 2)).unwrap()
}

fn run<B: QpBackend + Default>(
    model: &mut dyn NlpProblem,
    spec: &ProblemSpec,
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
    r
}

fn with(second_order_correction: bool, elastic_mode: bool) -> SqpOptions {
    SqpOptions {
        second_order_correction,
        elastic_mode,
        record_history: true,
        ..SqpOptions::default()
    }
}

#[test]
fn the_correction_accepts_full_steps_that_the_merit_rejects() {
    let circle = spec(Bounds::fixed(vec![1.0]));
    let start = [0.3_f64.cos(), 0.3_f64.sin()];
    let plain = run::<ActiveSetQp>(&mut Maratos, &circle, &start, with(false, false));
    let r = run::<ActiveSetQp>(&mut Maratos, &circle, &start, with(true, false));
    let x = r.candidate.as_ref().unwrap();
    assert!((x[0] - 1.0).abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
    assert!(r.report.second_order_steps >= 1, "{:?}", r.report);
    assert!(r.report.second_order_corrections >= r.report.second_order_steps);
    assert!(
        r.counts.functions < plain.counts.functions,
        "{} vs {}",
        r.counts.functions,
        plain.counts.functions
    );
    // Without the option nothing is tried.
    assert_eq!(plain.report.second_order_corrections, 0);
    // An accepted correction is a full step, flagged in the history.
    assert!(
        r.report
            .history
            .iter()
            .any(|h| h.second_order_correction && h.step_length == Some(1.0))
    );
    // The same holds with Clarabel and in elastic mode.
    for r in [
        run::<ClarabelQp>(&mut Maratos, &circle, &start, with(true, false)),
        run::<ActiveSetQp>(&mut Maratos, &circle, &start, with(true, true)),
    ] {
        assert!(r.report.second_order_steps >= 1, "{:?}", r.report);
    }
}

/// (x₀ − 3)² + (x₁ + 1)² + x₀⁴ with the linear row x₀ + x₁ = 1.
struct Linear;

impl NlpProblem for Linear {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] + x[1];
        Ok((x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2) + x[0].powi(4))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[2.0 * (x[0] - 3.0) + 4.0 * x[0].powi(3), 2.0 * (x[1] + 1.0)]);
        j.copy_from_slice(&[1.0, 1.0]);
        Ok(())
    }
}

#[test]
fn linear_rows_never_need_a_correction() {
    // Rejected steps here come from the objective; the corrected QP would
    // return the same step, so none is tried.
    let row = spec(Bounds::fixed(vec![1.0]));
    let r = run::<ActiveSetQp>(&mut Linear, &row, &[4.0, -3.0], with(true, false));
    assert!(r.report.rejected_trials > 0, "{:?}", r.report);
    assert_eq!(r.report.second_order_corrections, 0);
}
