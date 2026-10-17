//! The four options taken from SNOPT (docs/snopt-lessons.md): the
//! augmented-Lagrangian curvature repair, the augmented-Lagrangian merit with
//! the violation limit, linear rows kept feasible, and scaling.
use sqp_craft::qp::QpBackend;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, CurvatureUpdate, NlpProblem, Options, ProblemSpec,
    Result, SolveResult, SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{HessianUpdate, NativeSqp, SqpOptions};

/// Nocedal & Wright, Example 15.4: min 2(x₁² + x₂² − 1) − x₁ subject to
/// x₁² + x₂² = 1, solved by x = (1, 0). The Lagrangian Hessian at the
/// solution is 4I − 2λI with λ = 1.5, so the true curvature (I) is smaller
/// than what the objective alone suggests, and the circle's curvature is
/// what a repair has to add.
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

fn circle() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::free(2),
        Bounds::fixed(vec![1.0]),
        CsrPattern::dense(1, 2),
    )
    .unwrap()
}

fn solve<B: QpBackend + Default>(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x0: &[f64],
    settings: SqpOptions,
) -> SolveResult {
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
    .unwrap()
}

fn converged(r: &SolveResult) {
    assert_eq!(r.status, Status::Converged, "{}", r.message);
}

fn curvature_counts(r: &SolveResult, kind: CurvatureUpdate) -> u64 {
    r.report
        .history
        .iter()
        .filter(|h| h.curvature == Some(kind))
        .count() as u64
}

fn updating(hessian_update: HessianUpdate) -> SqpOptions {
    SqpOptions {
        hessian_update,
        record_history: true,
        ..SqpOptions::default()
    }
}

#[test]
fn the_repair_replaces_damping_on_a_curved_constraint() {
    let start = [2.0_f64.cos() * 1.5, 2.0_f64.sin() * 1.5];
    for backend in ["active-set", "clarabel"] {
        let run = |update| match backend {
            "active-set" => solve::<ActiveSetQp>(&circle(), &mut Maratos, &start, updating(update)),
            _ => solve::<ClarabelQp>(&circle(), &mut Maratos, &start, updating(update)),
        };
        let damped = run(HessianUpdate::Damped);
        let repaired = run(HessianUpdate::Augmented);
        converged(&damped);
        converged(&repaired);
        let x = repaired.candidate.as_ref().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
        // Damping fires in the default; every such step is repaired instead.
        assert!(damped.report.bfgs_damped_updates >= 1, "{backend}");
        assert_eq!(damped.report.bfgs_modified_updates, 0);
        assert!(repaired.report.bfgs_modified_updates >= 1, "{backend}");
        assert_eq!(repaired.report.bfgs_damped_updates, 0, "{backend}");
        // The history records every update, matching the counters.
        for r in [&damped, &repaired] {
            assert_eq!(
                curvature_counts(r, CurvatureUpdate::Damped),
                r.report.bfgs_damped_updates
            );
            assert_eq!(
                curvature_counts(r, CurvatureUpdate::Modified),
                r.report.bfgs_modified_updates
            );
            let steps = r.report.history.iter().filter(|h| h.step_length.is_some());
            assert!(steps.clone().all(|h| h.curvature.is_some()));
            assert_eq!(steps.count() as u64, r.iterations);
        }
    }
}

/// `0.01‖x‖²` on the line x₀ + x₁ = 1: the objective's curvature is far
/// below B₀ = I, so every early update is below the threshold, and the
/// only row is linear, so no repair exists.
struct Flat;

impl NlpProblem for Flat {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] + x[1];
        Ok(0.01 * (x[0] * x[0] + x[1] * x[1]) + 0.3 * x[0])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[0.02 * x[0] + 0.3, 0.02 * x[1]]);
        j.copy_from_slice(&[1.0, 1.0]);
        Ok(())
    }
}

#[test]
fn without_curved_rows_the_repair_falls_back_to_damping_exactly() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![-20.0; 2], vec![20.0; 2]),
        Bounds::fixed(vec![1.0]),
        CsrPattern::dense(1, 2),
    )
    .unwrap();
    let damped = solve::<ActiveSetQp>(
        &spec,
        &mut Flat,
        &[3.0, -1.0],
        updating(HessianUpdate::Damped),
    );
    let repaired = solve::<ActiveSetQp>(
        &spec,
        &mut Flat,
        &[3.0, -1.0],
        updating(HessianUpdate::Augmented),
    );
    converged(&damped);
    assert!(damped.report.bfgs_damped_updates >= 1);
    assert_eq!(repaired.report.bfgs_modified_updates, 0);
    assert_eq!(repaired.candidate, damped.candidate);
    assert_eq!(repaired.multipliers, damped.multipliers);
    assert_eq!(repaired.report.history, damped.report.history);
}
