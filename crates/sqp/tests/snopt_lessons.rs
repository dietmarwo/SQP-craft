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
use sqp_craft_solver::{HessianUpdate, Merit, NativeSqp, SqpOptions};

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

fn merit(merit: Merit, violation_limit: Option<f64>) -> SqpOptions {
    SqpOptions {
        merit,
        violation_limit,
        record_history: true,
        ..SqpOptions::default()
    }
}

fn step_lengths(r: &SolveResult) -> Vec<f64> {
    r.report
        .history
        .iter()
        .filter_map(|h| h.step_length)
        .collect()
}

#[test]
fn the_augmented_lagrangian_merit_takes_unit_steps_on_a_curved_constraint() {
    // From a feasible start the ℓ1 merit rejects the tangent QP steps (the
    // Maratos effect) and backtracks; the smooth merit accepts them.
    let start = [0.3_f64.cos(), 0.3_f64.sin()];
    for backend in ["active-set", "clarabel"] {
        let run = |settings| match backend {
            "active-set" => solve::<ActiveSetQp>(&circle(), &mut Maratos, &start, settings),
            _ => solve::<ClarabelQp>(&circle(), &mut Maratos, &start, settings),
        };
        let l1 = run(merit(Merit::L1, None));
        let smooth = run(merit(Merit::AugmentedLagrangian, None));
        converged(&l1);
        converged(&smooth);
        let x = smooth.candidate.as_ref().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
        assert!(
            step_lengths(&l1).contains(&0.25),
            "{backend}: {:?}",
            step_lengths(&l1)
        );
        assert!(
            step_lengths(&smooth).iter().all(|&a| a == 1.0),
            "{backend}: {:?}",
            step_lengths(&smooth)
        );
        assert!(smooth.counts.functions < l1.counts.functions, "{backend}");
        // The record shows the largest penalty ρ of the search.
        assert!(
            smooth
                .report
                .history
                .iter()
                .all(|h| h.max_penalty.is_finite())
        );
    }
}

/// min −10x₀ subject to x₀² + x₁² ≤ 1 from the origin, where J = 0: the
/// first QP sees no constraint and steps to x₀ = 10, a violation of 99.
struct Unbounded;

impl NlpProblem for Unbounded {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] * x[0] + x[1] * x[1];
        Ok(-10.0 * x[0])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[-10.0, 0.0]);
        j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        Ok(())
    }
}

#[test]
fn the_violation_limit_keeps_trial_points_near_the_constraints() {
    let disc = ProblemSpec::new(
        Bounds::free(2),
        Bounds::new(vec![f64::NEG_INFINITY], vec![1.0]),
        CsrPattern::dense(1, 2),
    )
    .unwrap();
    for kind in [Merit::L1, Merit::AugmentedLagrangian] {
        let free = solve::<ActiveSetQp>(&disc, &mut Unbounded, &[0.0, 0.0], merit(kind, None));
        let limited =
            solve::<ActiveSetQp>(&disc, &mut Unbounded, &[0.0, 0.0], merit(kind, Some(10.0)));
        converged(&free);
        converged(&limited);
        let worst = |r: &SolveResult| {
            r.report
                .history
                .iter()
                .fold(0.0_f64, |a, h| a.max(h.primal_violation))
        };
        assert_eq!(worst(&free), 99.0, "{kind:?}");
        // τ_V · max(1, 0) = 10 bounds every accepted point.
        assert!(worst(&limited) <= 10.0, "{kind:?}: {}", worst(&limited));
        assert!(limited.report.violation_limit_rejections >= 1);
        assert_eq!(free.report.violation_limit_rejections, 0);
        assert!(limited.iterations < free.iterations, "{kind:?}");
    }
}

#[test]
fn merit_settings_are_checked_before_any_model_call() {
    struct Untouchable;
    impl NlpProblem for Untouchable {
        fn functions(&mut self, _: &[f64], _: &mut [f64]) -> Result<f64> {
            panic!("model called")
        }
        fn derivatives(&mut self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<()> {
            panic!("model called")
        }
    }
    for settings in [
        SqpOptions {
            merit: Merit::AugmentedLagrangian,
            second_order_correction: true,
            ..SqpOptions::default()
        },
        merit(Merit::L1, Some(0.5)),
        merit(Merit::L1, Some(f64::INFINITY)),
        merit(Merit::L1, Some(f64::NAN)),
    ] {
        let error = NativeSqp::<ActiveSetQp> {
            settings,
            qp: ActiveSetQp::default(),
        }
        .solve(
            &circle(),
            &mut Untouchable,
            &[1.0, 0.0],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap_err();
        assert!(
            matches!(error, sqp_craft::Error::InvalidInput(_)),
            "{error}"
        );
    }
}

/// (x − 2)² with x² = 1 from x = 0, where J = 0: only elastic mode
/// converges. Its elastic step uses the ℓ1 merit, the rest the smooth one.
struct Square;

impl NlpProblem for Square {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] * x[0];
        Ok((x[0] - 2.0).powi(2))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] - 2.0);
        j[0] = 2.0 * x[0];
        Ok(())
    }
}

#[test]
fn elastic_steps_keep_the_l1_merit_under_the_augmented_lagrangian() {
    let spec = ProblemSpec::new(
        Bounds::free(1),
        Bounds::fixed(vec![1.0]),
        CsrPattern::dense(1, 1),
    )
    .unwrap();
    let r = solve::<ActiveSetQp>(
        &spec,
        &mut Square,
        &[0.0],
        SqpOptions {
            elastic_mode: true,
            ..merit(Merit::AugmentedLagrangian, None)
        },
    );
    converged(&r);
    assert!((r.candidate.as_ref().unwrap()[0] - 1.0).abs() < 1e-9);
    assert_eq!(r.report.elastic_steps, 1);
    // The elastic step's record shows the elastic weight as its penalty.
    let weight = r.report.elastic_weight.unwrap();
    assert_eq!(r.report.history[0].max_penalty, weight);
    assert!(r.report.history[1..].iter().all(|h| h.max_penalty < weight));
}

fn line_spec(rows: &[usize]) -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![-20.0; 2], vec![20.0; 2]),
        Bounds::fixed(vec![1.0]),
        CsrPattern::dense(1, 2),
    )
    .unwrap()
    .with_linear_rows(rows)
    .unwrap()
}

#[test]
fn a_start_off_the_linear_rows_is_projected_onto_them() {
    let plain = solve::<ActiveSetQp>(
        &line_spec(&[]),
        &mut Flat,
        &[3.0, -1.0],
        updating(HessianUpdate::Damped),
    );
    for backend in ["active-set", "clarabel"] {
        let settings = updating(HessianUpdate::Damped);
        let r = match backend {
            "active-set" => {
                solve::<ActiveSetQp>(&line_spec(&[0]), &mut Flat, &[3.0, -1.0], settings)
            }
            _ => solve::<ClarabelQp>(&line_spec(&[0]), &mut Flat, &[3.0, -1.0], settings),
        };
        converged(&r);
        assert!(r.report.linear_projection, "{backend}");
        // The smallest move onto x₀ + x₁ = 1 is (−½, −½); every iterate
        // after it is on the line.
        let first = r.report.history[0];
        assert!(first.primal_violation <= 1e-9, "{backend}: {first:?}");
        assert!(r.report.history.iter().all(|h| h.primal_violation <= 1e-9));
        // Projection plus one more evaluation each.
        assert_eq!(r.counts.functions, r.iterations + 2, "{backend}");
        let (a, b) = (
            r.candidate.as_ref().unwrap(),
            plain.candidate.as_ref().unwrap(),
        );
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }
    // A start on the rows is not moved.
    let r = solve::<ActiveSetQp>(
        &line_spec(&[0]),
        &mut Flat,
        &[0.5, 0.5],
        updating(HessianUpdate::Damped),
    );
    converged(&r);
    assert!(!r.report.linear_projection);
}

#[test]
fn inconsistent_linear_rows_end_the_solve_as_linear_infeasible() {
    // x₀ + x₁ ≥ 3 on the unit box.
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0; 2], vec![1.0; 2]),
        Bounds::new(vec![3.0], vec![f64::INFINITY]),
        CsrPattern::dense(1, 2),
    )
    .unwrap()
    .with_linear_rows(&[0])
    .unwrap();
    let a = solve::<ActiveSetQp>(&spec, &mut Flat, &[0.5, 0.5], SqpOptions::default());
    let b = solve::<ClarabelQp>(&spec, &mut Flat, &[0.5, 0.5], SqpOptions::default());
    for r in [a, b] {
        assert_eq!(r.status, Status::LinearInfeasible, "{}", r.message);
        assert_eq!(r.candidate.as_deref(), Some(&[0.5, 0.5][..]));
        assert_eq!(r.iterations, 0);
    }
}

/// min x² + (y − ½)² subject to x ≥ 1 (linear) and y² = x, from (1, 0).
/// The model is undefined for x < ½. The linearization −1 − p_x = 0 of the
/// parabola contradicts x ≥ 1, so the step QP is inconsistent.
struct Parabola {
    pull: f64,
}

impl NlpProblem for Parabola {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        if x[0] < 0.5 {
            return Err(sqp_craft::Error::UndefinedPoint("x < 1/2".into()));
        }
        c[0] = x[0];
        c[1] = x[1] * x[1] - x[0];
        Ok((x[0] - self.pull).powi(2) + (x[1] - 0.5).powi(2))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[2.0 * (x[0] - self.pull), 2.0 * (x[1] - 0.5)]);
        j.copy_from_slice(&[1.0, -1.0, 2.0 * x[1]]);
        Ok(())
    }
}

fn parabola(rows: &[usize]) -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![-5.0; 2], vec![5.0; 2]),
        Bounds::new(vec![1.0, 0.0], vec![f64::INFINITY, 0.0]),
        CsrPattern {
            rows: 2,
            cols: 2,
            indptr: vec![0, 1, 3],
            indices: vec![0, 0, 1],
        },
    )
    .unwrap()
    .with_linear_rows(rows)
    .unwrap()
}

#[test]
fn declared_rows_stay_hard_in_the_elastic_qp() {
    // The objective pulls x down. Undeclared, the elastic QP violates x ≥ 1
    // (at the same cost γ as the parabola) and the full step lands where the
    // model is undefined; declared, it keeps x = 1 and reaches (1, 1) at once.
    let elastic = SqpOptions {
        elastic_mode: true,
        record_history: true,
        ..SqpOptions::default()
    };
    let soft = solve::<ActiveSetQp>(
        &parabola(&[]),
        &mut Parabola { pull: 0.0 },
        &[1.0, 0.0],
        elastic.clone(),
    );
    let hard = solve::<ActiveSetQp>(
        &parabola(&[0]),
        &mut Parabola { pull: 0.0 },
        &[1.0, 0.0],
        elastic,
    );
    converged(&soft);
    converged(&hard);
    assert!(soft.report.undefined_trials >= 1);
    assert_eq!(hard.report.undefined_trials, 0);
    assert_eq!(hard.iterations, 1);
    let x = hard.candidate.as_ref().unwrap();
    assert!(
        (x[0] - 1.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9,
        "{x:?}"
    );
}

#[test]
fn restoration_keeps_declared_rows_satisfied() {
    // Restoration reduces the parabola's violation through x alone (its y
    // derivative is 0 at y = 0). Undeclared, it leaves x ≥ 1 for the
    // undefined region; declared, it stops at once without leaving it.
    let soft = solve::<ActiveSetQp>(
        &parabola(&[]),
        &mut Parabola { pull: 2.0 },
        &[1.0, 0.0],
        SqpOptions::default(),
    );
    let hard = solve::<ActiveSetQp>(
        &parabola(&[0]),
        &mut Parabola { pull: 2.0 },
        &[1.0, 0.0],
        SqpOptions::default(),
    );
    assert_eq!(soft.status, Status::RestorationFailed);
    assert_eq!(hard.status, Status::RestorationFailed);
    assert!(soft.candidate.as_ref().unwrap()[0] < 1.0);
    assert!(soft.report.undefined_trials >= 1);
    assert_eq!(hard.candidate.as_deref(), Some(&[1.0, 0.0][..]));
    assert_eq!(hard.report.undefined_trials, 0);
}

#[test]
fn a_row_wrongly_declared_linear_is_an_error() {
    // The parabola's Jacobian changes with y, which the first step changes.
    // (A wrong declaration can also make the start projection infeasible and
    // the status `linear_infeasible` wrong: the status relies on it.)
    let error = NativeSqp::<ActiveSetQp> {
        settings: SqpOptions::default(),
        qp: ActiveSetQp::default(),
    }
    .solve(
        &parabola(&[0, 1]),
        &mut Parabola { pull: 2.0 },
        &[1.0, 1.0],
        &Options::default(),
        &CancellationToken::default(),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(&error, sqp_craft::Error::InvalidInput(m) if m.contains("row 1 is declared linear")),
        "{error}"
    );
}
