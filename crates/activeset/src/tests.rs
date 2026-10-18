use super::*;
use sqp_craft::{Bounds, CsrMatrix, CsrPattern};

const INF: f64 = f64::INFINITY;

/// QP from a dense symmetric `h` (row-major), dense rows `a` and bounds.
pub(crate) fn qp(
    h: &[&[f64]],
    g: &[f64],
    variables: (&[f64], &[f64]),
    a: &[&[f64]],
    rows: (&[f64], &[f64]),
) -> QpModel {
    let n = g.len();
    let mut lower = (vec![0], vec![], vec![]);
    for (i, row) in h.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i + 1) {
            if v != 0.0 {
                lower.1.push(j);
                lower.2.push(v);
            }
        }
        lower.0.push(lower.1.len());
    }
    let mut matrix = (vec![0], vec![], vec![]);
    for row in a {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                matrix.1.push(j);
                matrix.2.push(v);
            }
        }
        matrix.0.push(matrix.1.len());
    }
    let csr = |rows, (indptr, indices, values): (Vec<usize>, Vec<usize>, Vec<f64>)| CsrMatrix {
        pattern: CsrPattern {
            rows,
            cols: n,
            indptr,
            indices,
        },
        values,
    };
    let model = QpModel {
        hessian_lower: csr(n, lower),
        gradient: g.to_vec(),
        variables: Bounds::new(variables.0.to_vec(), variables.1.to_vec()),
        matrix: csr(a.len(), matrix),
        rows: Bounds::new(rows.0.to_vec(), rows.1.to_vec()),
    };
    model.validate().expect("valid test QP");
    model
}

fn backend() -> ActiveSetQp {
    let mut b = ActiveSetQp::default();
    b.configure(&QpOptions::default()).unwrap();
    b
}

fn optimal(b: &mut ActiveSetQp, model: &QpModel, warm: Option<&QpWarmStart>) -> QpSolution {
    match b.solve(model, warm, &CancellationToken::default()).unwrap() {
        QpOutcome::Optimal(s) => {
            assert!(model.accepts(&s, 1e-9).unwrap().accepted, "{s:?}");
            s
        }
        other => panic!("expected an optimal solution, got {other:?}"),
    }
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} != {b:?}");
    }
}

#[test]
fn bound_constrained_qp_clips_the_unconstrained_minimizer() {
    // min ½‖p‖² − (3, −3, 0.5)ᵀp on [−1, 1]³: p = (1, −1, 0.5).
    let model = qp(
        &[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]],
        &[-3., 3., -0.5],
        (&[-1.; 3], &[1.; 3]),
        &[],
        (&[], &[]),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[1., -1., 0.5], 1e-14);
    // Shared convention: H p + g = z; a lower-side multiplier is positive.
    close(&s.multipliers.variables, &[-2., 2., 0.], 1e-14);
    let set = s.active_set.unwrap();
    assert_eq!(
        set.variables,
        [Activity::AtUpper, Activity::AtLower, Activity::Inactive]
    );
    assert_eq!(
        s.iterations, 0,
        "closed-form box solve changes no working set"
    );
}

#[test]
fn equality_and_inequality_rows_match_the_closed_form() {
    // min ½(p₀² + p₁² + p₂²) s.t. p₀ + p₁ + p₂ = 3, p₀ − p₁ ≥ 1:
    // p = (1.5, 0.5, 1) with λ = (1, 0.5).
    let model = qp(
        &[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]],
        &[0., 0., 0.],
        (&[-INF; 3], &[INF; 3]),
        &[&[1., 1., 1.], &[1., -1., 0.]],
        (&[3., 1.], &[3., INF]),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[1.5, 0.5, 1.0], 1e-14);
    close(&s.multipliers.constraints, &[1.0, 0.5], 1e-14);
    assert_eq!(
        s.active_set.unwrap().rows,
        [Activity::AtLower, Activity::AtLower]
    );
    // The equality enters the initial working set: one counted change.
    assert_eq!(s.iterations, 1);
}

#[test]
fn an_active_upper_row_and_a_fixed_variable_get_negative_multipliers() {
    // min ½‖p‖² − (2, 2)ᵀp s.t. p₀ + p₁ ≤ 1 and p₀ = 0.25.
    let model = qp(
        &[&[1., 0.], &[0., 1.]],
        &[-2., -2.],
        (&[0.25, -INF], &[0.25, INF]),
        &[&[1., 1.]],
        (&[-INF], &[1.]),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[0.25, 0.75], 1e-14);
    close(&s.multipliers.constraints, &[-1.25], 1e-14);
    close(&s.multipliers.variables, &[-0.5, 0.0], 1e-14);
    let set = s.active_set.unwrap();
    assert_eq!(set.rows, [Activity::AtUpper]);
    assert_ne!(set.variables[0], Activity::Inactive);
}

#[test]
fn inconsistent_constraints_are_infeasible() {
    let h: &[&[f64]] = &[&[2., 0.], &[0., 1.]];
    let rows: &[&[f64]] = &[&[1., 1.], &[1., 1.]];
    for (lower, upper) in [
        // p₀ + p₁ ≥ 2 and p₀ + p₁ ≤ 1.
        ([2., -INF], [INF, 1.]),
        // Two different values for the same equality.
        ([1., 1.5], [1., 1.5]),
    ] {
        let model = qp(
            h,
            &[1., 1.],
            (&[-INF; 2], &[INF; 2]),
            rows,
            (&lower, &upper),
        );
        let mut b = backend();
        let outcome = b
            .solve(&model, None, &CancellationToken::default())
            .unwrap();
        assert_eq!(outcome, QpOutcome::Infeasible, "{lower:?} {upper:?}");
        assert_eq!(b.report().raw_status.as_deref(), Some("infeasible"));
    }
    // Box and row disagree.
    let model = qp(
        h,
        &[0., 0.],
        (&[0., 0.], &[1., 1.]),
        &[&[1., 1.]],
        (&[2.5], &[INF]),
    );
    assert_eq!(
        backend()
            .solve(&model, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::Infeasible
    );
}

#[test]
fn dependent_and_duplicate_constraints_are_handled() {
    // The same row three times (twice as an equality), a zero row and a
    // variable bound repeated by a row.
    let model = qp(
        &[&[1., 0.], &[0., 1.]],
        &[-4., -4.],
        (&[-INF, -INF], &[1., INF]),
        &[&[1., 1.], &[1., 1.], &[2., 2.], &[0., 0.], &[1., 0.]],
        (&[3., 3., -INF, -1., -INF], &[3., 3., 6., 1., 1.]),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[1., 2.], 1e-13);
    // The split between duplicates is arbitrary; their sum is not.
    let l = &s.multipliers.constraints;
    assert!((l[0] + l[1] + 2.0 * l[2] + 2.0).abs() < 1e-12, "{l:?}");
    assert!((s.multipliers.variables[0] + l[4] + 1.0).abs() < 1e-12);
}

#[test]
fn not_positive_definite_is_an_outcome() {
    for h in [[1., 2., 2., 1.], [1., 1., 1., 1.], [0., 0., 0., 1.]] {
        let model = qp(
            &[&h[..2], &h[2..]],
            &[1., 1.],
            (&[-1.; 2], &[1.; 2]),
            &[],
            (&[], &[]),
        );
        let mut b = backend();
        let outcome = b
            .solve(&model, None, &CancellationToken::default())
            .unwrap();
        assert_eq!(outcome, QpOutcome::NotPositiveDefinite, "{h:?}");
        assert_eq!(b.report().setups, 1);
    }
}

fn chain(n: usize) -> QpModel {
    // A tridiagonal objective pulling neighbours apart, inside [−1, 1]ⁿ,
    // with the rows p_i − p_{i+1} ≤ 0.1: active bounds and active rows.
    let mut h = vec![vec![0.0; n]; n];
    let mut a = vec![vec![0.0; n]; n - 1];
    for i in 0..n {
        h[i][i] = 2.0 + i as f64 / n as f64;
        if i + 1 < n {
            h[i][i + 1] = -0.5;
            h[i + 1][i] = -0.5;
            a[i][i] = 1.0;
            a[i][i + 1] = -1.0;
        }
    }
    let g: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 4.0 } else { -4.0 } * (1.0 + (i % 3) as f64))
        .collect();
    let h: Vec<&[f64]> = h.iter().map(Vec::as_slice).collect();
    let a: Vec<&[f64]> = a.iter().map(Vec::as_slice).collect();
    qp(
        &h,
        &g,
        (&vec![-1.0; n], &vec![1.0; n]),
        &a,
        (&vec![-INF; n - 1], &vec![0.1; n - 1]),
    )
}

#[test]
fn warm_starts_from_the_solution_or_a_wrong_set_reach_the_same_point() {
    let model = chain(12);
    let mut b = backend();
    let cold = optimal(&mut b, &model, None);
    assert!(cold.iterations > 5, "{}", cold.iterations);
    let exact = QpWarmStart {
        active_set: cold.active_set.clone(),
        ..Default::default()
    };
    let warm = optimal(&mut b, &model, Some(&exact));
    assert_eq!(warm.iterations, 0);
    close(&warm.step, &cold.step, 1e-12);
    close(
        &warm.multipliers.variables,
        &cold.multipliers.variables,
        1e-11,
    );
    // Every bound at its lower side and every row at its upper side: mostly wrong.
    let wrong = QpWarmStart {
        active_set: Some(ActiveSet {
            variables: vec![Activity::AtLower; 12],
            rows: vec![Activity::AtUpper; 11],
        }),
        // A dual method cannot use a primal-dual point; it is ignored.
        step: Some(vec![0.3; 12]),
        multipliers: None,
    };
    let recovered = optimal(&mut b, &model, Some(&wrong));
    close(&recovered.step, &cold.step, 1e-12);
    assert_eq!(recovered.active_set, cold.active_set);
    assert_eq!(b.report().cold_retries, 0);
}

#[test]
fn reused_buffers_never_change_a_result() {
    let (small, large) = (chain(4), chain(9));
    let fresh = |m: &QpModel| optimal(&mut backend(), m, None);
    let mut b = backend();
    for model in [&large, &small, &large, &small] {
        assert_eq!(optimal(&mut b, model, None), fresh(model));
    }
}

#[test]
fn the_budget_charges_drops_and_reentries_only() {
    let model = chain(12);
    let token = CancellationToken::default();
    let run = |max_iterations: u32, warm: Option<&QpWarmStart>| {
        let mut b = ActiveSetQp::default();
        b.configure(&QpOptions {
            max_iterations,
            ..Default::default()
        })
        .unwrap();
        (b.solve(&model, warm, &token).unwrap(), b.report())
    };
    // Mostly wrong: every solve from here drops constraints.
    let wrong = QpWarmStart {
        active_set: Some(ActiveSet {
            variables: vec![Activity::AtLower; 12],
            rows: vec![Activity::AtUpper; 11],
        }),
        ..Default::default()
    };
    let minimal = (0..1000)
        .find(|&k| matches!(run(k, Some(&wrong)).0, QpOutcome::Optimal(_)))
        .unwrap();
    assert!(minimal > 0);
    let (outcome, report) = run(minimal - 1, Some(&wrong));
    assert_eq!(outcome, QpOutcome::LimitReached(None));
    assert_eq!(report.limit_reason.as_deref(), Some("iterations"));
    let (QpOutcome::Optimal(s), report) = run(minimal, Some(&wrong)) else {
        panic!("expected an optimal solution");
    };
    // First entries count as changes but are not charged.
    assert!(
        s.iterations > u64::from(minimal),
        "{} {minimal}",
        s.iterations
    );
    assert_eq!(report.iterations, s.iterations);
    // A diagonal box QP needs no working-set changes or iteration budget.
    let clip = qp(
        &[&[1., 0.], &[0., 1.]],
        &[-3., 3.],
        (&[-1.; 2], &[1.; 2]),
        &[],
        (&[], &[]),
    );
    let mut b = ActiveSetQp::default();
    b.configure(&QpOptions {
        max_iterations: 0,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        b.solve(&clip, None, &token).unwrap(),
        QpOutcome::Optimal(QpSolution { iterations: 0, .. })
    ));
}

#[test]
fn cancellation_and_an_exhausted_time_budget_stop_before_any_work() {
    let model = chain(5);
    let token = CancellationToken::default();
    token.cancel();
    let mut b = backend();
    assert_eq!(b.solve(&model, None, &token).unwrap(), QpOutcome::Cancelled);
    assert_eq!(b.report().setups, 0);
    b.configure(&QpOptions {
        time_limit_seconds: Some(0.0),
        ..Default::default()
    })
    .unwrap();
    let outcome = b
        .solve(&model, None, &CancellationToken::default())
        .unwrap();
    assert_eq!(outcome, QpOutcome::LimitReached(None));
    assert_eq!(b.report().limit_reason.as_deref(), Some("time"));
}

#[test]
fn invalid_input_is_an_error() {
    let model = chain(3);
    let bad = QpWarmStart {
        active_set: Some(ActiveSet {
            variables: vec![Activity::Inactive; 2],
            rows: vec![],
        }),
        ..Default::default()
    };
    let mut b = backend();
    assert!(matches!(
        b.solve(&model, Some(&bad), &CancellationToken::default()),
        Err(Error::InvalidInput(_))
    ));
    assert!(
        b.configure(&QpOptions {
            tolerance: 0.0,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn a_block_diagonal_hessian_factorizes_by_blocks() {
    let n = 6;
    let mut h = vec![vec![0.0; n]; n];
    for b in 0..3 {
        let (i, j) = (2 * b, 2 * b + 1);
        h[i][i] = 2.0;
        h[j][j] = 3.0;
        h[i][j] = 1.0;
        h[j][i] = 1.0;
    }
    let h: Vec<&[f64]> = h.iter().map(Vec::as_slice).collect();
    let model = qp(
        &h,
        &[-1.0; 6],
        (&[-INF; 6], &[0.1; 6]),
        &[&[1.0; 6]],
        (&[-INF], &[0.3]),
    );
    let s = optimal(&mut backend(), &model, None);
    assert!(s.step.iter().all(|&p| p <= 0.1 + 1e-15));
    assert!((s.step.iter().sum::<f64>() - 0.3).abs() < 1e-14);
}

#[test]
fn a_nearly_singular_hessian_does_not_make_independent_constraints_dependent() {
    // A step QP of the stress problem seed 4, case 86 (restoration off):
    // det H ≈ 1e-9, so L⁻¹ maps the equality normal and e₁ to nearly
    // parallel vectors. A relative dependence test called the QP infeasible;
    // Clarabel solves it, with p₀ at its lower bound.
    let s = 0.5593743535551771;
    let model = qp(
        &[&[1.9963077684903379, s], &[s, 0.15673919240012563]],
        &[-6.459257621824861, -1.823568371642865],
        (
            &[-0.17915685898120315, -2.5268324504685515],
            &[3.8208431410187966, 1.4731675495314485],
        ),
        &[
            &[-8.9596874857966, -2.272076449362157],
            &[-0.8521524321190789, 0.14032305686187965],
        ],
        (
            &[-0.08774310693758913, -INF],
            &[-0.08774310693758913, 1.746434542202251],
        ),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[-0.1791568589809212, 0.7451037023066871], 1e-9);
    assert_eq!(s.active_set.unwrap().variables[0], Activity::AtLower);
}

fn elastic(weight: f64) -> ActiveSetQp {
    let mut b = ActiveSetQp::default();
    b.configure(&QpOptions {
        elastic_weight: Some(weight),
        ..Default::default()
    })
    .unwrap();
    b
}

fn elastic_optimal(b: &mut ActiveSetQp, model: &QpModel, warm: Option<&QpWarmStart>) -> QpSolution {
    let weight = b.settings.elastic_weight.unwrap();
    match b.solve(model, warm, &CancellationToken::default()).unwrap() {
        QpOutcome::Optimal(s) => {
            let check = model.accepts_elastic(&s, 1e-9, weight).unwrap();
            assert!(check.accepted, "{:?} {s:?}", check.residuals);
            s
        }
        other => panic!("expected an optimal elastic solution, got {other:?}"),
    }
}

#[test]
fn contradicting_rows_are_both_violated_at_the_elastic_weight() {
    // p ≥ 1 and p ≤ −1: between the two the ℓ1 cost is constant, so p = 0.
    let model = qp(
        &[&[1.]],
        &[0.],
        (&[-INF], &[INF]),
        &[&[1.], &[1.]],
        (&[1., -INF], &[INF, -1.]),
    );
    assert_eq!(
        backend()
            .solve(&model, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::Infeasible
    );
    let s = elastic_optimal(&mut elastic(3.0), &model, None);
    close(&s.step, &[0.0], 1e-14);
    close(&s.multipliers.constraints, &[3.0, -3.0], 1e-14);
    assert_eq!(s.active_set.unwrap().rows, [Activity::Inactive; 2]);
}

#[test]
fn hard_rows_stay_satisfied_in_an_elastic_qp() {
    // p ≥ 1 (hard) and p ≤ −1 (soft, γ = 3): min ½p² + 3·max(p + 1, 0) on
    // p ≥ 1 gives p = 1, λ = (4, −3). The hard multiplier exceeds γ.
    let model = qp(
        &[&[1.]],
        &[0.],
        (&[-INF], &[INF]),
        &[&[1.], &[1.]],
        (&[1., -INF], &[INF, -1.]),
    );
    let mut b = ActiveSetQp::default();
    b.configure(&QpOptions {
        elastic_weight: Some(3.0),
        hard_rows: vec![true, false],
        ..Default::default()
    })
    .unwrap();
    let QpOutcome::Optimal(s) = b
        .solve(&model, None, &CancellationToken::default())
        .unwrap()
    else {
        panic!("expected an optimal solution")
    };
    close(&s.step, &[1.0], 1e-14);
    close(&s.multipliers.constraints, &[4.0, -3.0], 1e-14);
    let mut ws = QpResidualWorkspace::default();
    let hard = [true, false];
    assert!(
        model
            .accepts_elastic_with_hard_rows(&s, 1e-9, 3.0, &hard, &mut ws)
            .unwrap()
            .accepted
    );
    // As an all-soft solution it is wrong: |λ₀| > γ.
    assert!(!model.accepts_elastic(&s, 1e-9, 3.0).unwrap().accepted);
    // One flag per row, or none.
    b.configure(&QpOptions {
        elastic_weight: Some(3.0),
        hard_rows: vec![true],
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        b.solve(&model, None, &CancellationToken::default()),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn a_small_weight_caps_the_multiplier_and_lets_the_row_be_violated() {
    // min ½(p − 3)² s.t. p ≤ 1: λ = −2 in the ordinary QP. With γ = 1 the
    // row is violated: p = 2, λ = −1. With γ = 5 nothing changes.
    let model = qp(
        &[&[1.]],
        &[-3.],
        (&[-INF], &[INF]),
        &[&[1.]],
        (&[-INF], &[1.]),
    );
    let s = elastic_optimal(&mut elastic(1.0), &model, None);
    close(&s.step, &[2.0], 1e-14);
    close(&s.multipliers.constraints, &[-1.0], 1e-14);
    let s = elastic_optimal(&mut elastic(5.0), &model, None);
    close(&s.step, &[1.0], 1e-14);
    close(&s.multipliers.constraints, &[-2.0], 1e-14);
    // A soft equality: min ½p² + |p − 2| gives p = 1 and λ = 1.
    let model = qp(&[&[1.]], &[0.], (&[-INF], &[INF]), &[&[1.]], (&[2.], &[2.]));
    let s = elastic_optimal(&mut elastic(1.0), &model, None);
    close(&s.step, &[1.0], 1e-14);
    close(&s.multipliers.constraints, &[1.0], 1e-14);
}

#[test]
fn variable_bounds_stay_hard_in_an_elastic_qp() {
    // The row p₀ + p₁ ≥ 5 cannot be met inside the box [−1, 1]²; the box
    // holds and the row is violated with λ = γ.
    let model = qp(
        &[&[1., 0.], &[0., 2.]],
        &[0., 0.],
        (&[-1.; 2], &[1.; 2]),
        &[&[1., 1.]],
        (&[5.], &[INF]),
    );
    let s = elastic_optimal(&mut elastic(10.0), &model, None);
    close(&s.step, &[1.0, 1.0], 1e-14);
    close(&s.multipliers.constraints, &[10.0], 1e-14);
    close(&s.multipliers.variables, &[-9.0, -8.0], 1e-13);
}

#[test]
fn an_elastic_row_that_becomes_satisfied_re_enters() {
    // min ½(p₀² − 1.8 p₀p₁ + p₁²) + γ·max(0, 2 − p₀ − p₁), γ = 0.05, with the
    // hard bound p₀ ≥ 1.3. From p = 0 the row is the most violated and
    // becomes elastic at p = (0.5, 0.5). The bound then pulls p₁ along to
    // 0.9 p₀ + γ, past the row, which must re-enter and leave: the optimum
    // is p = (1.3, 1.17) with λ = 0 and z₀ = 1.3 − 0.9·1.17.
    let model = qp(
        &[&[1., -0.9], &[-0.9, 1.]],
        &[0., 0.],
        (&[1.3, -INF], &[INF, INF]),
        &[&[1., 1.]],
        (&[2.], &[INF]),
    );
    let s = elastic_optimal(&mut elastic(0.05), &model, None);
    close(&s.step, &[1.3, 1.17], 1e-13);
    close(&s.multipliers.constraints, &[0.0], 1e-13);
    close(&s.multipliers.variables, &[1.3 - 0.9 * 1.17, 0.0], 1e-13);
}

#[test]
fn a_large_weight_reproduces_the_ordinary_solution() {
    let model = chain(12);
    let ordinary = optimal(&mut backend(), &model, None);
    let s = elastic_optimal(&mut elastic(1e6), &model, None);
    close(&s.step, &ordinary.step, 1e-12);
    close(
        &s.multipliers.constraints,
        &ordinary.multipliers.constraints,
        1e-10,
    );
    assert_eq!(s.active_set, ordinary.active_set);
}

#[test]
fn clarabel_and_the_dummy_reject_an_elastic_configuration() {
    let options = QpOptions {
        elastic_weight: Some(1.0),
        ..Default::default()
    };
    assert!(ActiveSetQp::default().supports_elastic());
    let mut clarabel = sqp_craft_clarabel::ClarabelQp::default();
    assert!(!clarabel.supports_elastic());
    assert!(matches!(
        clarabel.configure(&options),
        Err(Error::Unsupported(_))
    ));
    for weight in [0.0, -1.0, f64::NAN, INF] {
        let bad = QpOptions {
            elastic_weight: Some(weight),
            ..Default::default()
        };
        assert!(ActiveSetQp::default().configure(&bad).is_err(), "{weight}");
    }
}

#[test]
fn source_lists_follow_mutated_bound_kinds() {
    let mut model = qp(
        &[&[1.0, 0.0], &[0.0, 1.0]],
        &[-1.0, -2.0],
        (&[-INF; 2], &[INF; 2]),
        &[&[1.0, 1.0]],
        (&[-INF], &[INF]),
    );
    let mut b = backend();
    optimal(&mut b, &model, None);
    assert!(b.work.bounded.is_empty());
    assert!(b.work.equalities.is_empty());
    model.rows = Bounds::fixed(vec![3.0]);
    optimal(&mut b, &model, None);
    assert_eq!(b.work.bounded, vec![2]);
    assert_eq!(b.work.equalities, vec![2]);
    model.variables = Bounds::fixed(vec![0.0; 2]);
    model.rows = Bounds::free(1);
    optimal(&mut b, &model, None);
    assert_eq!(b.work.bounded, vec![0, 1]);
    assert_eq!(b.work.equalities, vec![0, 1]);
}

#[test]
fn unconstrained_coupled_qp_uses_checked_triangular_solutions() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = [2.0 * scale, scale];
        let b = [scale, 3.0 * scale];
        let model = qp(
            &[&a, &b],
            &[-scale, -2.0 * scale],
            (&[-INF; 2], &[INF; 2]),
            &[],
            (&[], &[]),
        );
        let s = optimal(&mut backend(), &model, None);
        close(&s.step, &[0.2, 0.6], 1e-12);
        assert_eq!(s.iterations, 0);
    }
}

#[test]
fn diagonal_box_proposal_keeps_original_rows_and_falls_back_when_needed() {
    let model = qp(
        &[&[1.0, 0.0], &[0.0, 1.0]],
        &[-3.0, -3.0],
        (&[0.0; 2], &[1.0; 2]),
        &[&[1.0, 1.0]],
        (&[1.0], &[1.0]),
    );
    let s = optimal(&mut backend(), &model, None);
    close(&s.step, &[0.5, 0.5], 1e-12);
    let mut impossible = model.clone();
    impossible.rows = Bounds::fixed(vec![3.0]);
    assert!(matches!(
        backend()
            .solve(&impossible, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::Infeasible
    ));
}

#[test]
fn factor_cache_uses_complete_exact_keys_and_never_reuses_failure() {
    let mut b = backend();
    let mut model = chain(4);
    optimal(&mut b, &model, None);
    assert_eq!(b.report().setups, 1);
    optimal(&mut b, &model, None);
    assert_eq!(b.report().updates, 1);
    assert_eq!(b.report().setups, 0);
    model.gradient[0] += 0.2;
    let reused = optimal(&mut b, &model, None);
    let fresh = optimal(&mut backend(), &model, None);
    close(&reused.step, &fresh.step, 1e-12);
    assert_eq!(b.report().updates, 1);
    model.hessian_lower.values[0] += 0.5;
    optimal(&mut b, &model, None);
    assert_eq!(b.report().setups, 1);
    let mut bad = model.clone();
    bad.hessian_lower.values[0] = -1.0;
    assert!(matches!(
        b.solve(&bad, None, &CancellationToken::default()).unwrap(),
        QpOutcome::NotPositiveDefinite
    ));
    assert!(!b.work.factor_valid);
    optimal(&mut b, &model, None);
    assert_eq!(b.report().setups, 1);
    let zero = model.hessian_lower.pattern.indptr[1];
    model.hessian_lower.values[zero] = 0.0;
    optimal(&mut b, &model, None);
    assert_eq!(b.report().setups, 1);
    model.hessian_lower.values[zero] = -0.0;
    optimal(&mut b, &model, None);
    assert_eq!(b.report().setups, 1);
    let a = qp(
        &[&[1000.0, 0.0, 0.1], &[0.0, 1.0, 0.0], &[0.1, 0.0, 4.0]],
        &[-1.0, -2.0, -3.0],
        (&[-10.0; 3], &[10.0; 3]),
        &[],
        (&[], &[]),
    );
    let c = qp(
        &[&[1000.0, 1.0, 0.0], &[1.0, 0.1, 0.0], &[0.0, 0.0, 4.0]],
        &[-1.0, -2.0, -3.0],
        (&[-10.0; 3], &[10.0; 3]),
        &[],
        (&[], &[]),
    );
    assert_eq!(a.hessian_lower.values, c.hessian_lower.values);
    optimal(&mut b, &a, None);
    let reused = optimal(&mut b, &c, None);
    let fresh = optimal(&mut backend(), &c, None);
    close(&reused.step, &fresh.step, 1e-12);
    assert_eq!(b.report().setups, 1);
    let mut invalid = c.clone();
    invalid.hessian_lower.values[0] = f64::NAN;
    assert!(
        b.solve(&invalid, None, &CancellationToken::default())
            .is_err()
    );
    optimal(&mut b, &c, None);
    assert_eq!(b.report().updates, 1);
    let mut cloned = b.clone();
    optimal(&mut cloned, &c, None);
    assert_eq!(cloned.report().updates, 1);
}

#[test]
fn inverse_cache_promotes_on_reuse_and_tracks_hessian_not_row_data() {
    let mut b = backend();
    let mut model = chain(5);
    optimal(&mut b, &model, None);
    assert!(!b.work.initial_valid);
    assert!(b.work.initial_j.is_empty());
    optimal(&mut b, &model, None);
    assert!(b.work.initial_valid);
    let inverse = b.work.initial_j.clone();
    model.gradient[0] += 0.3;
    model.matrix.values[0] *= 0.75;
    model.rows.upper[0] = 0.2;
    let reused = optimal(&mut b, &model, None);
    let fresh = optimal(&mut backend(), &model, None);
    close(&reused.step, &fresh.step, 1e-12);
    assert!(b.work.initial_valid);
    assert_eq!(b.work.initial_j, inverse);
    model.hessian_lower.values[0] += 1.0;
    let reused = optimal(&mut b, &model, None);
    let fresh = optimal(&mut backend(), &model, None);
    close(&reused.step, &fresh.step, 1e-12);
    assert!(!b.work.initial_valid);
    optimal(&mut b, &model, None);
    assert!(b.work.initial_valid);
    let mut invalid = model.clone();
    invalid.hessian_lower.values[0] = -1.0;
    assert!(matches!(
        b.solve(&invalid, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::NotPositiveDefinite
    ));
    assert!(!b.work.initial_valid);
    optimal(&mut b, &model, None);
    assert!(!b.work.initial_valid);
}

#[test]
fn parallel_transformed_normals_match_the_original_dots() {
    let execution = sqp_craft_parallel::Execution::new(8).unwrap();
    for n in [128, 256] {
        let model = QpModel {
            hessian_lower: CsrMatrix {
                pattern: CsrPattern {
                    rows: n,
                    cols: n,
                    indptr: (0..=n).collect(),
                    indices: (0..n).collect(),
                },
                values: vec![2.; n],
            },
            gradient: vec![0.; n],
            variables: Bounds::free(n),
            matrix: CsrMatrix {
                pattern: CsrPattern {
                    rows: 2,
                    cols: n,
                    indptr: vec![0, n, n + 1],
                    indices: (0..n).chain([3]).collect(),
                },
                values: (0..n).map(|i| (i % 7) as f64 * 0.1).chain([0.3]).collect(),
            },
            rows: Bounds::free(2),
        };
        let norms = vec![1.; 2];
        let serial = Sources {
            model: &model,
            norms: &norms,
            execution: Default::default(),
        };
        let parallel = Sources {
            model: &model,
            norms: &norms,
            execution: execution.clone(),
        };
        let j: Vec<_> = (0..n * n).map(|i| (i % 13) as f64 * 0.001).collect();
        for source in [0, n, n + 1] {
            for sigma in [-1., 1.] {
                let mut a = vec![0.; n];
                let mut b = a.clone();
                serial.project(source, sigma, &j, &mut a);
                parallel.project(source, sigma, &j, &mut b);
                assert_eq!(a, b);
            }
        }
    }
}
