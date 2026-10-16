use sqp_craft::{qp::*, *};
use sqp_craft_clarabel::ClarabelQp;
fn q() -> QpModel {
    QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: 2,
                cols: 2,
                indptr: vec![0, 1, 3],
                indices: vec![0, 0, 1],
            },
            values: vec![2.0, 1.0, 2.0],
        },
        gradient: vec![-3.0, -3.0],
        variables: Bounds::free(2),
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(0, 2),
            values: vec![],
        },
        rows: Bounds::free(0),
    }
}
fn optimal(q: &QpModel) -> QpSolution {
    let outcome = ClarabelQp::default()
        .solve(q, None, &CancellationToken::default())
        .unwrap();
    let QpOutcome::Optimal(s) = outcome else {
        panic!("{outcome:?}")
    };
    assert!(q.residuals(&s).unwrap().max() < 1e-7);
    s
}
#[test]
fn off_diagonal_storage_is_not_doubled() {
    let s = optimal(&q());
    assert!(s.step.iter().all(|v| (v - 1.0).abs() < 1e-7));
}
#[test]
fn signed_lower_upper_and_fixed_duals() {
    let mut q = q();
    q.variables = Bounds::new(vec![2.0, -10.0], vec![2.0, 0.0]);
    let s = optimal(&q);
    assert!((s.step[0] - 2.0).abs() < 1e-7);
    assert!(s.multipliers.variables[1] < 0.0);
    q.variables = Bounds::new(vec![2.0, -10.0], vec![10.0, 10.0]);
    let s = optimal(&q);
    assert!(s.multipliers.variables[0] > 0.0);
}
#[test]
fn impossible_rows_and_zero_iteration_limit() {
    let mut q = q();
    q.matrix = CsrMatrix {
        pattern: CsrPattern::dense(1, 2),
        values: vec![0.0, 0.0],
    };
    q.rows = Bounds::fixed(vec![1.0]);
    assert!(matches!(
        ClarabelQp::default()
            .solve(&q, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::Infeasible
    ));
    let mut backend = ClarabelQp::default();
    backend
        .configure(&QpOptions {
            max_iterations: 0,
            ..Default::default()
        })
        .unwrap();
    let mut limited = self::q();
    limited.variables = Bounds::new(vec![2.0; 2], vec![10.0; 2]);
    assert!(matches!(
        backend
            .solve(&limited, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::LimitReached(_)
    ));
}
#[test]
fn cancellation_and_warm_start_validation() {
    let q = q();
    let token = CancellationToken::default();
    token.cancel();
    assert!(matches!(
        ClarabelQp::default().solve(&q, None, &token).unwrap(),
        QpOutcome::Cancelled
    ));
    let warm = QpWarmStart {
        step: Some(vec![f64::NAN; 2]),
        ..Default::default()
    };
    assert!(matches!(
        ClarabelQp::default().solve(&q, Some(&warm), &token),
        Err(Error::InvalidInput(_))
    ));
    let warm = QpWarmStart {
        step: Some(vec![0.0; 2]),
        ..Default::default()
    };
    assert!(matches!(
        ClarabelQp::default().solve(&q, Some(&warm), &token),
        Err(Error::Unsupported(_))
    ));
}

fn reuse_backend() -> ClarabelQp {
    let mut b = ClarabelQp::default();
    b.configure(&QpOptions {
        reuse_workspace: true,
        ..Default::default()
    })
    .unwrap();
    b
}

fn accepted(backend: &mut ClarabelQp, model: &QpModel) -> QpSolution {
    let QpOutcome::Optimal(s) = backend
        .solve(model, None, &CancellationToken::default())
        .unwrap()
    else {
        panic!("not optimal: {:?}", backend.report())
    };
    assert!(model.accepts(&s, 1e-7).unwrap().accepted);
    assert_eq!(s.iterations, backend.report().iterations);
    s
}

#[test]
fn updates_every_numeric_input_and_preserves_structural_zeros() {
    let mut b = reuse_backend();
    let mut model = q();
    model.variables = Bounds::new(vec![-4.0; 2], vec![4.0; 2]);
    model.matrix = CsrMatrix {
        pattern: CsrPattern::dense(1, 2),
        values: vec![0.0, 1.0],
    };
    model.rows = Bounds::fixed(vec![1.0]);
    model.hessian_lower.values[1] = 0.0;
    accepted(&mut b, &model);
    assert_eq!(b.report().setups, 1);
    for k in 1..=8 {
        let t = f64::from(k) / 8.0;
        model.hessian_lower.values = vec![2.0 + t, t / 2.0, 2.0 + t / 2.0];
        model.gradient = vec![-t, 1.0 + t];
        model.matrix.values = vec![t, 1.0 + t];
        model.rows = Bounds::fixed(vec![t]);
        model.variables.upper = vec![3.0 - t; 2];
        let updated = accepted(&mut b, &model);
        let cold = optimal(&model);
        for (&a, &z) in updated.step.iter().zip(&cold.step) {
            assert!((a - z).abs() < 1e-6);
        }
        assert_eq!(
            (
                b.report().setups,
                b.report().updates,
                b.report().cold_retries
            ),
            (0, 1, 0)
        );
    }
}

#[test]
fn structure_changes_rebuild_even_with_same_number_of_conic_rows() {
    let mut b = reuse_backend();
    let mut model = q();
    // Equality -> upper side has identical A pattern and dual-sign mapping.
    // The cone type is therefore an essential part of the cache key.
    model.variables = Bounds::new(vec![1.0, f64::NEG_INFINITY], vec![1.0, f64::INFINITY]);
    accepted(&mut b, &model);
    model.variables.lower[0] = f64::NEG_INFINITY;
    accepted(&mut b, &model);
    assert_eq!((b.report().setups, b.report().updates), (1, 0));
    model.variables.lower[0] = -2.0;
    model.variables.upper[0] = f64::INFINITY;
    accepted(&mut b, &model);
    assert_eq!((b.report().setups, b.report().updates), (1, 0));
    // An omitted off-diagonal zero differs structurally from an explicit zero.
    model.hessian_lower = CsrMatrix {
        pattern: CsrPattern {
            rows: 2,
            cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
        },
        values: vec![2.0, 2.0],
    };
    accepted(&mut b, &model);
    assert_eq!(b.report().setups, 1);
    let smaller = QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern::dense(1, 1),
            values: vec![1.0],
        },
        gradient: vec![-1.0],
        variables: Bounds::free(1),
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(0, 1),
            values: vec![],
        },
        rows: Bounds::free(0),
    };
    accepted(&mut b, &smaller);
    assert_eq!(b.report().setups, 1);
}

#[test]
fn cache_clone_disabling_and_cancellation_do_not_reuse_stale_state() {
    let mut b = reuse_backend();
    let model = q();
    accepted(&mut b, &model);
    let mut cloned = b.clone();
    accepted(&mut cloned, &model);
    assert_eq!(cloned.report().setups, 1);
    let old = CancellationToken::default();
    assert!(matches!(
        b.solve(&model, None, &old).unwrap(),
        QpOutcome::Optimal(_)
    ));
    old.cancel();
    accepted(&mut b, &model); // Replaces callback; old cancellation is irrelevant.
    assert_eq!(b.report().updates, 1);
    assert!(matches!(
        b.solve(&model, None, &old).unwrap(),
        QpOutcome::Cancelled
    ));
    assert_eq!((b.report().updates, b.report().setups), (0, 0));
    accepted(&mut b, &model);
    assert_eq!(b.report().setups, 1);
    b.configure(&QpOptions::default()).unwrap();
    accepted(&mut b, &model);
    accepted(&mut b, &model);
    assert_eq!((b.report().setups, b.report().updates), (1, 0));
}

#[test]
fn reused_infeasibility_is_confirmed_cold_and_not_cached() {
    let mut b = reuse_backend();
    let mut model = q();
    model.variables = Bounds::new(vec![0.0; 2], vec![2.0; 2]);
    model.matrix = CsrMatrix {
        pattern: CsrPattern::dense(1, 2),
        values: vec![1.0, 1.0],
    };
    model.rows = Bounds::fixed(vec![1.0]);
    accepted(&mut b, &model);
    model.rows = Bounds::fixed(vec![5.0]);
    assert!(matches!(
        b.solve(&model, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::Infeasible
    ));
    assert_eq!(
        (
            b.report().updates,
            b.report().setups,
            b.report().cold_retries
        ),
        (1, 1, 1)
    );
    assert!(
        b.report()
            .retry_reason
            .as_ref()
            .unwrap()
            .contains("Infeasible")
    );
    model.rows = Bounds::fixed(vec![1.0]);
    accepted(&mut b, &model);
    assert_eq!(b.report().setups, 1);
}

#[test]
fn changed_zero_time_and_iteration_limits_apply_to_cached_workspace() {
    let mut b = reuse_backend();
    let mut model = q();
    model.variables = Bounds::new(vec![2.0; 2], vec![10.0; 2]);
    accepted(&mut b, &model);
    b.configure(&QpOptions {
        reuse_workspace: true,
        max_iterations: 0,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        b.solve(&model, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::LimitReached(_)
    ));
    assert_eq!(b.report().limit_reason.as_deref(), Some("iterations"));
    assert_eq!(b.report().iterations, 0);
    assert_eq!(b.report().cold_retries, 0);
    b.configure(&QpOptions {
        reuse_workspace: true,
        time_limit_seconds: Some(0.0),
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        b.solve(&model, None, &CancellationToken::default())
            .unwrap(),
        QpOutcome::LimitReached(_)
    ));
    assert_eq!(b.report().limit_reason.as_deref(), Some("time"));
    assert_eq!((b.report().setups, b.report().updates), (0, 0));
}
