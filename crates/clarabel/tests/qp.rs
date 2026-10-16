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
