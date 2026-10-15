//! QP data layout, independent residuals and the placeholder backend.

use sqp_craft::qp::{ActiveSet, Activity, DummyQp, QpBackend, QpModel, QpSolution, QpWarmStart};
use sqp_craft::{Bounds, CancellationToken, CsrMatrix, CsrPattern, Error, Multipliers};

const INF: f64 = f64::INFINITY;

fn csr(rows: usize, cols: usize, indptr: &[usize], indices: &[usize], values: &[f64]) -> CsrMatrix {
    CsrMatrix {
        pattern: CsrPattern {
            rows,
            cols,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
        },
        values: values.to_vec(),
    }
}

fn no_rows(n: usize) -> (CsrMatrix, Bounds) {
    (csr(0, n, &[0], &[], &[]), Bounds::free(0))
}

fn solution(step: &[f64], rows: &[f64], vars: &[f64]) -> QpSolution {
    QpSolution {
        step: step.to_vec(),
        multipliers: Multipliers {
            constraints: rows.to_vec(),
            variables: vars.to_vec(),
        },
        active_set: None,
        iterations: 0,
    }
}

/// minimize 0.5 p^2 + p on [-0.5, 1]: the solution p = -0.5 has an active lower bound.
fn bound_qp() -> QpModel {
    let (matrix, rows) = no_rows(1);
    QpModel {
        hessian_lower: csr(1, 1, &[0, 1], &[0], &[1.0]),
        gradient: vec![1.0],
        variables: Bounds::new(vec![-0.5], vec![1.0]),
        matrix,
        rows,
    }
}

#[test]
fn bound_constrained_solution_has_zero_residuals() {
    let qp = bound_qp();
    qp.validate().unwrap();
    assert_eq!(qp.objective(&[-0.5]), -0.375);
    let r = qp.residuals(&solution(&[-0.5], &[], &[0.5])).unwrap();
    assert_eq!(r.max(), 0.0);
}

#[test]
fn row_constrained_solution_and_wrong_sign() {
    // minimize 0.5 |p|^2 subject to p0 + p1 >= 1: p = (0.5, 0.5), λ = 0.5.
    let qp = QpModel {
        hessian_lower: csr(2, 2, &[0, 1, 2], &[0, 1], &[1.0, 1.0]),
        gradient: vec![0.0, 0.0],
        variables: Bounds::free(2),
        matrix: csr(1, 2, &[0, 2], &[0, 1], &[1.0, 1.0]),
        rows: Bounds::new(vec![1.0], vec![INF]),
    };
    let ok = qp
        .residuals(&solution(&[0.5, 0.5], &[0.5], &[0.0, 0.0]))
        .unwrap();
    assert_eq!(ok.max(), 0.0);
    let wrong = qp
        .residuals(&solution(&[0.5, 0.5], &[-0.5], &[0.0, 0.0]))
        .unwrap();
    assert_eq!((wrong.stationarity, wrong.dual_sign), (1.0, 0.5));
}

#[test]
fn lower_triangle_expands_symmetrically() {
    // H = [[2, 1], [1, 2]] stored as its lower triangle; g = (-3, -3) gives p = (1, 1).
    let (matrix, rows) = no_rows(2);
    let qp = QpModel {
        hessian_lower: csr(2, 2, &[0, 1, 3], &[0, 0, 1], &[2.0, 1.0, 2.0]),
        gradient: vec![-3.0, -3.0],
        variables: Bounds::free(2),
        matrix,
        rows,
    };
    let mut hp = [0.0; 2];
    qp.hessian_product(&[1.0, 0.0], &mut hp);
    assert_eq!(hp, [2.0, 1.0]);
    let r = qp
        .residuals(&solution(&[1.0, 1.0], &[], &[0.0, 0.0]))
        .unwrap();
    assert_eq!(r.max(), 0.0);
}

#[test]
fn validation_rejects_upper_triangle_and_shape_errors() {
    let (matrix, rows) = no_rows(2);
    let upper = QpModel {
        hessian_lower: csr(2, 2, &[0, 2, 3], &[0, 1, 1], &[2.0, 1.0, 2.0]),
        gradient: vec![0.0, 0.0],
        variables: Bounds::free(2),
        matrix,
        rows,
    };
    let err = upper.validate().unwrap_err();
    assert!(matches!(&err, Error::InvalidInput(m) if m.contains("(0, 1) is above the diagonal")));

    let mut qp = bound_qp();
    qp.matrix = csr(1, 2, &[0, 1], &[0], &[1.0]);
    qp.rows = Bounds::free(1);
    assert!(qp.validate().is_err(), "matrix with 2 columns for n = 1");
    assert!(
        bound_qp()
            .residuals(&solution(&[0.0, 0.0], &[], &[0.0]))
            .is_err()
    );
}

#[test]
fn dummy_backend_never_fabricates_a_step() {
    let cancel = CancellationToken::default();
    let r = DummyQp.solve(&bound_qp(), None, &cancel);
    assert!(matches!(r, Err(Error::Unsupported(_))));

    let warm = QpWarmStart {
        active_set: Some(ActiveSet {
            variables: vec![Activity::AtLower],
            rows: vec![],
        }),
        ..QpWarmStart::default()
    };
    assert!(matches!(
        DummyQp.solve(&bound_qp(), Some(&warm), &cancel),
        Err(Error::Unsupported(_))
    ));

    let bad_warm = QpWarmStart {
        step: Some(vec![0.0, 0.0]),
        ..QpWarmStart::default()
    };
    let r = DummyQp.solve(&bound_qp(), Some(&bad_warm), &cancel);
    assert!(matches!(r, Err(Error::InvalidInput(_))));
}
