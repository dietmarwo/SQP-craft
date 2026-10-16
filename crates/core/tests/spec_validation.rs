//! Problem specification invariants and error messages.

use sqp_craft::{Bounds, CsrMatrix, CsrPattern, Error, ProblemSpec};

const INF: f64 = f64::INFINITY;

fn pattern(rows: usize, cols: usize, indptr: &[usize], indices: &[usize]) -> CsrPattern {
    CsrPattern {
        rows,
        cols,
        indptr: indptr.to_vec(),
        indices: indices.to_vec(),
    }
}

fn message(r: Result<ProblemSpec, Error>) -> String {
    match r {
        Err(Error::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn accepts_free_one_sided_and_equality_rows() {
    let spec = ProblemSpec::new(
        Bounds::free(3),
        Bounds::new(vec![-INF, 1.0, 2.0], vec![0.0, INF, 2.0]),
        pattern(3, 3, &[0, 1, 1, 3], &[2, 0, 2]),
    )
    .unwrap();
    assert_eq!((spec.n(), spec.m(), spec.nnz()), (3, 3, 3));
    assert_eq!(spec.jacobian().row(2), [0, 2]);
}

#[test]
fn accepts_zero_constraints() {
    let spec = ProblemSpec::new(Bounds::free(2), Bounds::free(0), CsrPattern::dense(0, 2)).unwrap();
    assert_eq!((spec.m(), spec.nnz()), (0, 0));
}

#[test]
fn rejects_zero_variables() {
    let m = message(ProblemSpec::new(
        Bounds::free(0),
        Bounds::free(0),
        CsrPattern::dense(0, 0),
    ));
    assert!(m.contains("at least one variable"));
}

#[test]
fn bound_errors_name_the_offending_entry() {
    for (lower, upper) in [(1.0, 0.0), (f64::NAN, 1.0), (INF, INF), (-INF, -INF)] {
        let m = message(ProblemSpec::new(
            Bounds::new(vec![0.0, lower], vec![1.0, upper]),
            Bounds::free(0),
            CsrPattern::dense(0, 2),
        ));
        assert!(m.starts_with("variable bounds: entry 1"), "{m}");
    }
    let m = message(ProblemSpec::new(
        Bounds::free(2),
        Bounds::new(vec![0.0], vec![]),
        CsrPattern::dense(1, 2),
    ));
    assert!(
        m.starts_with("constraint bounds: expected 1 entries"),
        "{m}"
    );
}

#[test]
fn jacobian_shape_must_match_bounds() {
    let m = message(ProblemSpec::new(
        Bounds::free(2),
        Bounds::free(1),
        CsrPattern::dense(1, 3),
    ));
    assert!(m.contains("shape is 1 x 3, expected m x n = 1 x 2"), "{m}");
}

#[test]
fn csr_invariants() {
    let bad = [
        (pattern(1, 2, &[0], &[]), "indptr has 1 entries"),
        (pattern(1, 2, &[1, 2], &[0, 1]), "must start at 0"),
        (pattern(1, 2, &[0, 3], &[0, 1]), "must start at 0 and end"),
        (pattern(3, 2, &[0, 2, 1, 2], &[0, 1]), "decreases at row 1"),
        (pattern(1, 2, &[0, 1], &[2]), "column index 2"),
        (pattern(1, 2, &[0, 2], &[0, 0]), "strictly increasing"),
        (pattern(1, 2, &[0, 2], &[1, 0]), "strictly increasing"),
    ];
    for (p, expected) in bad {
        let m = message(ProblemSpec::new(Bounds::free(2), Bounds::free(p.rows), p));
        assert!(
            m.starts_with("Jacobian pattern: ") && m.contains(expected),
            "{m}"
        );
    }
}

#[test]
fn check_point_requires_finite_in_bounds_points() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0, -INF], vec![1.0, INF]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    assert!(spec.check_point(&[0.0, -1e300], "x").is_ok());
    assert!(spec.check_point(&[1.0, 5.0], "x").is_ok());
    for bad in [&[0.5][..], &[0.5, INF], &[1.5, 0.0], &[-0.0 - 1e-300, 0.0]] {
        assert!(spec.check_point(bad, "x").is_err(), "{bad:?}");
    }
}

#[test]
fn csr_products() {
    // A = [[1, 0, 2], [0, 3, 0]]
    let p = pattern(2, 3, &[0, 2, 3], &[0, 2, 1]);
    let values = [1.0, 2.0, 3.0];
    let mut y = [0.0; 2];
    p.multiply(&values, &[1.0, 1.0, 1.0], &mut y);
    assert_eq!(y, [3.0, 3.0]);
    let mut z = [9.0; 3];
    p.multiply_transpose(&values, &[1.0, 2.0], &mut z);
    assert_eq!(z, [1.0, 6.0, 2.0]);
    let a = CsrMatrix {
        pattern: p,
        values: vec![1.0, f64::NAN, 3.0],
    };
    assert!(a.validate("A").is_err());
}

#[test]
fn violation_measures_largest_excess() {
    let b = Bounds::new(vec![0.0, -INF, 2.0], vec![1.0, 0.0, 2.0]);
    assert_eq!(b.violation(&[0.5, -5.0, 2.0]), 0.0);
    assert_eq!(b.violation(&[-0.25, 3.0, 2.5]), 3.0);
    assert_eq!(b.first_outside(&[0.5, 1.0, 2.0]), Some(1));
}

#[test]
fn error_display_has_category() {
    assert_eq!(
        Error::InvalidInput("x".into()).to_string(),
        "invalid input: x"
    );
    assert_eq!(Error::UndefinedPoint("x".into()).message(), "x");
}
