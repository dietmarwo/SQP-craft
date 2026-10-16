use sqp_craft::*;
#[test]
fn overflowing_kkt_product_does_not_disappear() {
    let spec = ProblemSpec::new(
        Bounds::free(1),
        Bounds::fixed(vec![0.0; 2]),
        CsrPattern::dense(2, 1),
    )
    .unwrap();
    let e = Evaluation {
        objective: 0.0,
        constraints: vec![0.0; 2],
        gradient: vec![1.0],
        jacobian_values: vec![1e308, -1e308],
        primal_violation: 0.0,
        feasible_candidate: true,
    };
    assert!(matches!(
        kkt::nlp_residuals(
            &spec,
            &[0.0],
            &e,
            &Multipliers {
                constraints: vec![1e308; 2],
                variables: vec![0.0]
            }
        ),
        Err(Error::Evaluation(_))
    ));
}
#[test]
fn raw_bounds_and_nonfinite_qp_starts_are_rejected() {
    assert!(
        Bounds::new(vec![0.0], vec![])
            .checked_violation(&[-1.0])
            .is_err()
    );
    let warm = qp::QpWarmStart {
        step: Some(vec![f64::NAN]),
        ..Default::default()
    };
    assert!(warm.validate(1, 0).is_err());
    let warm = qp::QpWarmStart {
        multipliers: Some(Multipliers {
            constraints: vec![],
            variables: vec![f64::INFINITY],
        }),
        ..Default::default()
    };
    assert!(warm.validate(1, 0).is_err());
}
