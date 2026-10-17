//! The multiplier sign convention, checked on problems with known solutions.

use sqp_craft::kkt::{KktResiduals, nlp_residuals};
use sqp_craft::{Bounds, CsrPattern, Error, Evaluation, Multipliers, ProblemSpec};

const INF: f64 = f64::INFINITY;

/// (x0 - 1)^2 + (x1 - 2)^2 with the single row c = x0 + x1, evaluated at x.
fn toy_evaluation(x: [f64; 2]) -> Evaluation {
    Evaluation {
        objective: (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        constraints: vec![x[0] + x[1]],
        gradient: vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)],
        jacobian_values: vec![1.0, 1.0],
        primal_violation: 0.0,
        feasible_candidate: true,
    }
}

fn toy_spec(row: Bounds) -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
        row,
        CsrPattern::dense(1, 2),
    )
    .unwrap()
}

fn multipliers(rows: &[f64], vars: &[f64]) -> Multipliers {
    Multipliers {
        constraints: rows.to_vec(),
        variables: vars.to_vec(),
    }
}

fn zero() -> KktResiduals {
    KktResiduals {
        primal: 0.0,
        stationarity: 0.0,
        complementarity: 0.0,
        dual_sign: 0.0,
    }
}

#[test]
fn equality_constraint_with_correct_and_flipped_sign() {
    // x0 + x1 = 2 has the solution (0.5, 1.5) with grad f = (-1, -1) = J^T λ, so λ = -1.
    let spec = toy_spec(Bounds::fixed(vec![2.0]));
    let (x, e) = ([0.5, 1.5], toy_evaluation([0.5, 1.5]));
    let r = nlp_residuals(&spec, &x, &e, &multipliers(&[-1.0], &[0.0, 0.0])).unwrap();
    assert_eq!(r, zero());
    let flipped = nlp_residuals(&spec, &x, &e, &multipliers(&[1.0], &[0.0, 0.0])).unwrap();
    assert_eq!(flipped.stationarity, 2.0);
    assert_eq!(flipped.dual_sign, 0.0, "an equality accepts either sign");
}

#[test]
fn upper_bound_activity_requires_negative_multiplier() {
    // x0 + x1 <= 2 is active at the same solution; λ = -1 selects the finite upper bound.
    let spec = toy_spec(Bounds::new(vec![-INF], vec![2.0]));
    let (x, e) = ([0.5, 1.5], toy_evaluation([0.5, 1.5]));
    assert_eq!(
        nlp_residuals(&spec, &x, &e, &multipliers(&[-1.0], &[0.0, 0.0])).unwrap(),
        zero()
    );
    let wrong = nlp_residuals(&spec, &x, &e, &multipliers(&[1.0], &[0.0, 0.0])).unwrap();
    assert_eq!((wrong.dual_sign, wrong.stationarity), (1.0, 2.0));
}

#[test]
fn lower_bound_activity_requires_positive_multiplier() {
    // x0 + x1 >= 4 is active at (1.5, 2.5) with grad f = (1, 1), so λ = +1.
    let spec = toy_spec(Bounds::new(vec![4.0], vec![INF]));
    let (x, e) = ([1.5, 2.5], toy_evaluation([1.5, 2.5]));
    assert_eq!(
        nlp_residuals(&spec, &x, &e, &multipliers(&[1.0], &[0.0, 0.0])).unwrap(),
        zero()
    );
}

fn shifted_square(x: f64) -> Evaluation {
    // f = (x + 1)^2 on [0, 4], no rows; the solution x = 0 has an active lower bound.
    Evaluation {
        objective: (x + 1.0).powi(2),
        constraints: vec![],
        gradient: vec![2.0 * (x + 1.0)],
        jacobian_values: vec![],
        primal_violation: 0.0,
        feasible_candidate: true,
    }
}

#[test]
fn variable_bound_multipliers_and_complementarity() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0], vec![4.0]),
        Bounds::free(0),
        CsrPattern::dense(0, 1),
    )
    .unwrap();
    let at = |x: f64, z: f64| {
        nlp_residuals(&spec, &[x], &shifted_square(x), &multipliers(&[], &[z])).unwrap()
    };
    assert_eq!(at(0.0, 2.0), zero());
    // Stationary but the multiplier claims an inactive bound.
    let interior = at(1.0, 4.0);
    assert_eq!(
        (interior.stationarity, interior.complementarity),
        (0.0, 4.0)
    );
    // A negative multiplier selects the upper bound, four units away.
    let wrong = at(0.0, -4.0);
    assert_eq!((wrong.stationarity, wrong.complementarity), (6.0, 16.0));
}

#[test]
fn primal_residual_and_input_checks() {
    let spec = toy_spec(Bounds::fixed(vec![2.0]));
    let e = toy_evaluation([1.0, 2.0]);
    let r = nlp_residuals(&spec, &[1.0, 2.0], &e, &multipliers(&[0.0], &[0.0, 0.0])).unwrap();
    assert_eq!(r.primal, 1.0);
    assert_eq!(r.max(), 1.0);
    let short = nlp_residuals(&spec, &[1.0, 2.0], &e, &multipliers(&[], &[0.0, 0.0]));
    assert!(matches!(short, Err(Error::InvalidInput(m)) if m.contains("row multipliers")));
    let nan = nlp_residuals(
        &spec,
        &[1.0, 2.0],
        &e,
        &multipliers(&[f64::NAN], &[0.0, 0.0]),
    );
    assert!(nan.is_err());
}

fn elastic(
    spec: &ProblemSpec,
    x: [f64; 2],
    m: &Multipliers,
    weight: f64,
    slack: f64,
) -> KktResiduals {
    let e = toy_evaluation(x);
    sqp_craft::kkt::elastic_residuals_with_workspace(
        &sqp_craft::kkt::KktPoint {
            x: &x,
            variable_bounds: spec.variables(),
            rows: &e.constraints,
            row_bounds: spec.constraints(),
            gradient: &e.gradient,
            jacobian: spec.jacobian(),
            jacobian_values: &e.jacobian_values,
        },
        m,
        weight,
        slack,
        &mut Bounds::free(0),
        &mut Vec::new(),
    )
    .unwrap()
}

#[test]
fn elastic_residuals_accept_a_violated_row_only_at_the_weight() {
    // x0 + x1 >= 5 at x = (1.5, 2.5): violated by 1, and ∇f = (1, 1) = 1·∇c.
    // That is first-order optimal for f + γ·violation exactly when γ = 1.
    let spec = toy_spec(Bounds::new(vec![5.0], vec![INF]));
    let m = multipliers(&[1.0], &[0.0, 0.0]);
    let at_one = elastic(&spec, [1.5, 2.5], &m, 1.0, 1e-7);
    assert_eq!(at_one, zero());
    // A larger weight leaves the multiplier one short of it on a row
    // violated by one; the ordinary residuals see only the violation.
    let at_two = elastic(&spec, [1.5, 2.5], &m, 2.0, 1e-7);
    assert_eq!(at_two.complementarity, 1.0);
    assert_eq!(at_two.primal, 0.0);
    let ordinary = nlp_residuals(&spec, &[1.5, 2.5], &toy_evaluation([1.5, 2.5]), &m).unwrap();
    assert_eq!(ordinary.primal, 1.0);
    // A multiplier above the weight is a sign-type violation.
    let above = elastic(&spec, [1.5, 2.5], &m, 0.5, 1e-7);
    assert_eq!(above.dual_sign, 0.5);
}

#[test]
fn elastic_residuals_treat_tolerance_level_violations_as_satisfied() {
    // On the bound to 1e-9 with an ordinary multiplier: within the slack the
    // row is satisfied and only the primal residual reports the 1e-9.
    let spec = toy_spec(Bounds::new(vec![4.0], vec![INF]));
    let x = [1.5, 2.5 - 1e-9];
    let lambda = 1.0 - 1e-9;
    let m = multipliers(&[lambda], &[0.0, 0.0]);
    let within = elastic(&spec, x, &m, 1e4, 1e-7);
    assert!(within.primal > 0.0 && within.primal < 2e-9, "{within:?}");
    assert!(within.complementarity < 2e-9, "{within:?}");
    // Without slack the same point would need λ = 10⁴ on a violated row.
    let strict = elastic(&spec, x, &m, 1e4, 0.0);
    assert!(strict.complementarity > 9e-6, "{strict:?}");
}

#[test]
fn hard_rows_are_tested_as_ordinary_rows_in_the_elastic_residuals() {
    // The violated row of the first test, but hard: its violation is primal
    // again, and a multiplier above the weight is no longer an error.
    let spec = toy_spec(Bounds::new(vec![5.0], vec![INF]));
    let x = [1.5, 2.5];
    let e = toy_evaluation(x);
    let point = sqp_craft::kkt::KktPoint {
        x: &x,
        variable_bounds: spec.variables(),
        rows: &e.constraints,
        row_bounds: spec.constraints(),
        gradient: &e.gradient,
        jacobian: spec.jacobian(),
        jacobian_values: &e.jacobian_values,
    };
    let m = multipliers(&[1.0], &[0.0, 0.0]);
    let hard = |weight| {
        sqp_craft::kkt::elastic_residuals_with_hard_rows(
            &point,
            &m,
            weight,
            &[true],
            1e-7,
            &mut Bounds::free(0),
            &mut Vec::new(),
        )
        .unwrap()
    };
    let ordinary = nlp_residuals(&spec, &x, &e, &m).unwrap();
    assert_eq!(hard(1.0), ordinary);
    assert_eq!(hard(0.5).dual_sign, 0.0);
    // One flag per row, or none.
    assert!(
        sqp_craft::kkt::elastic_residuals_with_hard_rows(
            &point,
            &m,
            1.0,
            &[true, false],
            1e-7,
            &mut Bounds::free(0),
            &mut Vec::new(),
        )
        .is_err()
    );
}
