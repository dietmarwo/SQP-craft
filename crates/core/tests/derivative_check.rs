use sqp_craft::{derivative_check::*, *};
struct Cached {
    last: Vec<f64>,
    wrong: bool,
}
impl NlpProblem for Cached {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.last = x.to_vec();
        c[0] = x[0] + 2.0 * x[1];
        Ok(x[0] * x[0] + x[1] * x[1])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        assert_eq!(x, self.last);
        g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        j.copy_from_slice(&[1.0, if self.wrong { -2.0 } else { 2.0 }]);
        Ok(())
    }
}
#[test]
fn grouped_cache_order_and_injected_derivative_error() {
    let s = ProblemSpec::new(
        Bounds::new(vec![0.0; 2], vec![2.0; 2]),
        Bounds::free(1),
        CsrPattern::dense(1, 2),
    )
    .unwrap();
    let r = check_derivatives(
        &s,
        &mut Cached {
            last: vec![],
            wrong: false,
        },
        &[0.0, 1.0],
        &DerivativeOptions::default(),
    )
    .unwrap();
    assert!(r.passed);
    assert_eq!(r.counts.derivatives, 1);
    let r = check_derivatives(
        &s,
        &mut Cached {
            last: vec![],
            wrong: true,
        },
        &[0.0, 1.0],
        &DerivativeOptions::default(),
    )
    .unwrap();
    assert!(!r.passed);
    assert!(r.issues.iter().any(|i| i.output == 1 && i.variable == 1));
}
#[test]
fn fixed_and_unvisited_coordinates_are_inconclusive() {
    let s = ProblemSpec::new(
        Bounds::fixed(vec![0.0, 1.0]),
        Bounds::free(1),
        CsrPattern::dense(1, 2),
    )
    .unwrap();
    let r = check_derivatives(
        &s,
        &mut Cached {
            last: vec![],
            wrong: false,
        },
        &[0.0, 1.0],
        &DerivativeOptions::default(),
    )
    .unwrap();
    assert!(!r.passed && r.inconclusive);
    assert_eq!(r.checked, 0);
    assert_eq!(r.fixed, vec![0, 1]);
    assert!(r.skipped.is_empty());
}

#[test]
fn fixed_variables_alone_do_not_make_a_check_inconclusive() {
    // Fixed initial states are common in trajectory transcriptions; the solver
    // never moves them, so they must not turn a correct model into "not passed".
    let s = ProblemSpec::new(
        Bounds::new(vec![0.0, -2.0], vec![0.0, 2.0]),
        Bounds::free(1),
        CsrPattern::dense(1, 2),
    )
    .unwrap();
    let mut model = Cached {
        last: vec![],
        wrong: false,
    };
    let r = check_derivatives(&s, &mut model, &[0.0, 1.0], &DerivativeOptions::default()).unwrap();
    assert!(r.passed && !r.inconclusive, "{r:?}");
    assert_eq!((r.checked, r.fixed.clone()), (1, vec![0]));
}

#[test]
fn max_directions_leaves_the_rest_skipped() {
    let s = ProblemSpec::new(Bounds::free(2), Bounds::free(1), CsrPattern::dense(1, 2)).unwrap();
    let options = DerivativeOptions {
        max_directions: 1,
        ..DerivativeOptions::default()
    };
    let mut model = Cached {
        last: vec![],
        wrong: false,
    };
    let r = check_derivatives(&s, &mut model, &[0.3, 0.4], &options).unwrap();
    assert!(!r.passed && r.inconclusive);
    assert_eq!((r.checked, r.skipped.clone()), (1, vec![1]));
}

/// Undefined for x0 > 0.5, as a propagation that fails past some point.
struct Partial;

impl NlpProblem for Partial {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        if x[0] > 0.5 {
            return Err(Error::UndefinedPoint("propagation failed".into()));
        }
        c[0] = x[0] + 2.0 * x[1];
        Ok(x[0] * x[0] + x[1] * x[1])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        j.copy_from_slice(&[1.0, 2.0]);
        Ok(())
    }
}

#[test]
fn undefined_perturbations_skip_the_coordinate_instead_of_aborting() {
    let s = ProblemSpec::new(Bounds::free(2), Bounds::free(1), CsrPattern::dense(1, 2)).unwrap();
    let r =
        check_derivatives(&s, &mut Partial, &[0.5, 1.0], &DerivativeOptions::default()).unwrap();
    assert!(!r.passed && r.inconclusive);
    assert_eq!((r.checked, r.skipped.clone()), (1, vec![0]));
    assert!(r.issues.is_empty());
    // An undefined base point still aborts: there is nothing to compare with.
    let err = check_derivatives(&s, &mut Partial, &[0.6, 1.0], &DerivativeOptions::default());
    assert!(matches!(err, Err(Error::UndefinedPoint(_))));
}

/// f = 1e6 + 10·x0: exact derivative 10, large value so roundoff matters.
struct Offset;

impl NlpProblem for Offset {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> Result<f64> {
        Ok(1e6 + 10.0 * x[0])
    }
    fn derivatives(&mut self, _: &[f64], g: &mut [f64], _: &mut [f64]) -> Result<()> {
        g[0] = 10.0;
        Ok(())
    }
}

#[test]
fn a_point_just_inside_a_bound_uses_a_full_one_sided_step() {
    // Central differences with h capped at the 1e-13 distance to the bound
    // give roundoff of ~1e3 here; a one-sided step of the requested size is exact
    // to ~1e-6.
    let s = ProblemSpec::new(
        Bounds::new(vec![0.0], vec![1.0]),
        Bounds::free(0),
        CsrPattern::dense(0, 1),
    )
    .unwrap();
    let r = check_derivatives(&s, &mut Offset, &[1e-13], &DerivativeOptions::default()).unwrap();
    assert!(r.passed, "{r:?}");
}
