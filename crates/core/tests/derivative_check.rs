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
}
