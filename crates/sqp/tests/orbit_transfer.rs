//! Maximum-radius low-thrust orbit transfer (Bryson & Ho) by trapezoidal
//! collocation: the closest fixture to the target application (sparse
//! Jacobian, fixed initial state, mixed nonlinear equality rows).
//!
//! The reference values r(tf) = 1.52206 (N = 20) and 1.52449 (N = 40) are the
//! SciPy SLSQP results on the same discretization.

use sqp_craft::derivative_check::{DerivativeOptions, check_derivatives};
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SqpSolver,
    Status,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

const THRUST: f64 = 0.1405;
const MASS_FLOW: f64 = 0.0749;
const FINAL_TIME: f64 = 3.32;

struct OrbitTransfer {
    nodes: usize,
    rows: Vec<Vec<usize>>,
}

impl OrbitTransfer {
    /// Spec, model and the usual crude initial guess for `n_intervals` intervals.
    fn new(n_intervals: usize) -> (ProblemSpec, Self, Vec<f64>) {
        let nodes = n_intervals + 1;
        let col = |block: usize, node: usize| block * nodes + node;
        let mut rows = Vec::new();
        for _state in 0..3 {
            for i in 0..n_intervals {
                let mut c: Vec<usize> = (0..4).flat_map(|b| [col(b, i), col(b, i + 1)]).collect();
                c.sort_unstable();
                rows.push(c);
            }
        }
        rows.push(vec![col(1, n_intervals)]); // u(tf) = 0
        rows.push(vec![col(0, n_intervals), col(2, n_intervals)]); // v(tf) = 1/sqrt(r(tf))
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        for r in &rows {
            indices.extend_from_slice(r);
            indptr.push(indices.len());
        }
        let n = 4 * nodes;
        let pi = std::f64::consts::PI;
        let ranges = [(0.5, 5.0), (-1.0, 1.0), (0.1, 2.0), (-pi, pi)];
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        for (lo, up) in ranges {
            lower.extend(std::iter::repeat_n(lo, nodes));
            upper.extend(std::iter::repeat_n(up, nodes));
        }
        for (block, value) in [(0, 1.0), (1, 0.0), (2, 1.0)] {
            lower[col(block, 0)] = value;
            upper[col(block, 0)] = value;
        }
        let m = rows.len();
        let spec = ProblemSpec::new(
            Bounds::new(lower, upper),
            Bounds::fixed(vec![0.0; m]),
            CsrPattern {
                rows: m,
                cols: n,
                indptr,
                indices,
            },
        )
        .expect("valid orbit-transfer spec");
        let s = |i: usize| i as f64 / n_intervals as f64;
        let mut x0 = Vec::with_capacity(n);
        x0.extend((0..nodes).map(|i| 1.0 + 0.5 * s(i)));
        x0.extend((0..nodes).map(|_| 0.0));
        x0.extend((0..nodes).map(|i| 1.0 - 0.2 * s(i)));
        x0.extend((0..nodes).map(|i| if 2 * i < n_intervals { 0.5 } else { -0.5 }));
        (spec, Self { nodes, rows }, x0)
    }

    fn h(&self) -> f64 {
        FINAL_TIME / (self.nodes - 1) as f64
    }

    fn mass(&self, node: usize) -> f64 {
        1.0 - MASS_FLOW * self.h() * node as f64
    }

    /// State derivative and its partials with respect to (r, u, v, phi).
    fn dynamics(&self, x: &[f64], node: usize) -> ([f64; 3], [[f64; 4]; 3]) {
        let k = self.nodes;
        let (r, u, v, phi) = (x[node], x[k + node], x[2 * k + node], x[3 * k + node]);
        let m = self.mass(node);
        let (s, c) = phi.sin_cos();
        let f = [
            u,
            v * v / r - 1.0 / (r * r) + THRUST * s / m,
            -u * v / r + THRUST * c / m,
        ];
        let d = [
            [0.0, 1.0, 0.0, 0.0],
            [
                -v * v / (r * r) + 2.0 / (r * r * r),
                0.0,
                2.0 * v / r,
                THRUST * c / m,
            ],
            [u * v / (r * r), -v / r, -u / r, -THRUST * s / m],
        ];
        (f, d)
    }
}

impl NlpProblem for OrbitTransfer {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        let (k, n_int, h) = (self.nodes, self.nodes - 1, self.h());
        for i in 0..n_int {
            let (fa, _) = self.dynamics(x, i);
            let (fb, _) = self.dynamics(x, i + 1);
            for state in 0..3 {
                let (a, b) = (x[state * k + i], x[state * k + i + 1]);
                c[state * n_int + i] = b - a - 0.5 * h * (fa[state] + fb[state]);
            }
        }
        let r_end = x[n_int];
        c[3 * n_int] = x[k + n_int];
        c[3 * n_int + 1] = x[2 * k + n_int] - 1.0 / r_end.sqrt();
        Ok(-r_end)
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        let (k, n_int, h) = (self.nodes, self.nodes - 1, self.h());
        g.fill(0.0);
        g[n_int] = -1.0;
        let mut pos = 0;
        for state in 0..3 {
            for i in 0..n_int {
                let (_, da) = self.dynamics(x, i);
                let (_, db) = self.dynamics(x, i + 1);
                for &column in &self.rows[state * n_int + i] {
                    let (block, node) = (column / k, column % k);
                    let partial = if node == i { da } else { db }[state][block];
                    let mut value = -0.5 * h * partial;
                    if block == state {
                        value += if node == i { -1.0 } else { 1.0 };
                    }
                    jac[pos] = value;
                    pos += 1;
                }
            }
        }
        jac[pos] = 1.0;
        jac[pos + 1] = 0.5 * x[n_int].powf(-1.5);
        jac[pos + 2] = 1.0;
        Ok(())
    }
}

#[test]
fn derivatives_are_consistent_and_fixed_states_do_not_make_the_check_inconclusive() {
    let (spec, mut model, x0) = OrbitTransfer::new(10);
    let report = check_derivatives(&spec, &mut model, &x0, &DerivativeOptions::default()).unwrap();
    assert!(report.passed, "{report:?}");
    assert_eq!(report.fixed, vec![0, 11, 22]);
    assert_eq!(report.checked, spec.n() - 3);
}

#[test]
fn converges_to_the_reference_radius() {
    // With the former nondecreasing penalty this ran into the 500-iteration
    // limit: the penalty kept the first iterate's multiplier (~30) while the
    // multipliers settled near 1.6, and most steps were cut to 1/32.
    let (spec, mut model, x0) = OrbitTransfer::new(20);
    let mut solver = NativeSqp::<ClarabelQp> {
        settings: SqpOptions {
            max_iterations: 200,
            record_history: true,
            ..SqpOptions::default()
        },
        ..NativeSqp::default()
    };
    let r = solver
        .solve(
            &spec,
            &mut model,
            &x0,
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    let radius = -r.evaluation.as_ref().unwrap().objective;
    assert!((radius - 1.52206).abs() < 5e-5, "r(tf) = {radius}");
    assert!(r.iterations < 150, "{} iterations", r.iterations);
    assert!(
        r.report.rejected_trials <= r.iterations / 4,
        "{:?}",
        r.report
    );
    let full_steps = r
        .report
        .history
        .iter()
        .filter(|h| h.step_length == Some(1.0))
        .count();
    assert!(
        full_steps as u64 * 10 >= r.iterations * 9,
        "{full_steps} full steps"
    );
}
