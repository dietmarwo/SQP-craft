//! The positive-definiteness gate of B, added in the review of the
//! performance iterations.
//!
//! Damping keeps the BFGS matrix positive definite in exact arithmetic, but
//! cancellation in the explicit rank-two terms can round it indefinite. The
//! delivered rank-updated Cholesky factor (used above 16 variables) kept
//! reporting a valid factor in that case, so the QP adapter rejected B and the
//! whole solve failed with `Unsupported`. The recorded seed-4 case-86 problem,
//! padded with decoupled variables to n = 22, shows it.

use sqp_craft::kkt::nlp_residuals;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SqpSolver,
    Status,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

const TARGET: [f64; 2] = [1.4087856698936339, 1.438616636289984];
// (a, s, y) of the rows a·x + s‖x − y‖² from docs/performance/regression-seed4-case86.json.
const ROWS: [([f64; 2], f64, [f64; 2]); 2] = [
    (
        [-0.48872841531358846, 0.5272826674624355],
        1.4156986884959413,
        [1.1709513486274008, 1.515517097693404],
    ),
    (
        [-0.7502177809032317, 0.3581332222226916],
        0.6277077422523949,
        [-1.7396471923675372, 0.7003289287653196],
    ),
];
const START: [f64; 2] = [-1.3259573972512375, -1.2370965122378164];

/// The two coupled variables plus `extra` decoupled ones, weighted `weight`.
struct Padded {
    extra: usize,
    weight: f64,
}

impl Padded {
    fn n(&self) -> usize {
        2 + self.extra
    }
    fn spec(&self) -> ProblemSpec {
        let n = self.n();
        ProblemSpec::new(
            Bounds::new(vec![-2.0; n], vec![2.0; n]),
            Bounds::new(
                vec![15.135470320677447, f64::NEG_INFINITY],
                vec![15.135470320677447, 3.3241726345546434],
            ),
            CsrPattern::dense(2, n),
        )
        .unwrap()
    }
    fn start(&self) -> Vec<f64> {
        START
            .iter()
            .copied()
            .chain(std::iter::repeat_n(1.5, self.extra))
            .collect()
    }
    fn weight(&self, k: usize) -> (f64, f64) {
        if k < 2 {
            (1.0, TARGET[k])
        } else {
            (self.weight, 0.3)
        }
    }
}

impl NlpProblem for Padded {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        for (ci, (a, s, y)) in c.iter_mut().zip(ROWS) {
            *ci = (0..2)
                .map(|k| a[k] * x[k] + s * (x[k] - y[k]).powi(2))
                .sum();
        }
        Ok((0..x.len())
            .map(|k| {
                let (w, t) = self.weight(k);
                w * (x[k] - t).powi(2)
            })
            .sum())
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        let n = x.len();
        for (k, gk) in g.iter_mut().enumerate() {
            let (w, t) = self.weight(k);
            *gk = 2.0 * w * (x[k] - t);
        }
        j.fill(0.0);
        for (i, (a, s, y)) in ROWS.iter().enumerate() {
            for k in 0..2 {
                j[i * n + k] = a[k] + 2.0 * s * (x[k] - y[k]);
            }
        }
        Ok(())
    }
}

#[test]
fn an_update_that_rounds_indefinite_resets_b_instead_of_failing_the_solve() {
    let mut resets = 0;
    for weight in [1e-3, 1.0, 1e3] {
        for reuse in [false, true] {
            let mut model = Padded { extra: 20, weight };
            let (spec, x0) = (model.spec(), model.start());
            let r = NativeSqp::<ClarabelQp> {
                settings: SqpOptions {
                    max_iterations: 300,
                    qp_reuse_workspace: reuse,
                    ..SqpOptions::default()
                },
                ..NativeSqp::default()
            }
            .solve(
                &spec,
                &mut model,
                &x0,
                &Options::default(),
                &CancellationToken::default(),
                None,
            )
            .unwrap_or_else(|e| panic!("weight {weight}, reuse {reuse}: {e}"));
            assert_eq!(
                r.status,
                Status::Converged,
                "{weight} {reuse}: {}",
                r.message
            );
            resets += r.report.hessian_resets;
            let k = nlp_residuals(
                &spec,
                r.candidate.as_ref().unwrap(),
                r.evaluation.as_ref().unwrap(),
                r.multipliers.as_ref().unwrap(),
            )
            .unwrap();
            assert!(
                k.primal <= 1e-7
                    && k.stationarity <= 1e-6
                    && k.complementarity <= 1e-6
                    && k.dual_sign <= 1e-8,
                "{k:?}"
            );
        }
    }
    // Five of the six runs reset B at least once after the QP gate fired.
    assert!(resets >= 1, "the gate was not exercised");
}
