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
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome, QpReport, QpWarmStart};
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SolveResult,
    SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
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
    // Both backends gate B exactly: Clarabel's adapter by a separate dense
    // Cholesky, the active-set backend by the factorization it solves with.
    padded_runs::<ClarabelQp>();
    padded_runs::<ActiveSetQp>();
}

fn padded_runs<B: QpBackend + Default>() {
    let (mut resets, mut rejections) = (0, 0);
    for weight in [1e-3, 1.0, 1e3] {
        for reuse in [false, true] {
            let mut model = Padded { extra: 20, weight };
            let (spec, x0) = (model.spec(), model.start());
            let r = NativeSqp::<B> {
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
            rejections += r.report.qp_not_positive_definite;
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
    // Most of the six runs reset B at least once after the QP gate fired
    // (five with Clarabel, four with the active-set backend).
    assert!(
        rejections >= 1 && resets >= rejections,
        "the gate was not exercised"
    );
}

/// Delegates to Clarabel but reports `NotPositiveDefinite` for the listed
/// (zero-based) calls, without touching the inner backend.
struct ScriptedGate {
    inner: ClarabelQp,
    reject: Vec<usize>,
    calls: usize,
    hessians: Vec<Vec<f64>>,
}

impl ScriptedGate {
    fn rejecting(reject: Vec<usize>) -> Self {
        Self {
            inner: ClarabelQp::default(),
            reject,
            calls: 0,
            hessians: vec![],
        }
    }
}

impl QpBackend for ScriptedGate {
    fn configure(&mut self, options: &QpOptions) -> Result<()> {
        self.inner.configure(options)
    }
    fn report(&self) -> QpReport {
        self.inner.report()
    }
    fn solve(
        &mut self,
        model: &QpModel,
        warm: Option<&QpWarmStart>,
        cancel: &CancellationToken,
    ) -> Result<QpOutcome> {
        let call = self.calls;
        self.calls += 1;
        self.hessians.push(model.hessian_lower.values.clone());
        if self.reject.contains(&call) {
            return Ok(QpOutcome::NotPositiveDefinite);
        }
        self.inner.solve(model, warm, cancel)
    }
}

/// Σ (i + 1)(x_i − i/2)² subject to Σ x_i = 2, x ∈ [−2, 2]⁶: convex and
/// benign, so a plain run sees no rejection; BFGS needs several steps.
struct Weighted;

impl NlpProblem for Weighted {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x.iter().sum();
        Ok(x.iter()
            .enumerate()
            .map(|(i, v)| (i + 1) as f64 * (v - i as f64 / 2.0).powi(2))
            .sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = 2.0 * (i + 1) as f64 * (x[i] - i as f64 / 2.0);
        }
        j.fill(1.0);
        Ok(())
    }
}

fn weighted_spec() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![-2.0; 6], vec![2.0; 6]),
        Bounds::fixed(vec![2.0]),
        CsrPattern::dense(1, 6),
    )
    .unwrap()
}

fn solve_with(gate: ScriptedGate, max_hessian_resets: u32) -> (SolveResult, ScriptedGate) {
    let spec = weighted_spec();
    let mut solver = NativeSqp {
        qp: gate,
        settings: SqpOptions {
            max_iterations: 300,
            max_hessian_resets,
            ..SqpOptions::default()
        },
    };
    let r = solver
        .solve(
            &spec,
            &mut Weighted,
            &[0.0; 6],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    (r, solver.qp)
}

#[test]
fn a_rejected_hessian_is_reset_and_the_same_qp_is_solved_again() {
    let (plain, _) = solve_with(ScriptedGate::rejecting(vec![]), 3);
    assert_eq!(plain.status, Status::Converged, "{}", plain.message);
    assert_eq!(
        (
            plain.report.qp_not_positive_definite,
            plain.report.hessian_resets
        ),
        (0, 0)
    );
    assert!(plain.iterations >= 3, "{} iterations", plain.iterations);
    assert_eq!(plain.report.qp_solves, plain.iterations + 1);
    // Reject the third QP: B has had two BFGS updates by then.
    let (r, gate) = solve_with(ScriptedGate::rejecting(vec![2]), 3);
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    assert_eq!(
        (r.report.qp_not_positive_definite, r.report.hessian_resets),
        (1, 1)
    );
    assert_eq!(r.report.qp_solves, gate.calls as u64);
    // The re-solve is the same QP with B = scale·I, and the rejected call
    // is not an iteration: the solve takes one call more than iterations
    // need (the plain run makes iterations + 1 calls, the last one at the
    // converged point).
    let identity: Vec<f64> = (0..6)
        .flat_map(|i| (0..=i).map(move |j| if i == j { 1.0 } else { 0.0 }))
        .collect();
    assert_ne!(gate.hessians[2], identity);
    assert_eq!(gate.hessians[3], identity);
    assert_eq!(gate.calls as u64, r.iterations + 2);
    let k = nlp_residuals(
        &weighted_spec(),
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        r.multipliers.as_ref().unwrap(),
    )
    .unwrap();
    assert!(k.primal <= 1e-7 && k.stationarity <= 1e-6, "{k:?}");
    let f = r.evaluation.as_ref().unwrap().objective;
    assert!((f - plain.evaluation.as_ref().unwrap().objective).abs() < 1e-9);
}

#[test]
fn repeated_rejections_stop_when_the_reset_budget_is_spent() {
    // Every QP after the first is rejected; two resets are allowed.
    let (r, gate) = solve_with(ScriptedGate::rejecting((1..100).collect()), 2);
    assert_eq!(r.status, Status::NumericalFailure, "{}", r.message);
    assert!(r.message.contains("not positive definite"), "{}", r.message);
    assert_eq!(r.report.hessian_resets, 2);
    assert_eq!(r.report.qp_not_positive_definite, 3);
    assert_eq!(gate.calls, 4);
    assert_eq!(r.iterations, 1);
}
