//! Random convex QPs solved by both backends: the same optimal point, the
//! same infeasibility findings, and warm starts that change nothing but work.
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome, QpSolution, QpWarmStart};
use sqp_craft::{Bounds, CancellationToken, CsrMatrix, CsrPattern};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        // Irwin–Hall approximation; the tails do not matter here.
        (0..12).map(|_| self.next()).sum::<f64>() - 6.0
    }
    fn below(&mut self, k: usize) -> usize {
        (self.next() * k as f64) as usize % k.max(1)
    }
}

fn csr(rows: usize, cols: usize, dense: &[Vec<f64>], lower: bool) -> CsrMatrix {
    let (mut indptr, mut indices, mut values) = (vec![0], vec![], vec![]);
    for (i, row) in dense.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 && (!lower || j <= i) {
                indices.push(j);
                values.push(v);
            }
        }
        indptr.push(indices.len());
    }
    CsrMatrix {
        pattern: CsrPattern {
            rows,
            cols,
            indptr,
            indices,
        },
        values,
    }
}

/// Bounds around `v` of a random kind: free, one-sided, two-sided or fixed,
/// often active at `v` (degenerate when several are).
fn bounds(rng: &mut Rng, v: f64) -> (f64, f64) {
    let gap = |rng: &mut Rng| if rng.next() < 0.4 { 0.0 } else { rng.next() };
    match rng.below(6) {
        0 => (f64::NEG_INFINITY, f64::INFINITY),
        1 => (v - gap(rng), f64::INFINITY),
        2 => (f64::NEG_INFINITY, v + gap(rng)),
        3 => (v, v),
        _ => (v - gap(rng), v + gap(rng)),
    }
}

/// A feasible random QP; with `infeasible`, one row is made to contradict
/// another clearly.
fn random_qp(rng: &mut Rng, infeasible: bool) -> QpModel {
    generate(rng, infeasible, 40, &[1e-6, 1e-4, 1e-2, 1.0])
}

fn generate(rng: &mut Rng, infeasible: bool, max_n: usize, conditioning: &[f64]) -> QpModel {
    let n = 1 + rng.below(max_n);
    let m = rng.below(n + 6);
    let rank = 1 + rng.below(n);
    let conditioning = conditioning[rng.below(conditioning.len())];
    let factor: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..rank).map(|_| rng.normal()).collect())
        .collect();
    let h: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let v: f64 = (0..rank).map(|k| factor[i][k] * factor[j][k]).sum();
                    v + if i == j { conditioning } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let density = 0.2 + 0.8 * rng.next();
    let mut a: Vec<Vec<f64>> = (0..m)
        .map(|_| {
            (0..n)
                .map(|_| {
                    if rng.next() < density {
                        rng.normal()
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    // Duplicate and scaled rows make the constraints dependent.
    if m >= 2 && rng.next() < 0.3 {
        let scale = [1.0, -2.0, 0.5][rng.below(3)];
        a[m - 1] = a[0].iter().map(|v| v * scale).collect();
    }
    let point: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let values: Vec<f64> = a
        .iter()
        .map(|row| row.iter().zip(&point).map(|(x, y)| x * y).sum())
        .collect();
    let (mut row_lower, mut row_upper): (Vec<f64>, Vec<f64>) =
        values.iter().map(|&v| bounds(rng, v)).unzip();
    let (lower, upper): (Vec<f64>, Vec<f64>) = point.iter().map(|&v| bounds(rng, v)).unzip();
    if infeasible && m >= 1 {
        a.push(a[0].clone());
        let v = values[0];
        row_lower[0] = v;
        row_upper[0] = f64::INFINITY;
        row_lower.push(f64::NEG_INFINITY);
        row_upper.push(v - 0.5 - rng.next());
        // Only meaningful when the row is not identically zero.
        if a[0].iter().all(|&x| x == 0.0) {
            a[0][0] = 1.0;
            a.last_mut().unwrap()[0] = 1.0;
            row_lower[0] = point[0];
            row_upper[m] = point[0] - 1.0;
        }
    }
    let rows = a.len();
    let model = QpModel {
        hessian_lower: csr(n, n, &h, true),
        gradient: (0..n).map(|_| 3.0 * rng.normal()).collect(),
        variables: Bounds::new(lower, upper),
        matrix: csr(rows, n, &a, false),
        rows: Bounds::new(row_lower, row_upper),
    };
    model.validate().expect("valid random QP");
    model
}

fn attempt(
    backend: &mut impl QpBackend,
    model: &QpModel,
    warm: Option<&QpWarmStart>,
) -> sqp_craft::Result<QpOutcome> {
    backend
        .configure(&QpOptions {
            acceptance_tolerance: 1e-7,
            ..Default::default()
        })
        .unwrap();
    backend.solve(model, warm, &CancellationToken::default())
}

fn solve(backend: &mut impl QpBackend, model: &QpModel, warm: Option<&QpWarmStart>) -> QpOutcome {
    attempt(backend, model, warm).unwrap_or_else(|e| panic!("{e}"))
}

fn optimal(outcome: QpOutcome) -> QpSolution {
    match outcome {
        QpOutcome::Optimal(s) => s,
        other => panic!("expected optimal, got {other:?}"),
    }
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |d, (x, y)| d.max((x - y).abs()))
}

#[test]
fn random_feasible_qps_match_clarabel() {
    let mut rng = Rng(2026);
    let (mut active, mut clarabel) = (ActiveSetQp::default(), ClarabelQp::default());
    let mut compared = 0;
    for case in 0..1000 {
        let model = random_qp(&mut rng, false);
        let ours = optimal(solve(&mut active, &model, None));
        let reference = match attempt(&mut clarabel, &model, None) {
            Ok(QpOutcome::Optimal(s)) => s,
            // The interior-point reference may fail on degenerate cases;
            // the active-set result was still checked independently.
            _ => continue,
        };
        let (f, f_ref) = (
            model.objective(&ours.step),
            model.objective(&reference.step),
        );
        assert!(
            f <= f_ref + 1e-7 * (1.0 + f_ref.abs()),
            "case {case}: objective {f} > {f_ref}"
        );
        assert!(
            (f - f_ref).abs() <= 1e-6 * (1.0 + f_ref.abs()),
            "case {case}: objective {f} vs {f_ref}"
        );
        let scale = 1.0 + reference.step.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            distance(&ours.step, &reference.step) <= 1e-4 * scale,
            "case {case}: steps differ by {}",
            distance(&ours.step, &reference.step)
        );
        compared += 1;
    }
    assert!(compared >= 950, "only {compared} cases compared");
}

#[test]
fn random_infeasible_qps_are_infeasible_for_both() {
    let mut rng = Rng(7);
    let (mut active, mut clarabel) = (ActiveSetQp::default(), ClarabelQp::default());
    let mut cases = 0;
    while cases < 150 {
        let model = random_qp(&mut rng, true);
        if model.m() == 0 {
            continue;
        }
        let outcome = solve(&mut active, &model, None);
        if outcome != QpOutcome::Infeasible {
            eprintln!(
                "case {cases}: n {} m {} {:?}",
                model.n(),
                model.m(),
                active.report()
            );
            eprintln!("{model:?}");
        }
        assert_eq!(outcome, QpOutcome::Infeasible);
        // The interior-point reference sometimes stops at AlmostSolved here.
        if let Ok(reference) = attempt(&mut clarabel, &model, None) {
            assert_eq!(reference, QpOutcome::Infeasible);
        }
        cases += 1;
    }
}

#[test]
fn warm_starts_reproduce_the_cold_solution_and_save_changes() {
    let mut rng = Rng(99);
    let mut backend = ActiveSetQp::default();
    let (mut cold_changes, mut warm_changes) = (0, 0);
    for case in 0..600 {
        let mut model = random_qp(&mut rng, false);
        let first = optimal(solve(&mut backend, &model, None));
        // The same QP from its own working set needs no change.
        let warm = QpWarmStart {
            active_set: first.active_set.clone(),
            ..Default::default()
        };
        let again = optimal(solve(&mut backend, &model, Some(&warm)));
        assert_eq!(again.iterations, 0, "case {case}");
        let scale = 1.0 + first.step.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            distance(&again.step, &first.step) <= 1e-9 * scale,
            "case {case}: {} at scale {scale}",
            distance(&again.step, &first.step)
        );
        // A nearby QP, as in consecutive SQP iterations.
        for g in &mut model.gradient {
            *g += 0.05 * rng.normal();
        }
        let cold = optimal(solve(&mut backend, &model, None));
        let warmed = optimal(solve(&mut backend, &model, Some(&warm)));
        assert_eq!(backend.report().cold_retries, 0, "case {case}");
        let scale = 1.0 + cold.step.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            distance(&warmed.step, &cold.step) <= 1e-7 * scale,
            "case {case}: {}",
            distance(&warmed.step, &cold.step)
        );
        cold_changes += cold.iterations;
        warm_changes += warmed.iterations;
    }
    assert!(
        2 * warm_changes < cold_changes,
        "warm {warm_changes} vs cold {cold_changes}"
    );
}

/// Outcome census on harder QPs (n up to 100, `H` regularized down to 1e-8):
///
///     SEED=1 COUNT=2000 cargo test --release -p sqp-craft-activeset \
///         --test against_clarabel -- --ignored --nocapture
///
/// Every fourth QP is made infeasible. Prints how often each pair of
/// outcomes occurs; `check` is a solution that failed the independent check.
#[test]
#[ignore = "census for the documentation; run explicitly"]
fn census() {
    let read = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let mut rng = Rng(read("SEED", 1));
    let (mut active, mut clarabel) = (ActiveSetQp::default(), ClarabelQp::default());
    let mut tally = std::collections::BTreeMap::<String, usize>::new();
    let label = |o: &sqp_craft::Result<QpOutcome>| match o {
        Ok(QpOutcome::Optimal(_)) => "optimal".to_string(),
        Ok(QpOutcome::Infeasible) => "infeasible".into(),
        Ok(QpOutcome::LimitReached(_)) => "limit".into(),
        Ok(other) => format!("{other:?}"),
        Err(e) if e.to_string().contains("independent check") => "check".into(),
        Err(_) => "error".into(),
    };
    let mut most_changes = 0;
    for case in 0..read("COUNT", 2000) {
        let infeasible = case % 4 == 3;
        let model = generate(&mut rng, infeasible, 100, &[1e-8, 1e-6, 1e-4, 1e-2, 1.0]);
        let ours = attempt(&mut active, &model, None);
        most_changes = most_changes.max(active.report().iterations);
        let reference = attempt(&mut clarabel, &model, None);
        let kind = if infeasible && model.m() > 0 {
            "infeasible QP"
        } else {
            "feasible QP"
        };
        let key = format!(
            "{kind}: active-set {}, clarabel {}",
            label(&ours),
            label(&reference)
        );
        *tally.entry(key).or_default() += 1;
    }
    for (key, count) in &tally {
        println!("{count:6} {key}");
    }
    println!("most working-set changes in one QP: {most_changes}");
}
