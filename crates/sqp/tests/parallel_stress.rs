use sqp_craft::{qp::*, *};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_solver::{NativeSqp, SqpOptions, batch::solve_many};
#[path = "../examples/support/propagation.rs"]
mod propagation;

fn specification(n: usize) -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![0.05; n], vec![2.; n]),
        Bounds::free(0),
        CsrPattern::dense(0, n),
    )
    .unwrap()
}
fn checked(spec: &ProblemSpec, r: &SolveResult) {
    assert_eq!(r.status, Status::Converged);
    let k = kkt::nlp_residuals(
        spec,
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
    assert!(
        r.candidate
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .all(|(i, &x)| (x - propagation::control(i)).abs() < 1e-4)
    );
}

#[test]
fn concurrent_native_callers_and_batch_keep_models_isolated() {
    let spec = specification(8);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for width in [1, 1, 2, 4] {
            let spec = &spec;
            handles.push(scope.spawn(move || {
                let mut solver = NativeSqp {
                    qp: ActiveSetQp::default(),
                    settings: SqpOptions::default(),
                };
                for i in 0..6 {
                    let mut model = propagation::Propagation::new(8, 256);
                    let result = solver
                        .solve(
                            spec,
                            &mut model,
                            &[0.1 + i as f64 * 0.02; 8],
                            &Options {
                                threads: width,
                                ..Default::default()
                            },
                            &CancellationToken::default(),
                            None,
                        )
                        .unwrap();
                    checked(spec, &result);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
    let starts = vec![vec![0.2; 8]; 24];
    let results = solve_many::<ActiveSetQp, _, _>(
        &spec,
        &starts,
        &SqpOptions::default(),
        &Options::default(),
        8,
        |_| propagation::Propagation::new(8, 256),
        &CancellationToken::default(),
    )
    .unwrap();
    for result in results {
        checked(&spec, &result.unwrap());
    }
}

fn model(n: usize) -> QpModel {
    QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr: (0..=n).map(|i| i * (i + 1) / 2).collect(),
                indices: (0..n).flat_map(|i| 0..=i).collect(),
            },
            values: (0..n)
                .flat_map(|i| (0..=i).map(move |j| if i == j { 2.0 } else { 0.001 }))
                .collect(),
        },
        gradient: vec![0.; n],
        variables: Bounds::free(n),
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(1, n),
            values: vec![1.; n],
        },
        rows: Bounds::fixed(vec![0.]),
    }
}
#[test]
fn concurrent_large_qps_keep_numeric_caches_private() {
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for width in [1, 1, 2, 4] {
            handles.push(scope.spawn(move || {
                let n = 512;
                let mut model = model(n);
                let mut backend = ActiveSetQp::default();
                backend.set_threads(width).unwrap();
                backend.configure(&QpOptions::default()).unwrap();
                for case in 0..6 {
                    if case % 2 == 0 {
                        model.hessian_lower.values[0] += 0.001;
                    }
                    let p: Vec<_> = (0..n)
                        .map(|i| (i % 7) as f64 * 0.001 + case as f64 * 0.00001)
                        .collect();
                    for (i, a) in model.matrix.values.iter_mut().enumerate() {
                        *a = ((i + case) % 5 + 1) as f64 / n as f64;
                    }
                    let rhs: f64 = model.matrix.values.iter().zip(&p).map(|(a, p)| a * p).sum();
                    model.rows = Bounds::fixed(vec![rhs]);
                    let mut hp = vec![0.; n];
                    model.hessian_product(&p, &mut hp);
                    for ((g, &a), h) in model.gradient.iter_mut().zip(&model.matrix.values).zip(hp)
                    {
                        *g = 0.7 * a - h;
                    }
                    let outcome = backend
                        .solve(&model, None, &CancellationToken::default())
                        .unwrap();
                    let QpOutcome::Optimal(solution) = outcome else {
                        panic!("{outcome:?}")
                    };
                    assert!(model.accepts(&solution, 1e-8).unwrap().accepted);
                    assert!(
                        solution
                            .step
                            .iter()
                            .zip(&p)
                            .all(|(x, y)| (x - y).abs() < 1e-8)
                    );
                    assert!((solution.multipliers.constraints[0] - 0.7).abs() < 1e-8);
                    if case == 2 {
                        let old = model.hessian_lower.values[0];
                        model.hessian_lower.values[0] = f64::NAN;
                        assert!(
                            backend
                                .solve(&model, None, &CancellationToken::default())
                                .is_err()
                        );
                        model.hessian_lower.values[0] = old;
                        let token = CancellationToken::default();
                        token.cancel();
                        assert!(matches!(
                            backend.solve(&model, None, &token).unwrap(),
                            QpOutcome::Cancelled
                        ));
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
