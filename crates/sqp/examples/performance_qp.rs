//! Coupled planted dense QPs, fresh Hessians, retained worker resources.
use sqp_craft::{qp::*, *};
use sqp_craft_activeset::ActiveSetQp;
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(256);
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(8);
    let p: Vec<f64> = (0..n).map(|i| (i % 11) as f64 * 0.001).collect();
    let mut model = QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr: (0..=n).map(|i| i * (i + 1) / 2).collect(),
                indices: (0..n).flat_map(|i| 0..=i).collect(),
            },
            values: (0..n)
                .flat_map(|i| {
                    (0..=i).map(move |j| {
                        let u = (i % 7 + 1) as f64 * 0.1 / (n as f64).sqrt();
                        let v = (j % 7 + 1) as f64 * 0.1 / (n as f64).sqrt();
                        u * v + if i == j { 2.0 } else { 0.0 }
                    })
                })
                .collect(),
        },
        gradient: vec![0.0; n],
        variables: Bounds::free(n),
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(1, n),
            values: vec![1.0; n],
        },
        rows: Bounds::fixed(vec![p.iter().sum()]),
    };
    let mut backend = ActiveSetQp::default();
    backend
        .configure(&QpOptions {
            acceptance_tolerance: 1e-8,
            ..Default::default()
        })
        .unwrap();
    let setup = Instant::now();
    let _resources = sqp_craft_parallel::Execution::new(threads).unwrap();
    backend.set_threads(threads).unwrap();
    let resource_setup = setup.elapsed().as_secs_f64();
    let mut seconds = 0.0;
    let mut iterations = 0;
    let mut objective = 0.0;
    for _ in 0..3 {
        model.hessian_lower.values[0] += 0.001;
        let mut hp = vec![0.0; n];
        model.hessian_product(&p, &mut hp);
        for (g, h) in model.gradient.iter_mut().zip(hp) {
            *g = 0.7 - h;
        }
        let began = Instant::now();
        let result = backend
            .solve(&model, None, &CancellationToken::default())
            .unwrap();
        seconds += began.elapsed().as_secs_f64();
        let QpOutcome::Optimal(solution) = result else {
            panic!("{result:?}")
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
        iterations += solution.iterations;
        objective = model.objective(&solution.step);
    }
    println!(
        "{{\"family\":\"qp\",\"size\":{n},\"threads\":{threads},\"seconds\":{seconds},\"resource_setup_seconds\":{resource_setup},\"objective\":{objective},\"iterations\":{iterations},\"fresh_hessian_calls\":3}}"
    );
    println!("{{\"kind\":\"batch\",\"jobs\":1,\"seconds\":{seconds}}}");
}
