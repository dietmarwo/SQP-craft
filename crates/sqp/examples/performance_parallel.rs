//! Fixed-callback release benchmark. Input generation is outside solver timing.
//!
//!     performance_parallel FAMILY SIZE MODE BLOCK BACKEND INNER_THREADS WORKERS JOBS
#[path = "support/orbit.rs"]
mod orbit;
use sqp_craft::qp::QpBackend;
use sqp_craft::*;
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};
use std::time::Instant;
struct Separable {
    restoration: bool,
}
impl NlpProblem for Separable {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        if self.restoration {
            for (v, &a) in c.iter_mut().zip(x) {
                *v = a * a;
            }
        } else {
            for (i, v) in c.iter_mut().enumerate() {
                *v = x[2 * i] + x[2 * i + 1];
            }
        }
        Ok(x.iter()
            .enumerate()
            .map(|(i, &a)| {
                let t = if self.restoration {
                    1.
                } else {
                    1. + (i % 3) as f64 * 0.1
                };
                let w = if self.restoration {
                    1.
                } else {
                    1. + (i % 10) as f64
                };
                0.5 * w * (a - t).powi(2)
            })
            .sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        for (i, v) in g.iter_mut().enumerate() {
            let t = if self.restoration {
                1.
            } else {
                1. + (i % 3) as f64 * 0.1
            };
            let w = if self.restoration {
                1.
            } else {
                1. + (i % 10) as f64
            };
            *v = w * (x[i] - t);
        }
        if self.restoration {
            for (v, &a) in j.iter_mut().zip(x) {
                *v = 2. * a;
            }
        } else {
            j.fill(1.);
        }
        Ok(())
    }
}
fn run_case(args: &[String]) -> String {
    let family = args.get(1).map(String::as_str).unwrap_or("orbit");
    let size = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let reuse = args.get(3).is_none_or(|s| s != "cold");
    let block = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    let backend = args.get(5).map(String::as_str).unwrap_or("clarabel");
    let (spec, mut model, x): (ProblemSpec, Box<dyn NlpProblem>, Vec<f64>) = if family == "orbit" {
        let (s, m, x) = orbit::OrbitTransfer::new(size);
        (s, Box::new(m), x)
    } else {
        let restoration = family == "restoration";
        assert!(restoration || family == "separable");
        let n = size;
        let m = if restoration { n } else { n / 2 };
        let (indptr, indices) = if restoration {
            ((0..=n).collect(), (0..n).collect())
        } else {
            ((0..=m).map(|i| 2 * i).collect(), (0..n).collect())
        };
        let rows = if restoration {
            vec![1.; m]
        } else {
            (0..m)
                .map(|i| 2. + ((2 * i) % 3 + (2 * i + 1) % 3) as f64 * 0.1)
                .collect()
        };
        let s = ProblemSpec::new(
            Bounds::new(vec![0.05; n], vec![2.; n]),
            Bounds::fixed(rows),
            CsrPattern {
                rows: m,
                cols: n,
                indptr,
                indices,
            },
        )
        .unwrap();
        (s, Box::new(Separable { restoration }), vec![0.1; n])
    };
    let settings = SqpOptions {
        max_iterations: 350,
        hessian_block_size: block,
        max_callback_calls: 3000,
        qp_reuse_workspace: reuse,
        ..Default::default()
    };
    let threads: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1);
    let (r, seconds) = match backend {
        "clarabel" => run::<ClarabelQp>(settings, &spec, model.as_mut(), &x, threads),
        "active-set" => run::<ActiveSetQp>(settings, &spec, model.as_mut(), &x, threads),
        other => panic!("unknown backend {other}"),
    };
    assert_eq!(r.status, Status::Converged, "{family} {size}: {r:?}");
    let e = r.evaluation.as_ref().unwrap();
    let k = sqp_craft::kkt::nlp_residuals(
        &spec,
        r.candidate.as_ref().unwrap(),
        e,
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
    if family == "orbit" && (size == 20 || size == 40) {
        let reference = if size == 20 { 1.52206 } else { 1.52449 };
        assert!((e.objective + reference).abs() < 5e-5);
    }
    if family != "orbit" {
        assert!(e.objective < 1e-9);
    }
    format!(
        "{{\"family\":\"{family}\",\"backend\":\"{backend}\",\"size\":{size},\"reuse\":{reuse},\"hessian_block_size\":{block},\"seconds\":{seconds},\"objective\":{},\"iterations\":{},\"qp_calls\":{},\"qp_iterations\":{},\"bfgs_seconds\":{},\"conversion_seconds\":{},\"engine_seconds\":{},\"setup_seconds\":{},\"check_seconds\":{},\"qp_seconds\":{},\"block_hessian_fallbacks\":{},\"setups\":{},\"updates\":{},\"primal\":{},\"stationarity\":{}}}",
        e.objective,
        r.iterations,
        r.report.qp_solves,
        r.report.qp_iterations,
        r.report.bfgs_seconds,
        r.report.qp_conversion_seconds,
        r.report.qp_solve_seconds,
        r.report.qp_setup_seconds,
        r.report.qp_check_seconds,
        r.report.qp_seconds,
        r.report.block_hessian_fallbacks,
        r.report.qp_setups,
        r.report.qp_updates,
        k.primal,
        k.stationarity
    )
}

fn run<B: QpBackend + Default>(
    settings: SqpOptions,
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    threads: usize,
) -> (SolveResult, f64) {
    let mut solver = NativeSqp::<B> {
        settings,
        qp: B::default(),
    };
    let t = Instant::now();
    let r = solver
        .solve(
            spec,
            model,
            x,
            &Options {
                threads,
                ..Options::default()
            },
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    (r, t.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let workers: usize = args.get(7).map(|v| v.parse().unwrap()).unwrap_or(1);
    let jobs: usize = args.get(8).map(|v| v.parse().unwrap()).unwrap_or(1);
    assert!(workers > 0 && workers <= 8 && jobs > 0);
    let began = Instant::now();
    let mut results = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let args = &args;
            handles.push(scope.spawn(move || {
                (worker..jobs)
                    .step_by(workers)
                    .map(|i| (i, run_case(args)))
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let wall = began.elapsed().as_secs_f64();
    results.sort_by_key(|(i, _)| *i);
    assert_eq!(results.len(), jobs);
    for (_, line) in results {
        println!("{line}");
    }
    println!(
        "{{\"kind\":\"batch\",\"workers\":{workers},\"jobs\":{jobs},\"seconds\":{wall},\"throughput\":{}}}",
        jobs as f64 / wall
    );
}
