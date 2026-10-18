//! Fixed-work kernels and QPs; production source is included without changing its API.
#[allow(dead_code)]
#[path = "../src/bfgs.rs"]
mod bfgs;
#[allow(dead_code)]
#[path = "../../activeset/src/dense.rs"]
mod dense;
use nalgebra::DMatrix;
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome, QpWarmStart};
use sqp_craft::*;
use sqp_craft_activeset::ActiveSetQp;
use std::{hint::black_box, time::Instant};

fn qp(n: usize, family: &str) -> QpModel {
    let diagonal = family != "qp-dense";
    let mut ptr = vec![0];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..=i {
            if diagonal && j != i {
                continue;
            }
            indices.push(j);
            values.push(if i == j {
                2.0 + (i % 7) as f64
            } else {
                0.01 * (((i + 3 * j) % 5) as f64 - 2.0)
            });
        }
        ptr.push(indices.len());
    }
    let paired = family == "qp-dense" || family == "qp-pairs";
    let m = if paired {
        n / 2
    } else if family == "qp-zero" {
        n
    } else {
        0
    };
    let rowptr = if paired {
        (0..=m).map(|i| 2 * i).collect()
    } else {
        (0..=m).collect()
    };
    let rowidx = if paired {
        (0..2 * m).collect()
    } else {
        (0..m).collect()
    };
    QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr: ptr,
                indices,
            },
            values,
        },
        gradient: if family == "qp-zero" {
            vec![0.0; n]
        } else {
            (0..n).map(|i| -0.1 - (i % 5) as f64).collect()
        },
        variables: if family == "qp-free" {
            Bounds::free(n)
        } else {
            Bounds::new(vec![0.0; n], vec![2.0; n])
        },
        matrix: CsrMatrix {
            pattern: CsrPattern {
                rows: m,
                cols: n,
                indptr: rowptr,
                indices: rowidx,
            },
            values: vec![1.0; if paired { 2 * m } else { m }],
        },
        rows: Bounds::fixed(vec![if family == "qp-zero" { 0.0 } else { 1.0 }; m]),
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let family = args.get(1).map(String::as_str).unwrap_or("hessian");
    let n: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(128);
    let loops: usize = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(1000);
    assert!(n > 0 && loops > 0);
    let mut q = qp(n, "qp-dense");
    if family == "csr-dense" || family == "kkt" {
        q.matrix = CsrMatrix {
            pattern: CsrPattern::dense(n / 2, n),
            values: vec![0.01; n / 2 * n],
        };
    }
    q.validate().unwrap();
    let x: Vec<f64> = (0..n).map(|i| 0.1 + (i % 13) as f64 * 0.01).collect();
    let mut output = vec![0.0; n];
    let mut b = DMatrix::identity(n, n);
    let mut curvature = bfgs::Workspace::new(n, false);
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + (i % 7) as f64))
        .collect();
    let mut solver = ActiveSetQp::default();
    solver.configure(&QpOptions::default()).unwrap();
    if family.starts_with("qp-") {
        q = qp(n, family);
    }
    let mut warm: Option<QpWarmStart> = None;
    let mut check_work = Vec::new();
    let mu = Multipliers {
        constraints: vec![0.0; q.m()],
        variables: vec![0.0; n],
    };
    let zero = vec![0.0; n];
    let row_zero = vec![0.0; q.m()];
    let point = sqp_craft::kkt::KktPoint {
        x: &zero,
        variable_bounds: &q.variables,
        rows: &row_zero,
        row_bounds: &q.rows,
        gradient: &zero,
        jacobian: &q.matrix.pattern,
        jacobian_values: &q.matrix.values,
    };
    let t = Instant::now();
    match family {
        "kkt" => {
            for _ in 0..loops {
                black_box(
                    sqp_craft::kkt::residuals_with_workspace(
                        black_box(&point),
                        &mu,
                        &mut check_work,
                    )
                    .unwrap(),
                );
            }
        }
        "hessian" => {
            for _ in 0..loops {
                q.hessian_product(black_box(&x), black_box(&mut output));
            }
        }
        "validate" => {
            for _ in 0..loops {
                black_box(&q).validate().unwrap();
            }
        }
        "csr" | "csr-dense" => {
            for _ in 0..loops {
                q.matrix.pattern.multiply(
                    &q.matrix.values,
                    black_box(&x),
                    black_box(&mut output[..q.m()]),
                );
            }
        }
        "transpose" => {
            for _ in 0..loops {
                q.matrix.pattern.multiply_transpose(
                    &q.matrix.values,
                    black_box(&x[..q.m()]),
                    black_box(&mut output),
                );
            }
        }
        "bfgs" => {
            for _ in 0..loops {
                assert!(
                    curvature
                        .update(black_box(&mut b), black_box(&x), black_box(&y), None)
                        .is_some()
                );
            }
        }
        "cholesky" => {
            let original: Vec<f64> = (0..n * n)
                .map(|k| if k / n == k % n { n as f64 + 1.0 } else { 0.01 })
                .collect();
            let mut a = original.clone();
            for _ in 0..loops {
                a.copy_from_slice(&original);
                assert!(dense::cholesky(black_box(&mut a), n));
            }
            output[0] = a[0];
        }
        "inverse" => {
            let l: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k / n == k % n {
                        2.0
                    } else if k % n > k / n {
                        0.001
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut j = vec![0.0; n * n];
            for _ in 0..loops {
                dense::inverse_transpose(black_box(&l), black_box(&mut j), n);
            }
            output[0] = j[0];
        }
        value if value.starts_with("qp-") => {
            for _ in 0..loops {
                let QpOutcome::Optimal(s) = solver
                    .solve(black_box(&q), warm.as_ref(), &CancellationToken::default())
                    .unwrap()
                else {
                    panic!("QP did not solve")
                };
                output.copy_from_slice(&s.step);
                warm = Some(QpWarmStart {
                    active_set: s.active_set,
                    ..Default::default()
                });
            }
        }
        _ => panic!("unknown kernel {family}"),
    }
    let seconds = t.elapsed().as_secs_f64();
    if family.starts_with("qp-") {
        let QpOutcome::Optimal(s) = solver
            .solve(&q, warm.as_ref(), &CancellationToken::default())
            .unwrap()
        else {
            panic!("check QP")
        };
        assert!(q.accepts(&s, 1e-7).unwrap().accepted);
    }
    if family == "bfgs" {
        assert!(b.clone().cholesky().is_some());
        output[0] = b[(0, 0)];
    }
    let digest = output.iter().sum::<f64>();
    assert!(digest.is_finite());
    println!(
        "{{\"family\":\"{family}\",\"size\":{n},\"loops\":{loops},\"seconds\":{seconds},\"digest\":{digest}}}"
    );
}
