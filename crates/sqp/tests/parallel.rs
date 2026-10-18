use sqp_craft::*;
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};
use std::{
    cell::Cell,
    rc::Rc,
    thread::{self, ThreadId},
};

struct LocalModel {
    owner: ThreadId,
    width: Rc<Cell<usize>>,
}
impl NlpProblem for LocalModel {
    fn set_threads(&mut self, threads: usize) -> Result<()> {
        assert_eq!(thread::current().id(), self.owner);
        self.width.set(threads);
        Ok(())
    }
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> Result<f64> {
        assert_eq!(thread::current().id(), self.owner);
        Ok(x.iter().map(|v| v * v).sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> Result<()> {
        assert_eq!(thread::current().id(), self.owner);
        for (v, x) in g.iter_mut().zip(x) {
            *v = 2.0 * x;
        }
        Ok(())
    }
}
fn check<B: qp::QpBackend + Default>() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![-3.; 2], vec![3.; 2]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    let width = Rc::new(Cell::new(0));
    let mut model = LocalModel {
        owner: thread::current().id(),
        width: width.clone(),
    };
    let mut solver = NativeSqp {
        qp: B::default(),
        settings: SqpOptions::default(),
    };
    for threads in [8, 1, 4] {
        let r = solver
            .solve(
                &spec,
                &mut model,
                &[1., -1.],
                &Options {
                    threads,
                    ..Default::default()
                },
                &CancellationToken::default(),
                None,
            )
            .unwrap();
        assert_eq!(width.get(), threads);
        assert_eq!(r.status, Status::Converged);
        assert!(r.candidate.unwrap().iter().all(|v| v.abs() < 1e-7));
    }
    width.set(0);
    assert!(
        solver
            .solve(
                &spec,
                &mut model,
                &[4., 0.],
                &Options {
                    threads: 8,
                    ..Default::default()
                },
                &CancellationToken::default(),
                None
            )
            .is_err()
    );
    assert_eq!(width.get(), 0);
}
#[test]
fn non_send_callbacks_stay_on_the_caller_for_both_backends() {
    check::<ActiveSetQp>();
    check::<ClarabelQp>();
}

#[test]
fn native_batch_keeps_order_errors_and_non_send_model_ownership() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![-3.; 2], vec![3.; 2]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    let mut starts: Vec<_> = (0..17).map(|i| vec![i as f64 * 0.01, -1.]).collect();
    starts[7][0] = 4.;
    let make = |_| LocalModel {
        owner: thread::current().id(),
        width: Rc::new(Cell::new(0)),
    };
    let results = sqp_craft_solver::batch::solve_many::<ActiveSetQp, _, _>(
        &spec,
        &starts,
        &SqpOptions::default(),
        &Options::default(),
        8,
        make,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(results.len(), 17);
    for (i, r) in results.iter().enumerate() {
        if i == 7 {
            assert!(r.is_err());
        } else {
            assert_eq!(r.as_ref().unwrap().status, Status::Converged);
        }
    }
    assert!(
        sqp_craft_solver::batch::solve_many::<ActiveSetQp, _, _>(
            &spec,
            &starts,
            &SqpOptions::default(),
            &Options {
                threads: 2,
                ..Default::default()
            },
            8,
            make,
            &CancellationToken::default()
        )
        .is_err()
    );
}

#[test]
fn batch_cancellation_retains_completed_results_and_skips_factories() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let spec = ProblemSpec::new(
        Bounds::new(vec![-3.; 2], vec![3.; 2]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    let starts = vec![vec![1., -1.]; 16];
    let calls = AtomicUsize::new(0);
    let token = CancellationToken::default();
    let make = |i| {
        calls.fetch_add(1, Ordering::SeqCst);
        if i == 3 {
            token.cancel();
        }
        LocalModel {
            owner: thread::current().id(),
            width: Rc::new(Cell::new(0)),
        }
    };
    let results = sqp_craft_solver::batch::solve_many::<ActiveSetQp, _, _>(
        &spec,
        &starts,
        &SqpOptions::default(),
        &Options::default(),
        1,
        make,
        &token,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    for (i, r) in results.into_iter().enumerate() {
        assert_eq!(
            r.unwrap().status,
            if i < 3 {
                Status::Converged
            } else {
                Status::Cancelled
            }
        );
    }
    calls.store(0, Ordering::SeqCst);
    let mut invalid = starts.clone();
    invalid[7][0] = 4.;
    let results = sqp_craft_solver::batch::solve_many::<ActiveSetQp, _, _>(
        &spec,
        &invalid,
        &SqpOptions::default(),
        &Options::default(),
        8,
        make,
        &token,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for (i, r) in results.into_iter().enumerate() {
        if i == 7 {
            assert!(r.is_err());
        } else {
            let r = r.unwrap();
            assert_eq!(r.status, Status::Cancelled);
            assert!(r.candidate.is_none());
            assert_eq!(r.counts.functions, 0);
        }
    }
}

#[test]
fn pilot_boundary_eight_budget_keeps_scalar_curvature_results() {
    struct Model;
    impl NlpProblem for Model {
        fn functions(&mut self, x: &[f64], _: &mut [f64]) -> Result<f64> {
            Ok(x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    0.5 * (1 + i % 9) as f64 * (v - (1.0 + (i % 3) as f64 * 0.01)).powi(2)
                })
                .sum())
        }
        fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> Result<()> {
            for (i, out) in g.iter_mut().enumerate() {
                *out = (1 + i % 9) as f64 * (x[i] - (1.0 + (i % 3) as f64 * 0.01));
            }
            Ok(())
        }
    }
    let n = 256;
    let spec = ProblemSpec::new(Bounds::free(n), Bounds::free(0), CsrPattern::dense(0, n)).unwrap();
    let run = |width| {
        let mut solver = NativeSqp {
            qp: ActiveSetQp::default(),
            settings: SqpOptions::default(),
        };
        solver
            .solve(
                &spec,
                &mut Model,
                &vec![0.; n],
                &Options {
                    threads: width,
                    ..Default::default()
                },
                &CancellationToken::default(),
                None,
            )
            .unwrap()
    };
    let a = run(1);
    let b = run(8);
    assert_eq!(a.status, Status::Converged);
    assert!(a.evaluation.as_ref().unwrap().objective < 1e-9);
    assert_eq!(a.status, b.status);
    assert_eq!(a.candidate, b.candidate);
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.counts.functions, b.counts.functions);
    assert_eq!(a.counts.derivatives, b.counts.derivatives);
}

#[test]
fn fallible_factories_return_ordered_errors_without_poisoning_workers() {
    let spec = ProblemSpec::new(
        Bounds::new(vec![-3.; 2], vec![3.; 2]),
        Bounds::free(0),
        CsrPattern::dense(0, 2),
    )
    .unwrap();
    let starts = vec![vec![1., -1.]; 24];
    let results = sqp_craft_solver::batch::try_solve_many::<ActiveSetQp, LocalModel, _>(
        &spec,
        &starts,
        &SqpOptions::default(),
        &Options::default(),
        8,
        |i| {
            if i % 5 == 0 {
                Err(Error::Evaluation(format!("factory {i}")))
            } else {
                Ok(LocalModel {
                    owner: thread::current().id(),
                    width: Rc::new(Cell::new(0)),
                })
            }
        },
        &CancellationToken::default(),
    )
    .unwrap();
    for (i, r) in results.into_iter().enumerate() {
        if i % 5 == 0 {
            assert!(matches!(r, Err(Error::Evaluation(_))));
        } else {
            assert_eq!(r.unwrap().status, Status::Converged);
        }
    }
}
