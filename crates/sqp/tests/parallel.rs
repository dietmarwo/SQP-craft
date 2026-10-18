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
