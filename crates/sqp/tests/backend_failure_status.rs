use sqp_craft::{qp::*, *};
use sqp_craft_solver::{NativeSqp, SqpOptions};
struct BadBackend {
    nonfinite: bool,
}
impl QpBackend for BadBackend {
    fn configure(&mut self, _: &QpOptions) -> Result<()> {
        Ok(())
    }
    fn solve(
        &mut self,
        q: &QpModel,
        _: Option<&QpWarmStart>,
        _: &CancellationToken,
    ) -> Result<QpOutcome> {
        Ok(QpOutcome::Optimal(QpSolution {
            step: if self.nonfinite {
                vec![f64::NAN; q.n()]
            } else {
                vec![]
            },
            multipliers: Multipliers {
                constraints: vec![0.0; q.m()],
                variables: vec![0.0; q.n()],
            },
            active_set: None,
            iterations: 1,
        }))
    }
}
struct Model;
impl NlpProblem for Model {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> Result<f64> {
        Ok(x[0] * x[0])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * x[0];
        Ok(())
    }
}
#[test]
fn malformed_backend_outputs_are_qp_failures() {
    let spec = ProblemSpec::new(Bounds::free(1), Bounds::free(0), CsrPattern::dense(0, 1)).unwrap();
    for nonfinite in [true, false] {
        let r = NativeSqp {
            qp: BadBackend { nonfinite },
            settings: SqpOptions::default(),
        }
        .solve(
            &spec,
            &mut Model,
            &[1.0],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
        assert_eq!(r.status, Status::QpFailure);
        assert_eq!(r.candidate, Some(vec![1.0]));
    }
}
