use sqp_craft::*;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::*;
struct Problem {
    kind: u8,
    last: Vec<f64>,
}
impl NlpProblem for Problem {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.last = x.to_vec();
        match self.kind {
            0 => Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)),
            1 => {
                c[0] = x[0] + x[1];
                Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
            }
            2 => {
                c[0] = x[0] * x[0] + x[1] * x[1];
                Ok(0.5 * ((x[0] - 2.0).powi(2) + x[1] * x[1]))
            }
            3 => Ok((x[0] + 1.0).powi(2)),
            4 => {
                c[0] = x[0] * x[0];
                Ok((x[0] - 2.0).powi(2))
            }
            _ => {
                if x[0] > 1.5 {
                    Err(Error::UndefinedPoint("domain".into()))
                } else {
                    Ok((x[0] - 1.0).powi(2))
                }
            }
        }
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        assert_eq!(x, self.last);
        match self.kind {
            0 | 1 => {
                g.copy_from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]);
                if self.kind == 1 {
                    j.copy_from_slice(&[1.0, 1.0]);
                }
            }
            2 => {
                g.copy_from_slice(&[x[0] - 2.0, x[1]]);
                j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            }
            3 => g[0] = 2.0 * (x[0] + 1.0),
            4 => {
                g[0] = 2.0 * (x[0] - 2.0);
                j[0] = 2.0 * x[0];
            }
            _ => g[0] = 2.0 * (x[0] - 1.0),
        }
        Ok(())
    }
}
fn spec(kind: u8) -> ProblemSpec {
    let n = if kind < 3 { 2 } else { 1 };
    let rows = match kind {
        1 | 4 => Bounds::fixed(vec![1.0]),
        2 => Bounds::new(vec![f64::NEG_INFINITY], vec![1.0]),
        _ => Bounds::free(0),
    };
    let vars = if kind == 3 {
        Bounds::new(vec![0.0], vec![f64::INFINITY])
    } else {
        Bounds::free(n)
    };
    ProblemSpec::new(vars, rows.clone(), CsrPattern::dense(rows.len(), n)).unwrap()
}
#[test]
fn analytic_suite_and_signed_multipliers() {
    for (kind, start, expected, objective, dual) in [
        (0, vec![0.0, 0.0], vec![1.0, 2.0], 0.0, None),
        (1, vec![0.0, 0.0], vec![0.0, 1.0], 2.0, Some(-2.0)),
        (2, vec![0.4, 0.2], vec![1.0, 0.0], 0.5, Some(-0.5)),
        (3, vec![1.0], vec![0.0], 1.0, None),
    ] {
        let spec = spec(kind);
        let mut model = Problem { kind, last: vec![] };
        let mut solver = NativeSqp::<ClarabelQp>::default();
        solver.settings.record_qp_limit = 2;
        let r = solver
            .solve(
                &spec,
                &mut model,
                &start,
                &Options::default(),
                &CancellationToken::default(),
                None,
            )
            .unwrap();
        assert_eq!(r.status, Status::Converged, "kind={kind}, {r:?}");
        let x = r.candidate.as_ref().unwrap();
        for (&a, &b) in x.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{r:?}");
        }
        assert!((r.evaluation.as_ref().unwrap().objective - objective).abs() < 1e-6);
        let k = kkt::nlp_residuals(
            &spec,
            x,
            r.evaluation.as_ref().unwrap(),
            r.multipliers.as_ref().unwrap(),
        )
        .unwrap();
        assert!(k.max() < 1e-6);
        if let Some(d) = dual {
            assert!((r.multipliers.as_ref().unwrap().constraints[0] - d).abs() < 1e-5);
        }
        assert!(r.report.qp_trace.len() <= 2);
        assert!(r.iterations > 0);
    }
}
#[test]
fn infeasible_linearization_is_not_nonlinear_infeasibility() {
    let mut solver = NativeSqp::<ClarabelQp>::default();
    let r = solver
        .solve(
            &spec(4),
            &mut Problem {
                kind: 4,
                last: vec![],
            },
            &[0.0],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    assert_eq!(r.status, Status::LinearizationInfeasible);
}
#[test]
fn undefined_trial_is_recovered() {
    let mut solver = NativeSqp::<ClarabelQp>::default();
    let r = solver
        .solve(
            &spec(5),
            &mut Problem {
                kind: 5,
                last: vec![],
            },
            &[0.0],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    assert_eq!(r.status, Status::Converged, "{r:?}");
    assert!(r.report.undefined_trials > 0);
}
#[test]
fn budgets_never_return_partial_trial() {
    for budget in [0, 1, 2, 3] {
        let mut solver = NativeSqp::<ClarabelQp>::default();
        solver.settings.max_callback_calls = budget;
        let r = solver
            .solve(
                &spec(0),
                &mut Problem {
                    kind: 0,
                    last: vec![],
                },
                &[0.0, 0.0],
                &Options::default(),
                &CancellationToken::default(),
                None,
            )
            .unwrap();
        assert_eq!(r.status, Status::EvaluationLimit);
        assert!(r.counts.functions + r.counts.derivatives <= budget);
        if budget < 2 {
            assert!(r.candidate.is_none());
        } else {
            assert_eq!(r.candidate, Some(vec![0.0, 0.0]));
            assert_eq!(r.evaluation.unwrap().objective, 5.0);
        }
    }
}
#[test]
fn observers_cancellation_and_limits() {
    let mut solver = NativeSqp::<ClarabelQp>::default();
    let token = CancellationToken::default();
    token.cancel();
    let r = solver
        .solve(
            &spec(0),
            &mut Problem {
                kind: 0,
                last: vec![],
            },
            &[0.0, 0.0],
            &Options::default(),
            &token,
            None,
        )
        .unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert_eq!(r.counts.functions, 0);
    let mut stop = |_: &Progress| Ok(Control::Stop);
    let r = solver
        .solve(
            &spec(0),
            &mut Problem {
                kind: 0,
                last: vec![],
            },
            &[0.0, 0.0],
            &Options::default(),
            &CancellationToken::default(),
            Some(&mut stop),
        )
        .unwrap();
    assert_eq!(r.status, Status::Cancelled);
    assert!(r.candidate.is_some());
    solver.settings.time_limit_seconds = Some(0.0);
    let r = solver
        .solve(
            &spec(0),
            &mut Problem {
                kind: 0,
                last: vec![],
            },
            &[0.0, 0.0],
            &Options::default(),
            &CancellationToken::default(),
            None,
        )
        .unwrap();
    assert_eq!(r.status, Status::TimeLimit);
}
