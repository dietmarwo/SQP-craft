//! Local bound-constrained Gauss–Newton feasibility restoration. No slack
//! variables, no NLP multiplier estimates, and no infeasibility certificate.
use super::*;
use sqp_craft::RestorationRecord;

/// A Gauss–Newton step that, without being cut by the trust bound, predicts
/// less than this fraction of ψ as decrease marks a stationary point of ψ.
///
/// Near a feasible point the predicted decrease is of the order of ψ itself
/// (the ratio is about 2 for a consistent linear model). At an infeasible
/// stationary point Gauss–Newton converges only linearly, so an absolute test
/// lets it creep until the step budget is spent and the solve reports
/// `restoration_limit`, as if more budget could help. The value is small
/// because the absolute regularization δ scales the ratio down to about
/// `2σ²/δ` on rows whose Jacobian singular values σ are small.
const STATIONARY_RATIO: f64 = 1e-8;

fn row_residuals(bounds: &Bounds, values: &[f64]) -> Result<Vec<f64>> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            finite(
                v - v.clamp(bounds.lower[i], bounds.upper[i]),
                "restoration row residual",
            )
        })
        .collect()
}
fn merit(bounds: &Bounds, values: &[f64]) -> Result<f64> {
    let mut total = 0.;
    for (i, &v) in values.iter().enumerate() {
        let r = finite(
            v - v.clamp(bounds.lower[i], bounds.upper[i]),
            "restoration row residual",
        )?;
        total = finite(total + finite(r * r, "dot product")?, "dot product")?;
    }
    finite(0.5 * total, "restoration merit")
}

fn model(spec: &ProblemSpec, x: &[f64], e: &Evaluation, settings: &SqpOptions) -> Result<QpModel> {
    let n = spec.n();
    let residuals = row_residuals(spec.constraints(), &e.constraints)?;
    let mut h = DMatrix::identity(n, n) * settings.restoration_regularization;
    let mut gradient = vec![0.0; n];
    let j = spec.jacobian();
    for (i, &residual) in residuals.iter().enumerate() {
        if residual == 0.0 && spec.constraints().lower[i] != spec.constraints().upper[i]
            || spec.linear_rows()[i]
        {
            continue;
        }
        let range = j.indptr[i]..j.indptr[i + 1];
        for a in range.clone() {
            let col = j.indices[a];
            let value = e.jacobian_values[a];
            gradient[col] = finite(
                gradient[col] + finite(value * residual, "restoration gradient")?,
                "restoration gradient",
            )?;
            for b in j.indptr[i]..=a {
                let other = j.indices[b];
                h[(col, other)] = finite(
                    h[(col, other)]
                        + finite(value * e.jacobian_values[b], "restoration curvature")?,
                    "restoration curvature",
                )?;
            }
        }
    }
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let v = h[(i, j)];
            if v != 0. {
                indices.push(j);
                values.push(v);
            }
        }
        indptr.push(indices.len());
    }
    let mut q = QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr,
                indices,
            },
            values,
        },
        gradient,
        variables: shifted(spec.variables(), x)?,
        matrix: CsrMatrix {
            pattern: CsrPattern::dense(0, n),
            values: vec![],
        },
        rows: Bounds::free(0),
    };
    // Rows declared linear are satisfied at x and stay hard.
    if spec.has_linear_rows() {
        (q.matrix, q.rows) = super::linear::rows(spec, e)?;
    }
    for k in 0..n {
        q.variables.lower[k] = q.variables.lower[k].max(-settings.restoration_step_limit);
        q.variables.upper[k] = q.variables.upper[k].min(settings.restoration_step_limit);
    }
    q.validate()?;
    Ok(q)
}

/// `None` means one complete step was accepted; `Some` is a terminal status.
pub(super) fn step<B: QpBackend>(
    backend: &mut B,
    spec: &ProblemSpec,
    problem: &mut dyn NlpProblem,
    calls: &mut Calls<'_>,
    result: &mut SolveResult,
    options: &Options,
) -> Result<Option<Status>> {
    if let Some(s) = calls.before_call() {
        return Ok(Some(s));
    }
    result.report.restoration_attempts += 1;
    let settings = calls.settings;
    let x = result.candidate.as_ref().expect("complete point");
    let e = result.evaluation.as_ref().expect("complete point");
    let before = merit(spec.constraints(), &e.constraints)?;
    let q = model(spec, x, e, settings)?;
    let acceptance = options
        .feasibility_tolerance
        .min(settings.stationarity_tolerance)
        .min(settings.complementarity_tolerance);
    backend.configure(&QpOptions {
        reuse_workspace: settings.qp_reuse_workspace,
        tolerance: settings.qp_tolerance,
        acceptance_tolerance: acceptance,
        max_iterations: settings.qp_max_iterations,
        time_limit_seconds: settings
            .time_limit_seconds
            .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0)),
        elastic_weight: None,
        hard_rows: Vec::new(),
    })?;
    let began = Instant::now();
    result.report.qp_solves += 1;
    let outcome = backend.solve(&q, None, calls.cancel);
    result.report.qp_seconds += began.elapsed().as_secs_f64();
    let report = backend.report();
    record_qp_work(&mut result.report, &report);
    result.report.qp_backend = report.backend;
    result.report.qp_status = report.raw_status;
    result.report.qp_iterations += report.iterations;
    if let Some(s) = calls.stop() {
        return Ok(Some(s));
    }
    let mut solution = match outcome {
        Ok(QpOutcome::Optimal(s)) => s,
        Ok(QpOutcome::Cancelled) => return Ok(Some(Status::Cancelled)),
        Ok(QpOutcome::LimitReached(_)) if report.limit_reason.as_deref() == Some("time") => {
            return Ok(Some(Status::TimeLimit));
        }
        Err(Error::Backend(message)) => {
            result.message = format!("restoration QP failed: {message}");
            return Ok(Some(Status::QpFailure));
        }
        Err(err) => return Err(err),
        Ok(other) => {
            result.message = format!("restoration QP returned {other:?}");
            return Ok(Some(Status::QpFailure));
        }
    };
    snap_step(
        &mut solution.step,
        &q.variables,
        spec.variables(),
        options.feasibility_tolerance,
        &mut result.report,
    );
    if !q.accepts(&solution, acceptance)?.accepted {
        result.message = "restoration QP failed independent acceptance".into();
        return Ok(Some(Status::QpFailure));
    }
    let p = &solution.step;
    let predicted = finite(-dot(&q.gradient, p)?, "restoration predicted decrease")?;
    let trust_limited = norm_inf(p) >= settings.restoration_step_limit * (1.0 - 1e-9);
    if predicted <= 0.0
        || norm_inf(p) <= 1e-14 * (1.0 + norm_inf(x))
        || (!trust_limited && predicted <= STATIONARY_RATIO * before)
    {
        result.message = format!(
            "restoration reached a stationary point of the squared constraint violation \
             (psi = {before:.3e}, predicted decrease {predicted:.3e}); nonlinear infeasibility \
             is not established"
        );
        return Ok(Some(Status::RestorationFailed));
    }
    let mut alpha = box_step_limit(spec.variables(), x, p);
    let mut trials = 0;
    for _ in 0..=settings.max_backtracks {
        if let Some(s) = calls.before_call() {
            return Ok(Some(s));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            break;
        }
        let trial = trial_point(spec.variables(), x, p, alpha);
        if spec.check_point(&trial, "restoration trial").is_err() {
            alpha *= 0.5;
            result.report.rejected_trials += 1;
            continue;
        }
        trials += 1;
        let values = match calls.functions(spec, problem, &trial) {
            Ok(v) => v,
            Err(Error::UndefinedPoint(_)) => {
                result.report.undefined_trials += 1;
                result.report.rejected_trials += 1;
                alpha *= 0.5;
                continue;
            }
            Err(err) => return Err(err),
        };
        let after = merit(spec.constraints(), &values.1)?;
        if after >= before || after > before - ARMIJO * alpha * predicted {
            result.report.rejected_trials += 1;
            alpha *= 0.5;
            continue;
        }
        if let Some(s) = calls.before_call() {
            return Ok(Some(s));
        }
        let evaluation =
            match calls.derivatives(spec, problem, &trial, values, options.feasibility_tolerance) {
                Ok(v) => v,
                Err(Error::UndefinedPoint(_)) => {
                    result.report.undefined_trials += 1;
                    result.report.rejected_trials += 1;
                    alpha *= 0.5;
                    continue;
                }
                Err(err) => return Err(err),
            };
        if let Some(s) = calls.stop() {
            return Ok(Some(s));
        }
        if settings.record_history {
            result.report.restoration_history.push(RestorationRecord {
                iteration: result.iterations,
                objective_before: e.objective,
                objective_after: evaluation.objective,
                primal_before: e.primal_violation,
                primal_after: evaluation.primal_violation,
                merit_before: before,
                merit_after: after,
                step_length: alpha,
                trials,
                qp_iterations: report.iterations,
            });
        }
        result.report.step_norm = Some(norm_inf(
            &trial
                .iter()
                .zip(x)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        ));
        result.candidate = Some(trial);
        if let Some(old) = result.evaluation.replace(evaluation) {
            calls.recycle(old);
        }
        result.multipliers = None;
        result.stationarity = None;
        result.complementarity = None;
        result.dual_sign = None;
        result.iterations += 1;
        result.report.restoration_steps += 1;
        return Ok(None);
    }
    result.message =
        "restoration could not complete a decreasing, defined feasibility trial".into();
    Ok(Some(Status::RestorationFailed))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagonal_restoration_has_only_diagonal_curvature() {
        let n = 8;
        let spec = ProblemSpec::new(
            Bounds::new(vec![0.05; n], vec![2.; n]),
            Bounds::fixed(vec![1.; n]),
            CsrPattern {
                rows: n,
                cols: n,
                indptr: (0..=n).collect(),
                indices: (0..n).collect(),
            },
        )
        .unwrap();
        let e = Evaluation {
            objective: 0.,
            constraints: vec![0.01; n],
            gradient: vec![0.; n],
            jacobian_values: vec![0.2; n],
            primal_violation: 0.99,
            feasible_candidate: false,
        };
        let q = model(&spec, &vec![0.1; n], &e, &SqpOptions::default()).unwrap();
        assert_eq!(q.hessian_lower.values.len(), n);
        assert_eq!(q.m(), 0);
        assert!(
            q.hessian_lower
                .values
                .iter()
                .all(|&v| (v - 0.0401).abs() < 1e-15)
        );
        assert!(q.gradient.iter().all(|&v| (v + 0.198).abs() < 1e-15));
        assert_eq!(q.variables.upper, vec![1.; n]);
    }
    #[test]
    fn overflowing_gauss_newton_curvature_is_an_error() {
        let spec = ProblemSpec::new(
            Bounds::free(1),
            Bounds::fixed(vec![1.0]),
            CsrPattern::dense(1, 1),
        )
        .unwrap();
        let evaluation = Evaluation {
            objective: 0.0,
            constraints: vec![0.0],
            gradient: vec![0.0],
            jacobian_values: vec![1e308],
            primal_violation: 1.0,
            feasible_candidate: false,
        };
        assert!(matches!(
            model(&spec, &[0.0], &evaluation, &SqpOptions::default()),
            Err(Error::Evaluation(_))
        ));
    }

    #[test]
    fn residual_merit_distinguishes_equalities_free_and_one_sided_rows() {
        let b = Bounds::new(
            vec![1.0, f64::NEG_INFINITY, -1.0, f64::NEG_INFINITY],
            vec![1.0, 2.0, f64::INFINITY, f64::INFINITY],
        );
        assert_eq!(
            row_residuals(&b, &[3.0, 4.0, -2.0, 8.0]).unwrap(),
            vec![2.0, 2.0, -1.0, 0.0]
        );
        assert_eq!(merit(&b, &[3.0, 4.0, -2.0, 8.0]).unwrap(), 4.5);
        assert!(merit(&Bounds::fixed(vec![-1e308]), &[1e308]).is_err());
    }
}
