//! Rows declared linear ([`ProblemSpec::with_linear_rows`]), handled as
//! SNOPT handles linear constraints (Gill, Murray and Saunders 2005, §1.1 and
//! §6.1): the start point is moved onto them by a proximal-point QP, and
//! every later iterate keeps them satisfied, so they need no merit weight.
use super::*;

/// Whether `e` violates a row declared linear by more than `tolerance`.
pub(super) fn violated(spec: &ProblemSpec, e: &Evaluation, tolerance: f64) -> bool {
    spec.linear_rows()
        .iter()
        .zip(&e.constraints)
        .enumerate()
        .any(|(i, (&linear, &v))| linear && row_violation(spec.constraints(), i, v) > tolerance)
}

/// The rows declared linear as a QP constraint block at `x` (values `e`):
/// their Jacobian rows, with the row bounds shifted to step coordinates.
pub(super) fn rows(spec: &ProblemSpec, e: &Evaluation) -> Result<(CsrMatrix, Bounds)> {
    let j = spec.jacobian();
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for (i, &linear) in spec.linear_rows().iter().enumerate() {
        if !linear {
            continue;
        }
        let range = j.indptr[i]..j.indptr[i + 1];
        indices.extend_from_slice(&j.indices[range.clone()]);
        values.extend_from_slice(&e.jacobian_values[range]);
        indptr.push(indices.len());
        let shift = |b: f64| {
            if b.is_finite() {
                finite(b - e.constraints[i], "shifted bound")
            } else {
                Ok(b)
            }
        };
        lower.push(shift(spec.constraints().lower[i])?);
        upper.push(shift(spec.constraints().upper[i])?);
    }
    let matrix = CsrMatrix {
        pattern: CsrPattern {
            rows: lower.len(),
            cols: spec.n(),
            indptr,
            indices,
        },
        values,
    };
    Ok((matrix, Bounds::new(lower, upper)))
}

/// SNOPT's proximal-point problem (PP2) at `x`: the smallest step
/// `min ½‖p‖²` onto the rows declared linear within the variable bounds.
fn proximal(spec: &ProblemSpec, x: &[f64], e: &Evaluation) -> Result<QpModel> {
    let n = spec.n();
    let (matrix, rows) = rows(spec, e)?;
    Ok(QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr: (0..=n).collect(),
                indices: (0..n).collect(),
            },
            values: vec![1.0; n],
        },
        gradient: vec![0.0; n],
        variables: shifted(spec.variables(), x)?,
        matrix,
        rows,
    })
}

/// Moves the start point onto the rows declared linear. `None` means the
/// candidate was replaced by the projected, fully evaluated point; `Some` is
/// a terminal status.
pub(super) fn project<B: QpBackend>(
    backend: &mut B,
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    calls: &mut Calls<'_>,
    result: &mut SolveResult,
    options: &Options,
    acceptance: f64,
) -> Result<Option<Status>> {
    let settings = calls.settings;
    let x = result.candidate.clone().expect("complete point");
    let q = proximal(
        spec,
        &x,
        result.evaluation.as_ref().expect("complete point"),
    )?;
    backend.configure(&QpOptions {
        reuse_workspace: false,
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
        Ok(QpOutcome::Infeasible) => return Ok(Some(Status::LinearInfeasible)),
        Ok(QpOutcome::Cancelled) => return Ok(Some(Status::Cancelled)),
        Ok(QpOutcome::LimitReached(_)) if report.limit_reason.as_deref() == Some("time") => {
            return Ok(Some(Status::TimeLimit));
        }
        Err(Error::Backend(message)) => {
            result.message = format!("the proximal-point QP failed: {message}");
            return Ok(Some(Status::QpFailure));
        }
        Err(err) => return Err(err),
        Ok(other) => {
            result.message = format!("the proximal-point QP returned {other:?}");
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
        result.message = "the proximal-point QP failed independent acceptance".into();
        return Ok(Some(Status::QpFailure));
    }
    let start = trial_point(spec.variables(), &x, &solution.step, 1.0);
    spec.check_point(&start, "projected start point")?;
    if let Some(s) = calls.before_call() {
        return Ok(Some(s));
    }
    let values = calls.functions(spec, model, &start)?;
    if let Some(s) = calls.before_call() {
        return Ok(Some(s));
    }
    let evaluation =
        calls.derivatives(spec, model, &start, values, options.feasibility_tolerance)?;
    result.candidate = Some(start);
    if let Some(old) = result.evaluation.replace(evaluation) {
        calls.recycle(old);
    }
    result.report.linear_projection = true;
    Ok(None)
}

/// Checks that the Jacobian entries of the rows declared linear still equal
/// `reference` (the start point's values) to within rounding.
pub(super) fn check_constant(spec: &ProblemSpec, reference: &[f64], e: &Evaluation) -> Result<()> {
    let j = spec.jacobian();
    for (i, &linear) in spec.linear_rows().iter().enumerate() {
        if !linear {
            continue;
        }
        let range = j.indptr[i]..j.indptr[i + 1];
        for ((&column, &was), &now) in j.indices[range.clone()]
            .iter()
            .zip(&reference[range.clone()])
            .zip(&e.jacobian_values[range])
        {
            if (now - was).abs() > 1e-9 * (1.0 + was.abs()) {
                return Err(Error::InvalidInput(format!(
                    "row {i} is declared linear, but its Jacobian entry for variable \
                     {column} changed from {was} to {now}"
                )));
            }
        }
    }
    Ok(())
}
