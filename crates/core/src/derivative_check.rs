//! Finite-difference checks of model derivatives along coordinate directions.
//!
//! The checker uses the grouped callback protocol exactly like a solver:
//! `derivatives` is called once, immediately after `functions` at the base
//! point; perturbed points only call `functions`. Every point lies within the
//! variable bounds. Every objective and constraint output is compared, so a
//! derivative missing from the declared Jacobian pattern is detected.

use crate::error::{Error, Result, finite};
use crate::{EvaluationCounts, NlpProblem, ProblemSpec};

/// Settings for [`check_derivatives`]; independent of any solver settings.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeOptions {
    /// Perturbation sizes relative to `max(1, |x_j|)`, tried in order; the
    /// estimate closest to the supplied derivative is kept. 1 to 16 entries.
    pub steps: Vec<f64>,
    /// Absolute comparison tolerance.
    pub absolute_tolerance: f64,
    /// Relative comparison tolerance, applied to `max(|analytic|, |estimate|)`.
    pub relative_tolerance: f64,
    /// Maximum number of coordinates to check.
    pub max_directions: usize,
}

impl Default for DerivativeOptions {
    fn default() -> Self {
        Self {
            steps: vec![1e-4, 1e-5, 1e-6],
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            max_directions: 256,
        }
    }
}

impl DerivativeOptions {
    /// Rejects empty or non-positive steps and invalid tolerances.
    pub fn validate(&self) -> Result<()> {
        let ok = !self.steps.is_empty()
            && self.steps.len() <= 16
            && self.steps.iter().all(|s| s.is_finite() && *s > 0.0)
            && self.absolute_tolerance.is_finite()
            && self.absolute_tolerance > 0.0
            && self.relative_tolerance.is_finite()
            && self.relative_tolerance >= 0.0
            && self.max_directions > 0;
        if ok {
            Ok(())
        } else {
            Err(Error::InvalidInput(
                "derivative check: need 1 to 16 positive finite steps, a positive absolute \
                 tolerance, a nonnegative relative tolerance and max_directions >= 1"
                    .into(),
            ))
        }
    }
}

/// One derivative entry that disagrees with every usable finite difference.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeIssue {
    /// Perturbed variable `j`.
    pub variable: usize,
    /// `0` for the objective, `i + 1` for constraint row `i`.
    pub output: usize,
    /// Supplied derivative (zero for an entry outside the Jacobian pattern).
    pub analytic: f64,
    /// Finite-difference estimate closest to `analytic`.
    pub estimated: f64,
    /// `|analytic - estimated| / (absolute + relative * max(|analytic|, |estimated|))`.
    pub scaled_error: f64,
}

/// Outcome of [`check_derivatives`]. Nothing unchecked is counted as passed.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeReport {
    /// At least one coordinate checked, no issues, and `skipped` empty.
    pub passed: bool,
    /// Some non-fixed coordinate could not be checked, or none was.
    pub inconclusive: bool,
    /// Number of coordinates checked.
    pub checked: usize,
    /// Fixed variables (equal bounds). They cannot be perturbed, and since the
    /// solver never moves them their derivatives do not affect its steps.
    pub fixed: Vec<usize>,
    /// Non-fixed coordinates left unchecked: beyond `max_directions`, without
    /// representable room inside the bounds, or undefined at every perturbation.
    pub skipped: Vec<usize>,
    /// Model calls made by the check.
    pub counts: EvaluationCounts,
    /// Disagreeing entries.
    pub issues: Vec<DerivativeIssue>,
}

/// `[f, c_0, ..., c_{m-1}]` at `x`, or `None` if the model is undefined there.
fn outputs(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    counts: &mut EvaluationCounts,
) -> Result<Option<Vec<f64>>> {
    spec.check_point(x, "derivative check point")?;
    let mut c = vec![f64::NAN; spec.m()];
    counts.functions += 1;
    let f = match model.functions(x, &mut c) {
        Ok(f) => f,
        Err(Error::UndefinedPoint(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut out = Vec::with_capacity(spec.m() + 1);
    out.push(finite(f, "objective")?);
    for (i, v) in c.into_iter().enumerate() {
        out.push(finite(v, &format!("constraint value {i}"))?);
    }
    Ok(Some(out))
}

/// Interval `[left, right]` around `x` for step `h` within `[lo, up]`:
/// central where both sides have room, otherwise one-sided towards the
/// roomier side, so a point just inside a bound does not force a tiny step.
fn interval(x: f64, lo: f64, up: f64, h: f64) -> Option<(f64, f64)> {
    let (down, upward) = (x - lo, up - x);
    let (left, right) = if down >= h && upward >= h {
        (x - h, x + h)
    } else if upward >= down {
        (x, x + h.min(upward))
    } else {
        (x - h.min(down), x)
    };
    (left.is_finite() && right.is_finite() && left < right && lo <= left && right <= up)
        .then_some((left, right))
}

/// Compares the supplied gradient and Jacobian with finite differences of
/// `functions` along each coordinate direction at `x`.
///
/// `x` must be finite and within the variable bounds. A perturbed point where
/// the model reports [`Error::UndefinedPoint`] is not used; a coordinate left
/// without any usable perturbation is reported in `skipped`. Any other model
/// error aborts the check, as an undefined base point does.
pub fn check_derivatives(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    options: &DerivativeOptions,
) -> Result<DerivativeReport> {
    options.validate()?;
    spec.check_point(x, "derivative check point")?;
    let (n, m) = (spec.n(), spec.m());
    let mut report = DerivativeReport {
        passed: false,
        inconclusive: false,
        checked: 0,
        fixed: vec![],
        skipped: vec![],
        counts: EvaluationCounts::default(),
        issues: vec![],
    };

    // Base point: functions, then derivatives at the same point.
    report.counts.functions += 1;
    let mut c = vec![f64::NAN; m];
    let f = model.functions(x, &mut c)?;
    let mut base = vec![finite(f, "objective")?];
    for (i, v) in c.into_iter().enumerate() {
        base.push(finite(v, &format!("constraint value {i}"))?);
    }
    let mut gradient = vec![f64::NAN; n];
    let mut jacobian = vec![f64::NAN; spec.nnz()];
    report.counts.derivatives += 1;
    model.derivatives(x, &mut gradient, &mut jacobian)?;
    for (k, &v) in gradient.iter().chain(&jacobian).enumerate() {
        finite(v, &format!("derivative output {k}"))?;
    }

    // Column view of the combined objective gradient and Jacobian.
    let mut columns: Vec<Vec<(usize, f64)>> = gradient.into_iter().map(|g| vec![(0, g)]).collect();
    let pattern = spec.jacobian();
    for row in 0..m {
        for k in pattern.indptr[row]..pattern.indptr[row + 1] {
            columns[pattern.indices[k]].push((row + 1, jacobian[k]));
        }
    }

    compare_columns(
        spec,
        x,
        options,
        &base,
        &columns,
        report,
        |point, counts| outputs(spec, model, point, counts),
    )
}

/// Checks an exact Lagrangian Hessian against differences of the analytic
/// Lagrangian gradient, including entries outside the declared sparse pattern.
/// The report's output indices are Lagrangian gradient coordinates. Fixed
/// variables can be output rows even though their columns cannot be perturbed.
pub fn check_lagrangian_hessian(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    lambda: &[f64],
    options: &DerivativeOptions,
) -> Result<DerivativeReport> {
    options.validate()?;
    spec.check_point(x, "Hessian check point")?;
    if lambda.len() != spec.m() || lambda.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "Hessian check requires {} finite constraint multipliers",
            spec.m()
        )));
    }
    let pattern = spec.lagrangian_hessian_pattern().ok_or_else(|| {
        Error::InvalidInput("Hessian check requires a declared Lagrangian Hessian pattern".into())
    })?;
    let mut report = DerivativeReport {
        passed: false,
        inconclusive: false,
        checked: 0,
        fixed: vec![],
        skipped: vec![],
        counts: EvaluationCounts::default(),
        issues: vec![],
    };
    let gradient_at = |model: &mut dyn NlpProblem, point: &[f64], counts: &mut EvaluationCounts| {
        if outputs(spec, model, point, counts)?.is_none() {
            return Ok(None);
        }
        let mut gradient = vec![f64::NAN; spec.n()];
        let mut jacobian = vec![f64::NAN; spec.nnz()];
        counts.derivatives += 1;
        match model.derivatives(point, &mut gradient, &mut jacobian) {
            Ok(()) => {}
            Err(Error::UndefinedPoint(_)) => return Ok(None),
            Err(e) => return Err(e),
        }
        for &v in &jacobian {
            finite(v, "Hessian check Jacobian")?;
        }
        let mut jt = vec![0.0; spec.n()];
        spec.jacobian()
            .multiply_transpose(&jacobian, lambda, &mut jt);
        for (g, v) in gradient.iter_mut().zip(jt) {
            *g = finite(*g - v, "Hessian check Lagrangian gradient")?;
        }
        Ok(Some(gradient))
    };
    let base = gradient_at(model, x, &mut report.counts)?
        .ok_or_else(|| Error::UndefinedPoint("Lagrangian Hessian check base point".into()))?;
    let mut lower = vec![f64::NAN; pattern.nnz()];
    model.lagrangian_hessian(x, lambda, &mut lower)?;
    let mut columns = vec![Vec::new(); spec.n()];
    for i in 0..spec.n() {
        let range = pattern.indptr[i]..pattern.indptr[i + 1];
        for (&j, &v) in pattern.row(i).iter().zip(&lower[range]) {
            let v = finite(v, "Lagrangian Hessian")?;
            columns[j].push((i, v));
            if i != j {
                columns[i].push((j, v));
            }
        }
    }
    compare_columns(
        spec,
        x,
        options,
        &base,
        &columns,
        report,
        |point, counts| gradient_at(model, point, counts),
    )
}

fn compare_columns(
    spec: &ProblemSpec,
    x: &[f64],
    options: &DerivativeOptions,
    base: &[f64],
    columns: &[Vec<(usize, f64)>],
    mut report: DerivativeReport,
    mut evaluate: impl FnMut(&[f64], &mut EvaluationCounts) -> Result<Option<Vec<f64>>>,
) -> Result<DerivativeReport> {
    let bounds = spec.variables();
    for (j, column) in columns.iter().enumerate() {
        let (lo, up) = (bounds.lower[j], bounds.upper[j]);
        if lo == up {
            report.fixed.push(j);
            continue;
        }
        if report.checked >= options.max_directions {
            report.skipped.push(j);
            continue;
        }
        let mut analytic = vec![0.0; base.len()];
        for &(row, value) in column {
            analytic[row] = value;
        }

        // (scaled error, estimate) of the best step per output.
        let mut best = vec![(f64::INFINITY, 0.0); base.len()];
        let mut sampled = false;
        for &step in &options.steps {
            let h = step * x[j].abs().max(1.0);
            let Some((left, right)) = interval(x[j], lo, up, h) else {
                continue;
            };
            let mut point = x.to_vec();
            let mut at = |value: f64, counts: &mut EvaluationCounts| -> Result<Option<Vec<f64>>> {
                if value == x[j] {
                    return Ok(Some(base.to_vec()));
                }
                point[j] = value;
                evaluate(&point, counts)
            };
            let Some(minus) = at(left, &mut report.counts)? else {
                continue;
            };
            let Some(plus) = at(right, &mut report.counts)? else {
                continue;
            };
            sampled = true;
            for i in 0..base.len() {
                let estimate = finite((plus[i] - minus[i]) / (right - left), "finite difference")?;
                let allowance = options.absolute_tolerance
                    + options.relative_tolerance * analytic[i].abs().max(estimate.abs());
                let error = finite((analytic[i] - estimate).abs() / allowance, "scaled error")?;
                if error < best[i].0 {
                    best[i] = (error, estimate);
                }
            }
        }
        if !sampled {
            report.skipped.push(j);
            continue;
        }
        report.checked += 1;
        for (i, &(error, estimate)) in best.iter().enumerate() {
            if error > 1.0 {
                report.issues.push(DerivativeIssue {
                    variable: j,
                    output: i,
                    analytic: analytic[i],
                    estimated: estimate,
                    scaled_error: error,
                });
            }
        }
    }
    report.inconclusive = !report.skipped.is_empty() || report.checked == 0;
    report.passed = report.issues.is_empty() && !report.inconclusive;
    Ok(report)
}
