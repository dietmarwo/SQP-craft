//! Bound-aware coordinate-direction checks of the grouped callback protocol.
use crate::{error::finite, *};

/// Settings for finite-difference checks, separate from optimizer accounting.
#[derive(Clone, Debug)]
pub struct DerivativeOptions {
    /// Relative perturbation sizes; the best resolvable estimate is retained.
    pub steps: Vec<f64>,
    /// Absolute comparison tolerance.
    pub absolute_tolerance: f64,
    /// Relative comparison tolerance.
    pub relative_tolerance: f64,
    /// Maximum coordinate directions checked.
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
/// One derivative component that failed every resolvable step size.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeIssue {
    /// Perturbed variable.
    pub variable: usize,
    /// 0 for objective, i+1 for constraint i.
    pub output: usize,
    /// Supplied derivative.
    pub analytic: f64,
    /// Best finite-difference estimate.
    pub estimated: f64,
    /// Error divided by absolute + relative allowance.
    pub scaled_error: f64,
}
/// No fixed/unrepresentable/unvisited coordinate is silently counted as passed.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeReport {
    /// All requested coordinates checked without failures or omissions.
    pub passed: bool,
    /// Some coordinate could not be checked (including fixed variables).
    pub inconclusive: bool,
    /// Number of checked coordinate directions.
    pub checked: usize,
    /// Fixed, unvisited or unrepresentable coordinates.
    pub skipped: Vec<usize>,
    /// Callback counts, separate from a solve.
    pub counts: EvaluationCounts,
    /// Failed components.
    pub issues: Vec<DerivativeIssue>,
}
fn functions(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    counts: &mut EvaluationCounts,
) -> Result<Vec<f64>> {
    spec.check_point(x, "derivative-check point")?;
    let mut c = vec![f64::NAN; spec.m()];
    counts.functions += 1;
    let f = model.functions(x, &mut c)?;
    let mut out = vec![finite(f, "checker objective")?];
    for v in c {
        out.push(finite(v, "checker constraint")?);
    }
    Ok(out)
}
/// Check every selected coordinate, using central differences inside the box
/// and one-sided differences at its boundary. Calls derivatives only immediately
/// after functions at the same point. No Jacobian densification is needed.
pub fn check_derivatives(
    spec: &ProblemSpec,
    model: &mut dyn NlpProblem,
    x: &[f64],
    options: &DerivativeOptions,
) -> Result<DerivativeReport> {
    spec.check_point(x, "checker x")?;
    if options.steps.is_empty()
        || options.steps.len() > 16
        || options.steps.iter().any(|s| !s.is_finite() || *s <= 0.0)
        || options.absolute_tolerance <= 0.0
        || !options.absolute_tolerance.is_finite()
        || options.relative_tolerance < 0.0
        || !options.relative_tolerance.is_finite()
        || options.max_directions == 0
    {
        return Err(Error::InvalidInput(
            "invalid derivative-check settings".into(),
        ));
    }
    let mut report = DerivativeReport {
        passed: false,
        inconclusive: false,
        checked: 0,
        skipped: vec![],
        counts: EvaluationCounts::default(),
        issues: vec![],
    };
    let base = functions(spec, model, x, &mut report.counts)?;
    let mut gradient = vec![f64::NAN; spec.n()];
    let mut jac = vec![f64::NAN; spec.nnz()];
    report.counts.derivatives += 1;
    model.derivatives(x, &mut gradient, &mut jac)?;
    for &v in gradient.iter().chain(&jac) {
        finite(v, "checker derivative")?;
    }
    for j in 0..spec.n() {
        if report.checked >= options.max_directions
            || spec.variables().lower[j] == spec.variables().upper[j]
        {
            report.skipped.push(j);
            continue;
        }
        let mut analytic = vec![gradient[j]];
        for row in 0..spec.m() {
            let range = spec.jacobian().indptr[row]..spec.jacobian().indptr[row + 1];
            let value = range
                .filter(|&k| spec.jacobian().indices[k] == j)
                .map(|k| jac[k])
                .sum::<f64>();
            analytic.push(value);
        }
        let mut best = vec![(f64::INFINITY, 0.0); spec.m() + 1];
        let mut sampled = false;
        for &step in &options.steps {
            let h = step * x[j].abs().max(1.0);
            if !h.is_finite() {
                continue;
            }
            let lo = spec.variables().lower[j];
            let up = spec.variables().upper[j];
            let central = x[j] > lo && x[j] < up;
            let (left, right) = if central {
                let h = h.min(x[j] - lo).min(up - x[j]);
                (x[j] - h, x[j] + h)
            } else if x[j] == lo {
                (x[j], (x[j] + h).min(up))
            } else {
                ((x[j] - h).max(lo), x[j])
            };
            if !left.is_finite() || !right.is_finite() || right <= left {
                continue;
            }
            let mut point = x.to_vec();
            point[j] = left;
            let minus = if left == x[j] {
                base.clone()
            } else {
                functions(spec, model, &point, &mut report.counts)?
            };
            point[j] = right;
            let plus = if right == x[j] {
                base.clone()
            } else {
                functions(spec, model, &point, &mut report.counts)?
            };
            sampled = true;
            for i in 0..analytic.len() {
                let estimate = finite((plus[i] - minus[i]) / (right - left), "finite difference")?;
                let allowance = finite(
                    options.absolute_tolerance
                        + finite(
                            options.relative_tolerance * analytic[i].abs().max(estimate.abs()),
                            "derivative allowance",
                        )?,
                    "derivative allowance",
                )?;
                let error = finite(
                    (analytic[i] - estimate).abs() / allowance,
                    "derivative discrepancy",
                )?;
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
