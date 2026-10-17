//! Solver-independent first-order (KKT) residuals.
//!
//! These functions do not trust any solver: given a point, its model values and
//! multipliers in the [`Multipliers`] sign convention, they recompute the
//! residuals in original, unscaled units. The same code checks NLP candidates
//! and QP subproblem solutions ([`QpModel::residuals`](crate::qp::QpModel::residuals)),
//! so every backend is held to one definition.

use crate::error::{Error, Result, ensure, first_non_finite};
use crate::solver::{Evaluation, Multipliers};
use crate::spec::{Bounds, CsrPattern, ProblemSpec};

/// First-order residuals, all absolute, unscaled and `>= 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KktResiduals {
    /// Largest bound violation of variables and rows.
    pub primal: f64,
    /// `‖gradient - J^T λ - z‖∞`.
    pub stationarity: f64,
    /// Largest `|μ| * |distance to the bound selected by the sign of μ|`, over
    /// all row and variable multipliers `μ`.
    pub complementarity: f64,
    /// Largest `|μ|` whose sign selects an infinite bound, for example `λ_i > 0`
    /// when row `i` has no lower bound. Zero for sign-consistent multipliers.
    pub dual_sign: f64,
}

impl KktResiduals {
    /// The largest of the four residuals.
    pub fn max(&self) -> f64 {
        self.primal
            .max(self.stationarity)
            .max(self.complementarity)
            .max(self.dual_sign)
    }
}

/// A point of `min f(x)  s.t.  row_bounds on r(x),  variable_bounds on x`
/// together with first-order information, as needed by [`residuals`].
#[derive(Clone, Copy, Debug)]
pub struct KktPoint<'a> {
    /// The point, length `n`.
    pub x: &'a [f64],
    /// Bounds on `x`.
    pub variable_bounds: &'a Bounds,
    /// Row values `r(x)`, length `m`.
    pub rows: &'a [f64],
    /// Bounds on the rows.
    pub row_bounds: &'a Bounds,
    /// Objective gradient at `x`, length `n`.
    pub gradient: &'a [f64],
    /// Pattern of the row Jacobian (`m x n`).
    pub jacobian: &'a CsrPattern,
    /// Row Jacobian values at `x`, in pattern order.
    pub jacobian_values: &'a [f64],
}

/// Computes [`KktResiduals`] for `point` and `multipliers`.
///
/// Returns [`Error::InvalidInput`] for inconsistent lengths, invalid bounds or
/// pattern, or non-finite inputs.
pub fn residuals(point: &KktPoint<'_>, multipliers: &Multipliers) -> Result<KktResiduals> {
    residuals_with_workspace(point, multipliers, &mut Vec::new())
}

/// Same validation and residuals as [`residuals`], reusing a transpose-product
/// buffer. Scratch is resized and overwritten; it holds no borrowed inputs.
pub fn residuals_with_workspace(
    point: &KktPoint<'_>,
    multipliers: &Multipliers,
    jt_lambda: &mut Vec<f64>,
) -> Result<KktResiduals> {
    let n = point.x.len();
    let m = point.rows.len();
    point.variable_bounds.validate(n, "KKT variable bounds")?;
    point.row_bounds.validate(m, "KKT row bounds")?;
    point.jacobian.validate("KKT Jacobian")?;
    let lengths = [
        ("gradient", point.gradient.len(), n),
        ("Jacobian columns", point.jacobian.cols, n),
        ("variable multipliers", multipliers.variables.len(), n),
        ("Jacobian rows", point.jacobian.rows, m),
        ("row multipliers", multipliers.constraints.len(), m),
        (
            "Jacobian values",
            point.jacobian_values.len(),
            point.jacobian.nnz(),
        ),
    ];
    for (what, got, expected) in lengths {
        ensure(got == expected, || {
            format!("KKT input: {what} has length {got}, expected {expected}")
        })?;
    }
    let vectors = [
        ("x", point.x),
        ("row values", point.rows),
        ("gradient", point.gradient),
        ("Jacobian values", point.jacobian_values),
        ("row multipliers", &multipliers.constraints[..]),
        ("variable multipliers", &multipliers.variables[..]),
    ];
    for (what, values) in vectors {
        if let Some((i, v)) = first_non_finite(values) {
            return Err(Error::InvalidInput(format!(
                "KKT input: {what} entry {i} is {v}"
            )));
        }
    }

    let primal = point
        .variable_bounds
        .checked_violation(point.x)?
        .max(point.row_bounds.checked_violation(point.rows)?);

    jt_lambda.resize(n, 0.);
    point
        .jacobian
        .multiply_transpose(point.jacobian_values, &multipliers.constraints, jt_lambda);
    let mut stationarity = 0.0_f64;
    for (j, &jt) in jt_lambda.iter().enumerate() {
        crate::error::finite(jt, "transpose product")?;
        let value = crate::error::finite(
            crate::error::finite(point.gradient[j] - jt, "stationarity")?
                - multipliers.variables[j],
            "stationarity",
        )?;
        stationarity = stationarity.max(value.abs());
    }

    let mut complementarity = 0.0_f64;
    let mut dual_sign = 0.0_f64;
    for (values, bounds, mu) in [
        (point.rows, point.row_bounds, &multipliers.constraints[..]),
        (point.x, point.variable_bounds, &multipliers.variables[..]),
    ] {
        for i in 0..values.len() {
            let (v, l, u, mu) = (values[i], bounds.lower[i], bounds.upper[i], mu[i]);
            if mu > 0.0 {
                if l.is_finite() {
                    complementarity = complementarity.max(crate::error::finite(
                        mu * crate::error::finite(v - l, "lower complementarity distance")?.abs(),
                        "complementarity",
                    )?);
                } else {
                    dual_sign = dual_sign.max(mu);
                }
            } else if mu < 0.0 {
                if u.is_finite() {
                    complementarity = complementarity.max(crate::error::finite(
                        -mu * crate::error::finite(u - v, "upper complementarity distance")?.abs(),
                        "complementarity",
                    )?);
                } else {
                    dual_sign = dual_sign.max(-mu);
                }
            }
        }
    }

    Ok(KktResiduals {
        primal,
        stationarity,
        complementarity,
        dual_sign,
    })
}

/// Residuals of the ℓ1-elastic problem, in which rows may be violated at a
/// cost of `weight` per unit:
///
/// ```text
/// minimize  f(x) + weight · Σ_i dist(r_i(x), [row_lower_i, row_upper_i])
/// subject to  variable_bounds on x
/// ```
///
/// Its first-order conditions are those of `point` with each violated row's
/// bounds relaxed to include `r_i(x)`, plus `|λ_i| <= weight`, with
/// `λ_i = weight` for a row below its lower bound and `λ_i = −weight` above
/// its upper bound. They are reported in the same four fields:
/// `complementarity` also holds `violation_i · |weight − σ_i λ_i|`
/// (σ_i = 1 below, −1 above), and `dual_sign` also holds
/// `max(|λ_i| − weight, 0)`.
///
/// A row violated by at most `slack` counts as satisfied: its violation is
/// reported in `primal`, as by [`residuals`]. Otherwise a rounding-level
/// violation (a QP solution on its bound to 1e-11, say) would demand
/// `|λ_i| = weight` and fail by `weight · 1e-11`. With `slack` at the primal
/// tolerance of the caller's acceptance test, `primal` stays within that
/// tolerance whenever the rows are elastic or satisfied. `relaxed` is
/// scratch; its contents are replaced.
///
/// At a point that violates no row by more than `slack` this equals
/// [`residuals`] apart from the `|λ_i| <= weight` test.
pub fn elastic_residuals_with_workspace(
    point: &KktPoint<'_>,
    multipliers: &Multipliers,
    weight: f64,
    slack: f64,
    relaxed: &mut Bounds,
    jt_lambda: &mut Vec<f64>,
) -> Result<KktResiduals> {
    ensure(weight.is_finite() && weight > 0.0, || {
        format!("elastic weight must be positive and finite, got {weight}")
    })?;
    ensure(slack.is_finite() && slack >= 0.0, || {
        format!("elastic slack must be finite and nonnegative, got {slack}")
    })?;
    let m = point.rows.len();
    point.row_bounds.validate(m, "KKT row bounds")?;
    ensure(multipliers.constraints.len() == m, || {
        format!(
            "KKT input: row multipliers has length {}, expected {m}",
            multipliers.constraints.len()
        )
    })?;
    relaxed.lower.clear();
    relaxed.upper.clear();
    for i in 0..m {
        // NaN rows are rejected by the residuals below; keep them in place.
        let (r, l, u) = (
            point.rows[i],
            point.row_bounds.lower[i],
            point.row_bounds.upper[i],
        );
        relaxed.lower.push(if l - r > slack { r } else { l });
        relaxed.upper.push(if r - u > slack { r } else { u });
    }
    let mut result = residuals_with_workspace(
        &KktPoint {
            row_bounds: relaxed,
            ..*point
        },
        multipliers,
        jt_lambda,
    )?;
    for i in 0..m {
        let (r, l, u) = (
            point.rows[i],
            point.row_bounds.lower[i],
            point.row_bounds.upper[i],
        );
        let lambda = multipliers.constraints[i];
        result.dual_sign = result.dual_sign.max(lambda.abs() - weight);
        let product = if l - r > slack {
            crate::error::finite((l - r) * (weight - lambda).abs(), "elastic complementarity")?
        } else if r - u > slack {
            crate::error::finite((r - u) * (weight + lambda).abs(), "elastic complementarity")?
        } else {
            0.0
        };
        result.complementarity = result.complementarity.max(product);
    }
    Ok(result)
}

/// [`residuals`] for an NLP candidate `x` with model values `evaluation`.
pub fn nlp_residuals(
    spec: &ProblemSpec,
    x: &[f64],
    evaluation: &Evaluation,
    multipliers: &Multipliers,
) -> Result<KktResiduals> {
    residuals(
        &KktPoint {
            x,
            variable_bounds: spec.variables(),
            rows: &evaluation.constraints,
            row_bounds: spec.constraints(),
            gradient: &evaluation.gradient,
            jacobian: spec.jacobian(),
            jacobian_values: &evaluation.jacobian_values,
        },
        multipliers,
    )
}
