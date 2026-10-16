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

    let mut jt_lambda = vec![0.0; n];
    point.jacobian.multiply_transpose(
        point.jacobian_values,
        &multipliers.constraints,
        &mut jt_lambda,
    );
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
