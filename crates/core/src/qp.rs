//! Provisional seam for the QP subproblem solver.
//!
//! No QP engine is implemented, and the dummy NLP backend never calls this
//! seam. It fixes the data layout, the multiplier convention and the possible
//! outcomes so that the first real backends (HiGHS active-set QP and an
//! interior-point candidate such as Clarabel) can be compared on recorded QP
//! sequences. Capability negotiation and backend-specific warm-start state
//! will be added with those backends.

use crate::error::{Error, Result, ensure, first_non_finite};
use crate::kkt::{KktPoint, KktResiduals, residuals};
use crate::solver::{CancellationToken, Multipliers};
use crate::spec::{Bounds, CsrMatrix};

/// A convex quadratic program in the SQP step `p`:
///
/// ```text
/// minimize    0.5 * p^T H p + g^T p
/// subject to  row_lower      <= A p <= row_upper
///             variable_lower <=  p  <= variable_upper
/// ```
///
/// At an SQP iterate `x_k`, `H` is the (convexified) Hessian model, `g =
/// grad f(x_k)`, `A = J(x_k)`, the row bounds are `constraint bounds - c(x_k)`
/// and the variable bounds are `variable bounds - x_k`: **all bounds apply to
/// the step, not to `x`**. With the shared [`Multipliers`] convention, the QP
/// multipliers are directly the next NLP multiplier estimates.
///
/// `H` is symmetric and stored as its lower triangle in CSR: an off-diagonal
/// entry `(i, j)`, `j < i`, stands for both `H[i][j]` and `H[j][i]` and is not
/// doubled. Validation checks structure, not positive semidefiniteness: the
/// SQP driver must convexify `H` before calling a convex backend.
#[derive(Clone, Debug, PartialEq)]
pub struct QpModel {
    /// Lower triangle of `H` (`n x n`).
    pub hessian_lower: CsrMatrix,
    /// Linear term `g`, length `n`.
    pub gradient: Vec<f64>,
    /// Bounds on the step `p`.
    pub variables: Bounds,
    /// Constraint matrix `A` (`m x n`).
    pub matrix: CsrMatrix,
    /// Bounds on `A p`.
    pub rows: Bounds,
}

impl QpModel {
    /// Number of step variables.
    pub fn n(&self) -> usize {
        self.gradient.len()
    }

    /// Number of linear constraint rows.
    pub fn m(&self) -> usize {
        self.matrix.pattern.rows
    }

    /// Checks shapes, finiteness and that `H` holds only its lower triangle.
    pub fn validate(&self) -> Result<()> {
        let n = self.n();
        ensure(n > 0, || "QP: at least one variable is required".into())?;
        if let Some((i, v)) = first_non_finite(&self.gradient) {
            return Err(Error::InvalidInput(format!(
                "QP gradient: entry {i} is {v}"
            )));
        }
        self.variables.validate(n, "QP variable bounds")?;
        self.matrix.validate("QP constraint matrix")?;
        ensure(self.matrix.pattern.cols == n, || {
            format!(
                "QP constraint matrix: {} columns, expected n = {n}",
                self.matrix.pattern.cols
            )
        })?;
        self.rows.validate(self.m(), "QP row bounds")?;
        self.hessian_lower.validate("QP Hessian")?;
        let h = &self.hessian_lower.pattern;
        ensure(h.rows == n && h.cols == n, || {
            format!(
                "QP Hessian: shape {} x {}, expected {n} x {n}",
                h.rows, h.cols
            )
        })?;
        for i in 0..n {
            if let Some(&j) = h.row(i).iter().find(|&&j| j > i) {
                return Err(Error::InvalidInput(format!(
                    "QP Hessian: entry ({i}, {j}) is above the diagonal; store the lower triangle only"
                )));
            }
        }
        Ok(())
    }

    /// `out = H p`, using the symmetric expansion of the stored lower triangle.
    ///
    /// # Panics
    /// On length mismatches or an invalid model.
    pub fn hessian_product(&self, p: &[f64], out: &mut [f64]) {
        let h = &self.hessian_lower;
        assert_eq!(p.len(), self.n(), "p length");
        assert_eq!(out.len(), self.n(), "out length");
        out.fill(0.0);
        for i in 0..self.n() {
            let range = h.pattern.indptr[i]..h.pattern.indptr[i + 1];
            for (&j, &v) in h.pattern.indices[range.clone()]
                .iter()
                .zip(&h.values[range])
            {
                out[i] += v * p[j];
                if j != i {
                    out[j] += v * p[i];
                }
            }
        }
    }

    /// Objective value `0.5 p^T H p + g^T p`.
    ///
    /// # Panics
    /// On a length mismatch or an invalid model.
    pub fn objective(&self, p: &[f64]) -> f64 {
        let mut hp = vec![0.0; self.n()];
        self.hessian_product(p, &mut hp);
        p.iter()
            .zip(&hp)
            .zip(&self.gradient)
            .map(|((&pi, &hpi), &gi)| pi * (0.5 * hpi + gi))
            .sum()
    }

    /// Recomputes the KKT residuals of `solution` from the model data alone,
    /// independently of the backend that produced it.
    pub fn residuals(&self, solution: &QpSolution) -> Result<KktResiduals> {
        self.validate()?;
        ensure(solution.step.len() == self.n(), || {
            format!(
                "QP solution: step has {} entries, expected {}",
                solution.step.len(),
                self.n()
            )
        })?;
        if let Some((i, v)) = first_non_finite(&solution.step) {
            return Err(Error::InvalidInput(format!(
                "QP solution: step entry {i} is {v}"
            )));
        }
        let mut gradient = vec![0.0; self.n()];
        self.hessian_product(&solution.step, &mut gradient);
        for (gi, &g) in gradient.iter_mut().zip(&self.gradient) {
            *gi += g;
        }
        let mut rows = vec![0.0; self.m()];
        self.matrix
            .pattern
            .multiply(&self.matrix.values, &solution.step, &mut rows);
        residuals(
            &KktPoint {
                x: &solution.step,
                variable_bounds: &self.variables,
                rows: &rows,
                row_bounds: &self.rows,
                gradient: &gradient,
                jacobian: &self.matrix.pattern,
                jacobian_values: &self.matrix.values,
            },
            &solution.multipliers,
        )
    }
}

/// Whether a bound is in the working (active) set.
///
/// For an equality (equal bounds) either `AtLower` or `AtUpper` marks it active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    /// Not in the working set.
    Inactive,
    /// The lower bound is active.
    AtLower,
    /// The upper bound is active.
    AtUpper,
}

/// Backend-neutral working set: one entry per variable and per row.
///
/// Active-set QP methods warm-start from the previous working set, which
/// near SQP convergence changes little between subproblems.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveSet {
    /// Length `n`.
    pub variables: Vec<Activity>,
    /// Length `m`.
    pub rows: Vec<Activity>,
}

/// Optional starting information. Backends use what their method can exploit:
/// a working set (active-set methods) or a primal-dual point (interior-point
/// methods).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QpWarmStart {
    /// Starting step.
    pub step: Option<Vec<f64>>,
    /// Starting multipliers.
    pub multipliers: Option<Multipliers>,
    /// Starting working set.
    pub active_set: Option<ActiveSet>,
}

impl QpWarmStart {
    /// Checks lengths against a QP with `n` variables and `m` rows.
    pub fn validate(&self, n: usize, m: usize) -> Result<()> {
        if let Some(step) = &self.step {
            ensure(step.len() == n, || {
                format!(
                    "QP warm start: step has {} entries, expected {n}",
                    step.len()
                )
            })?;
        }
        if let Some(mult) = &self.multipliers {
            ensure(
                mult.variables.len() == n && mult.constraints.len() == m,
                || format!("QP warm start: multipliers must have lengths {n} and {m}"),
            )?;
        }
        if let Some(set) = &self.active_set {
            ensure(set.variables.len() == n && set.rows.len() == m, || {
                format!("QP warm start: active set must have lengths {n} and {m}")
            })?;
        }
        Ok(())
    }
}

/// A QP solution as reported by a backend. Check it with
/// [`QpModel::residuals`] rather than trusting the backend's status.
#[derive(Clone, Debug, PartialEq)]
pub struct QpSolution {
    /// The step `p`.
    pub step: Vec<f64>,
    /// Row and variable multipliers in the shared convention
    /// `H p + g = A^T λ + z`.
    pub multipliers: Multipliers,
    /// Final working set, if the method maintains one.
    pub active_set: Option<ActiveSet>,
    /// Backend iterations.
    pub iterations: u64,
}

/// How a QP solve ended.
///
/// Infeasibility and unboundedness are outcomes, not errors: an SQP driver
/// responds to them (elastic mode or feasibility restoration for an
/// inconsistent linearization; Hessian regularization for unboundedness).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum QpOutcome {
    /// The backend reports an optimal solution.
    Optimal(QpSolution),
    /// The linearized constraints are inconsistent.
    Infeasible,
    /// The objective is unbounded below on the feasible set.
    Unbounded,
    /// An iteration or time limit was reached; the last iterate, if available.
    LimitReached(Option<QpSolution>),
    /// Stopped by the cancellation token.
    Cancelled,
}

/// A QP subproblem solver.
pub trait QpBackend {
    /// Solves `model`. [`Error`] is reserved for invalid input and backend
    /// failures; see [`QpOutcome`] for regular outcomes.
    fn solve(
        &mut self,
        model: &QpModel,
        warm_start: Option<&QpWarmStart>,
        cancel: &CancellationToken,
    ) -> Result<QpOutcome>;
}

/// Validates its input, then returns [`Error::Unsupported`]. It never
/// fabricates a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct DummyQp;

impl QpBackend for DummyQp {
    fn solve(
        &mut self,
        model: &QpModel,
        warm_start: Option<&QpWarmStart>,
        _cancel: &CancellationToken,
    ) -> Result<QpOutcome> {
        model.validate()?;
        if let Some(warm) = warm_start {
            warm.validate(model.n(), model.m())?;
        }
        Err(Error::Unsupported(
            "DummyQp is an interface placeholder; no QP backend is implemented".into(),
        ))
    }
}
