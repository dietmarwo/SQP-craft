//! Seam for the QP subproblem solver.
//!
//! It fixes the data layout, the multiplier convention, the possible outcomes
//! and one independent acceptance rule ([`QpModel::accepts`]). The first
//! backend is the Clarabel adapter in the `sqp-craft-clarabel` crate; the
//! dummy NLP backend never calls this seam. Capability negotiation and
//! backend-specific warm-start state are still to come.

use crate::error::{Error, Result, ensure, first_non_finite};
use crate::kkt::{KktPoint, KktResiduals, elastic_residuals_with_hard_rows, residuals_validated};
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

/// Owned scratch for independent QP checks. Separate instances can check the
/// same model; scratch contents never change the validation or acceptance rule.
#[derive(Clone, Debug, Default)]
pub struct QpResidualWorkspace {
    gradient: Vec<f64>,
    rows: Vec<f64>,
    transpose: Vec<f64>,
    relaxed_lower: Vec<f64>,
    relaxed_upper: Vec<f64>,
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
            if h.row(i).last().is_some_and(|&j| j > i) {
                let j = *h.row(i).iter().find(|&&j| j > i).expect("above diagonal");
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
            let pi = p[i];
            let mut value = out[i];
            if range.len() == i + 1 {
                // In a valid lower-only sorted row, full length proves 0..=i.
                let values = &h.values[range];
                for ((&v, &pj), outj) in values[..i].iter().zip(&p[..i]).zip(&mut out[..i]) {
                    value += v * pj;
                    *outj += v * pi;
                }
                out[i] = value + values[i] * pi;
                continue;
            }
            for (&j, &v) in h.pattern.indices[range.clone()]
                .iter()
                .zip(&h.values[range])
            {
                value += v * p[j];
                if j != i {
                    out[j] += v * pi;
                }
            }
            out[i] = value;
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
        self.residuals_with_workspace(solution, &mut QpResidualWorkspace::default())
    }

    /// [`Self::residuals`] with reusable owned work buffers and identical checks.
    pub fn residuals_with_workspace(
        &self,
        solution: &QpSolution,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<KktResiduals> {
        self.checked_residuals(solution, None, workspace)
    }

    /// Residuals of the elastic QP, in which rows may be violated at a cost of
    /// `weight` per unit; rows violated by at most `slack` count as
    /// satisfied (see [`QpOptions::elastic_weight`] and
    /// [`crate::kkt::elastic_residuals_with_workspace`]).
    pub fn elastic_residuals_with_workspace(
        &self,
        solution: &QpSolution,
        weight: f64,
        slack: f64,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<KktResiduals> {
        self.checked_residuals(solution, Some((weight, slack, &[])), workspace)
    }

    /// [`Self::elastic_residuals_with_workspace`] where the rows flagged in
    /// `hard` (see [`QpOptions::hard_rows`]) stay hard.
    pub fn elastic_residuals_with_hard_rows(
        &self,
        solution: &QpSolution,
        weight: f64,
        hard: &[bool],
        slack: f64,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<KktResiduals> {
        self.checked_residuals(solution, Some((weight, slack, hard)), workspace)
    }

    fn checked_residuals(
        &self,
        solution: &QpSolution,
        elastic: Option<(f64, f64, &[bool])>,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<KktResiduals> {
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
        let QpResidualWorkspace {
            gradient,
            rows,
            transpose,
            relaxed_lower,
            relaxed_upper,
        } = workspace;
        gradient.resize(self.n(), 0.);
        self.hessian_product(&solution.step, gradient);
        for (gi, &g) in gradient.iter_mut().zip(&self.gradient) {
            *gi += g;
        }
        rows.resize(self.m(), 0.);
        self.matrix
            .pattern
            .multiply(&self.matrix.values, &solution.step, rows);
        for &v in gradient.iter().chain(rows.iter()) {
            crate::error::finite(v, "QP residual arithmetic")?;
        }
        let point = KktPoint {
            x: &solution.step,
            variable_bounds: &self.variables,
            rows: rows.as_slice(),
            row_bounds: &self.rows,
            gradient: gradient.as_slice(),
            jacobian: &self.matrix.pattern,
            jacobian_values: &self.matrix.values,
        };
        let Some((weight, slack, hard)) = elastic else {
            for (what, values, expected) in [
                (
                    "variable multipliers",
                    solution.multipliers.variables.as_slice(),
                    self.n(),
                ),
                (
                    "row multipliers",
                    solution.multipliers.constraints.as_slice(),
                    self.m(),
                ),
            ] {
                ensure(values.len() == expected, || {
                    format!(
                        "KKT input: {what} has length {}, expected {expected}",
                        values.len()
                    )
                })?;
                if let Some((i, v)) = first_non_finite(values) {
                    return Err(Error::InvalidInput(format!(
                        "KKT input: {what} entry {i} is {v}"
                    )));
                }
            }
            return residuals_validated(&point, &solution.multipliers, transpose);
        };
        let mut relaxed = Bounds::new(std::mem::take(relaxed_lower), std::mem::take(relaxed_upper));
        let result = elastic_residuals_with_hard_rows(
            &point,
            &solution.multipliers,
            weight,
            hard,
            slack,
            &mut relaxed,
            transpose,
        );
        (*relaxed_lower, *relaxed_upper) = (relaxed.lower, relaxed.upper);
        result
    }

    /// Accepts or rejects `solution` from the model data alone, with one rule
    /// shared by every backend and the SQP driver.
    ///
    /// The primal residual (bound violation of `p` and `A p`) must not exceed
    /// `tolerance` in step units. Stationarity, complementarity and dual-sign
    /// residuals must not exceed `tolerance * scale` with
    /// `scale = 1 + max(‖g‖∞, ‖λ‖∞, ‖z‖∞)`. The scaling matters for
    /// interior-point solutions: their complementarity is `|μ| * distance`, so
    /// a multiplier of 500 with an active bound met to 1e-9 already gives 5e-7.
    /// An absolute threshold would reject such correct solutions while still
    /// catching mapping errors (wrong sign, transposition, doubling), which
    /// produce residuals of the order of `scale` itself.
    pub fn accepts(&self, solution: &QpSolution, tolerance: f64) -> Result<QpAcceptance> {
        self.accepts_with_workspace(solution, tolerance, &mut QpResidualWorkspace::default())
    }

    /// [`Self::accepts`] using reusable independent-check scratch.
    pub fn accepts_with_workspace(
        &self,
        solution: &QpSolution,
        tolerance: f64,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<QpAcceptance> {
        self.checked_acceptance(solution, tolerance, None, workspace)
    }

    /// [`Self::accepts_elastic_with_workspace`] with fresh scratch.
    pub fn accepts_elastic(
        &self,
        solution: &QpSolution,
        tolerance: f64,
        weight: f64,
    ) -> Result<QpAcceptance> {
        let mut workspace = QpResidualWorkspace::default();
        self.accepts_elastic_with_workspace(solution, tolerance, weight, &mut workspace)
    }

    /// [`Self::accepts`] for the elastic QP with `weight` (see
    /// [`QpOptions::elastic_weight`]): the same thresholds applied to
    /// [`Self::elastic_residuals_with_workspace`] with `slack = tolerance`.
    /// Rows may be violated; the primal residual covers the variable bounds
    /// and rows violated by at most the tolerance.
    pub fn accepts_elastic_with_workspace(
        &self,
        solution: &QpSolution,
        tolerance: f64,
        weight: f64,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<QpAcceptance> {
        self.checked_acceptance(
            solution,
            tolerance,
            Some((weight, tolerance, &[])),
            workspace,
        )
    }

    /// [`Self::accepts_elastic_with_workspace`] where the rows flagged in
    /// `hard` (see [`QpOptions::hard_rows`]) stay hard.
    pub fn accepts_elastic_with_hard_rows(
        &self,
        solution: &QpSolution,
        tolerance: f64,
        weight: f64,
        hard: &[bool],
        workspace: &mut QpResidualWorkspace,
    ) -> Result<QpAcceptance> {
        self.checked_acceptance(
            solution,
            tolerance,
            Some((weight, tolerance, hard)),
            workspace,
        )
    }

    fn checked_acceptance(
        &self,
        solution: &QpSolution,
        tolerance: f64,
        elastic: Option<(f64, f64, &[bool])>,
        workspace: &mut QpResidualWorkspace,
    ) -> Result<QpAcceptance> {
        ensure(tolerance.is_finite() && tolerance > 0.0, || {
            format!("QP acceptance tolerance must be positive and finite, got {tolerance}")
        })?;
        let residuals = self.checked_residuals(solution, elastic, workspace)?;
        let norm = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let scale = 1.0
            + norm(&self.gradient)
                .max(norm(&solution.multipliers.constraints))
                .max(norm(&solution.multipliers.variables));
        let dual = tolerance * scale;
        Ok(QpAcceptance {
            accepted: residuals.primal <= tolerance
                && residuals.stationarity <= dual
                && residuals.complementarity <= dual
                && residuals.dual_sign <= dual,
            residuals,
            scale,
        })
    }
}

/// Result of [`QpModel::accepts`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QpAcceptance {
    /// Whether every residual is within its threshold.
    pub accepted: bool,
    /// The independently recomputed residuals.
    pub residuals: KktResiduals,
    /// `1 + max(‖g‖∞, ‖λ‖∞, ‖z‖∞)`, the factor applied to the dual-side thresholds.
    pub scale: f64,
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
        for values in [
            self.step.as_deref(),
            self.multipliers.as_ref().map(|m| m.variables.as_slice()),
            self.multipliers.as_ref().map(|m| m.constraints.as_slice()),
        ]
        .into_iter()
        .flatten()
        {
            ensure(values.iter().all(|v| v.is_finite()), || {
                "QP warm start contains nonfinite values".into()
            })?;
        }
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
    /// The Hessian is not positive definite, so the model was not solved.
    /// A backend that requires a convex model reports this instead of an
    /// error; the SQP driver resets its Hessian approximation and re-solves.
    NotPositiveDefinite,
}

/// A QP subproblem solver.
pub trait QpBackend {
    /// Sets a maximum native worker budget. Custom backends default to one
    /// thread and explicitly reject an unsupported parallel request.
    fn set_threads(&mut self, threads: usize) -> Result<()> {
        if threads == 1 {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "parallel threads = {threads} unsupported by this QP backend"
            )))
        }
    }

    /// Configure tolerances and per-call budgets. Unsupported by the dummy.
    fn configure(&mut self, _options: &QpOptions) -> Result<()> {
        Err(Error::Unsupported("QP options are not implemented".into()))
    }
    /// Whether [`QpOptions::elastic_weight`] is supported. A backend that
    /// returns `false` rejects an elastic configuration with
    /// [`Error::Unsupported`].
    fn supports_elastic(&self) -> bool {
        false
    }
    /// Engine identity and last termination details.
    fn report(&self) -> QpReport {
        QpReport::default()
    }

    /// Solves `model`. [`Error`] is reserved for invalid input and backend
    /// failures; see [`QpOutcome`] for regular outcomes.
    ///
    /// A backend that needs a positive-definite Hessian must check every model
    /// exactly and report a failure as [`QpOutcome::NotPositiveDefinite`],
    /// not as an error. The SQP driver does not factorize its BFGS matrix: it
    /// relies on this outcome to reset the matrix, as after a failed update.
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

/// Per-call configuration for a convex QP backend.
#[derive(Clone, Debug, PartialEq)]
pub struct QpOptions {
    /// Permit compatible backend workspace reuse, not a primal/dual warm start.
    pub reuse_workspace: bool,
    /// Native feasibility and gap tolerances, in the backend's own (possibly
    /// scaled) measure.
    pub tolerance: f64,
    /// Tolerance for the independent check [`QpModel::accepts`].
    pub acceptance_tolerance: f64,
    /// Iteration budget.
    pub max_iterations: u32,
    /// Remaining cooperative time budget, including setup.
    pub time_limit_seconds: Option<f64>,
    /// Solve the ℓ1-elastic QP instead: every row may be violated at a cost
    /// of this weight per unit, so the QP is feasible whenever the variable
    /// bounds are. Variable bounds stay hard. Row multipliers then satisfy
    /// `|λ_i| <= weight`, with `|λ_i| = weight` on violated rows. Solutions
    /// are checked by [`QpModel::accepts_elastic_with_workspace`]. Only
    /// backends whose [`QpBackend::supports_elastic`] is true accept it.
    pub elastic_weight: Option<f64>,
    /// Rows that stay hard in an elastic QP (empty: none; otherwise one flag
    /// per row). The SQP driver passes the rows declared linear, which its
    /// iterates keep satisfied. Without an elastic weight every row is hard
    /// and this is ignored.
    pub hard_rows: Vec<bool>,
}
impl Default for QpOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            reuse_workspace: false,
            acceptance_tolerance: 1e-7,
            max_iterations: 200,
            time_limit_seconds: None,
            elastic_weight: None,
            hard_rows: Vec::new(),
        }
    }
}
impl QpOptions {
    /// Validate before backend allocation or cancellation.
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.tolerance.is_finite()
                && self.tolerance > 0.0
                && self.acceptance_tolerance.is_finite()
                && self.acceptance_tolerance > 0.0,
            || "QP tolerances must be positive and finite".into(),
        )?;
        ensure(
            self.time_limit_seconds
                .is_none_or(|s| s.is_finite() && s >= 0.0),
            || "QP time budget must be finite and nonnegative".into(),
        )?;
        ensure(
            self.elastic_weight.is_none_or(|w| w.is_finite() && w > 0.0),
            || "QP elastic weight must be positive and finite".into(),
        )
    }
}
/// Details not represented by the backend-neutral outcome enum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QpReport {
    /// Engine and version.
    pub backend: String,
    /// Native termination status.
    pub raw_status: Option<String>,
    /// If limited, time or iterations.
    pub limit_reason: Option<String>,
    /// Iterations including unsuccessful calls.
    pub iterations: u64,
    /// Fresh engine setup attempts, including a cold retry.
    pub setups: u64,
    /// Successful complete numeric workspace updates.
    pub updates: u64,
    /// Fresh retries after an unsuccessful workspace update or reused solve.
    pub cold_retries: u64,
    /// Reason requesting a cold retry, even if the remaining budget prevents it.
    pub retry_reason: Option<String>,
    /// Model conversion and positive-definiteness validation time.
    pub conversion_seconds: f64,
    /// Fresh engine setup time.
    pub setup_seconds: f64,
    /// Numeric data update time, including failed updates.
    pub update_seconds: f64,
    /// Total engine solve time across all attempts.
    pub solve_seconds: f64,
    /// Independent solution-check time across all attempts.
    pub check_seconds: f64,
}
