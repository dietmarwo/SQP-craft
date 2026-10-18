//! Native SQP driver: dense damped BFGS, convex QP subproblems behind
//! [`QpBackend`], and an L1 merit line search with per-constraint penalties.
//!
//! A bounded research pilot, not a general solver: at most 256 variables and
//! local Gauss–Newton restoration for inconsistent linearizations (disable it
//! with `max_restoration_steps = 0`). Opt-in options add elastic step QPs, a
//! second-order correction, and four mechanisms taken from SNOPT
//! (docs/snopt-lessons.md):
//! - SNOPT's BFGS repair ([`HessianUpdate::Augmented`]);
//! - an augmented-Lagrangian merit ([`Merit::AugmentedLagrangian`]);
//! - a violation limit;
//! - linear rows kept feasible ([`ProblemSpec::with_linear_rows`]);
//! - scaling ([`Scaling`]).
//!
//! Convergence means the first-order KKT residuals at the returned point are
//! within the declared tolerances, in original units; it is not a global or
//! second-order optimality certificate.
//!
//! Each iteration at `x_k`:
//!
//! 1. Solve the step QP `min ½pᵀBp + gᵀp` with bounds shifted to step
//!    coordinates, and accept it only through [`QpModel::accepts`].
//! 2. Compute KKT residuals at `x_k` with the QP's multipliers; stop with
//!    [`Status::Converged`] if all four are within tolerance.
//! 3. Update the penalty weights by Powell's rule
//!    `μ_i ← max(t_i, ½(μ_i + t_i))` with `t_i = 1.1·|λ_i|`, and backtrack on the merit
//!    `φ(x) = f(x) + Σ μ_i · violation_i(c(x))` with an Armijo test against the
//!    predicted decrease of the linearized merit.
//! 4. At the accepted point call `derivatives`, then update `B` by damped
//!    BFGS on the Lagrangian gradient with the new multipliers at both ends.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bfgs;
mod linear;
mod merit;
mod restoration;
mod scaling;

use scaling::Scales;
pub use scaling::Scaling;

use std::time::Instant;

use nalgebra::DMatrix;
use sqp_craft::kkt::nlp_residuals;
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome, QpSolution, QpWarmStart};
use sqp_craft::{
    Bounds, CancellationToken, Control, CsrMatrix, CsrPattern, CurvatureUpdate, Error, Evaluation,
    EvaluationCounts, IterationRecord, Multipliers, NlpProblem, Observer, Options, ProblemSpec,
    Progress, Result, SolveResult, SqpSolver, Status,
};

/// Largest number of variables the dense pilot accepts.
pub const DENSE_SIZE_CAP: usize = 256;
/// Largest number of QP models [`SqpOptions::record_qp_limit`] may keep.
pub const QP_TRACE_CAP: usize = 16;
/// Armijo sufficient-decrease constant.
const ARMIJO: f64 = 1e-4;
/// Penalty weights target `PENALTY_MARGIN · |λ_i|`. The margin above `|λ_i|`
/// keeps the predicted merit decrease positive at infeasible iterates although
/// interior-point QP solutions stop slightly short of active bounds.
const PENALTY_MARGIN: f64 = 1.1;
/// The rounding level of a merit value, in units of `ε` times the size of
/// its terms (see the augmented-Lagrangian noise steps).
const MERIT_ROUNDING: f64 = 100.0;
/// Consecutive steps allowed whose predicted merit decrease is non-positive
/// only within the QP's accuracy (see the line search).
const MAX_NOISE_STEPS: u32 = 3;
/// Times elastic mode multiplies its weight by ten at a stationary point of
/// the ℓ1 penalty with violated constraints before it reports
/// `locally_infeasible`.
const MAX_ELASTIC_INCREASES: u64 = 6;

/// How the BFGS update treats a step with too little curvature,
/// `sᵀy < 0.2·sᵀBs`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HessianUpdate {
    /// Powell's damping: blend `y` with `Bs` until the curvature is
    /// `0.2·sᵀBs` (default).
    #[default]
    Damped,
    /// SNOPT's second modification (Gill, Murray and Saunders 2005, §2.10):
    /// add the gradient change of the augmented modified Lagrangian,
    /// `(J(x⁺) − J(x))ᵀ Ω d_L` with the departure from linearity
    /// `d_L = c(x⁺) − c(x) − J(x)s` and minimum-norm weights `Ω ≥ 0`, so
    /// that the curvature becomes `0.2·sᵀBs`. If no such weights exist, or
    /// the change exceeds `100·‖Bs‖`, the update is damped as with
    /// [`Self::Damped`]. SNOPT skips the update instead; here that left a
    /// `B` that overestimated the curvature unchanged for hundreds of steps
    /// on complementarity problems.
    Augmented,
}

impl HessianUpdate {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            HessianUpdate::Damped => "damped",
            HessianUpdate::Augmented => "augmented",
        }
    }

    /// Parses [`Self::as_str`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "damped" => Ok(HessianUpdate::Damped),
            "augmented" => Ok(HessianUpdate::Augmented),
            _ => Err(Error::InvalidInput(format!(
                "hessian_update must be \"damped\" or \"augmented\", got {name:?}"
            ))),
        }
    }
}

/// Merit function of the line search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Merit {
    /// `f + Σ μᵢ·violationᵢ` with Powell's per-row penalties
    /// `μᵢ ← max(1.1|λᵢ|, ½(μᵢ + 1.1|λᵢ|))` (default).
    #[default]
    L1,
    /// SNOPT's augmented Lagrangian `f − πᵀ(c − s) + ½Σρᵢ(cᵢ − sᵢ)²`,
    /// searched jointly in the variables, the multiplier estimate `π` and
    /// the slacks `s` (Gill, Murray and Saunders 2005, §2.7). The penalties
    /// `ρ` are the smallest for which the linear model of the merit falls by
    /// `½pᵀBp` over the unit step, twice the paper's slope rule, and may
    /// decrease under its damping rule. It is differentiable, so it
    /// needs no second-order correction; the combination is rejected.
    /// Steps whose elastic QP leaves rows violated use the ℓ1 merit.
    AugmentedLagrangian,
}

impl Merit {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Merit::L1 => "l1",
            Merit::AugmentedLagrangian => "augmented-lagrangian",
        }
    }

    /// Parses [`Self::as_str`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "l1" => Ok(Merit::L1),
            "augmented-lagrangian" => Ok(Merit::AugmentedLagrangian),
            _ => Err(Error::InvalidInput(format!(
                "merit must be \"l1\" or \"augmented-lagrangian\", got {name:?}"
            ))),
        }
    }
}

/// Settings of [`NativeSqp`]. Every field is used; invalid values are rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct SqpOptions {
    /// Maximum number of accepted steps.
    pub max_iterations: u64,
    /// Maximum step halvings in one line search after the first trial.
    pub max_backtracks: u32,
    /// Maximum `functions` plus `derivatives` calls, including rejected and
    /// undefined trials.
    pub max_callback_calls: u64,
    /// Cooperative wall-time budget for the whole solve, in seconds.
    pub time_limit_seconds: Option<f64>,
    /// Absolute stationarity tolerance `‖∇f − Jᵀλ − z‖∞`.
    pub stationarity_tolerance: f64,
    /// Absolute complementarity tolerance.
    pub complementarity_tolerance: f64,
    /// Absolute tolerance on multipliers whose sign selects an infinite bound.
    pub dual_sign_tolerance: f64,
    /// Variables allowed for this solve, `1..=DENSE_SIZE_CAP`.
    pub dense_size_limit: usize,
    /// Initial and reset Hessian approximation `B = scale · I`.
    pub initial_hessian_scale: f64,
    /// Contiguous curvature block width; zero keeps full BFGS (default).
    pub hessian_block_size: usize,
    /// Total Hessian resets allowed (failed BFGS updates and non-descent retries).
    pub max_hessian_resets: u32,
    /// Native tolerance passed to the QP backend.
    pub qp_tolerance: f64,
    /// Iteration cap per QP.
    pub qp_max_iterations: u32,
    /// Opt in to compatible QP workspace reuse (not a primal/dual warm start).
    pub qp_reuse_workspace: bool,
    /// Pass the working set of the last step QP to the next one. Only
    /// backends that report a working set (active-set methods) get one, so
    /// this has no effect with Clarabel.
    pub qp_warm_start: bool,
    /// Solve every step QP as an ℓ1-elastic QP (`QpOptions::elastic_weight`):
    /// constraint rows may be violated at a cost, so the QP is always
    /// feasible and feasibility restoration is not used. Requires a backend
    /// with elastic support (`ActiveSetQp`).
    pub elastic_mode: bool,
    /// Initial elastic weight, relative to `max(1, ‖∇f(x₀)‖∞)`.
    pub elastic_weight: f64,
    /// When the full step fails the merit test and increases the constraint
    /// violation, solve the step QP once more with the constraints shifted by
    /// their second-order change along the step, and try that step before
    /// backtracking (against the Maratos effect).
    pub second_order_correction: bool,
    /// What to do when the curvature of a step is too small for a plain
    /// BFGS update (`sᵀy < 0.2·sᵀBs`).
    pub hessian_update: HessianUpdate,
    /// Merit function of the line search.
    pub merit: Merit,
    /// Scaling of the QP subproblems and the BFGS matrix. The merit, the line
    /// search and the convergence test stay in original units.
    pub scaling: Scaling,
    /// SNOPT's violation limit `τ_V ≥ 1`: a trial point whose violation of
    /// some row exceeds `τ_V · max(1, violation of that row at the start)`
    /// is rejected before its derivatives are evaluated. `None` (default)
    /// imposes no limit.
    pub violation_limit: Option<f64>,
    /// Number of step QPs to keep in the report, `0..=QP_TRACE_CAP`.
    pub record_qp_limit: usize,
    /// Keep one [`IterationRecord`] per iteration in the report.
    pub record_history: bool,
    /// Accepted restoration steps allowed across the solve; zero disables it.
    pub max_restoration_steps: u64,
    /// Infinity-norm trust bound on each restoration direction, in x units.
    pub restoration_step_limit: f64,
    /// Positive diagonal regularization of the restoration Gauss–Newton model.
    pub restoration_regularization: f64,
}

impl Default for SqpOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_backtracks: 25,
            max_callback_calls: 2000,
            time_limit_seconds: None,
            stationarity_tolerance: 1e-6,
            complementarity_tolerance: 1e-6,
            dual_sign_tolerance: 1e-8,
            dense_size_limit: DENSE_SIZE_CAP,
            initial_hessian_scale: 1.0,
            hessian_block_size: 0,
            max_hessian_resets: 3,
            qp_tolerance: 1e-10,
            qp_max_iterations: 200,
            qp_reuse_workspace: false,
            qp_warm_start: true,
            elastic_mode: false,
            elastic_weight: 1e4,
            second_order_correction: false,
            hessian_update: HessianUpdate::Damped,
            merit: Merit::L1,
            scaling: Scaling::None,
            violation_limit: None,
            record_qp_limit: 0,
            record_history: false,
            max_restoration_steps: 10,
            restoration_step_limit: 1.0,
            restoration_regularization: 1e-4,
        }
    }
}

impl SqpOptions {
    /// Rejects non-positive or non-finite tolerances and out-of-range caps.
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("stationarity_tolerance", self.stationarity_tolerance),
            ("complementarity_tolerance", self.complementarity_tolerance),
            ("dual_sign_tolerance", self.dual_sign_tolerance),
            ("initial_hessian_scale", self.initial_hessian_scale),
            ("qp_tolerance", self.qp_tolerance),
            ("elastic_weight", self.elastic_weight),
            ("restoration_step_limit", self.restoration_step_limit),
            (
                "restoration_regularization",
                self.restoration_regularization,
            ),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidInput(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if let Some(s) = self.time_limit_seconds {
            if !(s.is_finite() && s >= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "time_limit_seconds must be finite and nonnegative, got {s}"
                )));
            }
        }
        if !(1..=DENSE_SIZE_CAP).contains(&self.dense_size_limit) {
            return Err(Error::InvalidInput(format!(
                "dense_size_limit must be in 1..={DENSE_SIZE_CAP}, got {}",
                self.dense_size_limit
            )));
        }
        if self.hessian_block_size > DENSE_SIZE_CAP {
            return Err(Error::InvalidInput(
                "hessian_block_size must lie in 0..=256".into(),
            ));
        }
        if let Some(limit) = self.violation_limit {
            if !(limit.is_finite() && limit >= 1.0) {
                return Err(Error::InvalidInput(format!(
                    "violation_limit must be finite and at least 1, got {limit}"
                )));
            }
        }
        if self.second_order_correction && self.merit == Merit::AugmentedLagrangian {
            return Err(Error::InvalidInput(
                "second_order_correction applies to the l1 merit; the augmented-Lagrangian \
                 merit is differentiable and needs no correction"
                    .into(),
            ));
        }
        if self.record_qp_limit > QP_TRACE_CAP {
            return Err(Error::InvalidInput(format!(
                "record_qp_limit must be at most {QP_TRACE_CAP}, got {}",
                self.record_qp_limit
            )));
        }
        Ok(())
    }
}

/// The native SQP solver with QP backend `B`.
#[derive(Clone, Debug, Default)]
pub struct NativeSqp<B> {
    /// QP subproblem backend.
    pub qp: B,
    /// Driver settings.
    pub settings: SqpOptions,
}

fn finite(v: f64, what: &str) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::Evaluation(format!("{what} became nonfinite")))
    }
}

fn dot(a: &[f64], b: &[f64]) -> Result<f64> {
    let mut v = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        v = finite(v + finite(a * b, "dot product")?, "dot product")?;
    }
    Ok(v)
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

fn record_qp_work(run: &mut sqp_craft::RunReport, qp: &sqp_craft::qp::QpReport) {
    run.qp_setups += qp.setups;
    run.qp_updates += qp.updates;
    run.qp_cold_retries += qp.cold_retries;
    if qp.retry_reason.is_some() {
        run.qp_retry_reason.clone_from(&qp.retry_reason);
    }
    run.qp_conversion_seconds += qp.conversion_seconds;
    run.qp_setup_seconds += qp.setup_seconds;
    run.qp_update_seconds += qp.update_seconds;
    run.qp_solve_seconds += qp.solve_seconds;
    run.qp_check_seconds += qp.check_seconds;
}

/// `Σ weights_i · (max(0, lower_i − v_i) + max(0, v_i − upper_i))` over finite sides.
fn weighted_violation(bounds: &Bounds, values: &[f64], weights: &[f64]) -> Result<f64> {
    let mut total = 0.0;
    for (i, &v) in values.iter().enumerate() {
        let mut row = 0.0;
        if bounds.lower[i].is_finite() {
            row += finite(bounds.lower[i] - v, "merit violation")?.max(0.0);
        }
        if bounds.upper[i].is_finite() {
            row += finite(v - bounds.upper[i], "merit violation")?.max(0.0);
        }
        total = finite(
            total + finite(weights[i] * row, "merit violation")?,
            "merit",
        )?;
    }
    Ok(total)
}

/// Violation of row `i` by `v`, zero inside the bounds.
fn row_violation(bounds: &Bounds, i: usize, v: f64) -> f64 {
    (bounds.lower[i] - v).max(v - bounds.upper[i]).max(0.0)
}

/// Whether some row of `values` violates its bounds by more than its limit.
fn beyond_limits(bounds: &Bounds, values: &[f64], limits: &[f64]) -> bool {
    values
        .iter()
        .zip(limits)
        .enumerate()
        .any(|(i, (&v, &limit))| row_violation(bounds, i, v) > limit)
}

/// `pᵀBp`.
fn curvature_along(b: &DMatrix<f64>, p: &[f64]) -> Result<f64> {
    let pv = nalgebra::DVectorView::from_slice(p, p.len());
    finite((b * pv).dot(&pv), "step curvature")
}

/// Model calls with budget, cancellation and timing bookkeeping.
struct Calls<'a> {
    began: Instant,
    settings: &'a SqpOptions,
    cancel: &'a CancellationToken,
    counts: EvaluationCounts,
    seconds: f64,
    gradient_scratch: Vec<f64>,
    jacobian_scratch: Vec<f64>,
}

impl Calls<'_> {
    fn recycle(&mut self, e: Evaluation) {
        self.gradient_scratch = e.gradient;
        self.jacobian_scratch = e.jacobian_values;
    }
    fn stop(&self) -> Option<Status> {
        if self.cancel.is_cancelled() {
            Some(Status::Cancelled)
        } else if self
            .settings
            .time_limit_seconds
            .is_some_and(|s| self.began.elapsed().as_secs_f64() >= s)
        {
            Some(Status::TimeLimit)
        } else {
            None
        }
    }

    fn before_call(&self) -> Option<Status> {
        self.stop().or_else(|| {
            (self.counts.functions + self.counts.derivatives >= self.settings.max_callback_calls)
                .then_some(Status::EvaluationLimit)
        })
    }

    fn functions(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        spec.check_point(x, "callback point")?;
        let mut c = vec![f64::NAN; spec.m()];
        self.counts.functions += 1;
        let t = Instant::now();
        let f = model.functions(x, &mut c);
        self.seconds += t.elapsed().as_secs_f64();
        let f = finite(f?, "objective")?;
        for &v in &c {
            finite(v, "constraint value")?;
        }
        Ok((f, c))
    }

    fn derivatives(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x: &[f64],
        (f, c): (f64, Vec<f64>),
        tolerance: f64,
    ) -> Result<Evaluation> {
        spec.check_point(x, "callback point")?;
        let mut g = std::mem::take(&mut self.gradient_scratch);
        let mut j = std::mem::take(&mut self.jacobian_scratch);
        g.resize(spec.n(), f64::NAN);
        g.fill(f64::NAN);
        j.resize(spec.nnz(), f64::NAN);
        j.fill(f64::NAN);
        self.counts.derivatives += 1;
        let t = Instant::now();
        let result = model.derivatives(x, &mut g, &mut j);
        self.seconds += t.elapsed().as_secs_f64();
        if let Err(err) = result {
            self.gradient_scratch = g;
            self.jacobian_scratch = j;
            return Err(err);
        }
        for &v in g.iter().chain(&j) {
            finite(v, "derivative value")?;
        }
        let primal = spec
            .variables()
            .checked_violation(x)?
            .max(spec.constraints().checked_violation(&c)?);
        Ok(Evaluation {
            objective: f,
            constraints: c,
            gradient: g,
            jacobian_values: j,
            primal_violation: primal,
            feasible_candidate: primal <= tolerance,
        })
    }
}

/// Bounds shifted to step coordinates: `bound − value` on finite sides.
fn shifted(bounds: &Bounds, values: &[f64]) -> Result<Bounds> {
    let shift = |side: &[f64]| -> Result<Vec<f64>> {
        side.iter()
            .zip(values)
            .map(|(&b, &v)| {
                if b.is_finite() {
                    finite(b - v, "shifted bound")
                } else {
                    Ok(b)
                }
            })
            .collect()
    };
    Ok(Bounds::new(shift(&bounds.lower)?, shift(&bounds.upper)?))
}

/// Moves step components that exceed their QP step bounds by at most
/// `tolerance` (in original units: `tolerance / scale_j` for a step in
/// coordinates scaled by `scale`) onto the bound, counting each move.
///
/// Interior-point solutions satisfy bounds only to their tolerance. Near an
/// active variable bound the remaining room `u − x` can be far smaller than
/// that tolerance (it was 3e-14 in the case that exposed this), so an
/// overshoot of 1e-11 would cut the step length for every variable to almost
/// zero. Larger violations are left for the acceptance check to reject.
fn snap_step(
    step: &mut [f64],
    step_bounds: &Bounds,
    variables: &Bounds,
    tolerance: f64,
    scale: &[f64],
    report: &mut sqp_craft::RunReport,
) {
    for (j, p) in step.iter_mut().enumerate() {
        let (lo, up) = (step_bounds.lower[j], step_bounds.upper[j]);
        let tolerance = tolerance / scale[j];
        let target = if *p < lo && lo - *p <= tolerance {
            lo
        } else if *p > up && *p - up <= tolerance {
            up
        } else {
            continue;
        };
        *p = target;
        if variables.lower[j] == variables.upper[j] {
            report.fixed_step_corrections += 1;
        } else {
            report.bound_step_corrections += 1;
        }
    }
}

/// Largest `α ≤ 1` with `x + αp` inside the variable box.
fn box_step_limit(bounds: &Bounds, x: &[f64], p: &[f64]) -> f64 {
    let mut alpha = 1.0_f64;
    for j in 0..x.len() {
        if p[j] > 0.0 && bounds.upper[j].is_finite() {
            alpha = alpha.min((bounds.upper[j] - x[j]) / p[j]);
        }
        if p[j] < 0.0 && bounds.lower[j].is_finite() {
            alpha = alpha.min((bounds.lower[j] - x[j]) / p[j]);
        }
    }
    alpha
}

/// `x + αp`, where a component that floating-point rounding of this sum
/// would place beyond a bound (by at most a few ulps) is set to the bound.
/// A step that lands exactly on a bound, `p_j = u_j − x_j` with `α = 1`, can
/// otherwise round one ulp outside and fail the box contract.
fn trial_point(bounds: &Bounds, x: &[f64], p: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(p)
        .enumerate()
        .map(|(j, (&x, &p))| {
            let t = x + alpha * p;
            let rounding = 4.0 * f64::EPSILON * (x.abs() + (alpha * p).abs());
            let (lo, up) = (bounds.lower[j], bounds.upper[j]);
            if t > up && t - up <= rounding {
                up
            } else if t < lo && lo - t <= rounding {
                lo
            } else {
                t
            }
        })
        .collect()
}

/// The step QP at `x` with model values `e` and Hessian approximation `b`.
#[cfg(test)]
fn qp_model(spec: &ProblemSpec, x: &[f64], e: &Evaluation, b: &DMatrix<f64>) -> Result<QpModel> {
    qp_model_with_blocks(spec, x, e, b, 0)
}
fn qp_model_with_blocks(
    spec: &ProblemSpec,
    x: &[f64],
    e: &Evaluation,
    b: &DMatrix<f64>,
    block: usize,
) -> Result<QpModel> {
    let n = spec.n();
    let mut indptr = vec![0];
    let capacity = n * (if block == 0 { n } else { block.min(n) } + 1) / 2;
    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    for i in 0..n {
        let first = i.checked_div(block).unwrap_or(0) * block;
        for j in first..=i {
            indices.push(j);
            values.push(b[(i, j)]);
        }
        indptr.push(indices.len());
    }
    Ok(QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr,
                indices,
            },
            values,
        },
        gradient: e.gradient.clone(),
        variables: shifted(spec.variables(), x)?,
        matrix: CsrMatrix {
            pattern: spec.jacobian().clone(),
            values: e.jacobian_values.clone(),
        },
        rows: shifted(spec.constraints(), &e.constraints)?,
    })
}

/// Refresh numeric data in a per-solve, fixed-structure QP template.
fn refresh_qp(
    q: &mut QpModel,
    spec: &ProblemSpec,
    x: &[f64],
    e: &Evaluation,
    b: &DMatrix<f64>,
) -> Result<()> {
    let h = &mut q.hessian_lower;
    for i in 0..spec.n() {
        for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
            h.values[k] = b[(i, h.pattern.indices[k])];
        }
    }
    q.gradient.clone_from(&e.gradient);
    q.matrix.values.clone_from(&e.jacobian_values);
    for (dst, src, values) in [
        (&mut q.variables, spec.variables(), x),
        (&mut q.rows, spec.constraints(), e.constraints.as_slice()),
    ] {
        for (out, input) in [(&mut dst.lower, &src.lower), (&mut dst.upper, &src.upper)] {
            for (i, v) in out.iter_mut().enumerate() {
                *v = if input[i].is_finite() {
                    finite(input[i] - values[i], "shifted bound")?
                } else {
                    input[i]
                };
            }
        }
    }
    Ok(())
}

/// `J(x⁺) − J(x)` in pattern order and the departure from linearity
/// `d_L = c(x⁺) − c(x) − J(x)s` for [`bfgs::Repair`].
fn curvature_repair(
    spec: &ProblemSpec,
    e: &Evaluation,
    next: &Evaluation,
    s: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let change = next
        .jacobian_values
        .iter()
        .zip(&e.jacobian_values)
        .map(|(&a, &b)| finite(a - b, "Jacobian change"))
        .collect::<Result<Vec<f64>>>()?;
    let mut departure = vec![0.0; spec.m()];
    spec.jacobian()
        .multiply(&e.jacobian_values, s, &mut departure);
    for (d, (&after, &before)) in departure
        .iter_mut()
        .zip(next.constraints.iter().zip(&e.constraints))
    {
        *d = finite(after - before - *d, "departure from linearity")?;
    }
    Ok((change, departure))
}

/// `∇f − Jᵀλ − z`.
fn lagrangian_gradient(spec: &ProblemSpec, e: &Evaluation, m: &Multipliers) -> Result<Vec<f64>> {
    let mut jt = vec![0.0; spec.n()];
    spec.jacobian()
        .multiply_transpose(&e.jacobian_values, &m.constraints, &mut jt);
    e.gradient
        .iter()
        .zip(jt)
        .zip(&m.variables)
        .map(|((&g, j), &z)| finite(g - j - z, "Lagrangian gradient"))
        .collect()
}

/// Stores KKT residuals of the current candidate for multipliers `m`.
///
/// Any multiplier vector gives an upper bound on the best first-order residual
/// at the point, so this is meaningful even when `m` comes from the QP solved
/// at the previous iterate.
/// Rounding levels of the stationarity and complementarity residuals at `x`
/// with multipliers `m`: what floating-point evaluation of
/// `∇f − Jᵀλ − z` and of `|μ|·distance` can contribute on its own. Entry `j`
/// of the stationarity sums `k_j + 2` terms (`k_j` rows use variable `j`),
/// so its error is at most about `(k_j + 2) ε (|∇f_j| + Σ_i |J_ij λ_i| + |z_j|)`;
/// a distance `v − b` is exact to `ε (|v| + |b|)`.
///
/// A residual within its tolerance only by less than this level is not
/// evidence of a first-order point: with multipliers of 5·10⁹ (an
/// interior-point QP on a degenerate problem) the stationarity residual is
/// pure rounding at 10⁻⁶.
fn rounding_levels(spec: &ProblemSpec, x: &[f64], e: &Evaluation, m: &Multipliers) -> (f64, f64) {
    let n = spec.n();
    let mut size: Vec<f64> = (0..n)
        .map(|j| e.gradient[j].abs() + m.variables[j].abs())
        .collect();
    let mut terms = vec![2.0_f64; n];
    let jacobian = spec.jacobian();
    for (i, &lambda) in m.constraints.iter().enumerate() {
        for a in jacobian.indptr[i]..jacobian.indptr[i + 1] {
            let j = jacobian.indices[a];
            size[j] += (e.jacobian_values[a] * lambda).abs();
            terms[j] += 1.0;
        }
    }
    let stationarity = size
        .iter()
        .zip(&terms)
        .fold(0.0_f64, |a, (s, t)| a.max(t * f64::EPSILON * s));
    let mut complementarity = 0.0_f64;
    for (values, bounds, mu) in [
        (&e.constraints[..], spec.constraints(), &m.constraints[..]),
        (x, spec.variables(), &m.variables[..]),
    ] {
        for (i, (&v, &mu)) in values.iter().zip(mu).enumerate() {
            let bound = if mu > 0.0 {
                bounds.lower[i]
            } else {
                bounds.upper[i]
            };
            if mu != 0.0 && bound.is_finite() {
                complementarity =
                    complementarity.max(mu.abs() * f64::EPSILON * (v.abs() + bound.abs()));
            }
        }
    }
    (stationarity, complementarity)
}

fn store_residuals(spec: &ProblemSpec, r: &mut SolveResult, m: Multipliers) -> Result<()> {
    let (Some(x), Some(e)) = (r.candidate.as_ref(), r.evaluation.as_ref()) else {
        return Ok(());
    };
    let k = nlp_residuals(spec, x, e, &m)?;
    r.stationarity = Some(k.stationarity);
    r.complementarity = Some(k.complementarity);
    r.dual_sign = Some(k.dual_sign);
    r.multipliers = Some(m);
    Ok(())
}

fn noise_message(predicted: f64, noise: f64) -> String {
    format!(
        "the QP step is below the QP solution accuracy (predicted merit decrease \
         {predicted:.3e}, QP accuracy {noise:.3e}); residuals at the returned point are reported"
    )
}

fn status_message(status: Status) -> &'static str {
    match status {
        Status::Converged => "first-order KKT residuals within tolerance at the returned point",
        Status::IterationLimit => "iteration limit reached",
        Status::EvaluationLimit => "callback budget exhausted; returning the last complete point",
        Status::TimeLimit => "time limit reached; returning the last complete point",
        Status::Cancelled => "cancelled",
        Status::LinearizationInfeasible => {
            "the linearized constraints are inconsistent at this point; the nonlinear problem \
             may still be feasible (restoration is disabled)"
        }
        Status::RestorationFailed => {
            "local restoration failed; nonlinear infeasibility is not established"
        }
        Status::RestorationLimit => "feasibility restoration step budget exhausted",
        Status::LocallyInfeasible => {
            "elastic mode stopped at a stationary point of the l1 penalty with violated \
             constraints at its largest weight; nonlinear infeasibility is not established"
        }
        Status::LinearInfeasible => {
            "the rows declared linear and the variable bounds have no common point (the \
             proximal-point QP is infeasible)"
        }
        Status::LineSearchFailure => "no acceptable step length within the backtracking limit",
        Status::NoDescent => "QP step is not a descent direction for the merit function",
        Status::Stalled => "QP step too small to change the iterate",
        Status::NumericalFailure => "Hessian reset budget exhausted",
        Status::QpFailure => "QP subproblem not solved acceptably",
        _ => "",
    }
}

impl<B: QpBackend> SqpSolver for NativeSqp<B> {
    fn solve(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x0: &[f64],
        options: &Options,
        cancel: &CancellationToken,
        mut observer: Option<&mut dyn Observer>,
    ) -> Result<SolveResult> {
        options.validate()?;
        self.settings.validate()?;
        self.qp.set_threads(options.threads)?;
        let execution =
            sqp_craft_parallel::Execution::new(options.threads).map_err(Error::Unsupported)?;
        if self.settings.elastic_mode && !self.qp.supports_elastic() {
            return Err(Error::Unsupported(
                "elastic_mode needs a QP backend with elastic support, such as ActiveSetQp".into(),
            ));
        }
        spec.check_point(x0, "start point")?;
        if spec.n() > self.settings.dense_size_limit {
            return Err(Error::Unsupported(format!(
                "n = {} exceeds the dense pilot limit of {} variables",
                spec.n(),
                self.settings.dense_size_limit
            )));
        }
        let settings = &self.settings;
        let qp_acceptance = options
            .feasibility_tolerance
            .min(settings.stationarity_tolerance)
            .min(settings.complementarity_tolerance);
        let mut calls = Calls {
            began: Instant::now(),
            settings,
            cancel,
            counts: EvaluationCounts::default(),
            seconds: 0.0,
            gradient_scratch: Vec::new(),
            jacobian_scratch: Vec::new(),
        };
        let mut r = SolveResult::empty(Status::IterationLimit, "");
        r.report.method = "sqp".into();

        // Start point: a complete evaluation or nothing.
        let mut status = 'start: {
            if let Some(status) = calls.before_call() {
                break 'start Some(status);
            }
            let values = calls.functions(spec, model, x0)?;
            if let Some(status) = calls.before_call() {
                break 'start Some(status);
            }
            let evaluation =
                calls.derivatives(spec, model, x0, values, options.feasibility_tolerance)?;
            r.candidate = Some(x0.to_vec());
            r.evaluation = Some(evaluation);
            None
        };
        // Rows declared linear: their Jacobian at the start, against which
        // every later evaluation is checked, and a start on those rows.
        let linear_reference = r
            .evaluation
            .as_ref()
            .filter(|_| spec.has_linear_rows())
            .map(|e| e.jacobian_values.clone());
        // Scales of the rows declared linear (constant), used from the
        // projection on; in Jacobian mode recomputed below from every row.
        let mut scales = match (&r.evaluation, settings.scaling) {
            (Some(e), Scaling::Linear | Scaling::Jacobian) => {
                Scales::of(Scaling::Linear, spec, &e.jacobian_values)
            }
            _ => Scales::identity(spec.n(), spec.m()),
        };
        if status.is_none()
            && linear_reference.is_some()
            && linear::violated(
                spec,
                r.evaluation.as_ref().expect("evaluated"),
                options.feasibility_tolerance,
            )
        {
            status = linear::project(
                &mut self.qp,
                spec,
                model,
                &mut calls,
                &mut r,
                options,
                qp_acceptance,
                &scales,
            )?;
            if let (Some(reference), Some(e)) = (&linear_reference, &r.evaluation) {
                linear::check_constant(spec, reference, e)?;
            }
        }
        // SNOPT's Scale option 2 uses the first point on the linear rows.
        if let (Scaling::Jacobian, Some(e)) = (settings.scaling, &r.evaluation) {
            scales = Scales::of(Scaling::Jacobian, spec, &e.jacobian_values);
        }
        let scaled_identity = scales.is_identity();
        if settings.scaling != Scaling::None {
            r.report.variable_scales.clone_from(&scales.variables);
            r.report.row_scales.clone_from(&scales.rows);
        }
        // In elastic mode the rows declared linear stay hard.
        let hard_rows = if settings.elastic_mode && spec.has_linear_rows() {
            spec.linear_rows().to_vec()
        } else {
            Vec::new()
        };

        let n = spec.n();
        let mut b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
        let mut effective_block = if settings.hessian_block_size >= n {
            0
        } else {
            settings.hessian_block_size
        };
        let mut bfgs_work = bfgs::Curvature::new(n, effective_block);
        bfgs_work.set_execution(&execution);
        let mut qp_work: Option<QpModel> = None;
        // The last step QP's working set, if the backend reports one.
        let mut qp_warm: Option<QpWarmStart> = None;
        // Elastic mode: the weight γ, scaled by the start gradient.
        let mut elastic_weight = r.evaluation.as_ref().and_then(|e| {
            settings
                .elastic_mode
                .then(|| settings.elastic_weight * norm_inf(&e.gradient).max(1.0))
        });
        let mut relaxed = Bounds::free(0);
        let mut transpose = Vec::new();
        let mut qp_checks = sqp_craft::qp::QpResidualWorkspace::default();
        let mut penalties = vec![0.0; spec.m()];
        let mut have_penalties = false;
        let mut augmented = merit::AugmentedLagrangian::new(spec.m());
        // Rows kept out of the merit (linear rows kept feasible).
        let skip_rows = spec.linear_rows().to_vec();
        // SNOPT's violation limit, fixed at the start point.
        let limits: Option<Vec<f64>> = match (settings.violation_limit, r.evaluation.as_ref()) {
            (Some(tau), Some(e)) => Some(
                (0..spec.m())
                    .map(|i| tau * row_violation(spec.constraints(), i, e.constraints[i]).max(1.0))
                    .collect(),
            ),
            _ => None,
        };
        // The last step QP left rows violated (elastic mode).
        let mut elastic_phase = false;
        let mut retry_descent = false;
        let mut noise_steps = 0_u32;
        let mut notify = true;
        while status.is_none() {
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            if notify {
                if let Some(obs) = observer.as_deref_mut() {
                    let e = r.evaluation.as_ref().expect("evaluated");
                    let progress = Progress {
                        iteration: r.iterations,
                        objective: e.objective,
                        primal_violation: e.primal_violation,
                    };
                    if obs.observe(&progress)? == Control::Stop {
                        status = Some(Status::Cancelled);
                        r.message = "stopped by the observer".into();
                        break;
                    }
                }
                if let Some(s) = calls.stop() {
                    status = Some(s);
                    break;
                }
            }
            if r.iterations >= settings.max_iterations {
                status = Some(Status::IterationLimit);
                break;
            }

            // 1. Step QP.
            let x = r.candidate.as_ref().expect("evaluated");
            let e = r.evaluation.as_ref().expect("evaluated");
            if let Some(q) = qp_work.as_mut() {
                refresh_qp(q, spec, x, e, &b)?;
            } else {
                qp_work = Some(qp_model_with_blocks(spec, x, e, &b, effective_block)?);
            }
            if !scaled_identity {
                // B is kept in scaled coordinates; scale the rest.
                scales.scale_qp(qp_work.as_mut().expect("QP initialized"), &scales.rows);
            }
            let qp = qp_work.as_ref().expect("QP initialized");
            if r.report.qp_trace.len() < settings.record_qp_limit {
                r.report.qp_trace.push((*qp).clone());
            }
            let remaining = settings
                .time_limit_seconds
                .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0));
            self.qp.configure(&QpOptions {
                reuse_workspace: settings.qp_reuse_workspace,
                tolerance: settings.qp_tolerance,
                acceptance_tolerance: qp_acceptance,
                max_iterations: settings.qp_max_iterations,
                time_limit_seconds: remaining,
                elastic_weight,
                hard_rows: hard_rows.clone(),
            })?;
            let t = Instant::now();
            r.report.qp_solves += 1;
            let outcome = self.qp.solve(qp, qp_warm.as_ref(), cancel);
            r.report.qp_seconds += t.elapsed().as_secs_f64();
            let report = self.qp.report();
            record_qp_work(&mut r.report, &report);
            r.report.qp_backend = report.backend;
            r.report.qp_status = report.raw_status;
            r.report.qp_iterations += report.iterations;
            let qp_iterations = report.iterations;
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            let mut solution: QpSolution = match outcome {
                Ok(QpOutcome::Optimal(s)) => s,
                Ok(QpOutcome::Infeasible) => {
                    if settings.max_restoration_steps == 0 {
                        status = Some(Status::LinearizationInfeasible);
                        break;
                    }
                    if r.report.restoration_steps >= settings.max_restoration_steps {
                        status = Some(Status::RestorationLimit);
                        break;
                    }
                    let began = Instant::now();
                    let restored = restoration::step(
                        &mut self.qp,
                        spec,
                        model,
                        &mut calls,
                        &mut r,
                        options,
                        &scales,
                    );
                    r.report.restoration_seconds += began.elapsed().as_secs_f64();
                    status = restored?;
                    if let (Some(reference), Some(e)) = (&linear_reference, &r.evaluation) {
                        linear::check_constant(spec, reference, e)?;
                    }
                    if status.is_some() {
                        break;
                    }
                    // Feasibility steps do not carry an NLP multiplier estimate.
                    // Restart curvature/penalties without spending the BFGS failure budget.
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    penalties.fill(0.0);
                    have_penalties = false;
                    augmented.reset();
                    retry_descent = false;
                    noise_steps = 0;
                    notify = true;
                    continue;
                }
                Ok(QpOutcome::Cancelled) => {
                    status = Some(Status::Cancelled);
                    break;
                }
                Ok(QpOutcome::LimitReached(_)) => {
                    let reason = report.limit_reason.unwrap_or_else(|| "unknown".into());
                    status = Some(if reason == "time" {
                        Status::TimeLimit
                    } else {
                        Status::QpFailure
                    });
                    r.message = format!("QP subproblem stopped at its {reason} limit");
                    break;
                }
                Ok(QpOutcome::Unbounded) => {
                    status = Some(Status::QpFailure);
                    r.message = "QP subproblem reported unbounded although B is positive \
                                 definite; numerical trouble"
                        .into();
                    break;
                }
                Ok(QpOutcome::NotPositiveDefinite) => {
                    r.report.qp_not_positive_definite += 1;
                    // Damping keeps B positive definite in exact arithmetic,
                    // but rounding in the explicit update can leave it
                    // indefinite. The backend's exact check is the only
                    // positive-definiteness test of B: treat its rejection
                    // like a failed update (reset, or stop without budget)
                    // and re-solve this QP without counting an iteration.
                    if r.report.hessian_resets >= u64::from(settings.max_hessian_resets) {
                        status = Some(Status::NumericalFailure);
                        r.message = "the Hessian approximation is not positive definite and \
                                     the Hessian reset budget is exhausted"
                            .into();
                        break;
                    }
                    if effective_block > 0 {
                        effective_block = 0;
                        bfgs_work = bfgs::Curvature::new(n, 0);
                        bfgs_work.set_execution(&execution);
                        qp_work = None;
                        r.report.block_hessian_fallbacks += 1;
                    }
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    r.report.hessian_resets += 1;
                    retry_descent = false;
                    notify = false;
                    continue;
                }
                Ok(other) => {
                    status = Some(Status::QpFailure);
                    r.message = format!("unexpected QP outcome {other:?}");
                    break;
                }
                Err(Error::Backend(message)) => {
                    status = Some(Status::QpFailure);
                    r.message = message;
                    break;
                }
                Err(err) => return Err(err),
            };
            if settings.qp_warm_start {
                if let Some(active_set) = solution.active_set.take() {
                    qp_warm = Some(QpWarmStart {
                        active_set: Some(active_set),
                        ..Default::default()
                    });
                }
            }
            // Put tolerance-level overshoots of the step bounds (including the
            // zero-width bounds of fixed variables) on the bound; recheck below.
            snap_step(
                &mut solution.step,
                &qp.variables,
                spec.variables(),
                options.feasibility_tolerance,
                &scales.variables,
                &mut r.report,
            );
            let check = match elastic_weight {
                Some(weight) => qp.accepts_elastic_with_hard_rows(
                    &solution,
                    qp_acceptance,
                    weight,
                    &hard_rows,
                    &mut qp_checks,
                )?,
                None => qp.accepts_with_workspace(&solution, qp_acceptance, &mut qp_checks)?,
            };
            if !check.accepted {
                status = Some(Status::QpFailure);
                r.message = format!(
                    "QP solution failed the independent check (scale {:.3e}): {:?}",
                    check.scale, check.residuals
                );
                break;
            }
            if !scaled_identity {
                scales.unscale_solution(&mut solution, &scales.rows);
            }

            // 2. Convergence test at x_k with the fresh multipliers.
            store_residuals(spec, &mut r, solution.multipliers.clone())?;
            let e = r.evaluation.as_ref().expect("evaluated");
            let (stationarity, complementarity) = (
                r.stationarity.expect("stored"),
                r.complementarity.expect("stored"),
            );
            let mut record = IterationRecord {
                iteration: r.iterations,
                objective: e.objective,
                primal_violation: e.primal_violation,
                stationarity,
                complementarity,
                qp_step_norm: norm_inf(&solution.step),
                step_length: None,
                trials: 0,
                max_penalty: 0.0,
                qp_iterations,
                second_order_correction: false,
                curvature: None,
            };
            // The residuals must clear their tolerances by their own rounding
            // level, or the claim rests on rounding noise.
            let (stationarity_level, complementarity_level) = rounding_levels(
                spec,
                r.candidate.as_ref().expect("evaluated"),
                e,
                r.multipliers.as_ref().expect("stored"),
            );
            if e.primal_violation <= options.feasibility_tolerance
                && stationarity + stationarity_level <= settings.stationarity_tolerance
                && complementarity + complementarity_level <= settings.complementarity_tolerance
                && r.dual_sign.expect("stored") <= settings.dual_sign_tolerance
            {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::Converged);
                break;
            }
            let x = r.candidate.as_ref().expect("evaluated");
            let p = &solution.step;
            let mut linear = vec![0.0; spec.m()];
            spec.jacobian().multiply(&e.jacobian_values, p, &mut linear);
            for (l, &c) in linear.iter_mut().zip(&e.constraints) {
                *l = finite(*l + c, "linearized constraints")?;
            }
            // Rows the elastic QP leaves violated.
            let rows = spec.constraints();
            let elastic_rows: Vec<usize> = match elastic_weight {
                Some(_) => (0..spec.m())
                    .filter(|&i| {
                        !spec.linear_rows()[i]
                            && (rows.lower[i] - linear[i] > options.feasibility_tolerance
                                || linear[i] - rows.upper[i] > options.feasibility_tolerance)
                    })
                    .collect(),
                None => Vec::new(),
            };
            if let Some(weight) = elastic_weight.filter(|_| {
                !elastic_rows.is_empty() && e.primal_violation > options.feasibility_tolerance
            }) {
                // At a stationary point of f + γ·violation with violated rows
                // a larger weight may still reach feasibility; past the last
                // increase the point is reported as locally infeasible.
                let weights: Vec<f64> = scales.rows.iter().map(|&r| weight * r).collect();
                let k = sqp_craft::kkt::elastic_residuals_with_row_weights(
                    &sqp_craft::kkt::KktPoint {
                        x,
                        variable_bounds: spec.variables(),
                        rows: &e.constraints,
                        row_bounds: rows,
                        gradient: &e.gradient,
                        jacobian: spec.jacobian(),
                        jacobian_values: &e.jacobian_values,
                    },
                    &solution.multipliers,
                    &weights,
                    &hard_rows,
                    options.feasibility_tolerance,
                    &mut relaxed,
                    &mut transpose,
                )?;
                // A negligible step says the same as small residuals: the QP
                // finds x stationary for the penalty.
                let scale = 1.0 + weight * scales.largest_row();
                let tiny = norm_inf(p) <= 1e-14 * (1.0 + norm_inf(x));
                if tiny
                    || k.stationarity <= settings.stationarity_tolerance
                        && k.complementarity <= settings.complementarity_tolerance * scale
                        && k.dual_sign <= settings.dual_sign_tolerance * scale
                {
                    if r.report.elastic_weight_increases < MAX_ELASTIC_INCREASES {
                        elastic_weight = Some(10.0 * weight);
                        r.report.elastic_weight_increases += 1;
                        retry_descent = false;
                        notify = false;
                        continue;
                    }
                    if settings.record_history {
                        r.report.history.push(record);
                    }
                    status = Some(Status::LocallyInfeasible);
                    break;
                }
            }

            // 3. Merit line search.
            if elastic_phase && elastic_rows.is_empty() {
                // The linearization is consistent again. Penalties of γ from
                // the elastic phase would shorten every following step (as
                // after restoration, start Powell's rule afresh).
                have_penalties = false;
            }
            elastic_phase = !elastic_rows.is_empty();
            // The augmented-Lagrangian merit, unless the elastic QP left rows
            // violated: its step then minimizes the model of f + γ·violation.
            let search = if settings.merit == Merit::AugmentedLagrangian && elastic_rows.is_empty()
            {
                Some(augmented.search(
                    spec.constraints(),
                    e.objective,
                    &e.constraints,
                    &linear,
                    dot(&e.gradient, p)?,
                    curvature_along(&b, &scales.to_scaled(p))?,
                    &solution.multipliers.constraints,
                    &skip_rows,
                )?)
            } else {
                None
            };
            let (merit, predicted, sensitivity) = if let Some(search) = &search {
                record.max_penalty = search.max_penalty;
                (search.value, search.predicted, search.sensitivity)
            } else {
                for (mu, &lambda) in penalties.iter_mut().zip(&solution.multipliers.constraints) {
                    let target = PENALTY_MARGIN * lambda.abs();
                    *mu = if have_penalties {
                        target.max(0.5 * (*mu + target))
                    } else {
                        target
                    };
                    finite(*mu, "penalty")?;
                }
                have_penalties = true;
                // Rows declared linear stay satisfied along the step.
                for (mu, &linear) in penalties.iter_mut().zip(spec.linear_rows()) {
                    if linear {
                        *mu = 0.0;
                    }
                }
                if let Some(weight) = elastic_weight {
                    // The elastic QP minimizes the model of f + γ·violation, so
                    // its step is a descent direction for that merit: rows it
                    // leaves violated (|λ_i| = γ) are weighted by exactly γ.
                    for &i in &elastic_rows {
                        penalties[i] = weight * scales.rows[i];
                    }
                }
                record.max_penalty = norm_inf(&penalties);
                let violation = weighted_violation(spec.constraints(), &e.constraints, &penalties)?;
                let linear_violation = weighted_violation(spec.constraints(), &linear, &penalties)?;
                let predicted = finite(
                    -dot(&e.gradient, p)? + (violation - linear_violation),
                    "predicted decrease",
                )?;
                (
                    finite(e.objective + violation, "merit")?,
                    predicted,
                    record.max_penalty,
                )
            };
            if norm_inf(p) <= 1e-14 * (1.0 + norm_inf(x)) {
                if settings.record_history {
                    r.report.history.push(record);
                }
                status = Some(Status::Stalled);
                break;
            }
            // `Some(accuracy)` when this step is taken although its predicted
            // decrease is non-positive within the QP's accuracy.
            let mut noise_step = None;
            if predicted <= 0.0 {
                // First retry once from B = scale·I: a badly scaled B (for
                // example nearly zero curvature) can make an accurate step tiny.
                if !retry_descent
                    && r.report.hessian_resets < u64::from(settings.max_hessian_resets)
                {
                    b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                    r.report.hessian_resets += 1;
                    retry_descent = true;
                    notify = false;
                    continue;
                }
                // With exact QP solutions the predicted decrease is at least
                // pᵀBp > 0. An interior-point QP leaves active rows a little
                // short of their bounds, which costs up to the QP's
                // complementarity residual per entry. A non-positive value
                // within that accuracy means the step is at the QP's noise
                // level. Near a solution the step is then still the right one
                // (pᵀBp ~ 1e-12 while stationarity is just above tolerance),
                // so take it if the merit does not rise by more than that
                // accuracy, a bounded number of times in a row.
                let qp = check.residuals;
                // The QP residuals are in scaled units.
                let noise = (spec.m() + n) as f64 * qp.complementarity
                    + qp.stationarity
                        * p.iter()
                            .zip(&scales.variables)
                            .map(|(v, d)| (v / d).abs())
                            .sum::<f64>()
                    + sensitivity / scales.smallest_row() * spec.m() as f64 * qp.primal;
                if -predicted <= noise && noise_steps < MAX_NOISE_STEPS {
                    noise_step = Some(noise);
                } else {
                    if settings.record_history {
                        r.report.history.push(record);
                    }
                    if -predicted <= noise {
                        status = Some(Status::Stalled);
                        r.message = noise_message(predicted, noise);
                    } else {
                        status = Some(Status::NoDescent);
                    }
                    break;
                }
            }
            // The augmented Lagrangian changes only to second order in the
            // violation. Near a solution its predicted decrease can fall below
            // the rounding of the merit value itself: an interior-point QP
            // that leaves a violation of 10⁻⁶ predicts 10⁻¹⁴ with f ≈ 20. Such
            // a step is taken at full length if the merit does not rise by
            // more than that rounding level, as a noise-level step.
            if let (Some(search), None) = (&search, noise_step) {
                let rounding = MERIT_ROUNDING * f64::EPSILON * search.magnitude;
                if predicted <= rounding && noise_steps < MAX_NOISE_STEPS {
                    noise_step = Some(rounding);
                }
            }
            if let Some(search) = &search {
                augmented.commit(search);
            }
            let mut alpha = box_step_limit(spec.variables(), x, p);
            if !(alpha.is_finite() && alpha > 0.0) {
                status = Some(Status::Stalled);
                r.message = "the QP step leaves the variable box immediately".into();
                break;
            }
            let mut accepted = None;
            let max_backtracks = if noise_step.is_some() {
                0
            } else {
                settings.max_backtracks
            };
            for _ in 0..=max_backtracks {
                if let Some(s) = calls.before_call() {
                    status = Some(s);
                    break;
                }
                let trial = trial_point(spec.variables(), x, p, alpha);
                if spec.check_point(&trial, "trial").is_err() {
                    // Roundoff pushed the trial out of the box: shorten without a call.
                    alpha *= 0.5;
                    r.report.rejected_trials += 1;
                    continue;
                }
                record.trials += 1;
                let values = match calls.functions(spec, model, &trial) {
                    Ok(v) => v,
                    Err(Error::UndefinedPoint(_)) => {
                        r.report.undefined_trials += 1;
                        r.report.rejected_trials += 1;
                        alpha *= 0.5;
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                if limits
                    .as_ref()
                    .is_some_and(|l| beyond_limits(spec.constraints(), &values.1, l))
                {
                    r.report.violation_limit_rejections += 1;
                    r.report.rejected_trials += 1;
                    alpha *= 0.5;
                    continue;
                }
                let trial_merit = match &search {
                    Some(search) => search.value_at(&augmented.pi, alpha, values.0, &values.1)?,
                    None => finite(
                        values.0 + weighted_violation(spec.constraints(), &values.1, &penalties)?,
                        "trial merit",
                    )?,
                };
                let threshold = match noise_step {
                    Some(noise) => merit + noise,
                    None => merit - ARMIJO * alpha * predicted,
                };
                if trial_merit > threshold {
                    r.report.rejected_trials += 1;
                    // Second-order correction: once, for a rejected full step
                    // that increased the constraint violation through the
                    // curvature of the constraints (for linear rows the
                    // corrected QP would return the same step).
                    let try_correction = settings.second_order_correction
                        && alpha == 1.0
                        && noise_step.is_none()
                        && !record.second_order_correction;
                    let shift: Vec<f64> = if try_correction {
                        values
                            .1
                            .iter()
                            .zip(&linear)
                            .zip(spec.linear_rows())
                            .map(|((c, l), &exact)| if exact { 0.0 } else { c - l })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let corrected = if try_correction
                        && norm_inf(&shift) > options.feasibility_tolerance
                        && spec.constraints().checked_violation(&values.1)? >= e.primal_violation
                    {
                        record.second_order_correction = true;
                        let q = qp_work.as_mut().expect("QP initialized");
                        let saved = q.rows.clone();
                        // Replace c(x) + Jp by c(x + p): the step must also
                        // absorb the curvature of the constraints along p.
                        for (i, &shift) in shift.iter().enumerate() {
                            let shift = finite(shift * scales.rows[i], "second-order shift")?;
                            q.rows.lower[i] -= shift;
                            q.rows.upper[i] -= shift;
                        }
                        let remaining = settings
                            .time_limit_seconds
                            .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0));
                        self.qp.configure(&QpOptions {
                            reuse_workspace: settings.qp_reuse_workspace,
                            tolerance: settings.qp_tolerance,
                            acceptance_tolerance: qp_acceptance,
                            max_iterations: settings.qp_max_iterations,
                            time_limit_seconds: remaining,
                            elastic_weight,
                            hard_rows: hard_rows.clone(),
                        })?;
                        let t = Instant::now();
                        r.report.qp_solves += 1;
                        let outcome = self.qp.solve(q, qp_warm.as_ref(), cancel);
                        r.report.qp_seconds += t.elapsed().as_secs_f64();
                        let report = self.qp.report();
                        record_qp_work(&mut r.report, &report);
                        r.report.qp_iterations += report.iterations;
                        let step = match outcome {
                            Ok(QpOutcome::Optimal(mut s)) => {
                                snap_step(
                                    &mut s.step,
                                    &q.variables,
                                    spec.variables(),
                                    options.feasibility_tolerance,
                                    &scales.variables,
                                    &mut r.report,
                                );
                                let check = match elastic_weight {
                                    Some(weight) => q.accepts_elastic_with_hard_rows(
                                        &s,
                                        qp_acceptance,
                                        weight,
                                        &hard_rows,
                                        &mut qp_checks,
                                    )?,
                                    None => {
                                        q.accepts_with_workspace(&s, qp_acceptance, &mut qp_checks)?
                                    }
                                };
                                if !scaled_identity {
                                    scales.unscale_solution(&mut s, &scales.rows);
                                }
                                check.accepted.then_some(s.step)
                            }
                            // No correction; a stop is noticed before the
                            // next call.
                            Ok(_) | Err(Error::Backend(_)) => None,
                            Err(err) => return Err(err),
                        };
                        q.rows = saved;
                        step
                    } else {
                        None
                    };
                    if let Some(step) = corrected {
                        if let Some(s) = calls.before_call() {
                            status = Some(s);
                            break;
                        }
                        let trial = trial_point(spec.variables(), x, &step, 1.0);
                        if spec.check_point(&trial, "trial").is_ok() {
                            record.trials += 1;
                            r.report.second_order_corrections += 1;
                            let values = match calls.functions(spec, model, &trial) {
                                Ok(v) => Some(v),
                                Err(Error::UndefinedPoint(_)) => {
                                    r.report.undefined_trials += 1;
                                    None
                                }
                                Err(err) => return Err(err),
                            };
                            let values = values.filter(|v| {
                                let beyond = limits
                                    .as_ref()
                                    .is_some_and(|l| beyond_limits(spec.constraints(), &v.1, l));
                                r.report.violation_limit_rejections += u64::from(beyond);
                                !beyond
                            });
                            let mut evaluation = None;
                            if let Some(values) = values {
                                let corrected_merit = finite(
                                    values.0
                                        + weighted_violation(
                                            spec.constraints(),
                                            &values.1,
                                            &penalties,
                                        )?,
                                    "corrected merit",
                                )?;
                                if corrected_merit <= threshold {
                                    if let Some(s) = calls.before_call() {
                                        status = Some(s);
                                        break;
                                    }
                                    match calls.derivatives(
                                        spec,
                                        model,
                                        &trial,
                                        values,
                                        options.feasibility_tolerance,
                                    ) {
                                        Ok(e) => evaluation = Some(e),
                                        Err(Error::UndefinedPoint(_)) => {
                                            r.report.undefined_trials += 1;
                                        }
                                        Err(err) => return Err(err),
                                    }
                                }
                            }
                            if let Some(evaluation) = evaluation {
                                r.report.second_order_steps += 1;
                                accepted = Some((trial, evaluation));
                                break;
                            }
                            r.report.rejected_trials += 1;
                        }
                    }
                    alpha *= 0.5;
                    continue;
                }
                if let Some(s) = calls.before_call() {
                    status = Some(s);
                    break;
                }
                match calls.derivatives(spec, model, &trial, values, options.feasibility_tolerance)
                {
                    Ok(evaluation) => {
                        accepted = Some((trial, evaluation));
                        break;
                    }
                    Err(Error::UndefinedPoint(_)) => {
                        r.report.undefined_trials += 1;
                        r.report.rejected_trials += 1;
                        alpha *= 0.5;
                    }
                    Err(err) => return Err(err),
                }
            }
            if status.is_some() {
                break;
            }
            if let Some(s) = calls.stop() {
                status = Some(s);
                break;
            }
            let Some((next, next_evaluation)) = accepted else {
                if settings.record_history {
                    r.report.history.push(record);
                }
                if let Some(noise) = noise_step {
                    status = Some(Status::Stalled);
                    r.message = noise_message(predicted, noise);
                } else {
                    status = Some(Status::LineSearchFailure);
                }
                break;
            };
            if !elastic_rows.is_empty() {
                r.report.elastic_steps += 1;
            }
            noise_steps = if noise_step.is_some() {
                noise_steps + 1
            } else {
                0
            };
            record.step_length = Some(alpha);
            if settings.merit == Merit::AugmentedLagrangian {
                augmented.advance(&solution.multipliers.constraints, alpha);
            }

            // 4. Accept the step and update B.
            if let Some(reference) = &linear_reference {
                linear::check_constant(spec, reference, &next_evaluation)?;
            }
            let s: Vec<f64> = next.iter().zip(x).map(|(&a, &b)| a - b).collect();
            let old_l = lagrangian_gradient(spec, e, &solution.multipliers)?;
            let new_l = lagrangian_gradient(spec, &next_evaluation, &solution.multipliers)?;
            let y: Vec<f64> = new_l.iter().zip(old_l).map(|(&a, b)| a - b).collect();
            let t = Instant::now();
            let (mut change, mut departure) = match settings.hessian_update {
                HessianUpdate::Damped => (Vec::new(), Vec::new()),
                HessianUpdate::Augmented => curvature_repair(spec, e, &next_evaluation, &s)?,
            };
            // B lives in scaled coordinates: s̃ = s/D, ỹ = D y, J̃ = R J D.
            let (s_scaled, y) = if scaled_identity {
                (s.clone(), y)
            } else {
                let jacobian = spec.jacobian();
                for i in 0..spec.m() {
                    for a in jacobian.indptr[i]..jacobian.indptr[i + 1] {
                        if let Some(c) = change.get_mut(a) {
                            *c *= scales.rows[i] * scales.variables[jacobian.indices[a]];
                        }
                    }
                    if let Some(d) = departure.get_mut(i) {
                        *d *= scales.rows[i];
                    }
                }
                let y = y
                    .iter()
                    .zip(&scales.variables)
                    .map(|(v, d)| v * d)
                    .collect();
                (scales.to_scaled(&s), y)
            };
            let repair =
                (settings.hessian_update == HessianUpdate::Augmented).then(|| bfgs::Repair {
                    pattern: spec.jacobian(),
                    change: &change,
                    departure: &departure,
                });
            let updated = bfgs_work.update(&mut b, &s_scaled, &y, repair.as_ref());
            r.report.bfgs_seconds += t.elapsed().as_secs_f64();
            match updated {
                Some(CurvatureUpdate::Damped) => r.report.bfgs_damped_updates += 1,
                Some(CurvatureUpdate::Modified) => r.report.bfgs_modified_updates += 1,
                _ => {}
            }
            record.curvature = Some(updated.unwrap_or(CurvatureUpdate::Reset));
            if settings.record_history {
                r.report.history.push(record);
            }
            r.candidate = Some(next);
            if let Some(old) = r.evaluation.replace(next_evaluation) {
                calls.recycle(old);
            }
            r.iterations += 1;
            r.report.step_norm = Some(norm_inf(&s));
            store_residuals(spec, &mut r, solution.multipliers)?;
            if updated.is_none() {
                if r.report.hessian_resets >= u64::from(settings.max_hessian_resets) {
                    status = Some(Status::NumericalFailure);
                    r.message =
                        "BFGS update failed and the Hessian reset budget is exhausted".into();
                    break;
                }
                if effective_block > 0 {
                    effective_block = 0;
                    bfgs_work = bfgs::Curvature::new(n, 0);
                    bfgs_work.set_execution(&execution);
                    qp_work = None;
                    r.report.block_hessian_fallbacks += 1;
                }
                b = DMatrix::identity(n, n) * settings.initial_hessian_scale;
                r.report.hessian_resets += 1;
            }
            retry_descent = false;
            notify = true;
        }

        r.report.elastic_weight = elastic_weight;
        r.status = status.unwrap_or(Status::IterationLimit);
        if r.message.is_empty() {
            r.message = status_message(r.status).into();
        }
        r.counts = calls.counts;
        r.report.total_seconds = calls.began.elapsed().as_secs_f64();
        r.report.callback_seconds = calls.seconds;
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_levels_scale_with_the_terms_of_each_residual() {
        // One row with J = (1e-12, 2) and λ = −4e9. Variable 0's stationarity
        // sums |1| + |1e-12 · 4e9| + |0.5|, variable 1's |−3| + |2 · 4e9|.
        let spec = ProblemSpec::new(
            Bounds::new(vec![0.0, 0.0], vec![10.0, 10.0]),
            Bounds::new(vec![f64::NEG_INFINITY], vec![0.0]),
            CsrPattern::dense(1, 2),
        )
        .unwrap();
        let e = Evaluation {
            objective: 0.0,
            constraints: vec![2e-12],
            gradient: vec![1.0, -3.0],
            jacobian_values: vec![1e-12, 2.0],
            primal_violation: 2e-12,
            feasible_candidate: true,
        };
        let m = Multipliers {
            constraints: vec![-4e9],
            variables: vec![0.5, 0.0],
        };
        let (stationarity, complementarity) = rounding_levels(&spec, &[2.0, 1e-12], &e, &m);
        // Three terms for each variable: one row, ∇f and z.
        let expected = 3.0 * f64::EPSILON * (3.0 + 8e9);
        assert!(
            (stationarity / expected - 1.0).abs() < 1e-12,
            "{stationarity}"
        );
        assert!(stationarity > 5e-6);
        // |λ|·ε·(|c| + |u|) for the row, 0.5·ε·(2 + 0) for x0 at its lower bound.
        let row = 4e9 * f64::EPSILON * 2e-12;
        let variable = 0.5 * f64::EPSILON * 2.0;
        assert_eq!(complementarity, row.max(variable));
    }

    #[test]
    fn explicit_curvature_blocks_preserve_spd_and_sparse_pattern() {
        let n = 7;
        let spec =
            ProblemSpec::new(Bounds::free(n), Bounds::free(0), CsrPattern::dense(0, n)).unwrap();
        let e = Evaluation {
            objective: 0.,
            constraints: vec![],
            gradient: vec![0.; n],
            jacobian_values: vec![],
            primal_violation: 0.,
            feasible_candidate: true,
        };
        let mut b = DMatrix::identity(n, n);
        let mut work = bfgs::Curvature::new(n, 3);
        assert!(
            work.update(&mut b, &[1.; 7], &[2., 3., 4., 1., 2., 3., 4.], None)
                .is_some()
        );
        assert!(b.clone().cholesky().is_some());
        let q = qp_model_with_blocks(&spec, &[0.; 7], &e, &b, 3).unwrap();
        q.validate().unwrap();
        assert_eq!(q.hessian_lower.values.len(), 13);
        for i in 0..n {
            for j in 0..n {
                if i / 3 != j / 3 {
                    assert_eq!(b[(i, j)], 0.);
                }
            }
        }
        for i in 0..n {
            assert!(
                q.hessian_lower
                    .pattern
                    .row(i)
                    .iter()
                    .all(|&j| i / 3 == j / 3)
            );
        }
        assert!(
            SqpOptions {
                hessian_block_size: 257,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn recycled_callback_buffers_still_reject_missing_entries() {
        struct Partial {
            missing: bool,
        }
        impl NlpProblem for Partial {
            fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
                c[0] = x[0] + x[1];
                Ok(0.)
            }
            fn derivatives(&mut self, _: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
                if self.missing {
                    g[0] = 1.;
                } else {
                    g.fill(1.);
                    j.fill(1.);
                }
                Ok(())
            }
        }
        let spec =
            ProblemSpec::new(Bounds::free(2), Bounds::free(1), CsrPattern::dense(1, 2)).unwrap();
        let settings = SqpOptions::default();
        let cancel = CancellationToken::default();
        let mut calls = Calls {
            began: Instant::now(),
            settings: &settings,
            cancel: &cancel,
            counts: EvaluationCounts::default(),
            seconds: 0.,
            gradient_scratch: vec![],
            jacobian_scratch: vec![],
        };
        let mut model = Partial { missing: false };
        let x = [1., 1.];
        let v = calls.functions(&spec, &mut model, &x).unwrap();
        let e = calls.derivatives(&spec, &mut model, &x, v, 1e-7).unwrap();
        calls.recycle(e);
        model.missing = true;
        let v = calls.functions(&spec, &mut model, &x).unwrap();
        assert!(calls.derivatives(&spec, &mut model, &x, v, 1e-7).is_err());
    }
    #[test]
    fn reused_qp_template_matches_fresh_numeric_data() {
        let spec = ProblemSpec::new(
            Bounds::new(vec![0.; 3], vec![4.; 3]),
            Bounds::new(vec![-1., 0.], vec![2., 0.]),
            CsrPattern::dense(2, 3),
        )
        .unwrap();
        let mut e = Evaluation {
            objective: 0.,
            constraints: vec![0., 0.],
            gradient: vec![1., 2., 3.],
            jacobian_values: vec![1.; 6],
            primal_violation: 0.,
            feasible_candidate: true,
        };
        let mut b = DMatrix::identity(3, 3);
        let mut q = qp_model(&spec, &[1.; 3], &e, &b).unwrap();
        let pointers = (
            q.hessian_lower.pattern.indices.as_ptr(),
            q.matrix.pattern.indices.as_ptr(),
            q.hessian_lower.values.as_ptr(),
            q.rows.lower.as_ptr(),
        );
        for step in 1..5 {
            e.constraints[0] = step as f64 / 10.;
            e.jacobian_values[2] = step as f64;
            b[(1, 1)] = step as f64;
            let x = [0.5 + step as f64 / 10.; 3];
            refresh_qp(&mut q, &spec, &x, &e, &b).unwrap();
            assert_eq!(q, qp_model(&spec, &x, &e, &b).unwrap());
            assert_eq!(
                pointers,
                (
                    q.hessian_lower.pattern.indices.as_ptr(),
                    q.matrix.pattern.indices.as_ptr(),
                    q.hessian_lower.values.as_ptr(),
                    q.rows.lower.as_ptr()
                )
            );
        }
    }
    #[test]
    fn a_trial_capped_at_a_bound_stays_inside_the_box() {
        // With α = (u − x)/p, x + αp rounds to one ulp above u here (about 2% of
        // random cases do); the delivered line search then halved α.
        let (x, u, p) = (0.11408985987451992, 0.9812160792571979, 2.958067622106532);
        let bounds = Bounds::new(vec![-1.0], vec![u]);
        let alpha = box_step_limit(&bounds, &[x], &[p]);
        assert!(x + alpha * p > u);
        assert_eq!(trial_point(&bounds, &[x], &[p], alpha), vec![u]);
        // Interior trials are untouched, and a real overshoot is not hidden.
        assert_eq!(
            trial_point(&bounds, &[x], &[p], 0.5 * alpha),
            vec![x + 0.5 * alpha * p]
        );
        assert_eq!(trial_point(&bounds, &[x], &[p], 1.0), vec![x + p]);
    }

    #[test]
    fn snapping_moves_only_tolerance_level_overshoots() {
        let variables = Bounds::new(vec![0.0, 1.0, -5.0], vec![2.0, 1.0, 5.0]);
        let x = [2.0 - 3e-14, 1.0, 0.0];
        let room = 2.0 - x[0];
        let step_bounds = Bounds::new(vec![-1.0, 0.0, -1.0], vec![room, 0.0, 1.0]);
        let mut report = sqp_craft::RunReport::default();
        let mut step = vec![room + 1e-11, -2e-12, 1.5];
        assert!(box_step_limit(&variables, &x, &step) < 0.01);
        snap_step(
            &mut step,
            &step_bounds,
            &variables,
            1e-7,
            &[1.0; 3],
            &mut report,
        );
        assert_eq!(
            step,
            vec![room, 0.0, 1.5],
            "a 0.5 overshoot is left for the QP check"
        );
        assert_eq!(
            (report.bound_step_corrections, report.fixed_step_corrections),
            (1, 1)
        );
        assert_eq!(box_step_limit(&variables, &x, &step), 1.0);
    }
}
