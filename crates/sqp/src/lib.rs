//! Bounded dense-BFGS SQP pilot. No restoration, scaling or global optimality
//! claim. The model protocol and sparse input remain in dependency-free core.
#![forbid(unsafe_code)]
mod bfgs;
use nalgebra::DMatrix;
use sqp_craft::{kkt::nlp_residuals, qp::*, *};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SqpOptions {
    pub max_iterations: u64,
    pub max_backtracks: u32,
    pub max_callback_calls: u64,
    pub time_limit_seconds: Option<f64>,
    pub stationarity_tolerance: f64,
    pub complementarity_tolerance: f64,
    pub dual_sign_tolerance: f64,
    pub dense_size_limit: usize,
    pub initial_hessian_scale: f64,
    pub max_hessian_resets: u32,
    pub qp_tolerance: f64,
    pub qp_max_iterations: u32,
    pub record_qp_limit: usize,
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
            dense_size_limit: 256,
            initial_hessian_scale: 1.0,
            max_hessian_resets: 3,
            qp_tolerance: 1e-10,
            qp_max_iterations: 200,
            record_qp_limit: 0,
        }
    }
}
impl SqpOptions {
    pub fn validate(&self) -> Result<()> {
        if [
            self.stationarity_tolerance,
            self.complementarity_tolerance,
            self.dual_sign_tolerance,
            self.initial_hessian_scale,
            self.qp_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .time_limit_seconds
                .is_some_and(|s| !s.is_finite() || s < 0.0)
            || self.dense_size_limit == 0
            || self.dense_size_limit > 256
            || self.record_qp_limit > 16
        {
            return Err(Error::InvalidInput("invalid SQP options (dense cap 1..256, trace cap 0..16, positive finite tolerances)".into()));
        }
        Ok(())
    }
}
pub struct NativeSqp<B> {
    pub qp: B,
    pub settings: SqpOptions,
}
impl<B: Default> Default for NativeSqp<B> {
    fn default() -> Self {
        Self {
            qp: B::default(),
            settings: SqpOptions::default(),
        }
    }
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
        v = finite(v + finite(a * b, "dot product")?, "dot sum")?;
    }
    Ok(v)
}
fn inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}
fn violation_sum(bounds: &Bounds, values: &[f64]) -> Result<f64> {
    let mut v = 0.0;
    for (i, &x) in values.iter().enumerate() {
        if bounds.lower[i].is_finite() {
            v = finite(
                v + finite(bounds.lower[i] - x, "merit distance")?.max(0.0),
                "merit sum",
            )?;
        }
        if bounds.upper[i].is_finite() {
            v = finite(
                v + finite(x - bounds.upper[i], "merit distance")?.max(0.0),
                "merit sum",
            )?;
        }
    }
    Ok(v)
}
struct Calls<'a> {
    began: Instant,
    settings: &'a SqpOptions,
    cancel: &'a CancellationToken,
    counts: EvaluationCounts,
    seconds: f64,
}
impl Calls<'_> {
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
            finite(v, "constraint output")?;
        }
        Ok((f, c))
    }
    fn derivatives(
        &mut self,
        spec: &ProblemSpec,
        model: &mut dyn NlpProblem,
        x: &[f64],
        f: f64,
        c: Vec<f64>,
        tolerance: f64,
    ) -> Result<Evaluation> {
        spec.check_point(x, "derivative point")?;
        let mut g = vec![f64::NAN; spec.n()];
        let mut j = vec![f64::NAN; spec.nnz()];
        self.counts.derivatives += 1;
        let t = Instant::now();
        let result = model.derivatives(x, &mut g, &mut j);
        self.seconds += t.elapsed().as_secs_f64();
        result?;
        for &v in g.iter().chain(&j) {
            finite(v, "derivative output")?;
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
fn shifted(bounds: &Bounds, x: &[f64]) -> Result<Bounds> {
    let shift = |values: &[f64]| -> Result<Vec<f64>> {
        values
            .iter()
            .zip(x)
            .map(|(&b, &x)| {
                if b.is_finite() {
                    finite(b - x, "shifted bound")
                } else {
                    Ok(b)
                }
            })
            .collect()
    };
    Ok(Bounds::new(shift(&bounds.lower)?, shift(&bounds.upper)?))
}
fn qp_model(spec: &ProblemSpec, x: &[f64], e: &Evaluation, b: &DMatrix<f64>) -> Result<QpModel> {
    let n = spec.n();
    let mut indptr = vec![0];
    let mut indices = vec![];
    let mut values = vec![];
    for i in 0..n {
        for j in 0..=i {
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
fn lagrangian_gradient(spec: &ProblemSpec, e: &Evaluation, m: &Multipliers) -> Result<Vec<f64>> {
    let mut jt = vec![0.0; spec.n()];
    spec.jacobian()
        .multiply_transpose(&e.jacobian_values, &m.constraints, &mut jt);
    e.gradient
        .iter()
        .zip(jt)
        .zip(&m.variables)
        .map(|((&g, j), &z)| {
            finite(
                finite(g - j, "Lagrangian gradient")? - z,
                "Lagrangian gradient",
            )
        })
        .collect()
}
fn residuals(spec: &ProblemSpec, r: &mut SolveResult, m: Multipliers) -> Result<()> {
    let k = nlp_residuals(
        spec,
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        &m,
    )?;
    r.stationarity = Some(k.stationarity);
    r.complementarity = Some(k.complementarity);
    r.dual_sign = Some(k.dual_sign);
    r.multipliers = Some(m);
    Ok(())
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
        spec.check_point(x0, "x0")?;
        if spec.n() > self.settings.dense_size_limit {
            return Err(Error::Unsupported(format!(
                "n={} exceeds dense pilot limit {}",
                spec.n(),
                self.settings.dense_size_limit
            )));
        }
        let settings = &self.settings;
        let mut calls = Calls {
            began: Instant::now(),
            settings,
            cancel,
            counts: EvaluationCounts::default(),
            seconds: 0.0,
        };
        let mut r = SolveResult::empty(Status::IterationLimit, "iteration limit");
        r.report.method = "sqp".into();
        if let Some(status) = calls.before_call() {
            r.status = status;
            r.message = status.as_str().into();
            return Ok(r);
        }
        let (f, c) = calls.functions(spec, model, x0)?;
        if let Some(status) = calls.before_call() {
            r.status = status;
            r.counts = calls.counts;
            r.message = status.as_str().into();
            r.report.total_seconds = calls.began.elapsed().as_secs_f64();
            r.report.callback_seconds = calls.seconds;
            return Ok(r);
        }
        let evaluation = calls.derivatives(spec, model, x0, f, c, options.feasibility_tolerance)?;
        r.candidate = Some(x0.to_vec());
        r.evaluation = Some(evaluation);
        let mut b = DMatrix::identity(spec.n(), spec.n()) * settings.initial_hessian_scale;
        let mut rho = 1.0_f64;
        let mut retry_descent = false;
        let mut notify = true;
        'outer: loop {
            if let Some(status) = calls.stop() {
                r.status = status;
                break;
            }
            if notify {
                if let Some(obs) = observer.as_deref_mut() {
                    let e = r.evaluation.as_ref().unwrap();
                    if obs.observe(&Progress {
                        iteration: r.iterations,
                        objective: e.objective,
                        primal_violation: e.primal_violation,
                    })? == Control::Stop
                    {
                        r.status = Status::Cancelled;
                        r.message = "observer requested stop".into();
                        break;
                    }
                }
                if let Some(status) = calls.stop() {
                    r.status = status;
                    break;
                }
            }
            if r.iterations >= settings.max_iterations {
                r.status = Status::IterationLimit;
                break;
            }
            let x = r.candidate.as_ref().unwrap();
            let e = r.evaluation.as_ref().unwrap();
            let qp = qp_model(spec, x, e, &b)?;
            if r.report.qp_trace.len() < settings.record_qp_limit {
                r.report.qp_trace.push(qp.clone());
            }
            let remaining = settings
                .time_limit_seconds
                .map(|s| (s - calls.began.elapsed().as_secs_f64()).max(0.0));
            self.qp.configure(&QpOptions {
                tolerance: settings.qp_tolerance,
                acceptance_tolerance: options
                    .feasibility_tolerance
                    .min(settings.stationarity_tolerance)
                    .min(settings.complementarity_tolerance),
                max_iterations: settings.qp_max_iterations,
                time_limit_seconds: remaining,
            })?;
            let t = Instant::now();
            r.report.qp_solves += 1;
            let outcome = self.qp.solve(&qp, None, cancel);
            r.report.qp_seconds += t.elapsed().as_secs_f64();
            let report = self.qp.report();
            r.report.qp_backend = report.backend;
            r.report.qp_status = report.raw_status;
            r.report.qp_iterations += report.iterations;
            if let Some(status) = calls.stop() {
                r.status = status;
                break;
            }
            let mut solution = match outcome {
                Ok(QpOutcome::Optimal(s)) => s,
                Ok(QpOutcome::Infeasible) => {
                    r.status = Status::LinearizationInfeasible;
                    break;
                }
                Ok(QpOutcome::Cancelled) => {
                    r.status = Status::Cancelled;
                    break;
                }
                Ok(QpOutcome::LimitReached(_)) => {
                    r.status = if report.limit_reason.as_deref() == Some("time") {
                        Status::TimeLimit
                    } else {
                        Status::QpFailure
                    };
                    r.message = format!("QP limit: {:?}", report.limit_reason);
                    break;
                }
                Ok(_) => {
                    r.status = Status::QpFailure;
                    r.message = "QP did not return an accepted solution".into();
                    break;
                }
                Err(Error::Backend(message)) => {
                    r.status = Status::QpFailure;
                    r.message = message;
                    break;
                }
                Err(err) => return Err(err),
            };
            // Fixed-variable steps have exact zero mathematical bounds. Correct
            // only tolerance-sized roundoff, record it, and recheck the whole QP.
            for j in 0..spec.n() {
                if spec.variables().lower[j] == spec.variables().upper[j]
                    && solution.step[j] != 0.0
                    && solution.step[j].abs() <= options.feasibility_tolerance
                {
                    solution.step[j] = 0.0;
                    r.report.fixed_step_corrections += 1;
                }
            }
            let checked = qp.residuals(&solution)?;
            if checked.primal > options.feasibility_tolerance
                || checked.stationarity > settings.stationarity_tolerance
                || checked.complementarity > settings.complementarity_tolerance
                || checked.dual_sign > settings.dual_sign_tolerance
            {
                r.status = Status::QpFailure;
                r.message = "independent QP residual check failed".into();
                break;
            }
            residuals(spec, &mut r, solution.multipliers.clone())?;
            if r.evaluation.as_ref().unwrap().primal_violation <= options.feasibility_tolerance
                && r.stationarity.unwrap() <= settings.stationarity_tolerance
                && r.complementarity.unwrap() <= settings.complementarity_tolerance
                && r.dual_sign.unwrap() <= settings.dual_sign_tolerance
            {
                r.status = Status::Converged;
                break;
            }
            let x = r.candidate.as_ref().unwrap();
            let e = r.evaluation.as_ref().unwrap();
            let p = &solution.step;
            rho = rho.max(1.1 * inf(&solution.multipliers.constraints) + 1.0);
            finite(rho, "penalty")?;
            let v = violation_sum(spec.constraints(), &e.constraints)?;
            let mut linear = vec![0.0; spec.m()];
            spec.jacobian().multiply(&e.jacobian_values, p, &mut linear);
            for (v, &c) in linear.iter_mut().zip(&e.constraints) {
                *v = finite(*v + c, "linearized constraints")?;
            }
            let predicted = finite(
                -dot(&e.gradient, p)? + rho * (v - violation_sum(spec.constraints(), &linear)?),
                "predicted decrease",
            )?;
            if predicted <= 0.0 {
                if !retry_descent
                    && r.report.hessian_resets < u64::from(settings.max_hessian_resets)
                {
                    b = DMatrix::identity(spec.n(), spec.n()) * settings.initial_hessian_scale;
                    r.report.hessian_resets += 1;
                    retry_descent = true;
                    notify = false;
                    continue;
                }
                r.status = Status::NoDescent;
                break;
            }
            if inf(p) <= 1e-14 * (1.0 + inf(x)) {
                r.status = Status::Stalled;
                break;
            }
            let merit = finite(e.objective + rho * v, "merit")?;
            let mut alpha = 1.0_f64;
            for j in 0..spec.n() {
                if p[j] > 0.0 && spec.variables().upper[j].is_finite() {
                    alpha = alpha.min((spec.variables().upper[j] - x[j]) / p[j]);
                }
                if p[j] < 0.0 && spec.variables().lower[j].is_finite() {
                    alpha = alpha.min((spec.variables().lower[j] - x[j]) / p[j]);
                }
            }
            if alpha <= 0.0 || !alpha.is_finite() {
                r.status = Status::Stalled;
                r.message = "QP direction cannot remain in the variable box".into();
                break;
            }
            let mut accepted = None;
            for _ in 0..=settings.max_backtracks {
                if let Some(status) = calls.before_call() {
                    r.status = status;
                    break 'outer;
                }
                let trial: Vec<f64> = x.iter().zip(p).map(|(&x, &p)| x + alpha * p).collect();
                if spec.check_point(&trial, "trial").is_err() {
                    alpha *= 0.5;
                    r.report.rejected_trials += 1;
                    continue;
                }
                let (f, c) = match calls.functions(spec, model, &trial) {
                    Ok(v) => v,
                    Err(Error::UndefinedPoint(_)) => {
                        model.discard_undefined();
                        r.report.undefined_trials += 1;
                        r.report.rejected_trials += 1;
                        alpha *= 0.5;
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                let trial_merit = finite(
                    f + rho * violation_sum(spec.constraints(), &c)?,
                    "trial merit",
                )?;
                if trial_merit <= merit - 1e-4 * alpha * predicted {
                    if let Some(status) = calls.before_call() {
                        r.status = status;
                        break 'outer;
                    }
                    let evaluation = match calls.derivatives(
                        spec,
                        model,
                        &trial,
                        f,
                        c,
                        options.feasibility_tolerance,
                    ) {
                        Ok(e) => e,
                        Err(Error::UndefinedPoint(_)) => {
                            model.discard_undefined();
                            r.report.undefined_trials += 1;
                            r.report.rejected_trials += 1;
                            alpha *= 0.5;
                            continue;
                        }
                        Err(err) => return Err(err),
                    };
                    if let Some(status) = calls.stop() {
                        r.status = status;
                        break 'outer;
                    }
                    accepted = Some((trial, evaluation));
                    break;
                }
                r.report.rejected_trials += 1;
                alpha *= 0.5;
            }
            let Some((next, next_evaluation)) = accepted else {
                r.status = Status::LineSearchFailure;
                break;
            };
            let s: Vec<f64> = next.iter().zip(x).map(|(&a, &b)| a - b).collect();
            let old_l = lagrangian_gradient(spec, e, &solution.multipliers)?;
            let new_l = lagrangian_gradient(spec, &next_evaluation, &solution.multipliers)?;
            let y: Vec<f64> = new_l.iter().zip(old_l).map(|(&a, b)| a - b).collect();
            let t = Instant::now();
            let updated = bfgs::update(&mut b, &s, &y);
            r.report.bfgs_seconds += t.elapsed().as_secs_f64();
            r.candidate = Some(next);
            r.evaluation = Some(next_evaluation);
            r.iterations += 1;
            r.report.step_norm = Some(inf(&s));
            residuals(spec, &mut r, solution.multipliers)?;
            if !updated {
                if r.report.hessian_resets >= u64::from(settings.max_hessian_resets) {
                    r.status = Status::NumericalFailure;
                    r.message = "BFGS reset budget exhausted".into();
                    break;
                }
                b = DMatrix::identity(spec.n(), spec.n()) * settings.initial_hessian_scale;
                r.report.hessian_resets += 1;
            }
            retry_descent = false;
            notify = true;
        }
        if r.message == "iteration limit" {
            r.message = r.status.as_str().into();
        }
        r.counts = calls.counts;
        r.report.total_seconds = calls.began.elapsed().as_secs_f64();
        r.report.callback_seconds = calls.seconds;
        Ok(r)
    }
}
