//! Bounded sparse equality-KKT polish. Guessed active sides are never a
//! certificate: the caller still checks the complete original QP.
use clarabel::{
    algebra::CscMatrix,
    qdldl::{QDLDLFactorisation, QDLDLSettings},
};
use sqp_craft::{CancellationToken, Multipliers, qp::*, sparse};
use std::time::Instant;

fn active(value: f64, dual: f64, lo: f64, hi: f64) -> Option<f64> {
    if lo == hi {
        Some(lo)
    } else {
        let bound = if dual > 1e-12 {
            lo
        } else if dual < -1e-12 {
            hi
        } else {
            return None;
        };
        (bound.is_finite() && (value - bound).abs() <= 1e-8 * bound.abs().max(1.0)).then_some(bound)
    }
}

fn stopped(
    options: &QpOptions,
    began: Instant,
    cancel: &CancellationToken,
    report: &QpReport,
) -> bool {
    cancel.is_cancelled()
        || options
            .time_limit_seconds
            .is_some_and(|s| began.elapsed().as_secs_f64() >= s)
        || report.iterations >= u64::from(options.max_iterations)
}

pub(super) fn solve(
    model: &QpModel,
    candidate: &QpSolution,
    options: &QpOptions,
    began: Instant,
    cancel: &CancellationToken,
    report: &mut QpReport,
    checks: &mut QpResidualWorkspace,
) -> Option<QpSolution> {
    let mut excluded_variables = vec![false; model.n()];
    let mut excluded_rows = vec![false; model.m()];
    for _ in 0..3 {
        match attempt(
            model,
            candidate,
            options,
            began,
            cancel,
            report,
            checks,
            &excluded_variables,
            &excluded_rows,
        )? {
            Attempt::Accepted(solution) => return Some(solution),
            Attempt::Drop(variables, rows) => {
                for j in variables {
                    excluded_variables[j] = true;
                }
                for i in rows {
                    excluded_rows[i] = true;
                }
            }
        }
    }
    None
}

enum Attempt {
    Accepted(QpSolution),
    Drop(Vec<usize>, Vec<usize>),
}

#[allow(clippy::too_many_arguments)]
fn attempt(
    model: &QpModel,
    candidate: &QpSolution,
    options: &QpOptions,
    began: Instant,
    cancel: &CancellationToken,
    report: &mut QpReport,
    checks: &mut QpResidualWorkspace,
    excluded_variables: &[bool],
    excluded_rows: &[bool],
) -> Option<Attempt> {
    if stopped(options, began, cancel, report) {
        return None;
    }
    let n = model.n();
    let mut pins = vec![None; n];
    let mut free_index = vec![usize::MAX; n];
    let mut nf = 0;
    for j in 0..n {
        if !excluded_variables[j] {
            pins[j] = active(
                candidate.step[j],
                candidate.multipliers.variables[j],
                model.variables.lower[j],
                model.variables.upper[j],
            );
        }
        if pins[j].is_none() {
            free_index[j] = nf;
            nf += 1;
        }
    }
    let a = &model.matrix;
    let mut rows = Vec::new();
    for (i, &excluded) in excluded_rows.iter().enumerate() {
        if excluded {
            continue;
        }
        let value: f64 = (a.pattern.indptr[i]..a.pattern.indptr[i + 1])
            .map(|k| a.values[k] * candidate.step[a.pattern.indices[k]])
            .sum();
        if let Some(target) = active(
            value,
            candidate.multipliers.constraints[i],
            model.rows.lower[i],
            model.rows.upper[i],
        ) {
            rows.push((i, target));
        }
    }
    let dimension = nf.checked_add(rows.len())?;
    // The engine workspace is dropped before this call. Reserve a dense
    // worst-case factor (including symbolic scratch) before sparse allocation;
    // this is a cap, not a dense matrix or dense fallback.
    let fixed = n
        .saturating_add(model.m())
        .saturating_mul(192)
        .saturating_add(a.pattern.nnz().saturating_mul(64))
        .saturating_add(model.hessian_lower.pattern.nnz().saturating_mul(96));
    let factor_bound = dimension
        .checked_mul(dimension.checked_add(1)?)?
        .checked_div(2)?;
    if dimension == 0
        || sparse::check_memory(
            fixed.saturating_add(sparse::factor_bytes(factor_bound)),
            sparse::MAX_MEMORY_BYTES,
        )
        .is_err()
    {
        return None;
    }
    let mut ii = Vec::new();
    let mut jj = Vec::new();
    let mut vv = Vec::new();
    let mut rhs = vec![0.0; dimension];
    for j in 0..n {
        if pins[j].is_none() {
            rhs[free_index[j]] = -model.gradient[j];
        }
    }
    let h = &model.hessian_lower;
    for i in 0..n {
        for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
            let j = h.pattern.indices[k];
            let value = h.values[k];
            match (pins[i], pins[j]) {
                (None, None) => {
                    ii.push(free_index[j]);
                    jj.push(free_index[i]);
                    vv.push(value);
                }
                (None, Some(v)) => rhs[free_index[i]] -= value * v,
                (Some(v), None) => rhs[free_index[j]] -= value * v,
                _ => (),
            }
        }
    }
    let mut norms = Vec::with_capacity(rows.len());
    for (r, &(i, target)) in rows.iter().enumerate() {
        let range = a.pattern.indptr[i]..a.pattern.indptr[i + 1];
        let norm = range
            .clone()
            .filter(|&k| pins[a.pattern.indices[k]].is_none())
            .fold(0.0_f64, |v, k| v.max(a.values[k].abs()));
        if norm == 0.0 {
            return None;
        }
        norms.push(norm);
        rhs[nf + r] = target;
        for k in range {
            let j = a.pattern.indices[k];
            if let Some(v) = pins[j] {
                rhs[nf + r] -= a.values[k] * v;
            } else {
                ii.push(free_index[j]);
                jj.push(nf + r);
                vv.push(a.values[k] / norm);
            }
        }
        rhs[nf + r] /= norm;
        ii.push(nf + r);
        jj.push(nf + r);
        vv.push(-1e-12);
    }
    if !rhs.iter().chain(&vv).all(|v| v.is_finite()) || stopped(options, began, cancel, report) {
        return None;
    }
    let matrix = CscMatrix::new_from_triplets(dimension, dimension, ii, jj, vv);
    report.iterations += 1; // One factor/solve, then at most eight refinements.
    let signs = (0..dimension)
        .map(|j| if j < nf { 1 } else { -1 })
        .collect();
    let mut factor = QDLDLFactorisation::new(
        &matrix,
        Some(QDLDLSettings {
            Dsigns: Some(signs),
            regularize_eps: 1e-14,
            regularize_delta: 1e-12,
            ..Default::default()
        }),
    )
    .ok()?;
    report.factor_nnz = Some(report.factor_nnz.unwrap_or(0).max(factor.nnzL()));
    let mut point = rhs.clone();
    factor.solve(&mut point);
    for refinement in 0..=8 {
        if cancel.is_cancelled()
            || options
                .time_limit_seconds
                .is_some_and(|s| began.elapsed().as_secs_f64() >= s)
        {
            return None;
        }
        if !point.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut solution = QpSolution {
            step: vec![0.0; n],
            multipliers: Multipliers {
                constraints: vec![0.0; model.m()],
                variables: vec![0.0; n],
            },
            active_set: None,
            iterations: report.iterations,
        };
        for j in 0..n {
            solution.step[j] = pins[j]
                .unwrap_or_else(|| point[free_index[j]])
                .clamp(model.variables.lower[j], model.variables.upper[j]);
        }
        for (r, &(i, _)) in rows.iter().enumerate() {
            solution.multipliers.constraints[i] = -point[nf + r] / norms[r];
        }
        let mut residual = model.gradient.clone();
        for i in 0..n {
            for k in h.pattern.indptr[i]..h.pattern.indptr[i + 1] {
                let j = h.pattern.indices[k];
                residual[i] += h.values[k] * solution.step[j];
                if i != j {
                    residual[j] += h.values[k] * solution.step[i];
                }
            }
        }
        for i in 0..model.m() {
            for k in a.pattern.indptr[i]..a.pattern.indptr[i + 1] {
                residual[a.pattern.indices[k]] -= a.values[k] * solution.multipliers.constraints[i];
            }
        }
        for j in 0..n {
            if pins[j].is_some() {
                solution.multipliers.variables[j] = residual[j];
            }
        }
        let t = Instant::now();
        let check = model
            .accepts_with_workspace(&solution, options.acceptance_tolerance, checks)
            .ok()?;
        report.check_seconds += t.elapsed().as_secs_f64();
        if check.accepted {
            return Some(Attempt::Accepted(solution));
        }
        if check.residuals.primal <= options.acceptance_tolerance
            && check.residuals.stationarity <= options.acceptance_tolerance * check.scale
        {
            let wrong = |value: f64, dual: f64, lo: f64, hi: f64| {
                lo != hi
                    && ((value == lo && dual < -options.acceptance_tolerance * check.scale)
                        || (value == hi && dual > options.acceptance_tolerance * check.scale))
            };
            let variables: Vec<_> = pins
                .iter()
                .enumerate()
                .filter_map(|(j, v)| {
                    v.filter(|&v| {
                        wrong(
                            v,
                            solution.multipliers.variables[j],
                            model.variables.lower[j],
                            model.variables.upper[j],
                        )
                    })
                    .map(|_| j)
                })
                .collect();
            let wrong_rows: Vec<_> = rows
                .iter()
                .filter_map(|&(i, target)| {
                    wrong(
                        target,
                        solution.multipliers.constraints[i],
                        model.rows.lower[i],
                        model.rows.upper[i],
                    )
                    .then_some(i)
                })
                .collect();
            if !variables.is_empty() || !wrong_rows.is_empty() {
                return Some(Attempt::Drop(variables, wrong_rows));
            }
        }
        if refinement == 8 || stopped(options, began, cancel, report) {
            return None;
        }
        // Refine against the unregularized equality KKT system. The tiny
        // negative dual diagonal belongs to the factor only.
        let mut correction = rhs.clone();
        for col in 0..dimension {
            for k in matrix.colptr[col]..matrix.colptr[col + 1] {
                let row = matrix.rowval[k];
                if row == col && row >= nf {
                    continue;
                }
                correction[row] -= matrix.nzval[k] * point[col];
                if row != col {
                    correction[col] -= matrix.nzval[k] * point[row];
                }
            }
        }
        report.iterations += 1;
        factor.solve(&mut correction);
        for (value, delta) in point.iter_mut().zip(correction) {
            *value += delta;
        }
    }
    None
}
