//! Sparse QP transport for the branch-structured QP experiment.
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome};
use sqp_craft::sparse::{MAX_MEMORY_BYTES, Symbolic, kkt_fill};
use sqp_craft::{Bounds, CancellationToken, CsrMatrix, CsrPattern};
use sqp_craft_clarabel::ClarabelQp;
use std::error::Error;
use std::io::{self, Read};
use std::str::{FromStr, SplitWhitespace};

const TOLERANCE: f64 = 1e-11;
const ACCEPTANCE: f64 = 1e-8;

fn take<T: FromStr>(tokens: &mut SplitWhitespace<'_>, what: &str) -> Result<T, Box<dyn Error>> {
    let token = tokens.next().ok_or_else(|| format!("missing {what}"))?;
    token
        .parse()
        .map_err(|_| format!("invalid {what}: {token}").into())
}

fn vector<T: FromStr>(
    tokens: &mut SplitWhitespace<'_>,
    count: usize,
    what: &str,
) -> Result<Vec<T>, Box<dyn Error>> {
    (0..count).map(|_| take(tokens, what)).collect()
}

fn matrix(
    tokens: &mut SplitWhitespace<'_>,
    rows: usize,
    cols: usize,
    entries: usize,
) -> Result<CsrMatrix, Box<dyn Error>> {
    let pointers = rows.checked_add(1).ok_or("too many CSR rows")?;
    Ok(CsrMatrix {
        pattern: CsrPattern {
            rows,
            cols,
            indptr: vector(tokens, pointers, "row pointer")?,
            indices: vector(tokens, entries, "column index")?,
        },
        values: vector(tokens, entries, "matrix coefficient")?,
    })
}

fn parse(input: &str) -> Result<QpModel, Box<dyn Error>> {
    let mut t = input.split_whitespace();
    if t.next() != Some("fan-qp/1") {
        return Err("expected fan-qp/1 header".into());
    }
    let n = take(&mut t, "primal dimension")?;
    let m = take(&mut t, "row dimension")?;
    let h_nnz = take(&mut t, "Hessian entry count")?;
    let a_nnz = take(&mut t, "Jacobian entry count")?;
    let model = QpModel {
        hessian_lower: matrix(&mut t, n, n, h_nnz)?,
        matrix: matrix(&mut t, m, n, a_nnz)?,
        gradient: vector(&mut t, n, "gradient")?,
        variables: Bounds::new(
            vector(&mut t, n, "lower bound")?,
            vector(&mut t, n, "upper bound")?,
        ),
        rows: Bounds::new(
            vector(&mut t, m, "row lower")?,
            vector(&mut t, m, "row upper")?,
        ),
    };
    if t.next().is_some() {
        return Err("unexpected trailing data".into());
    }
    model.validate()?;
    if (0..n).any(|i| model.hessian_lower.pattern.row(i).last() != Some(&i)) {
        return Err("each Hessian row must contain its diagonal".into());
    }
    Ok(model)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let wire_bytes = input.len();
    let model = parse(&input)?;
    drop(input);

    let nptr = model.hessian_lower.pattern.indptr.len() + model.matrix.pattern.indptr.len();
    let indices = model.hessian_lower.values.len() + model.matrix.values.len();
    let csr_bytes = (nptr + indices) * size_of::<usize>() + indices * size_of::<f64>();
    let budget = MAX_MEMORY_BYTES;
    let hessian_nnz = Symbolic::of(&model.hessian_lower.pattern, budget)?.nnz();
    let kkt_nnz = kkt_fill(
        &model.hessian_lower.pattern,
        &model.matrix.pattern,
        &model.rows,
        &model.variables,
        budget,
    )?;

    let mut solver = ClarabelQp::default();
    solver.configure(&QpOptions {
        tolerance: TOLERANCE,
        acceptance_tolerance: ACCEPTANCE,
        ..Default::default()
    })?;
    let outcome = solver.solve(&model, None, &CancellationToken::default())?;
    let report = solver.report();
    let raw = report
        .raw_status
        .as_deref()
        .map_or_else(|| "null".to_string(), |s| format!("{s:?}"));
    let factor = report
        .factor_nnz
        .map_or_else(|| "null".to_string(), |n| n.to_string());
    print!(
        "{{\"backend\":{:?},\"raw_status\":{raw},\"wire_bytes\":{wire_bytes},\"csr_payload_bytes\":{csr_bytes},\
         \"hessian_factor_nnz\":{hessian_nnz},\"kkt_factor_nnz\":{kkt_nnz},\
         \"qp_factor_nnz\":{factor},\"cold_retries\":{},\"setups\":{},\"iterations\":{},\
         \"backend_tolerance\":{TOLERANCE},\"acceptance_tolerance\":{ACCEPTANCE},",
        report.backend, report.cold_retries, report.setups, report.iterations
    );
    let (status, candidate) = match outcome {
        QpOutcome::Optimal(s) => ("optimal", Some(s)),
        QpOutcome::LimitReached(s) => ("limit", s),
        QpOutcome::Infeasible => ("primal_infeasible", None),
        QpOutcome::Unbounded => ("dual_infeasible", None),
        QpOutcome::Cancelled => ("cancelled", None),
        QpOutcome::NotPositiveDefinite => ("not_positive_definite", None),
        _ => ("unknown_outcome", None),
    };
    print!("\"status\":\"{status}\"");
    if let Some(s) = candidate {
        let check = model.accepts(&s, ACCEPTANCE)?;
        let r = check.residuals;
        print!(
            ",\"x\":{:?},\"lambda\":{:?},\"z\":{:?},\"objective\":{},\
             \"accepted_by_qp_model\":{},\"primal\":{},\"stationarity\":{},\
             \"complementarity\":{},\"dual_sign\":{}",
            s.step,
            s.multipliers.constraints,
            s.multipliers.variables,
            model.objective(&s.step),
            check.accepted,
            r.primal,
            r.stationarity,
            r.complementarity,
            r.dual_sign
        );
    }
    println!("}}");
    Ok(())
}
