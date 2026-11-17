//! Standalone QP seam for the fan-structure experiment; whitespace input.
use sqp_craft::qp::{QpBackend, QpModel, QpOptions, QpOutcome};
use sqp_craft::{Bounds, CancellationToken, CsrMatrix, CsrPattern};
use sqp_craft_clarabel::ClarabelQp;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let numbers: Vec<f64> = input
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    // Header n, m; then 4n + 2m vectors and the dense m-by-n row matrix.
    let dimension = |v: Option<&f64>| match v {
        Some(&v) if v >= 0.0 && v.fract() == 0.0 && v <= 1e6 => Ok(v as usize),
        _ => Err("the input must start with two nonnegative integer dimensions n and m"),
    };
    let n = dimension(numbers.first())?;
    let m = dimension(numbers.get(1))?;
    let expected = 2 + 4 * n + 2 * m + n * m;
    if numbers.len() != expected {
        return Err(format!(
            "expected {expected} numbers for n = {n}, m = {m}, found {}",
            numbers.len()
        )
        .into());
    }
    let mut offset = 2;
    let mut take = |count: usize| {
        let v = numbers[offset..offset + count].to_vec();
        offset += count;
        v
    };
    let diagonal = take(n);
    let gradient = take(n);
    let lower = take(n);
    let upper = take(n);
    let row_lower = take(m);
    let row_upper = take(m);
    let dense = take(n * m);
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in dense.chunks(n) {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                indices.push(j);
                values.push(value);
            }
        }
        indptr.push(indices.len());
    }
    let model = QpModel {
        hessian_lower: CsrMatrix {
            pattern: CsrPattern {
                rows: n,
                cols: n,
                indptr: (0..=n).collect(),
                indices: (0..n).collect(),
            },
            values: diagonal,
        },
        gradient,
        variables: Bounds::new(lower, upper),
        matrix: CsrMatrix {
            pattern: CsrPattern {
                rows: m,
                cols: n,
                indptr,
                indices,
            },
            values,
        },
        rows: Bounds::new(row_lower, row_upper),
    };
    model.validate()?;
    let mut solver = ClarabelQp::default();
    solver.configure(&QpOptions {
        tolerance: 1e-11,
        acceptance_tolerance: 1e-8,
        ..Default::default()
    })?;
    let t = std::time::Instant::now();
    match solver.solve(&model, None, &CancellationToken::default())? {
        QpOutcome::Optimal(s) => {
            let elapsed = t.elapsed().as_secs_f64();
            let r = model.residuals(&s)?;
            println!(
                "{{\"status\":\"optimal\",\"x\":{:?},\"lambda\":{:?},\"z\":{:?},\"objective\":{},\"seconds\":{},\"primal\":{},\"stationarity\":{},\"complementarity\":{},\"dual_sign\":{}}}",
                s.step,
                s.multipliers.constraints,
                s.multipliers.variables,
                model.objective(&s.step),
                elapsed,
                r.primal,
                r.stationarity,
                r.complementarity,
                r.dual_sign
            );
        }
        other => return Err(format!("QP outcome: {other:?}").into()),
    }
    Ok(())
}
