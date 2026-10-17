//! Row and column scaling of the QP subproblems and the BFGS matrix, after
//! SNOPT's `Scale option` 1 and 2 (Gill, Murray and Saunders 2005, §7.1) and
//! the iterative geometric-mean scaling of MINOS (Fourer 1982).
//!
//! With `x = D x̃` and `c̃ = R c` the step QP is solved in the scaled
//! coordinates and its solution mapped back: `p = D p̃`, `λ = R λ̃`,
//! `z = z̃ / D`. Every factor is a power of two, so these maps are exact in
//! floating point, and the identity scaling reproduces the unscaled
//! arithmetic bit for bit. The merit function, the line search and the
//! convergence test stay in original units.
use super::{CsrPattern, ProblemSpec};
use sqp_craft::qp::{QpModel, QpSolution};

/// Passes of the geometric-mean iteration at most.
const MAX_PASSES: usize = 20;
/// The iteration stops when a pass reduces the largest column ratio
/// `max|ã|/min|ã|` by less than this factor (MINOS's scale tolerance 0.9).
const IMPROVEMENT: f64 = 0.9;
/// Factors are confined to `2^±LIMIT`.
const LIMIT: i32 = 60;

/// Which constraint rows determine the scales.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling (default).
    #[default]
    None,
    /// SNOPT's `Scale option 1`: the rows declared linear and the variables
    /// that occur in them, from their constant coefficients. Without
    /// declared linear rows this is the identity.
    Linear,
    /// SNOPT's `Scale option 2`: every row, from the Jacobian at the first
    /// point that satisfies the rows declared linear (the start, or its
    /// projection). Best with a good start on a mildly nonlinear problem.
    Jacobian,
}

impl Scaling {
    /// Stable lowercase name, as used by the Python binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Scaling::None => "none",
            Scaling::Linear => "linear",
            Scaling::Jacobian => "jacobian",
        }
    }

    /// Parses [`Self::as_str`].
    pub fn parse(name: &str) -> sqp_craft::Result<Self> {
        match name {
            "none" => Ok(Scaling::None),
            "linear" => Ok(Scaling::Linear),
            "jacobian" => Ok(Scaling::Jacobian),
            _ => Err(sqp_craft::Error::InvalidInput(format!(
                "scaling must be \"none\", \"linear\" or \"jacobian\", got {name:?}"
            ))),
        }
    }
}

/// Factors `D` (variables) and `R` (rows).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Scales {
    pub variables: Vec<f64>,
    pub rows: Vec<f64>,
}

fn power_of_two(v: f64) -> f64 {
    if !(v.is_finite() && v > 0.0) {
        return 1.0;
    }
    let exponent = (v.log2().round() as i32).clamp(-LIMIT, LIMIT);
    2.0_f64.powi(exponent)
}

impl Scales {
    pub fn identity(n: usize, m: usize) -> Self {
        Self {
            variables: vec![1.0; n],
            rows: vec![1.0; m],
        }
    }

    pub fn is_identity(&self) -> bool {
        self.variables.iter().chain(&self.rows).all(|&v| v == 1.0)
    }

    /// Geometric-mean scaling of the Jacobian entries `values` restricted to
    /// the rows with `selected[i]`. Rows not selected and columns without a
    /// nonzero in the selected rows keep the factor 1.
    pub fn geometric(pattern: &CsrPattern, values: &[f64], selected: &[bool]) -> Self {
        let (m, n) = (pattern.rows, pattern.cols);
        let mut rows = vec![1.0_f64; m];
        let mut columns = vec![1.0_f64; n];
        let entries: Vec<(usize, usize, f64)> = (0..m)
            .filter(|&i| selected[i])
            .flat_map(|i| {
                (pattern.indptr[i]..pattern.indptr[i + 1])
                    .map(move |a| (i, pattern.indices[a], values[a].abs()))
            })
            .filter(|&(_, _, v)| v.is_finite() && v > 0.0)
            .collect();
        if entries.is_empty() {
            return Self::identity(n, m);
        }
        let extremes = |rows: &[f64], columns: &[f64], by_row: bool, len: usize| {
            let mut low = vec![f64::INFINITY; len];
            let mut high = vec![0.0_f64; len];
            for &(i, j, v) in &entries {
                let scaled = rows[i] * v * columns[j];
                let k = if by_row { i } else { j };
                low[k] = low[k].min(scaled);
                high[k] = high[k].max(scaled);
            }
            (low, high)
        };
        let ratio = |rows: &[f64], columns: &[f64]| {
            let (low, high) = extremes(rows, columns, false, n);
            low.iter()
                .zip(&high)
                .filter(|&(&l, _)| l.is_finite())
                .fold(1.0_f64, |r, (&l, &h)| r.max(h / l))
        };
        let mut previous = ratio(&rows, &columns);
        for _ in 0..MAX_PASSES {
            let (low, high) = extremes(&rows, &columns, true, m);
            for i in 0..m {
                if low[i].is_finite() {
                    rows[i] /= (low[i] * high[i]).sqrt();
                }
            }
            let (low, high) = extremes(&rows, &columns, false, n);
            for j in 0..n {
                if low[j].is_finite() {
                    columns[j] /= (low[j] * high[j]).sqrt();
                }
            }
            let current = ratio(&rows, &columns);
            if current > IMPROVEMENT * previous {
                break;
            }
            previous = current;
        }
        Self {
            variables: columns.into_iter().map(power_of_two).collect(),
            rows: rows.into_iter().map(power_of_two).collect(),
        }
    }

    /// Scales the gradient, the constraint matrix and the bounds of a QP
    /// built in original units (`row_factors` per QP row), but not its
    /// Hessian, which the callers keep in scaled coordinates.
    pub fn scale_qp(&self, q: &mut QpModel, row_factors: &[f64]) {
        let d = &self.variables;
        for (g, &dj) in q.gradient.iter_mut().zip(d) {
            *g *= dj;
        }
        let pattern = &q.matrix.pattern;
        for (i, &factor) in row_factors.iter().enumerate().take(pattern.rows) {
            for a in pattern.indptr[i]..pattern.indptr[i + 1] {
                q.matrix.values[a] *= factor * d[pattern.indices[a]];
            }
        }
        for (j, &dj) in d.iter().enumerate() {
            for v in [&mut q.variables.lower[j], &mut q.variables.upper[j]] {
                if v.is_finite() {
                    *v /= dj;
                }
            }
        }
        for (i, &factor) in row_factors.iter().enumerate().take(q.rows.len()) {
            for v in [&mut q.rows.lower[i], &mut q.rows.upper[i]] {
                if v.is_finite() {
                    *v *= factor;
                }
            }
        }
    }

    /// Maps a solution of a QP scaled by [`Self::scale_qp`] back:
    /// `p = D p̃`, `λ = R λ̃`, `z = z̃ / D`.
    pub fn unscale_solution(&self, s: &mut QpSolution, row_factors: &[f64]) {
        for (p, &dj) in s.step.iter_mut().zip(&self.variables) {
            *p *= dj;
        }
        for (l, &factor) in s.multipliers.constraints.iter_mut().zip(row_factors) {
            *l *= factor;
        }
        for (z, &dj) in s.multipliers.variables.iter_mut().zip(&self.variables) {
            *z /= dj;
        }
    }

    /// `v / D` (a displacement in scaled coordinates).
    pub fn to_scaled(&self, v: &[f64]) -> Vec<f64> {
        v.iter()
            .zip(&self.variables)
            .map(|(&a, &d)| a / d)
            .collect()
    }

    /// The smallest row factor (1 without rows).
    pub fn smallest_row(&self) -> f64 {
        self.rows.iter().copied().reduce(f64::min).unwrap_or(1.0)
    }

    /// The largest row factor (1 without rows).
    pub fn largest_row(&self) -> f64 {
        self.rows.iter().copied().reduce(f64::max).unwrap_or(1.0)
    }

    /// The scales of `mode` for `spec` from the Jacobian `values`.
    pub fn of(mode: Scaling, spec: &ProblemSpec, values: &[f64]) -> Self {
        match mode {
            Scaling::None => Self::identity(spec.n(), spec.m()),
            Scaling::Linear => Self::geometric(spec.jacobian(), values, spec.linear_rows()),
            Scaling::Jacobian => Self::geometric(spec.jacobian(), values, &vec![true; spec.m()]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_are_powers_of_two_and_equalize_a_badly_scaled_matrix() {
        // Rows of very different size: [1e4, 2e4; 3e-3, 1e-3].
        let pattern = CsrPattern::dense(2, 2);
        let values = [1e4, 2e4, 3e-3, 1e-3];
        let s = Scales::geometric(&pattern, &values, &[true, true]);
        for v in s.variables.iter().chain(&s.rows) {
            assert_eq!(v.log2().fract(), 0.0, "{v}");
        }
        let scaled: Vec<f64> = (0..4)
            .map(|k| values[k] * s.rows[k / 2] * s.variables[k % 2])
            .collect();
        let (low, high) = scaled
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(high / low < 16.0, "{scaled:?}");
    }

    #[test]
    fn unselected_rows_and_empty_columns_keep_the_factor_one() {
        // Row 0 selected, using only column 0; row 1 not selected.
        let pattern = CsrPattern {
            rows: 2,
            cols: 3,
            indptr: vec![0, 1, 3],
            indices: vec![0, 1, 2],
        };
        let s = Scales::geometric(&pattern, &[1e6, 5.0, 7.0], &[true, false]);
        assert_eq!(s.rows[1], 1.0);
        assert_eq!(&s.variables[1..], &[1.0, 1.0]);
        assert!((s.rows[0] * 1e6 * s.variables[0]).log2().abs() <= 1.0);
        // Nothing selected, or only zeros: the identity.
        assert!(Scales::geometric(&pattern, &[1e6, 5.0, 7.0], &[false, false]).is_identity());
        assert!(Scales::geometric(&pattern, &[0.0, 5.0, 7.0], &[true, false]).is_identity());
    }

    #[test]
    fn scaling_names_round_trip() {
        for mode in [Scaling::None, Scaling::Linear, Scaling::Jacobian] {
            assert_eq!(Scaling::parse(mode.as_str()).unwrap(), mode);
        }
        assert!(Scaling::parse("auto").is_err());
        assert_eq!(power_of_two(3.0), 4.0);
        assert_eq!(power_of_two(f64::NAN), 1.0);
        assert_eq!(power_of_two(1e300), 2.0_f64.powi(LIMIT));
    }
}
