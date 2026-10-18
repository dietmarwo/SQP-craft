//! Bounds, sparse structure and the validated problem specification.

use crate::error::{Result, ensure, first_non_finite};

/// Elementwise bounds `lower[i] <= v[i] <= upper[i]`.
///
/// Equal finite bounds express an equality. `-inf` lower or `+inf` upper
/// bounds express one-sided or free entries. NaN, `lower = +inf`,
/// `upper = -inf` and `lower > upper` are invalid.
#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    /// Lower bounds; `f64::NEG_INFINITY` for none.
    pub lower: Vec<f64>,
    /// Upper bounds; `f64::INFINITY` for none.
    pub upper: Vec<f64>,
}

impl Bounds {
    /// Bounds from two vectors. Validation happens where the bounds are used.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self { lower, upper }
    }

    /// `n` free entries.
    pub fn free(n: usize) -> Self {
        Self::new(vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n])
    }

    /// `n` equalities `v[i] = value[i]`.
    pub fn fixed(values: Vec<f64>) -> Self {
        Self::new(values.clone(), values)
    }

    /// Number of entries (length of `lower`).
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Checks lengths against `n` and every pair for consistency. `what` names
    /// the bounds in the error message.
    pub fn validate(&self, n: usize, what: &str) -> Result<()> {
        ensure(self.lower.len() == n && self.upper.len() == n, || {
            format!(
                "{what}: expected {n} entries, got lower={} and upper={}",
                self.lower.len(),
                self.upper.len()
            )
        })?;
        for (i, (&l, &u)) in self.lower.iter().zip(&self.upper).enumerate() {
            let ok = !l.is_nan()
                && !u.is_nan()
                && l != f64::INFINITY
                && u != f64::NEG_INFINITY
                && l <= u;
            ensure(ok, || {
                format!(
                    "{what}: entry {i} has lower={l}, upper={u}; need lower <= upper, \
                     no NaN, lower < +inf and upper > -inf"
                )
            })?;
        }
        Ok(())
    }

    /// Largest amount by which `values` exceed the bounds, or `0.0` when all
    /// entries are within them. Absolute and in the units of `values`.
    ///
    /// # Panics
    /// If bounds/point are invalid or a finite-bound distance overflows. Use
    /// [`Self::checked_violation`] for a fallible call.
    pub fn violation(&self, values: &[f64]) -> f64 {
        self.checked_violation(values)
            .expect("invalid bounds/point or nonfinite violation")
    }

    /// Checked bound violation, rejecting malformed inputs and arithmetic overflow.
    pub fn checked_violation(&self, values: &[f64]) -> Result<f64> {
        self.validate(values.len(), "bounds")?;
        self.violation_with_valid_bounds(values)
    }

    /// Bounds/lengths must have been validated by the caller. Values and arithmetic remain checked.
    pub(crate) fn violation_with_valid_bounds(&self, values: &[f64]) -> Result<f64> {
        debug_assert_eq!(values.len(), self.lower.len());
        debug_assert_eq!(values.len(), self.upper.len());
        let mut result = 0.0_f64;
        for (i, &v) in values.iter().enumerate() {
            ensure(v.is_finite(), || format!("point entry {i} must be finite"))?;
            if self.lower[i].is_finite() {
                result = result.max(crate::error::finite(
                    self.lower[i] - v,
                    "lower-bound distance",
                )?);
            }
            if self.upper[i].is_finite() {
                result = result.max(crate::error::finite(
                    v - self.upper[i],
                    "upper-bound distance",
                )?);
            }
        }
        Ok(result)
    }

    /// Index of the first entry of `values` outside the bounds, if any.
    ///
    /// # Panics
    /// If bounds/point are invalid or a finite-bound distance overflows. Use
    /// [`Self::checked_violation`] for a fallible call.
    pub fn first_outside(&self, values: &[f64]) -> Option<usize> {
        assert_eq!(values.len(), self.len(), "bounds/values length mismatch");
        (0..values.len()).find(|&i| !(self.lower[i] <= values[i] && values[i] <= self.upper[i]))
    }
}

/// Structure of a sparse matrix in zero-based compressed sparse row (CSR) form.
///
/// Row `r` holds the column indices `indices[indptr[r]..indptr[r + 1]]`, which
/// must be strictly increasing (sorted, no duplicates) and smaller than `cols`.
/// Values belonging to this pattern are stored in the same order as `indices`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrPattern {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row pointers, `rows + 1` entries, starting at 0 and ending at `indices.len()`.
    pub indptr: Vec<usize>,
    /// Column index of every stored entry.
    pub indices: Vec<usize>,
}

impl CsrPattern {
    /// Pattern of a dense `rows x cols` matrix, stored row by row.
    pub fn dense(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            indptr: (0..=rows).map(|r| r * cols).collect(),
            indices: (0..rows).flat_map(|_| 0..cols).collect(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Column indices of row `r`.
    ///
    /// # Panics
    /// If `r >= self.rows` or the pattern is invalid.
    pub fn row(&self, r: usize) -> &[usize] {
        &self.indices[self.indptr[r]..self.indptr[r + 1]]
    }

    /// Checks the CSR invariants. `what` names the matrix in the error message.
    pub fn validate(&self, what: &str) -> Result<()> {
        ensure(self.rows.checked_add(1) == Some(self.indptr.len()), || {
            format!(
                "{what}: indptr has {} entries, expected rows + 1 with rows = {}",
                self.indptr.len(),
                self.rows
            )
        })?;
        ensure(
            self.indptr[0] == 0 && self.indptr[self.rows] == self.indices.len(),
            || {
                format!(
                    "{what}: indptr must start at 0 and end at the number of indices ({}), \
                     got {} and {}",
                    self.indices.len(),
                    self.indptr[0],
                    self.indptr[self.rows]
                )
            },
        )?;
        for r in 0..self.rows {
            let (start, end) = (self.indptr[r], self.indptr[r + 1]);
            ensure(start <= end && end <= self.indices.len(), || {
                format!("{what}: indptr decreases at row {r} ({start} > {end})")
            })?;
            let row = &self.indices[start..end];
            ensure(row.windows(2).all(|p| p[0] < p[1]), || {
                format!("{what}: row {r} column indices must be strictly increasing, got {row:?}")
            })?;
            if row.last().is_some_and(|&j| j >= self.cols) {
                let j = *row
                    .iter()
                    .find(|&&j| j >= self.cols)
                    .expect("last index outside");
                return Err(crate::Error::InvalidInput(format!(
                    "{what}: row {r} has column index {j}, but there are only {} columns",
                    self.cols
                )));
            }
        }
        Ok(())
    }

    /// `y = A x`, where `A` has this pattern and `values`.
    ///
    /// # Panics
    /// On length mismatches.
    pub fn multiply(&self, values: &[f64], x: &[f64], y: &mut [f64]) {
        assert_eq!(values.len(), self.nnz(), "values length");
        assert_eq!(x.len(), self.cols, "x length");
        assert_eq!(y.len(), self.rows, "y length");
        for (r, out) in y.iter_mut().enumerate() {
            let range = self.indptr[r]..self.indptr[r + 1];
            *out = if range.len() == 1 {
                0.0 + values[range.start] * x[self.indices[range.start]]
            } else if range.len() == self.cols {
                // A full, valid, strictly sorted row has columns 0..cols.
                values[range].iter().zip(x).map(|(&a, &v)| a * v).sum()
            } else {
                self.indices[range.clone()]
                    .iter()
                    .zip(&values[range])
                    .map(|(&j, &a)| a * x[j])
                    .sum()
            };
        }
    }

    /// `y = A^T x`, where `A` has this pattern and `values`.
    ///
    /// # Panics
    /// On length mismatches.
    pub fn multiply_transpose(&self, values: &[f64], x: &[f64], y: &mut [f64]) {
        assert_eq!(values.len(), self.nnz(), "values length");
        assert_eq!(x.len(), self.rows, "x length");
        assert_eq!(y.len(), self.cols, "y length");
        y.fill(0.0);
        for (r, &xr) in x.iter().enumerate() {
            let range = self.indptr[r]..self.indptr[r + 1];
            for (&j, &a) in self.indices[range.clone()].iter().zip(&values[range]) {
                y[j] += a * xr;
            }
        }
    }
}

/// A sparse matrix: a [`CsrPattern`] plus one value per stored entry.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    /// Structure.
    pub pattern: CsrPattern,
    /// Values in the order of `pattern.indices`.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Checks the pattern, the number of values and their finiteness.
    pub fn validate(&self, what: &str) -> Result<()> {
        self.pattern.validate(what)?;
        ensure(self.values.len() == self.pattern.nnz(), || {
            format!(
                "{what}: {} values for {} stored entries",
                self.values.len(),
                self.pattern.nnz()
            )
        })?;
        match first_non_finite(&self.values) {
            Some((k, v)) => Err(crate::Error::InvalidInput(format!(
                "{what}: value {k} is {v}"
            ))),
            None => Ok(()),
        }
    }
}

/// The immutable, validated part of a nonlinear program
///
/// ```text
/// minimize f(x)  subject to  constraint_lower <= c(x) <= constraint_upper,
///                            variable_lower   <= x    <= variable_upper,
/// ```
///
/// with `n >= 1` variables, `m >= 0` constraints and the fixed sparsity
/// pattern of the constraint Jacobian `J[i][j] = dc_i/dx_j`.
///
/// A `ProblemSpec` can only be obtained through [`ProblemSpec::new`], so every
/// instance satisfies its invariants and solvers do not re-validate it. It can
/// be shared (for example behind an `Arc`) by any number of independent solves.
#[derive(Clone, Debug, PartialEq)]
pub struct ProblemSpec {
    variables: Bounds,
    constraints: Bounds,
    jacobian: CsrPattern,
    linear: Vec<bool>,
}

impl ProblemSpec {
    /// Validates and assembles a specification.
    ///
    /// The number of variables is `variables.len()`, the number of constraints
    /// `constraints.len()`; the Jacobian pattern must be `m x n`.
    ///
    /// The pattern must contain every entry that can be nonzero anywhere in the
    /// domain, because its order defines the layout of the Jacobian values the
    /// model returns.
    pub fn new(variables: Bounds, constraints: Bounds, jacobian: CsrPattern) -> Result<Self> {
        let n = variables.len();
        let m = constraints.len();
        ensure(n > 0, || "at least one variable is required".into())?;
        variables.validate(n, "variable bounds")?;
        constraints.validate(m, "constraint bounds")?;
        ensure(jacobian.rows == m && jacobian.cols == n, || {
            format!(
                "Jacobian pattern: shape is {} x {}, expected m x n = {m} x {n}",
                jacobian.rows, jacobian.cols
            )
        })?;
        jacobian.validate("Jacobian pattern")?;
        Ok(Self {
            variables,
            constraints,
            jacobian,
            linear: vec![false; m],
        })
    }

    /// Declares the rows `rows` linear: `c_i(x) = a_iᵀx + b_i`, so their
    /// Jacobian rows are constant. A solver may then keep them satisfied at
    /// every iterate and leave them out of its merit function; the native
    /// SQP ends a solve with `InvalidInput` if a declared row's Jacobian
    /// entries change. Indices must be distinct and smaller than `m`.
    pub fn with_linear_rows(mut self, rows: &[usize]) -> Result<Self> {
        let mut linear = vec![false; self.m()];
        for &i in rows {
            ensure(i < self.m(), || {
                format!(
                    "linear rows: index {i} is out of range for m = {}",
                    self.m()
                )
            })?;
            ensure(!linear[i], || format!("linear rows: index {i} is repeated"))?;
            linear[i] = true;
        }
        self.linear = linear;
        Ok(self)
    }

    /// Per row, whether it was declared linear (length `m`).
    pub fn linear_rows(&self) -> &[bool] {
        &self.linear
    }

    /// Whether any row was declared linear.
    pub fn has_linear_rows(&self) -> bool {
        self.linear.iter().any(|&l| l)
    }

    /// Number of variables `n`.
    pub fn n(&self) -> usize {
        self.variables.len()
    }

    /// Number of constraints `m`.
    pub fn m(&self) -> usize {
        self.constraints.len()
    }

    /// Number of structural nonzeros of the Jacobian.
    pub fn nnz(&self) -> usize {
        self.jacobian.nnz()
    }

    /// Variable bounds.
    pub fn variables(&self) -> &Bounds {
        &self.variables
    }

    /// Constraint bounds.
    pub fn constraints(&self) -> &Bounds {
        &self.constraints
    }

    /// Jacobian sparsity pattern (`m x n`).
    pub fn jacobian(&self) -> &CsrPattern {
        &self.jacobian
    }

    /// Checks that `x` is a finite point of length `n` within the variable bounds.
    ///
    /// Solvers call this for a start point, so the model is never evaluated
    /// outside the box. A point is rejected, never silently projected.
    pub fn check_point(&self, x: &[f64], what: &str) -> Result<()> {
        ensure(x.len() == self.n(), || {
            format!("{what}: expected {} entries, got {}", self.n(), x.len())
        })?;
        if let Some((i, v)) = first_non_finite(x) {
            return Err(crate::Error::InvalidInput(format!(
                "{what}: entry {i} is {v}"
            )));
        }
        if let Some(i) = self.variables.first_outside(x) {
            return Err(crate::Error::InvalidInput(format!(
                "{what}: entry {i} = {} lies outside the variable bounds [{}, {}]",
                x[i], self.variables.lower[i], self.variables.upper[i]
            )));
        }
        Ok(())
    }
}
