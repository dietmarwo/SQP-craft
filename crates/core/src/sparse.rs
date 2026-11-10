//! Internal sparse-factor utilities shared by the SQP driver and QP adapter.

use crate::{Bounds, CsrPattern, Error, Result};
use std::collections::BTreeSet;

/// Maximum estimated sparse QP storage supported by the native adapters.
pub const MAX_MEMORY_BYTES: usize = 128 * 1024 * 1024;

/// A conservative charge per graph/factor entry, including symbolic scratch.
const ENTRY_BYTES: usize = 96;

/// Estimated bytes for one symbolic or numeric structure of this size.
pub fn factor_bytes(entries: usize) -> usize {
    entries.saturating_mul(ENTRY_BYTES)
}

/// Checks an estimate before a sparse allocation or graph expansion.
pub fn check_memory(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        return Err(Error::Unsupported(format!(
            "structured memory estimate {bytes} bytes exceeds the {limit}-byte budget"
        )));
    }
    Ok(())
}

/// Lower-triangular graph stored by column, in the supplied variable order.
pub struct Graph {
    columns: Vec<BTreeSet<usize>>,
    entries: usize,
    limit: usize,
}

impl Graph {
    /// Starts with the diagonal, including structurally empty variables.
    pub fn new(n: usize, limit: usize) -> Result<Self> {
        check_memory(n.saturating_mul(ENTRY_BYTES), limit)?;
        Ok(Self {
            columns: (0..n).map(|j| BTreeSet::from([j])).collect(),
            entries: n,
            limit,
        })
    }

    /// Adds a symmetric entry, given its lower-triangle coordinates.
    pub fn insert(&mut self, row: usize, column: usize) -> Result<()> {
        if self.columns[column].insert(row) {
            self.entries += 1;
            check_memory(self.estimated_bytes(), self.limit)?;
        }
        Ok(())
    }

    /// Adds the clique of a sorted list of columns.
    pub fn clique(&mut self, columns: &[usize]) -> Result<()> {
        for (i, &row) in columns.iter().enumerate() {
            for &column in &columns[..i] {
                self.insert(row, column)?;
            }
        }
        Ok(())
    }

    /// Estimate including both the graph and the flattened factor.
    pub fn estimated_bytes(&self) -> usize {
        factor_bytes(self.entries)
    }

    /// Flattens the unfilled graph to lower-triangular CSR.
    pub fn into_pattern(self) -> CsrPattern {
        let n = self.columns.len();
        let mut rows = vec![Vec::new(); n];
        for (j, column) in self.columns.into_iter().enumerate() {
            for i in column {
                rows[i].push(j);
            }
        }
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(self.entries);
        indptr.push(0);
        for row in rows {
            indices.extend(row);
            indptr.push(indices.len());
        }
        CsrPattern {
            rows: n,
            cols: n,
            indptr,
            indices,
        }
    }

    /// Builds the natural-order Cholesky pattern, including diagonal entries.
    /// A pivot forwards its later neighbors to its first later neighbor.
    /// That neighbor propagates the rest of the clique before any of those
    /// columns is factored.
    pub fn factor(mut self) -> Result<Symbolic> {
        let mut indptr = Vec::with_capacity(self.columns.len() + 1);
        let mut indices = Vec::with_capacity(self.entries);
        indptr.push(0);
        for j in 0..self.columns.len() {
            let column = std::mem::take(&mut self.columns[j]);
            let mut higher = column.iter().copied().skip(1);
            if let Some(parent) = higher.next() {
                for row in higher {
                    self.insert(row, parent)?;
                }
            }
            indices.extend(column);
            indptr.push(indices.len());
        }
        Ok(Symbolic { indptr, indices })
    }
}

/// Symbolic natural-order Cholesky in lower-triangular CSC.
pub struct Symbolic {
    indptr: Vec<usize>,
    indices: Vec<usize>,
}

impl Symbolic {
    /// Analyses a validated lower-triangular CSR pattern.
    pub fn of(pattern: &CsrPattern, budget: usize) -> Result<Self> {
        let mut graph = Graph::new(pattern.rows, budget)?;
        for i in 0..pattern.rows {
            for &j in pattern.row(i) {
                graph.insert(i, j)?;
            }
        }
        graph.factor()
    }

    /// Number of factor entries, including the diagonal.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn slot(&self, row: usize, column: usize) -> usize {
        let start = self.indptr[column];
        start
            + self.indices[start..self.indptr[column + 1]]
                .binary_search(&row)
                .expect("symbolic Cholesky entry")
    }

    /// Runs numeric Cholesky on the analysed pattern. No regularization,
    /// pivot replacement, or dense fallback is used.
    pub fn positive_definite(
        &self,
        input: &CsrPattern,
        values: &[f64],
        factor: &mut Vec<f64>,
    ) -> bool {
        factor.resize(self.nnz(), 0.0);
        factor.fill(0.0);
        for i in 0..input.rows {
            for k in input.indptr[i]..input.indptr[i + 1] {
                factor[self.slot(i, input.indices[k])] = values[k];
            }
        }
        for j in 0..input.rows {
            let diagonal = self.indptr[j];
            let end = self.indptr[j + 1];
            let pivot = factor[diagonal];
            if !pivot.is_finite() || pivot <= 0.0 {
                return false;
            }
            let root = pivot.sqrt();
            for value in &mut factor[diagonal + 1..end] {
                *value /= root;
            }
            for a in diagonal + 1..end {
                let c = self.indices[a];
                for b in a..end {
                    let k = self.slot(self.indices[b], c);
                    factor[k] -= factor[a] * factor[b];
                }
            }
        }
        true
    }
}

/// Estimates the structural KKT factor with bounded-row duals eliminated
/// first, then variables in input order. Box-bound duals only add diagonal
/// contributions. This is a budgeting estimate, not the backend's ordering.
pub fn kkt_fill(
    hessian: &CsrPattern,
    jacobian: &CsrPattern,
    rows: &Bounds,
    variables: &Bounds,
    budget: usize,
) -> Result<usize> {
    fn sides(lo: f64, hi: f64) -> usize {
        if lo == hi {
            1
        } else {
            usize::from(lo.is_finite()) + usize::from(hi.is_finite())
        }
    }
    let mut dual_entries = 0_usize;
    for (&lo, &hi) in variables.lower.iter().zip(&variables.upper) {
        dual_entries = dual_entries.saturating_add(2 * sides(lo, hi));
    }
    for i in 0..jacobian.rows {
        dual_entries = dual_entries.saturating_add(
            (jacobian.row(i).len() + 1).saturating_mul(sides(rows.lower[i], rows.upper[i])),
        );
    }
    let dual_bytes = dual_entries.saturating_mul(ENTRY_BYTES);
    check_memory(dual_bytes, budget)?;
    let mut graph = Graph::new(hessian.rows, budget - dual_bytes)?;
    for i in 0..hessian.rows {
        for &j in hessian.row(i) {
            graph.insert(i, j)?;
        }
    }
    for i in 0..jacobian.rows {
        if sides(rows.lower[i], rows.upper[i]) > 0 {
            graph.clique(jacobian.row(i))?;
        }
    }
    Ok(graph.factor()?.nnz().saturating_add(dual_entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_fill_matches_explicit_elimination() {
        let n = 23;
        for seed in 0..31 {
            let mut graph = Graph::new(n, MAX_MEMORY_BYTES).unwrap();
            let mut dense = vec![vec![false; n]; n];
            for (j, column) in dense.iter_mut().enumerate() {
                for (i, entry) in column.iter_mut().enumerate().skip(j + 1) {
                    if (i * 31 + j * 17 + i * j + seed * 13) % 31 < 5 {
                        graph.insert(i, j).unwrap();
                        *entry = true;
                    }
                }
            }
            let symbolic = graph.factor().unwrap();
            for j in 0..n {
                let higher: Vec<_> = (j + 1..n).filter(|&i| dense[j][i]).collect();
                let mut expected = vec![j];
                expected.extend(&higher);
                assert_eq!(
                    symbolic.indices[symbolic.indptr[j]..symbolic.indptr[j + 1]],
                    expected
                );
                for (k, &row) in higher.iter().enumerate() {
                    for &col in &higher[..k] {
                        dense[col][row] = true;
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_cholesky_checks_a_connected_nondominant_matrix() {
        let mut graph = Graph::new(3, MAX_MEMORY_BYTES).unwrap();
        graph.insert(1, 0).unwrap();
        graph.insert(2, 1).unwrap();
        let pattern = graph.into_pattern();
        let symbolic = Symbolic::of(&pattern, MAX_MEMORY_BYTES).unwrap();
        let mut scratch = vec![];
        assert!(symbolic.positive_definite(&pattern, &[4., 2., 3., 2., 4.], &mut scratch));
        assert!(!symbolic.positive_definite(&pattern, &[4., 2., 2., 2., 4.], &mut scratch));
    }
}
