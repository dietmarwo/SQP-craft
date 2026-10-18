//! Column-major dense kernels on `n × n` buffers: Cholesky, `L⁻ᵀ` and Givens
//! rotations. Plain loops over contiguous column slices, so the compiler can
//! vectorize them without unsafe code or a BLAS.

/// Overwrites the lower triangle of `a` (column-major `n × n`) with its
/// Cholesky factor `L`, `A = L Lᵀ`. Only the lower triangle is read.
///
/// Returns `false` as soon as a pivot is not positive and finite: then `A`
/// is not positive definite in floating point and `a` is left partly
/// overwritten. Zero entries of `L` are skipped, so a block-diagonal `A`
/// costs only its blocks.
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> bool {
    debug_assert_eq!(a.len(), n * n);
    for k in 0..n {
        let (done, rest) = a.split_at_mut(k * n);
        let column = &mut rest[..n];
        for p in 0..k {
            let factor = &done[p * n..(p + 1) * n];
            let f = factor[k];
            if f != 0.0 {
                axpy(-f, &factor[k..], &mut column[k..]);
            }
        }
        let pivot = column[k];
        if !(pivot > 0.0 && pivot.is_finite()) {
            return false;
        }
        let d = pivot.sqrt();
        column[k] = d;
        for v in &mut column[k + 1..] {
            *v /= d;
        }
    }
    true
}

/// `j = L⁻ᵀ` (upper triangular) from the factor in the lower triangle of `l`.
pub(crate) fn inverse_transpose(l: &[f64], j: &mut [f64], n: usize) {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(j.len(), n * n);
    j.fill(0.0);
    // Column k solves Lᵀ y = e_k; y vanishes below row k.
    for k in 0..n {
        let column = &mut j[k * n..(k + 1) * n];
        column[k] = 1.0 / l[k * n + k];
        for i in (0..k).rev() {
            let li = &l[i * n..(i + 1) * n];
            column[i] = -dot(&li[i + 1..=k], &column[i + 1..=k]) / li[i];
        }
    }
}

/// `Σ a_i b_i` with four partial sums, so the loop vectorizes.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0.0; 4];
    let (a4, a1) = a.split_at(a.len() - a.len() % 4);
    let (b4, b1) = b.split_at(a4.len());
    for (x, y) in a4.chunks_exact(4).zip(b4.chunks_exact(4)) {
        for lane in 0..4 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let tail: f64 = a1.iter().zip(b1).map(|(x, y)| x * y).sum();
    (sums[0] + sums[1]) + (sums[2] + sums[3]) + tail
}

/// `y += alpha x`.
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Euclidean norm, robust against overflow only as far as `f64` allows;
/// QP data are finite and moderately scaled.
pub(crate) fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Givens rotation `(c, s, r)` with `[c s; -s c] [a; b] = [r; 0]`, `r ≥ 0`.
pub(crate) fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    let r = a.hypot(b);
    if r == 0.0 {
        (1.0, 0.0, 0.0)
    } else {
        (a / r, b / r, r)
    }
}

/// Applies `[c s; -s c]` to the pairs `(x_i, y_i)`.
pub(crate) fn rotate(x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
    debug_assert_eq!(x.len(), y.len());
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (*xi, *yi);
        *xi = c * a + s * b;
        *yi = c * b - s * a;
    }
}

/// Applies a Householder reflector to the columns `q..` of `j` (column-major,
/// `n` rows) so that `tail = Jᵀn` restricted to those columns becomes
/// `(‖tail‖, 0, …, 0)`: `J₂ ← J₂ H`, `tail ← H tail`. It costs `4 n·len`
/// operations, against `6 n·len` for a chain of Givens rotations. The first
/// column is negated if needed, so the leading entry is nonnegative. `w` is
/// scratch of length `n`.
pub(crate) fn reflect(j: &mut [f64], n: usize, q: usize, tail: &mut [f64], w: &mut [f64]) {
    let columns = &mut j[q * n..];
    debug_assert_eq!(columns.len(), tail.len() * n);
    let used = tail.iter().rposition(|&v| v != 0.0).map_or(1, |i| i + 1);
    let columns = &mut columns[..used * n];
    if used > 1 {
        let size = norm(tail);
        let tail = &mut tail[..used];
        let alpha = tail[0];
        let beta = if alpha >= 0.0 { -size } else { size };
        // H = I − τ v vᵀ with v = tail − β e₁ and vᵀv = 2‖tail‖(‖tail‖ + |α|).
        tail[0] = alpha - beta;
        let tau = 1.0 / (size * (size + alpha.abs()));
        w.fill(0.0);
        for (&v, column) in tail.iter().zip(columns.chunks_exact(n)) {
            if v != 0.0 {
                axpy(v, column, w);
            }
        }
        for (&v, column) in tail.iter().zip(columns.chunks_exact_mut(n)) {
            if v != 0.0 {
                axpy(-tau * v, w, column);
            }
        }
        tail[0] = beta;
        tail[1..].fill(0.0);
    }
    if tail[0] < 0.0 {
        tail[0] = -tail[0];
        for v in &mut columns[..n] {
            *v = -*v;
        }
    }
}

/// Columns `a < b` of a column-major matrix with `n` rows, both mutable.
pub(crate) fn column_pair(m: &mut [f64], n: usize, a: usize, b: usize) -> (&mut [f64], &mut [f64]) {
    debug_assert!(a < b);
    let (low, high) = m.split_at_mut(b * n);
    (&mut low[a * n..(a + 1) * n], &mut high[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Vec<f64> {
        // A = M Mᵀ + n I for a fixed pseudo-random M.
        let mut seed = 12345_u64;
        let mut m = vec![0.0; n * n];
        for v in &mut m {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        }
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[j * n + i] = (0..n).map(|k| m[k * n + i] * m[k * n + j]).sum::<f64>()
                    + if i == j { n as f64 } else { 0.0 };
            }
        }
        a
    }

    #[test]
    fn cholesky_and_inverse_transpose_reproduce_the_identity() {
        for n in [1, 2, 5, 17] {
            let a = spd(n);
            let mut l = a.clone();
            assert!(cholesky(&mut l, n));
            let mut j = vec![0.0; n * n];
            inverse_transpose(&l, &mut j, n);
            // Jᵀ A J = I with A from its full (symmetric) storage.
            for p in 0..n {
                for q in 0..n {
                    let mut v = 0.0;
                    for i in 0..n {
                        for k in 0..n {
                            v += j[p * n + i] * a[k * n + i] * j[q * n + k];
                        }
                    }
                    let expected = if p == q { 1.0 } else { 0.0 };
                    assert!((v - expected).abs() < 1e-12, "n={n} ({p},{q}) {v}");
                }
            }
        }
    }

    #[test]
    fn cholesky_rejects_indefinite_singular_and_nonfinite_matrices() {
        let indefinite = [1.0, 2.0, 2.0, 1.0];
        let singular = [1.0, 1.0, 1.0, 1.0];
        let nan = [f64::NAN, 0.0, 0.0, 1.0];
        for a in [indefinite, singular, nan] {
            assert!(!cholesky(&mut a.to_vec(), 2), "{a:?}");
        }
        // The upper triangle is never read.
        assert!(cholesky(&mut [4.0, 2.0, f64::NAN, 5.0], 2));
    }

    #[test]
    fn reflect_maps_the_tail_to_its_norm_and_keeps_jt_n_consistent() {
        let n = 5;
        let mut j: Vec<f64> = spd(n);
        let normal = [0.3, -1.0, 0.0, 2.0, 0.5];
        let project = |j: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|k| dot(&j[k * n..(k + 1) * n], &normal))
                .collect()
        };
        for q in 0..n {
            let mut m = j.clone();
            let mut d = project(&m);
            let size = norm(&d[q..]);
            let mut w = vec![0.0; n];
            reflect(&mut m, n, q, &mut d[q..], &mut w);
            assert!((d[q] - size).abs() < 1e-12 * size, "q={q}");
            assert!(d[q + 1..].iter().all(|&v| v == 0.0));
            // The stored d is Jᵀn for the reflected J, and columns before q
            // are untouched.
            for (a, b) in d.iter().zip(project(&m)) {
                assert!((a - b).abs() < 1e-11 * size, "q={q}: {a} vs {b}");
            }
            assert_eq!(m[..q * n], j[..q * n]);
        }
        j.fill(0.0);
    }

    #[test]
    fn givens_zeroes_the_second_entry_and_dot_matches_the_plain_sum() {
        let (c, s, r) = givens(3.0, -4.0);
        let (mut x, mut y) = ([3.0], [-4.0]);
        rotate(&mut x, &mut y, c, s);
        assert_eq!(r, 5.0);
        assert!((x[0] - 5.0).abs() < 1e-15 && y[0].abs() < 1e-15);
        assert_eq!(givens(0.0, 0.0), (1.0, 0.0, 0.0));
        let a: Vec<f64> = (0..11).map(|i| i as f64).collect();
        assert_eq!(dot(&a, &a), (0..11).map(|i| (i * i) as f64).sum::<f64>());
    }
}
