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
    inverse_transpose_impl(l, j, n, false);
}

pub(crate) fn inverse_transpose_tiled(l: &[f64], j: &mut [f64], n: usize) {
    inverse_transpose_impl(l, j, n, true);
}

fn inverse_transpose_impl(l: &[f64], j: &mut [f64], n: usize, tiled: bool) {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(j.len(), n * n);
    j.fill(0.0);
    if n == 0 {
        return;
    }
    if l.chunks_exact(n)
        .enumerate()
        .all(|(k, col)| col[k + 1..].iter().all(|&v| v == 0.0))
    {
        // The general recurrence below gives −0.0 above the diagonal
        // (−dot/l with dot = +0.0 and l > 0); keep those bits.
        for k in 0..n {
            j[k * n..k * n + k].fill(-0.0);
            j[k * n + k] = 1.0 / l[k * n + k];
        }
        return;
    }
    if tiled && (80..=256).contains(&n) {
        inverse_panels(l, j, n);
        return;
    }
    inverse_columns(l, j, n);
}

fn inverse_columns(l: &[f64], j: &mut [f64], n: usize) {
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

// Traverse several right-hand sides while a factor column is still in
// cache. Keep the original dot product and within-column dependency order.
fn inverse_panels(l: &[f64], j: &mut [f64], n: usize) {
    const WIDTH: usize = 8;
    for start in (0..n).step_by(WIDTH) {
        let end = (start + WIDTH).min(n);
        let panel = &mut j[start * n..end * n];
        for (k, column) in (start..end).zip(panel.chunks_exact_mut(n)) {
            column[k] = 1.0 / l[k * n + k];
        }
        for i in (0..end - 1).rev() {
            let factor = &l[i * n..(i + 1) * n];
            let first = (i + 1).max(start);
            for (k, column) in (first..end).zip(panel[(first - start) * n..].chunks_exact_mut(n)) {
                column[i] = -dot(&factor[i + 1..=k], &column[i + 1..=k]) / factor[i];
            }
        }
    }
}

/// Independent inverse columns with unchanged within-column arithmetic.
pub(crate) fn inverse_transpose_parallel(
    l: &[f64],
    j: &mut [f64],
    n: usize,
    execution: &sqp_craft_parallel::Execution,
) {
    if n < 512
        || execution.threads() == 1
        || l.chunks_exact(n)
            .enumerate()
            .all(|(k, col)| col[k + 1..].iter().all(|&v| v == 0.0))
    {
        inverse_transpose(l, j, n);
        return;
    }
    j.fill(0.0);
    execution.chunks_mut(j, n, n * n * n / 3, |start, column| {
        let k = start / n;
        column[k] = 1.0 / l[k * n + k];
        for i in (0..k).rev() {
            let li = &l[i * n..(i + 1) * n];
            column[i] = -dot(&li[i + 1..=k], &column[i + 1..=k]) / li[i];
        }
    });
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
#[cfg(test)]
pub(crate) fn reflect(j: &mut [f64], n: usize, q: usize, tail: &mut [f64], w: &mut [f64]) {
    reflect_parallel(j, n, q, tail, w, &sqp_craft_parallel::Execution::default());
}

/// Disjoint reflector target columns, with an ordered unchanged accumulator.
pub(crate) fn reflect_parallel(
    j: &mut [f64],
    n: usize,
    q: usize,
    tail: &mut [f64],
    w: &mut [f64],
    execution: &sqp_craft_parallel::Execution,
) {
    let columns = &mut j[q * n..];
    debug_assert_eq!(columns.len(), tail.len() * n);
    let used = tail.iter().rposition(|&v| v != 0.0).map_or(1, |i| i + 1);
    let columns = &mut columns[..used * n];
    if used > 1 {
        let original_size = norm(tail);
        let tail = &mut tail[..used];
        // Tiny HS25NE normals made size^2 underflow and tau infinite. Outside
        // the safe product range, divide by the largest magnitude: unlike the
        // plain norm, it neither underflows to zero nor overflows (`used > 1`
        // means a nonzero entry). The ordinary path keeps its arithmetic and
        // its parallel accumulation order.
        let product = original_size * (original_size + tail[0].abs());
        let normalized = !product.is_finite() || product < f64::MIN_POSITIVE;
        let scale = if normalized {
            tail.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        } else {
            1.0
        };
        if normalized {
            for value in tail.iter_mut() {
                *value /= scale;
            }
        }
        let size = if normalized {
            norm(tail)
        } else {
            original_size
        };
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
        if n >= 512 && execution.threads() > 1 {
            execution.chunks_mut(columns, n, n * used * 3, |start, column| {
                let v = tail[start / n];
                if v != 0.0 {
                    axpy(-tau * v, w, column);
                }
            });
        } else {
            for (&v, column) in tail.iter().zip(columns.chunks_exact_mut(n)) {
                if v != 0.0 {
                    axpy(-tau * v, w, column);
                }
            }
        }

        tail[0] = if normalized { beta * scale } else { beta };
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
    fn tiled_inverse_preserves_each_column_at_partial_panel_boundaries() {
        for n in [80, 81, 164, 196, 255, 256] {
            let mut factor = spd(n);
            assert!(cholesky(&mut factor, n));
            let mut reference = vec![0.0; n * n];
            let mut result = vec![0.0; n * n];
            inverse_columns(&factor, &mut reference, n);
            inverse_transpose_tiled(&factor, &mut result, n);
            assert!(
                reference
                    .iter()
                    .zip(&result)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
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
    fn reflect_handles_tails_whose_squared_norm_underflows_or_overflows() {
        // The reflector depends only on the tail's direction: a tail scaled
        // by c rotates the columns exactly as the unscaled one and keeps c
        // times its norm. At 1e-170 every square underflows to zero, at
        // 1e170 the squared norm overflows; a plain norm divides by 0 or ∞.
        let n = 4;
        let j = spd(n);
        let unit = [0.6, -0.25, 0.5, 0.75];
        let mut expected = j.clone();
        let mut d = unit;
        let mut w = vec![0.0; n];
        reflect(&mut expected, n, 0, &mut d, &mut w);
        let size = d[0];
        for c in [1e-170, 1e-300, 1e170, 1e300] {
            let mut m = j.clone();
            let mut tail: Vec<f64> = unit.iter().map(|v| v * c).collect();
            reflect(&mut m, n, 0, &mut tail, &mut w);
            assert!(m.iter().all(|v| v.is_finite()), "c={c}");
            for (a, b) in m.iter().zip(&expected) {
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                    "c={c}: {a} vs {b}"
                );
            }
            assert!(
                (tail[0] / c - size).abs() <= 1e-12 * size,
                "c={c}: {}",
                tail[0]
            );
            assert!(tail[1..].iter().all(|&v| v == 0.0));
        }
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

#[cfg(test)]
#[test]
fn diagonal_inverse_is_exact_and_tiny_coupling_keeps_the_general_path() {
    for n in [0, 1, 7, 32] {
        let mut l = vec![0.0; n * n];
        for k in 0..n {
            l[k * n + k] = 2.0 + k as f64;
        }
        let mut j = vec![f64::NAN; n * n];
        inverse_transpose(&l, &mut j, n);
        for col in 0..n {
            for row in 0..n {
                // As the general recurrence: −0.0 above the diagonal.
                let expected = match row.cmp(&col) {
                    std::cmp::Ordering::Less => -0.0,
                    std::cmp::Ordering::Equal => 1.0 / (2.0 + col as f64),
                    std::cmp::Ordering::Greater => 0.0,
                };
                assert_eq!(j[col * n + row].to_bits(), f64::to_bits(expected));
            }
        }
    }
    let mut j = vec![0.0; 4];
    inverse_transpose(&[2.0, 1e-12, f64::NAN, 3.0], &mut j, 2);
    assert!((j[2] + 1e-12 / 6.0).abs() < 1e-28);
}

/// Solves L L^T x = rhs in place, using the checked lower factor.
pub(crate) fn solve_factored(l: &[f64], x: &mut [f64], n: usize) {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(x.len(), n);
    for i in 0..n {
        let mut value = x[i];
        for k in 0..i {
            value -= l[k * n + i] * x[k];
        }
        x[i] = value / l[i * n + i];
    }
    for i in (0..n).rev() {
        let col = &l[i * n..(i + 1) * n];
        x[i] = (x[i] - dot(&col[i + 1..], &x[i + 1..])) / col[i];
    }
}

#[cfg(test)]
mod parallel_inverse_tests {
    use super::*;
    #[test]
    fn inverse_columns_are_bit_identical_at_dispatch_boundaries() {
        let execution = sqp_craft_parallel::Execution::new(8).unwrap();
        for n in [0, 1, 31, 127, 128, 129, 256, 511, 512, 768] {
            let mut l = vec![0.0; n * n];
            for k in 0..n {
                l[k * n + k] = 1.0 + k as f64 * 0.01;
                for i in k + 1..n {
                    l[k * n + i] = ((i + k) % 17) as f64 * 0.001;
                }
            }
            let mut a = vec![f64::NAN; n * n];
            let mut b = a.clone();
            inverse_transpose(&l, &mut a, n);
            inverse_transpose_parallel(&l, &mut b, n, &execution);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(test)]
mod parallel_reflector_tests {
    use super::*;
    #[test]
    fn reflector_target_columns_are_bit_identical() {
        let execution = sqp_craft_parallel::Execution::new(8).unwrap();
        for n in [17, 128, 256, 512] {
            for q in [0, n / 2, n - 1] {
                let mut a: Vec<_> = (0..n * n).map(|i| (i % 97) as f64 * 0.001 - 0.02).collect();
                let mut b = a.clone();
                let mut ta: Vec<_> = (0..n - q)
                    .map(|i| {
                        if i % 3 == 0 {
                            0.0
                        } else {
                            (i % 13) as f64 * 0.01 - 0.02
                        }
                    })
                    .collect();
                if ta.iter().all(|&x| x == 0.0) {
                    ta[0] = 0.5;
                }
                let mut tb = ta.clone();
                let mut wa = vec![0.0; n];
                let mut wb = wa.clone();
                reflect(&mut a, n, q, &mut ta, &mut wa);
                reflect_parallel(&mut b, n, q, &mut tb, &mut wb, &execution);
                assert_eq!(a, b);
                assert_eq!(ta, tb);
                assert_eq!(wa, wb);
            }
        }
    }
}
