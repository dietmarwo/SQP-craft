//! SNOPT's augmented-Lagrangian merit function (Gill, Murray and Saunders
//! 2005, §2.7):
//!
//! ```text
//! M_ρ(x, π, s) = f(x) − πᵀ(c(x) − s) + ½ Σ ρᵢ (cᵢ(x) − sᵢ)²
//! ```
//!
//! searched along `(x, π, s) + α (p, π̄ − π, ŝ − s)` toward the QP solution
//! `p` with multipliers `π̄` and linearized slacks `ŝ`. The slacks `s` minimize
//! `M` for the current `(x, π, ρ)` within the row bounds, and the penalties
//! come from the minimum-norm problem (LSPρ) with the damped update (2.4).
use super::{Bounds, Result, finite};

/// Multiplier estimate and penalties carried from one line search to the
/// next.
pub(crate) struct AugmentedLagrangian {
    /// `π`, one per row.
    pub pi: Vec<f64>,
    /// `ρ ≥ 0`, one per row.
    pub rho: Vec<f64>,
    /// `Δρ` of the damped update; doubles when the trend of `‖ρ‖` reverses.
    delta: f64,
    /// Sign of the last change of `‖ρ‖` (0 before the first change).
    trend: i8,
}

/// One line search: the path and the merit value and slope at `α = 0`.
pub(crate) struct Search {
    skip: Vec<bool>,
    slack: Vec<f64>,
    slack_step: Vec<f64>,
    pi_step: Vec<f64>,
    rho: Vec<f64>,
    /// `M` at `α = 0` with the updated penalties.
    pub value: f64,
    /// `−φ′(0)`, at least `½pᵀBp` when the penalties could enforce it.
    pub predicted: f64,
    /// `max ρ̄ᵢ`, for the iteration record.
    pub max_penalty: f64,
    /// `max |πᵢ| + |π̄ᵢ| + ρ̄ᵢ|rᵢ|`: how much `M` changes per unit of row
    /// value, which scales the effect of QP inaccuracy.
    pub sensitivity: f64,
    /// `|f| + Σ|πᵢrᵢ| + ½Σρ̄ᵢrᵢ²`, the size of the terms summed into `M`,
    /// which sets the rounding level of `M` itself.
    pub magnitude: f64,
}

/// The slack that minimizes `−π(c − s) + ½ρ(c − s)²` over `[lower, upper]`.
fn best_slack(c: f64, pi: f64, rho: f64, lower: f64, upper: f64) -> f64 {
    let target = if rho > 0.0 { c - pi / rho } else { c };
    target.clamp(lower, upper)
}

impl AugmentedLagrangian {
    pub fn new(m: usize) -> Self {
        Self {
            pi: vec![0.0; m],
            rho: vec![0.0; m],
            delta: 1.0,
            trend: 0,
        }
    }

    /// Forgets the multiplier estimate and the penalties (after restoration,
    /// whose steps carry no multipliers).
    pub fn reset(&mut self) {
        self.pi.fill(0.0);
        self.rho.fill(0.0);
        self.delta = 1.0;
        self.trend = 0;
    }

    /// Prepares the search from `x` (objective `f`, rows `c`, gradient `g`)
    /// along the QP step `p` with linearized rows `linear = c + Jp`,
    /// curvature `pbp = pᵀBp` and QP multipliers `pi_bar`. Rows flagged in
    /// `skip` (linear rows kept feasible) are left out of the merit. The
    /// penalties are not stored until [`Self::commit`].
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        bounds: &Bounds,
        f: f64,
        c: &[f64],
        linear: &[f64],
        gp: f64,
        pbp: f64,
        pi_bar: &[f64],
        skip: &[bool],
    ) -> Result<Search> {
        let m = c.len();
        let mut slack = vec![0.0; m];
        let mut slack_step = vec![0.0; m];
        let mut pi_step = vec![0.0; m];
        let mut residual = vec![0.0; m];
        // q_i = −r_i r'_i, where r' = Jp − (ŝ − s) is the rate of change of
        // the residual r = c − s along the path; φ′(0) = base − Σ ρ_i q_i.
        let mut q = vec![0.0; m];
        let mut base = gp;
        for i in 0..m {
            if skip.get(i).copied().unwrap_or(false) {
                slack[i] = c[i];
                continue;
            }
            let (lower, upper) = (bounds.lower[i], bounds.upper[i]);
            let s = best_slack(c[i], self.pi[i], self.rho[i], lower, upper);
            let target = linear[i].clamp(lower, upper);
            let r = finite(c[i] - s, "merit residual")?;
            let rate = finite((linear[i] - c[i]) - (target - s), "merit residual rate")?;
            slack[i] = s;
            slack_step[i] = target - s;
            pi_step[i] = pi_bar[i] - self.pi[i];
            residual[i] = r;
            q[i] = -r * rate;
            base = finite(base - pi_step[i] * r - self.pi[i] * rate, "merit slope")?;
        }
        // (LSPρ): minimize ½‖ρ‖² subject to φ′(0) = −½pᵀBp, ρ ≥ 0. With
        // ξ = base + ½pᵀBp the solution is ρ* = ξ q⁺/‖q⁺‖² (0 if ξ ≤ 0).
        let xi = base + 0.5 * pbp;
        let norm2: f64 = q.iter().map(|&v| v.max(0.0).powi(2)).sum();
        let mut rho = vec![0.0; m];
        for i in 0..m {
            let star = if xi > 0.0 && norm2 > 0.0 {
                xi * q[i].max(0.0) / norm2
            } else {
                0.0
            };
            // Damped update (2.4): keep ρᵢ unless it exceeds 4(ρ*ᵢ + Δρ),
            // then move to the geometric mean of ρᵢ and ρ*ᵢ + Δρ.
            let old = self.rho[i];
            let kept = if old < 4.0 * (star + self.delta) {
                old
            } else {
                (old * (star + self.delta)).sqrt()
            };
            rho[i] = finite(star.max(kept), "merit penalty")?;
        }
        let mut value = f;
        let mut slope = base;
        let mut sensitivity = 0.0_f64;
        let mut magnitude = f.abs();
        for i in 0..m {
            let r = residual[i];
            magnitude += (self.pi[i] * r).abs() + 0.5 * rho[i] * r * r;
            value = finite(
                value - self.pi[i] * r + 0.5 * rho[i] * r * r,
                "augmented-Lagrangian merit",
            )?;
            slope = finite(slope - rho[i] * q[i], "merit slope")?;
            sensitivity = sensitivity.max(self.pi[i].abs() + pi_bar[i].abs() + rho[i] * r.abs());
        }
        Ok(Search {
            skip: (0..m)
                .map(|i| skip.get(i).copied().unwrap_or(false))
                .collect(),
            max_penalty: rho.iter().fold(0.0_f64, |a, &b| a.max(b)),
            slack,
            slack_step,
            pi_step,
            rho,
            value,
            predicted: -slope,
            sensitivity,
            magnitude,
        })
    }

    /// Stores the penalties of `search`, doubling `Δρ` when the trend of
    /// `‖ρ‖` reverses.
    pub fn commit(&mut self, search: &Search) {
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (before, after) = (norm(&self.rho), norm(&search.rho));
        let trend = if after > before {
            1
        } else if after < before {
            -1
        } else {
            0
        };
        if trend != 0 {
            if self.trend != 0 && trend != self.trend {
                self.delta *= 2.0;
            }
            self.trend = trend;
        }
        self.rho.clone_from(&search.rho);
    }

    /// `π ← π + α(π̄ − π)` after a step of length `alpha`.
    pub fn advance(&mut self, pi_bar: &[f64], alpha: f64) {
        for (pi, &target) in self.pi.iter_mut().zip(pi_bar) {
            *pi += alpha * (target - *pi);
        }
    }
}

impl Search {
    /// `M` at the trial point `x + αp` with objective `f` and rows `c`.
    pub fn value_at(&self, pi: &[f64], alpha: f64, f: f64, c: &[f64]) -> Result<f64> {
        let mut value = f;
        for (i, &ci) in c.iter().enumerate() {
            if self.skip[i] {
                continue;
            }
            let r = finite(
                ci - (self.slack[i] + alpha * self.slack_step[i]),
                "merit residual",
            )?;
            let multiplier = pi[i] + alpha * self.pi_step[i];
            value = finite(
                value - multiplier * r + 0.5 * self.rho[i] * r * r,
                "augmented-Lagrangian merit",
            )?;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// φ(α) along the path for one row, from the definition.
    fn merit(search: &Search, al: &AugmentedLagrangian, alpha: f64, f: f64, c: f64) -> f64 {
        search.value_at(&al.pi, alpha, f, &[c]).unwrap()
    }

    #[test]
    fn the_slope_matches_a_finite_difference_of_the_merit() {
        // f(x) = x², c(x) = x³ on 1 ≤ c ≤ 2, from x = 0.5 along p = 0.7,
        // with an earlier estimate π = 0.3 and ρ = 2.
        let bounds = Bounds::new(vec![1.0], vec![2.0]);
        let (x, p) = (0.5_f64, 0.7_f64);
        let mut al = AugmentedLagrangian::new(1);
        al.pi[0] = 0.3;
        al.rho[0] = 2.0;
        let linear = x.powi(3) + 3.0 * x * x * p;
        let search = al
            .search(
                &bounds,
                x * x,
                &[x.powi(3)],
                &[linear],
                2.0 * x * p,
                1.3 * p * p,
                &[0.9],
                &[],
            )
            .unwrap();
        let h = 1e-7;
        let at = |a: f64| merit(&search, &al, a, (x + a * p).powi(2), (x + a * p).powi(3));
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        assert!(
            (numeric + search.predicted).abs() < 1e-6,
            "{numeric} vs {}",
            -search.predicted
        );
        assert!((at(0.0) - search.value).abs() < 1e-15);
        // The penalty enforces φ′(0) ≤ −½pᵀBp.
        assert!(search.predicted >= 0.5 * 1.3 * p * p - 1e-12);
    }

    #[test]
    fn the_penalty_is_the_minimum_norm_solution_and_its_decrease_is_damped() {
        // One violated equality c = 3 > 1 with π = π̄ = 0 and gᵀp = 1, a step
        // that satisfies the linearization: r = 2, r′ = −2, q = 4, and
        // φ′(0) = 1 − 4ρ ≤ −½pᵀBp = −0.5 needs ρ* = 1.5/4.
        let bounds = Bounds::new(vec![1.0], vec![1.0]);
        let al = AugmentedLagrangian::new(1);
        let search = al
            .search(&bounds, 0.0, &[3.0], &[1.0], 1.0, 1.0, &[0.0], &[])
            .unwrap();
        assert!((search.rho[0] - 0.375).abs() < 1e-15);
        assert!((search.predicted - 0.5).abs() < 1e-15);
        // A large old penalty is kept below 4(ρ* + Δρ), and moved to the
        // geometric mean of ρ and ρ* + Δρ above it.
        let mut al = AugmentedLagrangian::new(1);
        al.rho[0] = 5.0;
        let search = al
            .search(&bounds, 0.0, &[3.0], &[1.0], 1.0, 1.0, &[0.0], &[])
            .unwrap();
        assert_eq!(search.rho[0], 5.0);
        al.rho[0] = 100.0;
        let search = al
            .search(&bounds, 0.0, &[3.0], &[1.0], 1.0, 1.0, &[0.0], &[])
            .unwrap();
        assert!((search.rho[0] - (100.0_f64 * 1.375).sqrt()).abs() < 1e-12);
        // Δρ doubles when ‖ρ‖ turns from increasing to decreasing.
        let mut al = AugmentedLagrangian::new(1);
        let up = Search {
            rho: vec![2.0],
            ..search_with(0.0)
        };
        al.commit(&up);
        al.commit(&Search {
            rho: vec![1.0],
            ..search_with(0.0)
        });
        assert_eq!(al.delta, 2.0);
        al.commit(&Search {
            rho: vec![0.5],
            ..search_with(0.0)
        });
        assert_eq!(al.delta, 2.0);
    }

    fn search_with(value: f64) -> Search {
        Search {
            skip: vec![false],
            slack: vec![0.0],
            slack_step: vec![0.0],
            pi_step: vec![0.0],
            rho: vec![0.0],
            value,
            predicted: 0.0,
            max_penalty: 0.0,
            sensitivity: 0.0,
            magnitude: 0.0,
        }
    }

    #[test]
    fn slacks_minimize_the_merit_within_the_row_bounds() {
        assert_eq!(best_slack(3.0, 0.0, 0.0, 1.0, 2.0), 2.0);
        assert_eq!(best_slack(1.5, 0.0, 0.0, 1.0, 2.0), 1.5);
        // c − π/ρ = 1.5 − 1 = 0.5, clipped to the lower bound.
        assert_eq!(best_slack(1.5, 2.0, 2.0, 1.0, 2.0), 1.0);
        assert_eq!(best_slack(1.5, -1.0, 4.0, 1.0, 2.0), 1.75);
    }

    #[test]
    fn skipped_rows_do_not_enter_the_merit() {
        let bounds = Bounds::new(vec![1.0], vec![1.0]);
        let mut al = AugmentedLagrangian::new(1);
        al.pi[0] = 4.0;
        al.rho[0] = 3.0;
        let search = al
            .search(&bounds, 2.0, &[3.0], &[1.0], 1.0, 1.0, &[5.0], &[true])
            .unwrap();
        assert_eq!(search.value, 2.0);
        assert_eq!(search.predicted, -1.0);
        assert_eq!(search.value_at(&al.pi, 0.5, 7.0, &[9.0]).unwrap(), 7.0);
    }
}
