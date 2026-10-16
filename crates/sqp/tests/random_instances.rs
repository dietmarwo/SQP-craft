//! Three instances from a random-problem stress test (feasible by
//! construction) that the delivered iteration-2 solver did not handle.
//!
//! Every instance: minimize ‖x − target‖² on [−2, 2]ⁿ subject to rows
//! `c_i(x) = a_i·x + s_i‖x − y_i‖²` with the given bounds. SciPy SLSQP solves
//! the first two; the third has no feasible point near its start.

use sqp_craft::kkt::nlp_residuals;
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SolveResult,
    SqpSolver, Status,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

const INF: f64 = f64::INFINITY;

struct Row {
    lower: f64,
    upper: f64,
    s: f64,
    a: &'static [f64],
    y: &'static [f64],
}

struct Instance {
    target: &'static [f64],
    rows: &'static [Row],
}

impl Instance {
    fn spec(&self) -> ProblemSpec {
        let n = self.target.len();
        ProblemSpec::new(
            Bounds::new(vec![-2.0; n], vec![2.0; n]),
            Bounds::new(
                self.rows.iter().map(|r| r.lower).collect(),
                self.rows.iter().map(|r| r.upper).collect(),
            ),
            CsrPattern::dense(self.rows.len(), n),
        )
        .unwrap()
    }
}

impl NlpProblem for &Instance {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        assert!(
            x.iter().all(|v| (-2.0..=2.0).contains(v)),
            "outside the box: {x:?}"
        );
        for (ci, row) in c.iter_mut().zip(self.rows) {
            *ci = (0..x.len())
                .map(|k| row.a[k] * x[k] + row.s * (x[k] - row.y[k]).powi(2))
                .sum();
        }
        Ok((0..x.len()).map(|k| (x[k] - self.target[k]).powi(2)).sum())
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        let n = x.len();
        for k in 0..n {
            g[k] = 2.0 * (x[k] - self.target[k]);
        }
        for (i, row) in self.rows.iter().enumerate() {
            for k in 0..n {
                j[i * n + k] = row.a[k] + 2.0 * row.s * (x[k] - row.y[k]);
            }
        }
        Ok(())
    }
}

fn solve(instance: &Instance, x0: &[f64], settings: SqpOptions) -> SolveResult {
    NativeSqp::<ClarabelQp> {
        settings,
        ..NativeSqp::default()
    }
    .solve(
        &instance.spec(),
        &mut &*instance,
        x0,
        &Options::default(),
        &CancellationToken::default(),
        None,
    )
    .unwrap()
}

fn assert_kkt(instance: &Instance, r: &SolveResult) {
    let k = nlp_residuals(
        &instance.spec(),
        r.candidate.as_ref().unwrap(),
        r.evaluation.as_ref().unwrap(),
        r.multipliers.as_ref().unwrap(),
    )
    .unwrap();
    assert!(
        k.primal <= 1e-7
            && k.stationarity <= 1e-6
            && k.complementarity <= 1e-6
            && k.dual_sign <= 1e-8,
        "{k:?}"
    );
}

/// Seed 4, problem 227. After one restoration step, x₀ ends up 3e-14 below
/// its upper bound; the interior-point QP step then overshoots that room by
/// ~1e-11, the step-length cap (u − x)/p cut every component to α ≈ 0.006
/// and then to 0, and the delivered solver stopped `stalled` at primal 0.06.
static BOX_STALL: Instance = Instance {
    target: &[0.555934167758755, 2.9023291482846325, -1.8776113948481934],
    rows: &[
        Row {
            lower: 10.886736055262936,
            upper: 10.886736055262936,
            s: 0.6882913336729005,
            a: &[
                -0.7096826272786517,
                -0.05539448867346282,
                -0.5635973429867314,
            ],
            y: &[-1.2633197994012235, 1.9235759274090292, 0.2442032422102458],
        },
        Row {
            lower: -INF,
            upper: 2.6564986245671016,
            s: 0.36186723000124943,
            a: &[1.1656651101121196, 1.0909499146510846, 2.625589644948277],
            y: &[1.8023946850669263, -0.955815009028318, -0.5266155277546729],
        },
    ],
};

#[test]
fn a_step_overshooting_a_nearly_active_bound_by_roundoff_does_not_stall() {
    let x0 = [
        -0.9172013240221495,
        -0.14300968224875232,
        1.0534262688563278,
    ];
    let r = solve(&BOX_STALL, &x0, SqpOptions::default());
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    assert_kkt(&BOX_STALL, &r);
    // SLSQP reaches the same point.
    let f = r.evaluation.as_ref().unwrap().objective;
    assert!((f - 5.277495770176593).abs() < 1e-6, "f = {f}");
    assert!(r.report.bound_step_corrections > 0, "{:?}", r.report);
    let x = r.candidate.unwrap();
    assert!(
        2.0 - x[0] < 1e-10 && x[2] + 2.0 < 1e-10,
        "both bounds active: {x:?}"
    );
}

/// Seed 4, problem 284. Near the solution pᵀBp ≈ 1e-12 fell below the QP's
/// complementarity noise, so the predicted merit decrease was −1.6e-12, and
/// the delivered solver stopped `stalled` with stationarity 1.18e-6, just
/// above the 1e-6 tolerance. One more full step converges.
static NOISE_STALL: Instance = Instance {
    target: &[
        1.125433686457722,
        1.7178331655560095,
        2.666929049037476,
        -1.3499521392453744,
        -1.7442956660683002,
    ],
    rows: &[
        Row {
            lower: 7.6053598046684865,
            upper: 7.6053598046684865,
            s: 1.0343735143950343,
            a: &[
                1.356667930479474,
                -0.5348284582720713,
                0.5368620478491556,
                1.720854327308658,
                -1.6166478842762422,
            ],
            y: &[
                0.3293252498748851,
                -1.268405101698952,
                -0.9448030622130368,
                1.0636370781012365,
                0.06769006300321667,
            ],
        },
        Row {
            lower: -INF,
            upper: 3.7680362998064645,
            s: 0.1424026388010147,
            a: &[
                -1.1068653004238782,
                0.14727602560537603,
                1.7047272731358345,
                -0.16161751970803467,
                -1.546129163169602,
            ],
            y: &[
                1.1266236770161804,
                -0.3860776279735907,
                -0.45596248098768166,
                -0.9784100632566002,
                -0.40333076871606366,
            ],
        },
        Row {
            lower: -INF,
            upper: -1.5425758082064371,
            s: 0.30196204509451063,
            a: &[
                0.35844728647372726,
                2.6170893043920165,
                -0.07234139485458482,
                -0.3899168709641961,
                -1.6747896635708106,
            ],
            y: &[
                -1.419617616627327,
                0.36442642882487997,
                1.341385047620423,
                -0.0036869693561181194,
                0.29084622669321725,
            ],
        },
        Row {
            lower: -INF,
            upper: 4.307461620739662,
            s: 0.976496886301876,
            a: &[
                -0.43252399478735,
                -1.0657595263251938,
                -0.12870532115440547,
                0.42814257165435415,
                0.10151997487143341,
            ],
            y: &[
                -0.45385423213150755,
                -1.3857424684285014,
                0.7145257124071458,
                0.3485820717985231,
                -0.8646926976421798,
            ],
        },
    ],
};

#[test]
fn a_step_at_the_qp_noise_level_is_taken_instead_of_stalling() {
    let x0 = [
        1.9919230152141014,
        -0.957355001969189,
        -1.140286581325051,
        1.720681247232151,
        1.740218982271721,
    ];
    let r = solve(&NOISE_STALL, &x0, SqpOptions::default());
    assert_eq!(r.status, Status::Converged, "{}", r.message);
    assert_kkt(&NOISE_STALL, &r);
    let f = r.evaluation.as_ref().unwrap().objective;
    assert!((f - 14.695432839604383).abs() < 1e-8, "f = {f}");
}

/// Seed 1, problem 118: no feasible point near the start; restoration
/// approaches a local minimizer of ψ (ψ ≈ 0.5923) where the Gauss–Newton
/// model predicts ever smaller decreases. The delivered absolute test
/// `predicted <= 0` crept until the 10-step budget was spent and reported
/// `restoration_limit`, as if more budget could help.
static PSI_MINIMUM: Instance = Instance {
    target: &[-2.0011813739941924, -0.520212886909948],
    rows: &[
        Row {
            lower: 2.1649964816620697,
            upper: 2.1649964816620697,
            s: 0.4073821023679465,
            a: &[0.7241920324022969, 0.42114350934917233],
            y: &[0.9144317205233956, 1.7156090622361182],
        },
        Row {
            lower: -INF,
            upper: 2.1320054894525287,
            s: 1.4625429447403433,
            a: &[-2.173249658650527, 0.13067901569983878],
            y: &[0.831267653389665, 1.9085727074750851],
        },
        Row {
            lower: -INF,
            upper: 5.4970189671967145,
            s: 0.9235026265033833,
            a: &[1.0388912844332436, 1.8616213451678116],
            y: &[-0.4069316587264886, 0.06045087859767495],
        },
    ],
};

#[test]
fn an_infeasible_stationary_point_of_psi_ends_restoration_promptly() {
    let x0 = [-0.2920809095680075, 1.3168771200811542];
    for budget in [10, 100] {
        let r = solve(
            &PSI_MINIMUM,
            &x0,
            SqpOptions {
                max_restoration_steps: budget,
                record_history: true,
                ..SqpOptions::default()
            },
        );
        assert_eq!(
            r.status,
            Status::RestorationFailed,
            "budget {budget}: {}",
            r.message
        );
        assert!(r.message.contains("stationary point"), "{}", r.message);
        assert!(
            r.report.restoration_steps < 10,
            "{} steps",
            r.report.restoration_steps
        );
        assert!(r.multipliers.is_none() && r.stationarity.is_none());
        let last = r.report.restoration_history.last().unwrap();
        assert!(
            (last.merit_after - 0.5923).abs() < 1e-3,
            "psi = {}",
            last.merit_after
        );
        assert!(!r.evaluation.unwrap().feasible_candidate);
    }
}
