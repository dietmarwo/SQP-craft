use sqp_craft::qp::{QpBackend, QpModel, QpOutcome};
use sqp_craft::{Bounds, CancellationToken, CsrMatrix, CsrPattern, Error};
use sqp_craft_activeset::ActiveSetQp;
include!("fixtures/hs25ne.rs");
#[test]
fn hs25ne_returns_a_status_or_backend_failure_without_nan() {
    let q = hs25ne();
    let out = ActiveSetQp::default().solve(&q, None, &CancellationToken::default());
    println!("HS25NE backend outcome: {out:?}");
    match out {
        Ok(QpOutcome::Optimal(s)) => assert!(q.accepts(&s, 1e-7).unwrap().accepted),
        Ok(QpOutcome::Infeasible | QpOutcome::LimitReached(_)) | Err(Error::Backend(_)) => (),
        other => panic!("HS25NE first QP: {other:?}"),
    }
}
