use optimization_job_api::{JobSpec, worker::Worker};
use serde_json::json;
use sqp_craft::{qp::QpModel, *};
use sqp_job_worker::{
    Adapter,
    registry::{Instance, Model, ModelInfo, NoConfig, Registry},
};
include!("../../activeset/tests/fixtures/hs25ne.rs");
#[derive(Clone)]
struct LinearFixture(Vec<f64>);
impl NlpProblem for LinearFixture {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        for (c, row) in c.iter_mut().zip(self.0.chunks_exact(x.len())) {
            *c = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        Ok(0.0)
    }
    fn derivatives(&mut self, _: &[f64], g: &mut [f64], j: &mut [f64]) -> Result<()> {
        g.fill(0.0);
        j.copy_from_slice(&self.0);
        Ok(())
    }
}
#[test]
fn hs25ne_like_dependent_rows_return_a_task_result_without_a_claim() {
    let mut registry = Registry::new();
    registry.register(
        ModelInfo::new("app.dependent", "1", "HS25NE first linearization").dimensions(3..=3),
        |_: &Instance, _: NoConfig| {
            let q = hs25ne();
            let spec = ProblemSpec::new(q.variables, q.rows, q.matrix.pattern)?;
            Ok(Model::new(spec, LinearFixture(q.matrix.values)))
        },
    );
    let adapter = Adapter::new(registry);
    let mut job: JobSpec = serde_json::from_value(json!({
     "schema_version":"optimization.job/0.1","request_key":"numerical-status","worker":"app",
     "payload_schema":"sqp.solve/1","payload":{"problem":"app.dependent","dimension":3,
      "starts":[[0.0,0.0,0.0]],"backend":"active_set","elastic_mode":false,"max_iterations":50,
      "max_callback_calls":1000,"feasibility_tolerance":1e-7,"stationarity_tolerance":1e-6},
     "resources":{"cpu_slots":1,"outer_workers":1,"inner_threads":1},
     "limits":{"wall_time_seconds":30.,"grace_seconds":1.},"inputs":[]
    }))
    .unwrap();
    let dir = std::env::temp_dir();
    let prepared = adapter.prepare(&mut job, &dir).unwrap();
    let result = prepared.solve(0, 30.0).unwrap();
    assert_eq!(result.outcome.claim, "none", "{:?}", result.outcome);
    assert!(!result.outcome.termination.is_empty());
    result.outcome.check().unwrap();
}
