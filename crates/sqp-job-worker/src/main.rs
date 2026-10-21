//! The stock SQP job worker: the three benchmark models.
#![forbid(unsafe_code)]
fn main() {
    optimization_job_api::worker::run(sqp_job_worker::Adapter::default());
}
