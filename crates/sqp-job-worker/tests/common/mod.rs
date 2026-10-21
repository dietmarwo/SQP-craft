//! Helpers of the worker's integration tests: temporary job directories and
//! jobs with staged, hashed inputs.
#![allow(dead_code)]
use optimization_job_api::{JobSpec, storage::digest};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

/// A directory removed when dropped.
pub struct TempDir(PathBuf);
impl TempDir {
    pub fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "sqp-job-worker-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A job on `worker` with the given resources, and inputs `(name, file,
/// text)` written into `dir` with their SHA-256.
pub fn job(
    dir: &Path,
    profile: &str,
    payload: Value,
    resources: Value,
    inputs: &[(&str, &str, &str)],
) -> JobSpec {
    let inputs: Vec<Value> = inputs
        .iter()
        .map(|(name, file, text)| {
            fs::write(dir.join(file), text).unwrap();
            json!({"name":name,"file":file,"sha256":digest(text.as_bytes())})
        })
        .collect();
    let s: JobSpec = serde_json::from_value(json!({"schema_version":"optimization.job/0.1",
        "request_key":"test","worker":"sqp","payload_schema":profile,"payload":payload,
        "resources":resources,"limits":{"wall_time_seconds":30.0,"grace_seconds":1.0},
        "inputs":inputs}))
    .unwrap();
    s.validate().unwrap();
    s
}

/// One CPU slot.
pub fn one_slot() -> Value {
    json!({"cpu_slots":1,"outer_workers":1,"inner_threads":1})
}
