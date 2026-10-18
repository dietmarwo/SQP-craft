//! Explicit local worker budgets. No global Rayon pool is configured.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
static POOLS: OnceLock<Mutex<BTreeMap<usize, Weak<rayon::ThreadPool>>>> = OnceLock::new();

/// Reusable owned execution resources; one means no worker pool.
#[derive(Clone, Debug, Default)]
pub struct Execution {
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Execution {
    /// Creates at most `threads` workers. Supported budgets are 1 through 8.
    pub fn new(threads: usize) -> Result<Self, String> {
        if !(1..=8).contains(&threads) {
            return Err(format!("parallel threads must lie in 1..=8, got {threads}"));
        }
        if threads == 1 {
            return Ok(Self::default());
        }
        let mut pools = POOLS
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| "worker registry lock poisoned".to_string())?;
        if let Some(pool) = pools.get(&threads).and_then(Weak::upgrade) {
            return Ok(Self { pool: Some(pool) });
        }
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("sqp-worker-{i}"))
                .build()
                .map_err(|e| format!("native worker creation: {e}"))?,
        );
        pools.insert(threads, Arc::downgrade(&pool));
        Ok(Self { pool: Some(pool) })
    }

    /// Configured maximum numerical workers.
    pub fn threads(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    /// Runs disjoint chunks with deterministic chunk offsets. Scalar reductions
    /// inside a chunk remain the responsibility of the kernel.
    pub fn chunks_mut<T, F>(&self, data: &mut [T], chunk: usize, work: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        assert!(chunk > 0);
        if let Some(pool) = &self.pool {
            if work >= 32768 {
                pool.install(|| {
                    data.par_chunks_mut(chunk)
                        .enumerate()
                        .for_each(|(i, part)| f(i * chunk, part))
                });
                return;
            }
        }
        for (i, part) in data.chunks_mut(chunk).enumerate() {
            f(i * chunk, part);
        }
    }

    /// Maps independent tasks, returning results in input-index order.
    pub fn map<T, F>(&self, count: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Send + Sync,
    {
        match &self.pool {
            Some(pool) => pool.install(|| (0..count).into_par_iter().map(f).collect()),
            None => (0..count).map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn width_order_disjoint_outputs_and_budget() {
        for width in [1, 2, 4, 8] {
            let exec = Execution::new(width).unwrap();
            assert_eq!(exec.threads(), width);
            assert_eq!(
                exec.map(97, |i| i * i),
                (0..97).map(|i| i * i).collect::<Vec<_>>()
            );
            let mut out = vec![0; 103];
            exec.chunks_mut(&mut out, 7, 100_000, |start, part| {
                for (i, v) in part.iter_mut().enumerate() {
                    *v = start + i;
                }
            });
            assert_eq!(out, (0..103).collect::<Vec<_>>());
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            exec.map(64, |_| {
                let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(n, Ordering::SeqCst);
                std::thread::yield_now();
                active.fetch_sub(1, Ordering::SeqCst);
            });
            assert!(peak.load(Ordering::SeqCst) <= width);
        }
        assert!(Execution::new(0).is_err());
        assert!(Execution::new(9).is_err());
    }
    #[test]
    fn panic_returns_to_caller_without_poisoning_pool() {
        let exec = Execution::new(8).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exec.map(16, |i| {
                assert_ne!(i, 7);
                i
            })))
            .is_err()
        );
        assert_eq!(exec.map(4, |i| i), vec![0, 1, 2, 3]);
    }
}

#[cfg(test)]
mod configuration_tests {
    #[test]
    fn explicit_widths_are_bounded() {
        for width in 1..=8 {
            assert_eq!(super::Execution::new(width).unwrap().threads(), width);
        }
    }
}

#[cfg(test)]
mod sharing_tests {
    use super::*;
    #[test]
    fn compatible_owners_share_without_permanent_retention() {
        let a = Execution::new(8).unwrap();
        let b = Execution::new(8).unwrap();
        assert!(Arc::ptr_eq(
            a.pool.as_ref().unwrap(),
            b.pool.as_ref().unwrap()
        ));
        let weak = Arc::downgrade(a.pool.as_ref().unwrap());
        drop(a);
        assert!(weak.upgrade().is_some());
        drop(b);
        let c = Execution::new(4).unwrap();
        assert_eq!(c.threads(), 4);
        let d = Execution::new(8).unwrap();
        assert!(!Arc::ptr_eq(
            c.pool.as_ref().unwrap(),
            d.pool.as_ref().unwrap()
        ));
    }
}
