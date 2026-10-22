//! Contracts for SQP solvers of smooth constrained problems with sparse (CSR)
//! constraint Jacobians: problem specification, model callbacks, solver and
//! observer traits, typed results, KKT residuals and a QP seam.
//!
//! This core has no dependencies. The solver is the sibling crate
//! `sqp-craft-solver` (dense damped BFGS, at most 256 variables) with the QP
//! backends `sqp-craft-clarabel` or `sqp-craft-activeset`. [`DummySolver`]
//! is a contract fixture: it evaluates the start point and never claims
//! convergence.
//!
//! ```
//! use sqp_craft::{
//!     Bounds, CancellationToken, CsrPattern, DummySolver, NlpProblem, Options,
//!     ProblemSpec, Result, SqpSolver, Status,
//! };
//!
//! /// minimize (x0 - 1)^2 + (x1 - 2)^2  subject to  x0 + x1 = 3,  0 <= x <= 4
//! struct Toy;
//!
//! impl NlpProblem for Toy {
//!     fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
//!         c[0] = x[0] + x[1];
//!         Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
//!     }
//!     fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
//!         g.copy_from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]);
//!         jac.copy_from_slice(&[1.0, 1.0]);
//!         Ok(())
//!     }
//! }
//!
//! let spec = ProblemSpec::new(
//!     Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
//!     Bounds::fixed(vec![3.0]),
//!     CsrPattern::dense(1, 2),
//! )?;
//! let result = DummySolver.solve(
//!     &spec, &mut Toy, &[1.0, 2.0], &Options::default(),
//!     &CancellationToken::default(), None,
//! )?;
//! assert_eq!(result.status, Status::Stub); // never a convergence claim
//! assert!(result.stationarity.is_none());
//! # Ok::<(), sqp_craft::Error>(())
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod derivative_check;
mod dummy;
mod error;
pub mod kkt;
mod problem;
pub mod qp;
mod solver;
mod spec;

pub use dummy::DummySolver;
pub use error::{Error, Result};
pub use problem::NlpProblem;
pub use solver::{
    CancellationToken, Control, CurvatureUpdate, Evaluation, EvaluationCounts, IterationRecord,
    Multipliers, Observer, Options, Progress, RestorationRecord, RunReport, SolveResult, SqpSolver,
    Status,
};
pub use spec::{Bounds, CsrMatrix, CsrPattern, ProblemSpec};
