//! The error type shared by all components.

use std::fmt;

/// Why a call was aborted.
///
/// Errors never carry a partial result: a solve either returns a
/// [`SolveResult`](crate::SolveResult) or one of these.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Malformed problem data, start point or options. Always detected before
    /// the model is called.
    InvalidInput(String),
    /// The model failed, or returned a non-finite value.
    Evaluation(String),
    /// The model reported that it is not defined at the requested point, for
    /// example because a trajectory propagation failed.
    ///
    /// This is distinct from [`Error::Evaluation`] so that a line search can
    /// later recover by shortening its step, as SNOPT and Ipopt do. At the start
    /// point there is nothing to fall back to, so it aborts the solve.
    UndefinedPoint(String),
    /// A well-formed request that this component does not implement.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Error::Evaluation(m) => write!(f, "evaluation failed: {m}"),
            Error::UndefinedPoint(m) => write!(f, "model undefined at point: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// The message without the category prefix that [`Display`](fmt::Display) adds.
    pub fn message(&self) -> &str {
        match self {
            Error::InvalidInput(m)
            | Error::Evaluation(m)
            | Error::UndefinedPoint(m)
            | Error::Unsupported(m) => m,
        }
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Returns `Err(InvalidInput)` with a lazily built message unless `ok` holds.
pub(crate) fn ensure(ok: bool, message: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidInput(message()))
    }
}

/// Index and value of the first non-finite entry, if any.
pub(crate) fn first_non_finite(values: &[f64]) -> Option<(usize, f64)> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
}
