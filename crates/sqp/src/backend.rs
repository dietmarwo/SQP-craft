//! Backend selection shared by frontends; the native driver remains generic.

use crate::Hessian;
use sqp_craft::{Error, Result};

/// Requested QP backend. Explicit choices are never replaced by a fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QpSolver {
    /// Active-set with elastic steps for full/block BFGS; Clarabel otherwise.
    #[default]
    Auto,
    /// Dense Goldfarb–Idnani backend.
    ActiveSet,
    /// Clarabel interior-point backend, including sparse structured QPs.
    Clarabel,
}

/// Concrete frontend options, resolved before model evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedQpOptions {
    /// Concrete backend; never [`QpSolver::Auto`].
    pub solver: QpSolver,
    /// Whether every step QP uses the elastic formulation.
    pub elastic_mode: bool,
}

impl QpSolver {
    /// Stable Python/API spelling. Jobs spell `active-set` as `active_set`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ActiveSet => "active-set",
            Self::Clarabel => "clarabel",
        }
    }

    /// Parses [`Self::as_str`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "auto" => Ok(Self::Auto),
            "active-set" => Ok(Self::ActiveSet),
            "clarabel" => Ok(Self::Clarabel),
            _ => Err(Error::InvalidInput(format!(
                "qp_solver must be \"auto\", \"clarabel\" or \"active-set\", got {name:?}"
            ))),
        }
    }

    /// Resolves the policy without changing the Hessian mode or size limits.
    ///
    /// An omitted elastic setting enables elastic steps only for automatic
    /// full/block BFGS. Explicit backends retain their legacy nonelastic
    /// default. An explicit boolean always wins; incompatible requests fail.
    /// Native `NativeSqp<B>` callers retain their explicit backend and settings.
    pub fn resolve(self, hessian: Hessian, elastic: Option<bool>) -> Result<ResolvedQpOptions> {
        let solver = match self {
            Self::Auto if hessian == Hessian::Full => Self::ActiveSet,
            Self::Auto => Self::Clarabel,
            explicit => explicit,
        };
        let elastic_mode = elastic.unwrap_or(self == Self::Auto && solver == Self::ActiveSet);
        if elastic_mode && solver == Self::Clarabel {
            return Err(Error::Unsupported(
                "elastic_mode needs the active-set backend; Clarabel has no elastic QP".into(),
            ));
        }
        Ok(ResolvedQpOptions {
            solver,
            elastic_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_and_explicit_choices_keep_distinct_defaults() {
        for hessian in [Hessian::Full, Hessian::Partitioned, Hessian::Exact] {
            let auto = QpSolver::Auto.resolve(hessian, None).unwrap();
            let dense = hessian == Hessian::Full;
            assert_eq!(
                auto.solver,
                if dense {
                    QpSolver::ActiveSet
                } else {
                    QpSolver::Clarabel
                }
            );
            assert_eq!(auto.elastic_mode, dense);
            assert!(
                !QpSolver::Auto
                    .resolve(hessian, Some(false))
                    .unwrap()
                    .elastic_mode
            );
            for solver in [QpSolver::ActiveSet, QpSolver::Clarabel] {
                let explicit = solver.resolve(hessian, None).unwrap();
                assert_eq!(explicit.solver, solver);
                assert!(!explicit.elastic_mode);
            }
        }
        assert!(QpSolver::Auto.resolve(Hessian::Exact, Some(true)).is_err());
        assert!(
            QpSolver::Clarabel
                .resolve(Hessian::Full, Some(true))
                .is_err()
        );
        assert!(
            QpSolver::ActiveSet
                .resolve(Hessian::Full, Some(true))
                .unwrap()
                .elastic_mode
        );
    }
}
