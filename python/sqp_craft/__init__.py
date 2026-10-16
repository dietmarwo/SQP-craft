"""Python interface to sqp-craft: a native Rust SQP solver for smooth
constrained problems.

``solve(spec, model, x0, method="sqp")`` runs the native dense-BFGS SQP with
Clarabel QP subproblems (at most 256 variables). The default
``method="dummy"`` only evaluates the start point and returns
``status == "stub"``; it is kept as a contract fixture.

A model provides two methods::

    functions(x)   -> (objective, constraint_values)
    derivatives(x) -> (gradient, jacobian_values)

``x`` is a fresh ``list[float]`` of length ``n`` within the variable bounds.
Returned sequences may be lists, tuples or 1-D NumPy arrays. Jacobian values
follow the fixed CSR pattern of the ``ProblemSpec``. ``derivatives`` is only
called at the point of the immediately preceding ``functions`` call. Callbacks
run serially on the calling thread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Protocol

from ._native import (
    Cancellation,
    EvaluationError,
    ProblemSpec,
    UndefinedPointError,
)
from ._native import check_derivatives as _check_derivatives
from ._native import kkt_residuals as _kkt_residuals
from ._native import solve_native as _solve_native

__version__ = "0.2.0"

__all__ = [
    "Cancellation",
    "DerivativeIssue",
    "DerivativeReport",
    "Evaluation",
    "EvaluationCounts",
    "EvaluationError",
    "IterationRecord",
    "KktResiduals",
    "Model",
    "Multipliers",
    "Options",
    "ProblemSpec",
    "Progress",
    "RunReport",
    "SolveResult",
    "SqpOptions",
    "Status",
    "UndefinedPointError",
    "check_derivatives",
    "kkt_residuals",
    "solve",
]

Status = Literal[
    "stub",
    "cancelled",
    "converged",
    "iteration_limit",
    "evaluation_limit",
    "time_limit",
    "qp_failure",
    "linearization_infeasible",
    "line_search_failure",
    "no_descent",
    "stalled",
    "numerical_failure",
]
"""How a solve ended. Only ``"converged"`` claims a first-order KKT point."""


class Model(Protocol):
    """The callback contract. Raise ``UndefinedPointError`` where the model is
    not defined (for example a failed propagation): at a line-search trial the
    solver shortens the step, at the start point the solve aborts. Any other
    exception aborts the solve and is re-raised unchanged."""

    def functions(self, x: list[float]) -> tuple[float, Iterable[float]]: ...

    def derivatives(self, x: list[float]) -> tuple[Iterable[float], Iterable[float]]: ...


@dataclass(frozen=True, kw_only=True)
class Options:
    """Settings shared by all methods."""

    feasibility_tolerance: float = 1e-7
    """Absolute tolerance, in original units, on the largest bound violation."""
    threads: int = 1
    """Only 1 is supported; run independent solves for parallelism."""


@dataclass(frozen=True, kw_only=True)
class SqpOptions:
    """Settings of ``method="sqp"``. Every field is honoured; invalid values
    raise ``ValueError``."""

    max_iterations: int = 100
    """Maximum number of accepted steps."""
    max_backtracks: int = 25
    """Maximum step halvings in one line search after the first trial."""
    max_callback_calls: int = 2000
    """Maximum ``functions`` plus ``derivatives`` calls, including rejected trials."""
    time_limit_seconds: Optional[float] = None
    """Cooperative wall-time budget; a running callback is not interrupted."""
    stationarity_tolerance: float = 1e-6
    """Absolute tolerance on ``‖∇f − Jᵀλ − z‖∞``."""
    complementarity_tolerance: float = 1e-6
    """Absolute complementarity tolerance."""
    dual_sign_tolerance: float = 1e-8
    """Tolerance on multipliers whose sign selects an infinite bound."""
    dense_size_limit: int = 256
    """Variables allowed, 1 to 256."""
    initial_hessian_scale: float = 1.0
    """Initial and reset Hessian approximation ``scale · I``."""
    max_hessian_resets: int = 3
    """Total Hessian resets allowed."""
    qp_tolerance: float = 1e-10
    """Native tolerance of the QP backend."""
    qp_max_iterations: int = 200
    """Iteration cap per QP."""
    record_qp_limit: int = 0
    """Number of step QPs to keep in ``report.qp_trace`` (at most 16)."""
    record_history: bool = False
    """Keep one ``IterationRecord`` per iteration in ``report.history``."""


@dataclass(frozen=True)
class Progress:
    """Passed to the observer. ``iteration == 0`` reports the start point."""

    iteration: int
    objective: float
    primal_violation: float


@dataclass(frozen=True)
class EvaluationCounts:
    functions: int
    derivatives: int


@dataclass(frozen=True)
class Evaluation:
    """Model values at the candidate. ``feasible_candidate`` compares
    ``primal_violation`` with the tolerance; it says nothing about optimality."""

    objective: float
    constraints: list[float]
    gradient: list[float]
    jacobian_values: list[float]
    primal_violation: float
    feasible_candidate: bool


@dataclass(frozen=True)
class Multipliers:
    """Convention: ``grad f = J^T constraints + variables`` at a KKT point;
    positive entries belong to active lower bounds, negative to upper bounds."""

    constraints: list[float]
    variables: list[float]


@dataclass(frozen=True)
class IterationRecord:
    """One SQP iteration: the iterate where a QP was solved and the step taken
    from it. ``step_length`` is ``None`` when no step was taken."""

    iteration: int
    objective: float
    primal_violation: float
    stationarity: float
    complementarity: float
    qp_step_norm: float
    step_length: Optional[float]
    trials: int
    max_penalty: float
    qp_iterations: int


@dataclass(frozen=True)
class RunReport:
    """Diagnostics. Times are inclusive elapsed seconds per phase."""

    method: str
    qp_backend: str
    qp_status: Optional[str]
    qp_solves: int
    qp_iterations: int
    rejected_trials: int
    undefined_trials: int
    hessian_resets: int
    fixed_step_corrections: int
    step_norm: Optional[float]
    total_seconds: float
    callback_seconds: float
    qp_seconds: float
    bfgs_seconds: float
    qp_trace: list[dict[str, Any]]
    history: list[IterationRecord]


@dataclass(frozen=True)
class SolveResult:
    status: Status
    candidate: Optional[list[float]]
    evaluation: Optional[Evaluation]
    multipliers: Optional[Multipliers]
    counts: EvaluationCounts
    iterations: int
    stationarity: Optional[float]
    complementarity: Optional[float]
    dual_sign: Optional[float]
    message: str
    report: RunReport


Observer = Callable[[Progress], Optional[bool]]


def _adapt_observer(observer: Observer) -> Callable[[dict[str, Any]], bool]:
    def observe(data: dict[str, Any]) -> bool:
        verdict = observer(Progress(**data))
        if verdict is None:
            return False
        if isinstance(verdict, bool):
            return verdict
        raise TypeError(
            "observer must return None or False to continue, or True to stop; "
            f"got {type(verdict).__name__}"
        )

    return observe


def solve(
    spec: ProblemSpec,
    model: Model,
    x0: Iterable[float],
    *,
    method: Literal["dummy", "sqp"] = "dummy",
    method_options: SqpOptions | None = None,
    options: Options | None = None,
    cancellation: Cancellation | None = None,
    observer: Observer | None = None,
) -> SolveResult:
    """Solve ``spec`` with ``model`` from ``x0``.

    ``method="sqp"`` runs the native solver, configured by ``method_options``.
    ``x0`` must be finite and within the variable bounds; it is never
    projected. ``observer(progress)`` may return ``True`` to stop the solve
    (status ``"cancelled"``, last complete point kept); ``None`` or ``False``
    continue, so a print-only observer works.

    Raises ``ValueError`` for malformed input, ``EvaluationError`` for failing
    or malformed model output, ``NotImplementedError`` for unsupported options
    or a method that was not compiled, and re-raises exceptions from the model
    or observer unchanged. Solver failures that are not errors (limits, an
    infeasible linearization, ...) are reported through ``status``.
    """
    options = options or Options()
    raw = _solve_native(
        spec,
        model.functions,
        model.derivatives,
        x0,
        options.feasibility_tolerance,
        options.threads,
        cancellation if cancellation is not None else Cancellation(),
        None if observer is None else _adapt_observer(observer),
        method,
        None if method_options is None else asdict(method_options),
    )
    evaluation = raw["evaluation"]
    multipliers = raw["multipliers"]
    report = raw["report"]
    report["history"] = [IterationRecord(**h) for h in report["history"]]
    return SolveResult(
        status=raw["status"],
        candidate=raw["candidate"],
        evaluation=None if evaluation is None else Evaluation(**evaluation),
        multipliers=None if multipliers is None else Multipliers(**multipliers),
        counts=EvaluationCounts(**raw["counts"]),
        iterations=raw["iterations"],
        stationarity=raw["stationarity"],
        complementarity=raw["complementarity"],
        dual_sign=raw["dual_sign"],
        message=raw["message"],
        report=RunReport(**report),
    )


@dataclass(frozen=True)
class KktResiduals:
    """Absolute first-order residuals in original units."""

    primal: float
    stationarity: float
    complementarity: float
    dual_sign: float


def kkt_residuals(
    spec: ProblemSpec, x: Iterable[float], evaluation: Evaluation, multipliers: Multipliers
) -> KktResiduals:
    """Recompute KKT residuals independently of any solver, in the multiplier
    convention ``grad f = J^T constraints + variables``. Calls no model code."""
    return KktResiduals(
        **_kkt_residuals(
            spec,
            list(x),
            evaluation.objective,
            evaluation.constraints,
            evaluation.gradient,
            evaluation.jacobian_values,
            multipliers.constraints,
            multipliers.variables,
        )
    )


@dataclass(frozen=True)
class DerivativeIssue:
    """A derivative entry that disagrees with every usable finite difference.
    ``output`` is 0 for the objective and ``i + 1`` for constraint row ``i``."""

    variable: int
    output: int
    analytic: float
    estimated: float
    scaled_error: float


@dataclass(frozen=True)
class DerivativeReport:
    """``passed`` requires at least one checked coordinate, no issues and no
    skipped coordinate. Fixed variables are listed in ``fixed``: they cannot be
    perturbed and do not affect the solver's steps."""

    passed: bool
    inconclusive: bool
    checked: int
    fixed: list[int]
    skipped: list[int]
    counts: EvaluationCounts
    issues: list[DerivativeIssue]


def check_derivatives(
    spec: ProblemSpec,
    model: Model,
    x: Iterable[float],
    *,
    steps: Iterable[float] = (1e-4, 1e-5, 1e-6),
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
    max_directions: int = 256,
) -> DerivativeReport:
    """Compare the model's derivatives at ``x`` with finite differences along
    each coordinate, staying within the variable bounds (central differences
    where both sides have room, one-sided otherwise). Every objective and
    constraint output is compared, so entries missing from the Jacobian
    pattern are found. A perturbation where the model raises
    ``UndefinedPointError`` is not used. Callback counts are reported
    separately from any solve."""
    raw = _check_derivatives(
        spec,
        model.functions,
        model.derivatives,
        x,
        list(steps),
        absolute_tolerance,
        relative_tolerance,
        max_directions,
    )
    raw["counts"] = EvaluationCounts(**raw["counts"])
    raw["issues"] = [DerivativeIssue(**i) for i in raw["issues"]]
    return DerivativeReport(**raw)
