"""Python interface to the sqp-craft SQP contracts.

**Iteration 1.** Select ``method="sqp"`` for the native dense BFGS pilot.
The compatibility default runs a dummy backend that evaluates the start point once and returns it with
``status == "stub"``. Model callbacks run serially on the calling thread.

A model provides two methods::

    functions(x)   -> (objective, constraint_values)
    derivatives(x) -> (gradient, jacobian_values)

``x`` is a fresh ``list[float]`` of length ``n`` within the variable bounds.
Returned sequences may be lists, tuples or 1-D NumPy arrays. Jacobian values
follow the fixed CSR pattern of the ``ProblemSpec``. ``derivatives`` is only
called at the point of the immediately preceding ``functions`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Protocol

from ._native import (
    Cancellation,
    EvaluationError,
    ProblemSpec,
    UndefinedPointError,
)
from ._native import solve_native as _solve_native
from ._native import check_derivatives as _check_derivatives
from ._native import kkt_residuals as _kkt_residuals

__version__ = "0.2.0"

__all__ = [
    "Cancellation",
    "Evaluation",
    "EvaluationCounts",
    "EvaluationError",
    "Model",
    "Multipliers",
    "Options",
    "ProblemSpec",
    "Progress",
    "SolveResult",
    "UndefinedPointError",
    "solve",
    "SqpOptions",
    "RunReport",
    "check_derivatives",
    "kkt_residuals",
    "DerivativeReport",
    "KktResiduals",
]


class Model(Protocol):
    """The callback contract. Raise ``UndefinedPointError`` where the model is
    not defined (for example a failed propagation); any other exception aborts
    the solve and is re-raised unchanged."""

    def functions(self, x: list[float]) -> tuple[float, Iterable[float]]: ...

    def derivatives(self, x: list[float]) -> tuple[Iterable[float], Iterable[float]]: ...


@dataclass(frozen=True, kw_only=True)
class Options:
    """Solver settings. Only settings a backend honours are exposed."""

    feasibility_tolerance: float = 1e-7
    """Absolute tolerance, in original units, on the largest bound violation."""
    threads: int = 1
    """Only 1 is supported; run independent solves for parallelism."""


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
class SolveResult:
    status: str
    candidate: Optional[list[float]]
    evaluation: Optional[Evaluation]
    multipliers: Optional[Multipliers]
    counts: EvaluationCounts
    iterations: int
    stationarity: Optional[float]
    complementarity: Optional[float]
    message: str
    dual_sign: Optional[float]
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
    options: Options | None = None,
    cancellation: Cancellation | None = None,
    observer: Observer | None = None,
    method: Literal["dummy", "sqp"] = "dummy",
    method_options: SqpOptions | None = None,
) -> SolveResult:
    """Solve ``spec`` with ``model`` from ``x0``.

    ``x0`` must be finite and within the variable bounds; it is never projected.
    ``observer(progress)`` may return ``True`` to stop the solve (status
    ``"cancelled"``, candidate kept); ``None`` or ``False`` continue, so a
    print-only observer works.

    Raises ``ValueError`` for malformed input, ``EvaluationError`` for failing
    or malformed model output, ``NotImplementedError`` for unsupported options,
    and re-raises exceptions from the model or observer unchanged.
    """
    options = options or Options()
    from dataclasses import asdict
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
    return SolveResult(
        status=raw["status"],
        candidate=raw["candidate"],
        evaluation=None if evaluation is None else Evaluation(**evaluation),
        multipliers=None if multipliers is None else Multipliers(**multipliers),
        counts=EvaluationCounts(**raw["counts"]),
        iterations=raw["iterations"],
        stationarity=raw["stationarity"],
        complementarity=raw["complementarity"],
        message=raw["message"],
        dual_sign=raw["dual_sign"],
        report=RunReport(**raw["report"]),
    )


@dataclass(frozen=True, kw_only=True)
class SqpOptions:
    """Dense BFGS pilot controls. Call budgets count grouped functions/derivatives."""
    max_iterations: int = 100
    max_backtracks: int = 25
    max_callback_calls: int = 2000
    time_limit_seconds: Optional[float] = None
    stationarity_tolerance: float = 1e-6
    complementarity_tolerance: float = 1e-6
    dual_sign_tolerance: float = 1e-8
    dense_size_limit: int = 256
    initial_hessian_scale: float = 1.0
    max_hessian_resets: int = 3
    qp_tolerance: float = 1e-10
    qp_max_iterations: int = 200
    record_qp_limit: int = 0


@dataclass(frozen=True)
class RunReport:
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


@dataclass(frozen=True)
class KktResiduals:
    primal: float
    stationarity: float
    complementarity: float
    dual_sign: float


def kkt_residuals(spec: ProblemSpec, x: Iterable[float], evaluation: Evaluation, multipliers: Multipliers) -> KktResiduals:
    """Independent floating-point KKT residuals, in original units and signs."""
    return KktResiduals(**_kkt_residuals(spec, list(x), evaluation.objective, evaluation.constraints,
        evaluation.gradient, evaluation.jacobian_values, multipliers.constraints, multipliers.variables))


@dataclass(frozen=True)
class DerivativeReport:
    passed: bool
    inconclusive: bool
    checked: int
    skipped: list[int]
    counts: EvaluationCounts
    issues: list[dict[str, Any]]


def check_derivatives(spec: ProblemSpec, model: Model, x: Iterable[float], *,
                      steps: Iterable[float] = (1e-4, 1e-5, 1e-6), absolute_tolerance: float = 1e-5,
                      relative_tolerance: float = 1e-4, max_directions: int = 256) -> DerivativeReport:
    """Bound-aware coordinate finite differences; fixed/unvisited coordinates are inconclusive.
    This utility has separate callback counts and does not change optimizer settings.
    """
    raw = _check_derivatives(spec, model.functions, model.derivatives, x, list(steps),
                            absolute_tolerance, relative_tolerance, max_directions)
    raw["counts"] = EvaluationCounts(**raw["counts"])
    return DerivativeReport(**raw)
