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

__version__ = "0.16.0"

__all__ = [
    "Cancellation",
    "DerivativeIssue",
    "DerivativeReport",
    "Evaluation",
    "EvaluationCounts",
    "EvaluationError",
    "IterationRecord",
    "RestorationRecord",
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
    "restoration_failed",
    "restoration_limit",
    "locally_infeasible",
    "linear_infeasible",
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
    hessian_block_size: int = 0
    """Explicit contiguous BFGS block width; 0 keeps full BFGS. Failed block updates switch to full BFGS within the reset budget."""
    max_hessian_resets: int = 3
    """Total Hessian resets allowed."""
    qp_tolerance: float = 1e-10
    """Native tolerance of the QP backend."""
    qp_max_iterations: int = 200
    """Iteration cap per QP. For ``"active-set"`` it caps drops and re-entries
    of constraints; first entries are not charged."""
    qp_reuse_workspace: bool = False
    """Opt in to compatible workspace reuse; primal/dual iterates still start cold."""
    qp_solver: str = "clarabel"
    """QP backend: ``"clarabel"`` (interior point) or ``"active-set"`` (dense
    Goldfarb–Idnani dual active set, warm-started from the last working set)."""
    qp_warm_start: bool = True
    """Pass the last step QP's working set to the next; only ``"active-set"`` uses it."""
    elastic_mode: bool = False
    """Solve every step QP as an l1-elastic QP: rows may be violated at a cost, so
    the QP is always feasible and restoration is not used. Needs
    ``qp_solver="active-set"``."""
    second_order_correction: bool = False
    """When the full step fails the merit test and increases the constraint
    violation, re-solve the step QP with the constraints shifted by their
    second-order change along the step and try that step before backtracking."""
    merit: Literal["l1", "augmented-lagrangian"] = "l1"
    """Line-search merit: ``"l1"`` (``f`` plus Powell-weighted violations) or
    ``"augmented-lagrangian"`` (SNOPT's ``f − πᵀ(c − s) + ½Σρᵢ(cᵢ − sᵢ)²``,
    searched jointly in ``x``, the multiplier estimate and the slacks, with the
    smallest penalties whose linear model falls by ½pᵀBp over the unit step,
    twice SNOPT's slope rule). The latter is differentiable and is not combined
    with ``second_order_correction``; elastic steps that leave rows violated use
    the l1 merit."""
    scaling: Literal["none", "linear", "jacobian"] = "none"
    """Scaling of the QP subproblems and the BFGS matrix by powers of two:
    ``"linear"`` from the rows declared linear (SNOPT's Scale option 1; the
    identity without such rows), ``"jacobian"`` from every row's Jacobian at the
    first point on the linear rows (Scale option 2). The merit, the line search
    and the convergence test stay in original units."""
    violation_limit: Optional[float] = None
    """SNOPT's ``τ_V ≥ 1``: reject a trial point that violates some row by more
    than ``τ_V · max(1, its violation at the start)`` before evaluating its
    derivatives. ``None`` imposes no limit; SNOPT uses 10."""
    hessian_update: Literal["damped", "augmented"] = "damped"
    """What to do when a step has too little curvature for a plain BFGS update
    (``sᵀy < 0.2·sᵀBs``): ``"damped"`` blends ``y`` with ``Bs`` (Powell);
    ``"augmented"`` adds SNOPT's augmented-Lagrangian term
    ``(J(x⁺) − J(x))ᵀ Ω d_L`` built from the departure from linearity of the
    constraints (SNOPT), and damps when no such term exists."""
    elastic_weight: float = 1e4
    """Initial elastic weight relative to ``max(1, ‖∇f(x0)‖∞)``; multiplied by 10
    (at most six times) at a stationary point of the penalty with violated rows."""
    record_qp_limit: int = 0
    """Number of step QPs to keep in ``report.qp_trace`` (at most 16)."""
    max_restoration_steps: int = 10
    """Accepted restoration steps allowed; zero keeps the old infeasible-QP stop."""
    restoration_step_limit: float = 1.0
    """Infinity-norm trust bound on restoration directions in variable units."""
    restoration_regularization: float = 1e-4
    """Positive diagonal regularization of the Gauss–Newton feasibility model."""
    record_history: bool = False
    """Keep normal records in ``history`` and feasibility steps in ``restoration_history``."""


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
    second_order_correction: bool
    curvature: Optional[str]
    """How ``B`` was updated after the step: ``"exact"``, ``"damped"``,
    ``"modified"``, ``"negligible"`` or ``"reset"``; ``None`` when no step was
    taken."""


@dataclass(frozen=True)
class RestorationRecord:
    """Accepted feasibility step; its QP duals are not NLP multipliers."""

    iteration: int
    objective_before: float
    objective_after: float
    primal_before: float
    primal_after: float
    merit_before: float
    merit_after: float
    step_length: float
    trials: int
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
    bfgs_damped_updates: int
    bfgs_modified_updates: int
    block_hessian_fallbacks: int
    qp_not_positive_definite: int
    elastic_steps: int
    elastic_weight: Optional[float]
    elastic_weight_increases: int
    linear_projection: bool
    variable_scales: list[float]
    row_scales: list[float]
    violation_limit_rejections: int
    second_order_corrections: int
    second_order_steps: int
    fixed_step_corrections: int
    bound_step_corrections: int
    step_norm: Optional[float]
    total_seconds: float
    callback_seconds: float
    qp_seconds: float
    qp_setups: int
    qp_updates: int
    qp_cold_retries: int
    qp_retry_reason: Optional[str]
    qp_conversion_seconds: float
    qp_setup_seconds: float
    qp_update_seconds: float
    qp_solve_seconds: float
    qp_check_seconds: float
    bfgs_seconds: float
    qp_trace: list[dict[str, Any]]
    history: list[IterationRecord]
    restoration_attempts: int
    restoration_steps: int
    restoration_seconds: float
    restoration_history: list[RestorationRecord]


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
    report["restoration_history"] = [RestorationRecord(**h) for h in report["restoration_history"]]
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
