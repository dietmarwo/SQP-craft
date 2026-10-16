from typing import Any, Callable, Iterable, Optional

class EvaluationError(RuntimeError):
    """The model failed or returned malformed or non-finite values."""

class UndefinedPointError(EvaluationError):
    """Raised by a model that is not defined at the requested point."""

class ProblemSpec:
    """Validated problem structure: variable and constraint bounds plus the
    zero-based CSR pattern of the constraint Jacobian. Immutable; picklable."""

    def __init__(
        self,
        *,
        lower: Iterable[float],
        upper: Iterable[float],
        constraint_lower: Iterable[float],
        constraint_upper: Iterable[float],
        jacobian_indptr: Iterable[int],
        jacobian_indices: Iterable[int],
    ) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def lower(self) -> list[float]: ...
    @property
    def upper(self) -> list[float]: ...
    @property
    def constraint_lower(self) -> list[float]: ...
    @property
    def constraint_upper(self) -> list[float]: ...
    @property
    def jacobian_indptr(self) -> list[int]: ...
    @property
    def jacobian_indices(self) -> list[int]: ...

class Cancellation:
    """Cooperative cancellation flag; checked between model calls."""

    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...

def solve_native(
    spec: ProblemSpec,
    functions: Callable[[list[float]], Any],
    derivatives: Callable[[list[float]], Any],
    x0: Iterable[float],
    feasibility_tolerance: float,
    threads: int,
    cancellation: Cancellation,
    observer: Optional[Callable[[dict[str, Any]], bool]] = None,
    method: str = "dummy",
    method_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]: ...

def check_derivatives(spec: ProblemSpec, functions: Any, derivatives: Any, x: Iterable[float], steps: list[float], absolute_tolerance: float, relative_tolerance: float, max_directions: int) -> dict[str, Any]: ...
def kkt_residuals(spec: ProblemSpec, x: list[float], objective: float, constraints: list[float], gradient: list[float], jacobian_values: list[float], row_multipliers: list[float], variable_multipliers: list[float]) -> dict[str, float]: ...
